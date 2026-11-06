"""Minimal OpenAI-compatible chat client with on-disk cache, backoff and provider fallback.

Groq's free tier is ~8k tokens/minute, so every response is cached in SQLite keyed by
(model, messages, params). Re-running an evaluation after a rate-limit only pays for the
calls that never finished. When the primary provider keeps returning 429 the client
switches to the fallback provider (NVIDIA Build serves the same gpt-oss-20b weights).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    raw: dict = field(default_factory=dict, repr=False)


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, json.dumps(value)))
            self._db.commit()


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str
    model: str | None = None  # override the client's model for this provider (fallback chain)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.key_env)


class ChatClient:
    def __init__(self, model: str, providers: list[Provider], cache_path: Path | None = None,
                 temperature: float = 0.0, max_tokens: int = 900, timeout_s: float = 90,
                 max_retries: int = 8, min_fit_tokens: int = 220, transport: httpx.BaseTransport | None = None):
        if not providers:
            raise ValueError("need at least one provider")
        self.model = model
        self.providers = providers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.min_fit_tokens = min_fit_tokens
        self.cache = ResponseCache(cache_path) if cache_path else None
        self._http = httpx.Client(timeout=timeout_s, transport=transport)

    def _key(self, messages: list[dict], max_tokens: int) -> str:
        blob = json.dumps({"m": self.model, "msg": messages, "t": self.temperature, "mt": max_tokens},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], max_tokens: int | None = None, use_cache: bool = True) -> LLMResponse:
        max_tokens = max_tokens or self.max_tokens
        key = self._key(messages, max_tokens)
        if use_cache and self.cache and (hit := self.cache.get(key)):
            return LLMResponse(text=hit["text"], model=hit["model"], provider=hit["provider"],
                               prompt_tokens=hit.get("pt", 0), completion_tokens=hit.get("ct", 0),
                               latency_s=0.0, cached=True)
        errors = []
        for prov in self.providers:
            if not prov.api_key:
                errors.append(f"{prov.name}: no key in ${prov.key_env}")
                continue
            try:
                resp = self._call(prov, messages, max_tokens)
            except LLMError as e:
                errors.append(f"{prov.name}: {e}")
                log.warning("provider %s failed: %s", prov.name, e)
                continue
            if self.cache and resp.text.strip():
                self.cache.put(key, {"text": resp.text, "model": resp.model, "provider": resp.provider,
                                     "pt": resp.prompt_tokens, "ct": resp.completion_tokens})
            return resp
        raise LLMError("; ".join(errors))

    def _call(self, prov: Provider, messages: list[dict], max_tokens: int) -> LLMResponse:
        model = prov.model or self.model
        body = {"model": model, "messages": messages, "temperature": self.temperature,
                "max_tokens": max_tokens}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
        headers = {"Authorization": f"Bearer {prov.api_key}", "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post(f"{prov.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"network: {e}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                if not (msg.get("content") or "").strip() and data["choices"][0].get("finish_reason") == "length":
                    # reasoning used the whole budget: an empty answer is not a refusal, try the next provider
                    raise LLMError("empty completion (finish_reason=length)")
                return LLMResponse(text=(msg.get("content") or "").strip(), model=data.get("model", model),
                                   provider=prov.name, prompt_tokens=usage.get("prompt_tokens", 0),
                                   completion_tokens=usage.get("completion_tokens", 0),
                                   latency_s=time.perf_counter() - t0, raw=data)
            if r.status_code == 413 and (m := re.search(r"Limit (\d+), Requested (\d+)", r.text)):
                # request bigger than the model's tokens-per-minute cap (e.g. 2,000 TPM on gpt-oss-safeguard-20b):
                # Groq counts prompt + max_tokens, so shrink max_tokens to fit if a useful budget is left
                fit = body["max_tokens"] - (int(m.group(2)) - int(m.group(1))) - 16
                if fit >= self.min_fit_tokens and attempt < self.max_retries:
                    log.info("%s: TPM cap, retrying with max_tokens=%d", prov.name, fit)
                    body["max_tokens"] = fit
                    continue
                raise LLMError(f"request exceeds TPM cap even with max_tokens={fit}: {r.text[:200]}")
            if r.status_code == 429 and "per day" in r.text:
                # daily quota: waiting won't help within this run, move on to the next provider
                raise LLMError(f"daily quota exhausted: {r.text[:200]}")
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                ra = r.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else delay
                if m := re.search(r"try again in (\d+(?:\.\d+)?)s", r.text):
                    wait = max(wait, float(m.group(1)) + 1)
                wait = min(wait, 60)
                log.info("%s %s, sleeping %.1fs: %s", prov.name, r.status_code, wait, r.text[:200])
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        raise LLMError("retries exhausted")


def default_client(settings) -> ChatClient:
    providers = [Provider("groq", settings.llm_base_url, settings.llm_key_env),
                 # same weights on a second provider, then the larger sibling on Groq (separate daily quota)
                 Provider("nvidia", settings.llm_fallback_base_url, settings.llm_fallback_key_env),
                 Provider("groq-120b", settings.llm_base_url, settings.llm_key_env, model="openai/gpt-oss-120b"),
                 Provider("groq-qwen", settings.llm_base_url, settings.llm_key_env, model="qwen/qwen3.8-27b"),
                 # gpt-oss-safeguard-20b: same gpt-oss-20b base, separate Groq daily quota
                 Provider("groq-safeguard-20b", settings.llm_base_url, settings.llm_key_env,
                          model="openai/gpt-oss-safeguard-20b")]
    return ChatClient(settings.llm_model, providers, cache_path=settings.cache_dir / "llm_cache.sqlite",
                      max_tokens=settings.llm_max_tokens)
