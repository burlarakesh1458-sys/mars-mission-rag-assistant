import json

import httpx
import pytest

from marsrag.llm import ChatClient, LLMError, Provider


def ok_body(text="hello"):
    return {"model": "m", "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("K1", "key-one")
    monkeypatch.setenv("K2", "key-two")
    monkeypatch.setattr("marsrag.llm.time.sleep", lambda s: None)


def client(handler, tmp_path=None, providers=None, **kw):
    providers = providers or [Provider("p1", "https://p1/v1", "K1")]
    return ChatClient("openai/gpt-oss-20b", providers, cache_path=(tmp_path / "c.sqlite") if tmp_path else None,
                      transport=httpx.MockTransport(handler), **kw)


def test_success_and_reasoning_effort_sent():
    seen = {}

    def h(req):
        seen.update(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer key-one"
        return httpx.Response(200, json=ok_body("hi"))

    r = client(h).chat([{"role": "user", "content": "x"}])
    assert r.text == "hi" and r.provider == "p1" and r.prompt_tokens == 3
    assert seen["reasoning_effort"] == "low" and seen["temperature"] == 0.0


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "slow down"})
        return httpx.Response(200, json=ok_body())

    assert client(h).chat([{"role": "user", "content": "x"}]).text == "hello"
    assert calls["n"] == 3


def test_cache_avoids_second_call(tmp_path):
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        return httpx.Response(200, json=ok_body("cached text"))

    c = client(h, tmp_path)
    c.chat([{"role": "user", "content": "same"}])
    r2 = c.chat([{"role": "user", "content": "same"}])
    assert r2.cached and r2.text == "cached text" and calls["n"] == 1


def test_falls_back_to_second_provider():
    def h(req):
        if req.url.host == "p1":
            return httpx.Response(401, json={"error": "bad key"})
        return httpx.Response(200, json=ok_body("from p2"))

    provs = [Provider("p1", "https://p1/v1", "K1"), Provider("p2", "https://p2/v1", "K2")]
    r = client(h, providers=provs).chat([{"role": "user", "content": "x"}])
    assert r.provider == "p2" and r.text == "from p2"


def test_missing_keys_raise(monkeypatch):
    monkeypatch.delenv("K1")
    with pytest.raises(LLMError, match="no key"):
        client(lambda r: httpx.Response(200, json=ok_body())).chat([{"role": "user", "content": "x"}])


def test_daily_quota_skips_retries_and_uses_model_override():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body["model"])
        if body["model"] == "openai/gpt-oss-20b":
            return httpx.Response(429, text="Rate limit reached on tokens per day (TPD)")
        return httpx.Response(200, json=ok_body("big model"))

    provs = [Provider("p1", "https://p1/v1", "K1"), Provider("p1-big", "https://p1/v1", "K1", model="openai/gpt-oss-120b")]
    r = client(h, providers=provs).chat([{"role": "user", "content": "x"}])
    assert r.text == "big model" and r.provider == "p1-big"
    assert seen == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # no retry loop on a daily quota
