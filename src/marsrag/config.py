"""Central settings. Everything can be overridden with MARSRAG_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _calibrated_threshold(default: float) -> float:
    """Refusal threshold written by scripts/calibrate.py (eval/refusal_threshold.json)."""
    import json

    p = ROOT / "eval" / "refusal_threshold.json"
    try:
        return float(json.loads(p.read_text())["threshold"])
    except (OSError, KeyError, ValueError):
        return default


def _env(name: str, default):
    raw = os.environ.get(f"MARSRAG_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "data" / "index"))))
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", str(ROOT / ".cache"))))

    # chunking (whitespace tokens)
    chunk_words: int = field(default_factory=lambda: _env("CHUNK_WORDS", 220))
    chunk_overlap: int = field(default_factory=lambda: _env("CHUNK_OVERLAP", 40))

    # models (small, CPU friendly)
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # retrieval
    k_dense: int = field(default_factory=lambda: _env("K_DENSE", 30))
    k_bm25: int = field(default_factory=lambda: _env("K_BM25", 30))
    rrf_k: int = field(default_factory=lambda: _env("RRF_K", 60))
    k_rerank_pool: int = field(default_factory=lambda: _env("K_RERANK_POOL", 20))
    k_context: int = field(default_factory=lambda: _env("K_CONTEXT", 5))
    use_reranker: bool = field(default_factory=lambda: _env("USE_RERANKER", True))

    # refusal gate: if the best cross-encoder logit is below this, decline without calling the LLM.
    # Calibrated on data/calibration.jsonl (disjoint from the golden set) by scripts/calibrate.py.
    refusal_threshold: float = field(default_factory=lambda: _env("REFUSAL_THRESHOLD", _calibrated_threshold(-2.0)))

    # LLM (OpenAI-compatible). Groq first, NVIDIA Build as fallback, both serving gpt-oss-20b.
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-oss-20b"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_key_env: str = field(default_factory=lambda: _env("LLM_KEY_ENV", "GROQ_API_KEY"))
    llm_fallback_base_url: str = field(
        default_factory=lambda: _env("LLM_FALLBACK_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    llm_fallback_key_env: str = field(default_factory=lambda: _env("LLM_FALLBACK_KEY_ENV", "NVIDIA_API_KEY"))
    llm_max_tokens: int = field(default_factory=lambda: _env("LLM_MAX_TOKENS", 900))

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"


def get_settings() -> Settings:
    return Settings()
