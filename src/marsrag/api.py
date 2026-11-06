"""FastAPI service: POST /ask, GET /search, GET /health, GET /reports, and the web UI at /."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import ROOT, get_settings

log = logging.getLogger("marsrag.api")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Mars Mission Research Assistant", version="0.1.0",
              description="Hybrid-retrieval RAG over NASA Technical Reports Server PDFs")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class CitationOut(BaseModel):
    n: int
    doc_id: str
    title: str
    page: int
    url: str
    label: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    contexts: list[dict]
    timings_ms: dict


@lru_cache(maxsize=1)
def get_pipeline():
    from .pipeline import RAGPipeline

    return RAGPipeline()


@app.get("/health")
def health() -> dict:
    s = get_settings()
    n_docs = sum(1 for _ in s.manifest_path.open()) if s.manifest_path.exists() else 0
    return {"status": "ok", "documents": n_docs, "chunks_file": s.chunks_path.exists(),
            "index_dir": s.index_dir.exists(), "llm_model": s.llm_model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        a = get_pipeline().answer(req.question.strip())
    except Exception as e:  # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(status_code=503, detail=f"answering failed: {e}") from e
    d = a.to_dict()
    return AskResponse(question=d["question"], answer=d["answer"], refused=d["refused"],
                       refusal_reason=d["refusal_reason"], citations=d["citations"],
                       contexts=[{k: c[k] for k in ("doc_id", "title", "page", "url", "rerank_score")}
                                 | {"snippet": c["text"][:300]} for c in d["contexts"]],
                       timings_ms=d["timings_ms"])


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)) -> dict:
    p = get_pipeline()
    hits = p.retriever.retrieve(q, k=k, mode="hybrid", rerank=p.use_reranker, rerank_mode=p.rerank_mode)
    return {"query": q, "hits": [{"doc_id": h.chunk.doc_id, "title": h.chunk.title, "page": h.chunk.page,
                                  "url": h.chunk.url, "rerank_score": h.rerank_score,
                                  "snippet": h.chunk.text[:300]} for h in hits]}


@app.get("/reports")
def reports() -> dict:
    """The NTRS reports in the corpus (data provenance)."""
    s = get_settings()
    if not s.manifest_path.exists():
        return {"reports": []}
    rows = [json.loads(line) for line in s.manifest_path.open() if line.strip()]
    return {"reports": [{k: r.get(k) for k in ("ntrs_id", "title", "url", "topic", "year", "pages")} for r in rows]}


@app.get("/eval")
def eval_results() -> dict:
    out = {}
    for name in ("retrieval_eval", "e2e_eval", "ragas_eval"):
        p = ROOT / "reports" / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if name == "retrieval_eval":
                data = {k: v["metrics"] for k, v in data.get("configs", {}).items()}
            out[name] = data
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
