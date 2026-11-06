"""Retrieval ablation on the golden set (no LLM): BM25 vs dense vs hybrid (RRF) vs hybrid+rerank.

    python -m marsrag.eval.run_retrieval                 # all four configs -> reports/retrieval_eval.json
    python -m marsrag.eval.run_retrieval --bm25-only     # torch-free, used by the CI eval gate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import ROOT, get_settings
from ..ingest import load_chunks
from .metrics import load_jsonl, percentile, retrieval_metrics

CONFIGS = {
    "bm25": dict(mode="bm25", rerank=False),
    "dense": dict(mode="dense", rerank=False),
    "hybrid_rrf": dict(mode="hybrid", rerank=False),
    "hybrid_rrf+rerank": dict(mode="hybrid", rerank=True),
    "dense+rerank": dict(mode="dense", rerank=True),
    "hybrid_rrf+rerank_fused": dict(mode="hybrid", rerank=True, rerank_mode="fuse"),
}


def evaluate(retriever, golden: list[dict], configs: dict, k: int = 10) -> dict:
    answerable = [g for g in golden if g["answerable"]]
    results = {}
    for name, cfg in configs.items():
        runs, lat, per_q = [], [], []
        for g in answerable:
            t0 = time.perf_counter()
            hits = retriever.retrieve(g["question"], k=k, **cfg)
            lat.append((time.perf_counter() - t0) * 1000)
            retrieved = [(h.chunk.doc_id, h.chunk.page) for h in hits]
            runs.append((retrieved, g))
            per_q.append({"id": g["id"], "top5": [f"{d}:p{p}" for d, p in retrieved[:5]]})
        m = retrieval_metrics(runs)
        m["latency_ms_p50"] = percentile(lat, 50)
        m["latency_ms_p95"] = percentile(lat, 95)
        results[name] = {"metrics": m, "per_question": per_q}
        print(f"{name:<20} page_hit@5={m['page_hit@5']:.3f} doc_hit@5={m['doc_hit@5']:.3f} "
              f"page_mrr={m['page_mrr@10']:.3f} p50={m['latency_ms_p50']:.0f}ms")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "retrieval_eval.json"))
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--min-doc-hit5", type=float, default=None, help="fail (exit 1) below this BM25 doc_hit@5")
    ap.add_argument("--configs", default="", help="comma-separated subset of configs to run")
    ap.add_argument("--merge", action="store_true", help="merge into an existing results file")
    args = ap.parse_args(argv)

    s = get_settings()
    golden = load_jsonl(args.golden)
    chunks = load_chunks(s.chunks_path)
    from ..retrieval import HybridRetriever

    if args.bm25_only:
        retriever = HybridRetriever(chunks, None, None, s.k_dense, s.k_bm25, s.rrf_k, s.k_rerank_pool)
        configs = {"bm25": CONFIGS["bm25"]}
    else:
        from ..retrieval import DenseIndex, Embedder, Reranker

        retriever = HybridRetriever(chunks, DenseIndex(s.index_dir, Embedder(s.embed_model)),
                                    Reranker(s.rerank_model), s.k_dense, s.k_bm25, s.rrf_k, s.k_rerank_pool)
        retriever.retrieve("warm up", k=3, mode="hybrid", rerank=True)
        configs = CONFIGS
    if args.configs:
        configs = {k: v for k, v in configs.items() if k in args.configs.split(",")}
    res = evaluate(retriever, golden, configs)
    if args.merge and Path(args.out).exists() and not args.bm25_only:
        old = json.loads(Path(args.out).read_text())["configs"]
        res = {**old, **res}
    payload = {"n_questions": sum(1 for g in golden if g["answerable"]), "n_chunks": len(chunks),
               "settings": {"k_dense": s.k_dense, "k_bm25": s.k_bm25, "rrf_k": s.rrf_k,
                            "k_rerank_pool": s.k_rerank_pool, "embed_model": s.embed_model,
                            "rerank_model": s.rerank_model}, "configs": res}
    if not args.bm25_only:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    if args.min_doc_hit5 is not None:
        got = res["bm25"]["metrics"]["doc_hit@5"]
        if got < args.min_doc_hit5:
            print(f"EVAL GATE FAILED: bm25 doc_hit@5 {got:.3f} < {args.min_doc_hit5}")
            return 1
        print(f"eval gate ok: bm25 doc_hit@5 {got:.3f} >= {args.min_doc_hit5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
