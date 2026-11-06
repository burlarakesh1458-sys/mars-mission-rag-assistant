"""Command line interface.

    marsrag ingest            # PDFs in data/pdfs -> data/chunks.jsonl
    marsrag index             # chunks -> Chroma (bge-small embeddings)
    marsrag search "query"    # retrieval only, no LLM
    marsrag ask "question"    # grounded answer with citations
    marsrag serve             # FastAPI + web UI on :8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import get_settings


def cmd_ingest(args) -> int:
    from .ingest import build_chunks

    s = get_settings()
    stats = build_chunks(s.manifest_path, s.pdf_dir, s.chunks_path, s.chunk_words, s.chunk_overlap)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_index(args) -> int:
    from .ingest import load_chunks
    from .retrieval import DenseIndex, Embedder

    s = get_settings()
    chunks = load_chunks(s.chunks_path)
    t0 = time.perf_counter()
    idx = DenseIndex(s.index_dir, Embedder(s.embed_model))
    idx.build(chunks, emb_cache=s.data_dir / "embeddings.npz")
    print(f"indexed {idx.count()} chunks in {time.perf_counter() - t0:.1f}s -> {s.index_dir}")
    return 0


def _retriever(rerank: bool):
    from .ingest import load_chunks
    from .retrieval import DenseIndex, Embedder, HybridRetriever, Reranker

    s = get_settings()
    return HybridRetriever(load_chunks(s.chunks_path), DenseIndex(s.index_dir, Embedder(s.embed_model)),
                           Reranker(s.rerank_model) if rerank else None, s.k_dense, s.k_bm25, s.rrf_k,
                           s.k_rerank_pool)


def cmd_search(args) -> int:
    r = _retriever(not args.no_rerank)
    for i, h in enumerate(r.retrieve(args.query, k=args.k, mode=args.mode, rerank=not args.no_rerank), 1):
        rs = f"{h.rerank_score:.2f}" if h.rerank_score is not None else "-"
        print(f"{i}. [{h.chunk.doc_id} p.{h.chunk.page}] rerank={rs} {h.chunk.title[:70]}")
        print(f"   {h.chunk.text[:200]}...")
    return 0


def cmd_ask(args) -> int:
    from .pipeline import RAGPipeline

    p = RAGPipeline(use_reranker=not args.no_rerank)
    a = p.answer(args.question)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))
        return 0
    print(a.answer)
    if a.citations:
        print("\nSources:")
        for c in a.citations:
            print(f"  [{c.n}] {c.label}")
    if a.refused:
        print(f"\n(declined: {a.refusal_reason})")
    print(f"\n({a.timings_ms['total']:.0f} ms)")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("marsrag.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marsrag", description="Mars Mission Research Assistant (NASA NTRS RAG)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    sp.add_argument("--no-rerank", action="store_true")
    sp.set_defaults(fn=cmd_search)
    ap_ask = sub.add_parser("ask")
    ap_ask.add_argument("question")
    ap_ask.add_argument("--json", action="store_true")
    ap_ask.add_argument("--no-rerank", action="store_true")
    ap_ask.set_defaults(fn=cmd_ask)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
