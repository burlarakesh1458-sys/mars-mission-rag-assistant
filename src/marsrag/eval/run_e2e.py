"""End-to-end evaluation: answers, citation accuracy, refusal accuracy, latency.

Resumable: every finished question is appended to reports/answers_<config>.jsonl and
skipped on the next run, and LLM responses are cached on disk, so a run interrupted by a
rate limit continues where it stopped.

    python -m marsrag.eval.run_e2e --config rerank       # hybrid + reranker + refusal gate
    python -m marsrag.eval.run_e2e --config no_rerank    # hybrid only, LLM-only refusal
    python -m marsrag.eval.run_e2e --summarize           # recompute reports/e2e_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import ROOT
from .metrics import citation_correct, load_jsonl, percentile, refusal_metrics

REPORTS = ROOT / "reports"


def answers_path(config: str) -> Path:
    return REPORTS / f"answers_{config}.jsonl"


def run(config: str, golden: list[dict], limit: int | None = None, patience_s: float = 0) -> None:
    from ..pipeline import RAGPipeline

    # rerank: contexts ordered by cross-encoder; rerank_fused: RRF(BM25, dense, cross-encoder);
    # no_rerank: plain hybrid RRF and no retrieval-side refusal gate (LLM-only refusal)
    pipe = RAGPipeline(use_reranker=config != "no_rerank",
                       rerank_mode="fuse" if config == "rerank_fused" else "replace")
    out = answers_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
    todo = [g for g in golden if g["id"] not in done][:limit]
    print(f"{config}: {len(done)} done, {len(todo)} to go")
    pipe.retriever.retrieve("warm up", k=3, mode="hybrid", rerank=pipe.use_reranker)
    deadline = time.time() + patience_s
    with out.open("a") as f:
        while todo:
            skipped = []
            for g in todo:
                try:
                    a = pipe.answer(g["question"])
                except Exception as e:  # noqa: BLE001 - every provider rate limited: skip, retry next pass
                    print(f"  {g['id']} skipped this pass: {str(e)[:120]}", flush=True)
                    skipped.append(g)
                    continue
                row = {"id": g["id"], "answerable": g["answerable"], "question": g["question"],
                       "gold_doc_id": g.get("gold_doc_id"), "gold_pages": g.get("gold_pages"),
                       "gold_sources": g.get("gold_sources"),
                       "reference": g.get("reference"), **a.to_dict(), "ts": time.time()}
                f.write(json.dumps(row) + "\n")
                f.flush()
                tag = "REFUSED" if a.refused else f"{len(a.citations)} cites"
                print(f"  {g['id']} {tag} {a.timings_ms['total']:.0f}ms", flush=True)
            todo = skipped
            if todo and time.time() + 120 < deadline:
                print(f"{len(todo)} questions waiting for LLM quota; next pass in 120s", flush=True)
                time.sleep(120)
            elif todo:
                print(f"stopped with {len(todo)} unanswered (no LLM quota): {[g['id'] for g in todo]}")
                break


def summarize_config(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    golds = [{"gold_doc_id": r["gold_doc_id"], "gold_pages": r["gold_pages"],
              "gold_sources": r.get("gold_sources")} for r in answered]
    cit_doc = [citation_correct(r["citations"], g, "doc") for r, g in zip(answered, golds)]
    cit_page = [citation_correct(r["citations"], g, "page") for r, g in zip(answered, golds)]
    with_cites = [r for r in answered if r["citations"]]
    # latency: only rows where the LLM was actually called live (not served from cache) or gated
    live = [r for r in rows if not r.get("llm", {}).get("cached", False)]
    def no_wait(r):  # retrieval + the successful LLM request only (drops rate-limit sleeps / failed providers)
        llm = r.get("llm") or {}
        return r["timings_ms"]["retrieve"] + llm.get("provider_latency_ms", 0.0) if llm else r["timings_ms"]["total"]

    m = {
        "n": len(rows),
        **refusal_metrics(rows),
        "answered_answerable": len(answered),
        "citation_present_rate": len(with_cites) / max(len(answered), 1),
        "citation_accuracy_doc": sum(cit_doc) / max(len(answered), 1),
        "citation_accuracy_page": sum(cit_page) / max(len(answered), 1),
        "citation_accuracy_doc_over_all_answerable": sum(cit_doc) / max(len(ans), 1),
        "latency_n_live": len(live),
        "latency_total_ms_p50": percentile([r["timings_ms"]["total"] for r in live], 50),
        "latency_total_ms_p95": percentile([r["timings_ms"]["total"] for r in live], 95),
        "latency_total_excl_ratelimit_ms_p50": percentile([no_wait(r) for r in live], 50),
        "latency_total_excl_ratelimit_ms_p95": percentile([no_wait(r) for r in live], 95),
        "latency_retrieve_ms_p50": percentile([r["timings_ms"]["retrieve"] for r in rows], 50),
        "latency_retrieve_ms_p95": percentile([r["timings_ms"]["retrieve"] for r in rows], 95),
        "latency_llm_ms_p50": percentile([r["timings_ms"]["llm"] for r in live if r["timings_ms"]["llm"] > 0], 50),
        "latency_llm_ms_p95": percentile([r["timings_ms"]["llm"] for r in live if r["timings_ms"]["llm"] > 0], 95),
        "refusal_reasons": {},
        "answered_by": {},
    }
    for r in rows:
        llm = r.get("llm") or {}
        key = llm.get("model", "?") if llm else "retrieval_gate (no LLM call)"
        m["answered_by"][key] = m["answered_by"].get(key, 0) + 1
    for r in rows:
        if r["refused"]:
            key = (r["refusal_reason"] or "").split("(")[0]
            m["refusal_reasons"][key] = m["refusal_reasons"].get(key, 0) + 1
    return m


def summarize(configs: list[str]) -> dict:
    out = {}
    for c in configs:
        p = answers_path(c)
        if p.exists():
            out[c] = summarize_config(load_jsonl(p))
    (REPORTS / "e2e_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("--config", choices=["rerank", "rerank_fused", "no_rerank"], default="rerank_fused")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--patience-s", type=float, default=0,
                    help="keep re-trying rate-limited questions in passes for up to this many seconds")
    args = ap.parse_args(argv)
    if not args.summarize:
        run(args.config, load_jsonl(args.golden), args.limit, args.patience_s)
    summarize(["rerank_fused", "rerank", "no_rerank"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
