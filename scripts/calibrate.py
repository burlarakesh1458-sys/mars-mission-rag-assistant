"""Calibrate the retrieval-side refusal gate on eval/calibration.jsonl (disjoint from the
golden set): for each question take the top cross-encoder score after hybrid retrieval,
then pick the threshold that best separates answerable from unanswerable questions
(balanced accuracy; ties go to the lower threshold so we refuse less).

Writes eval/refusal_threshold.json, which config.py picks up as the default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marsrag.config import get_settings  # noqa: E402
from marsrag.eval.metrics import best_threshold, load_jsonl  # noqa: E402
from marsrag.ingest import load_chunks  # noqa: E402
from marsrag.retrieval import DenseIndex, Embedder, HybridRetriever, Reranker  # noqa: E402


def top_scores(retriever, rows):
    out = []
    for r in rows:
        hits = retriever.retrieve(r["question"], k=1, mode="hybrid", rerank=True)
        out.append(hits[0].rerank_score if hits else -99.0)
    return out


def main() -> int:
    s = get_settings()
    retriever = HybridRetriever(load_chunks(s.chunks_path), DenseIndex(s.index_dir, Embedder(s.embed_model)),
                                Reranker(s.rerank_model), s.k_dense, s.k_bm25, s.rrf_k, s.k_rerank_pool)
    cal = load_jsonl(ROOT / "eval" / "calibration.jsonl")
    pos = [r for r in cal if r["answerable"]]
    neg = [r for r in cal if not r["answerable"]]
    sp, sn = top_scores(retriever, pos), top_scores(retriever, neg)
    t, bal = best_threshold(sp, sn)
    # place the threshold a margin below the weakest answerable score when classes separate
    # perfectly, so borderline real questions are not refused (the LLM still declines on its own).
    margin_t = t
    if min(sp) > max(sn):
        margin_t = max(sn) + 0.25 * (min(sp) - max(sn))
    payload = {"threshold": round(margin_t, 3), "balanced_accuracy_on_calibration": bal,
               "answerable_scores": [round(x, 3) for x in sp], "unanswerable_scores": [round(x, 3) for x in sn],
               "n_answerable": len(sp), "n_unanswerable": len(sn), "rerank_model": s.rerank_model}
    (ROOT / "eval" / "refusal_threshold.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
