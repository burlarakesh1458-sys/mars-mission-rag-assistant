"""Pure metric functions (no models, no network) so they are unit-testable."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. Matches numpy's default method."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def gold_sources(gold: dict) -> list[tuple[str, int]]:
    """All (doc_id, page) locations that contain the gold evidence. Falls back to the
    primary gold_doc_id / gold_pages when no explicit list is given."""
    if gold.get("gold_sources"):
        return [(str(d), int(p)) for d, p in gold["gold_sources"]]
    return [(str(gold["gold_doc_id"]), int(p)) for p in gold["gold_pages"]]


def is_relevant(doc_id: str, page: int, gold: dict, level: str = "page") -> bool:
    """A retrieved chunk is relevant if it comes from a report that contains the gold evidence
    (doc level) and, at page level, from a page that contains it."""
    srcs = gold_sources(gold)
    if level == "doc":
        return doc_id in {d for d, _ in srcs}
    return (doc_id, page) in set(srcs)


def first_relevant_rank(retrieved: list[tuple[str, int]], gold: dict, level: str = "page") -> int | None:
    for rank, (doc_id, page) in enumerate(retrieved, start=1):
        if is_relevant(doc_id, page, gold, level):
            return rank
    return None


def retrieval_metrics(runs: list[tuple[list[tuple[str, int]], dict]], ks=(1, 3, 5, 10)) -> dict:
    """runs: list of (retrieved [(doc_id, page)...], gold). Returns hit@k and MRR at page and doc level."""
    out: dict = {"n": len(runs)}
    for level in ("page", "doc"):
        ranks = [first_relevant_rank(r, g, level) for r, g in runs]
        for k in ks:
            out[f"{level}_hit@{k}"] = sum(1 for x in ranks if x is not None and x <= k) / max(len(runs), 1)
        out[f"{level}_mrr@10"] = sum(1.0 / x for x in ranks if x is not None and x <= 10) / max(len(runs), 1)
    return out


def citation_correct(citations: list[dict], gold: dict, level: str = "page") -> bool:
    """True if at least one citation points to the gold report (and gold page at page level)."""
    return any(is_relevant(c["doc_id"], c["page"], gold, level) for c in citations)


def refusal_metrics(rows: list[dict]) -> dict:
    """rows: dicts with keys answerable (bool) and refused (bool)."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    tp = sum(1 for r in una if r["refused"])  # correctly declined
    fp = sum(1 for r in ans if r["refused"])  # wrongly declined
    correct = tp + sum(1 for r in ans if not r["refused"])
    return {
        "refusal_accuracy": correct / max(len(rows), 1),
        "unanswerable_declined_rate": tp / max(len(una), 1),
        "answerable_false_refusal_rate": fp / max(len(ans), 1),
        "refusal_precision": tp / max(tp + fp, 1),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
    }


def best_threshold(scores_pos: list[float], scores_neg: list[float]) -> tuple[float, float]:
    """Pick the refusal threshold t (refuse if top score < t) that maximises balanced accuracy
    between answerable (pos) and unanswerable (neg) questions. Ties -> the lowest t, which
    refuses least (we prefer answering over a false refusal). Returns (t, balanced_acc)."""
    if not scores_pos or not scores_neg:
        raise ValueError("need both classes")
    cands = sorted(set(scores_pos) | set(scores_neg))
    # thresholds midway between observed scores, plus the extremes
    ts = [cands[0] - 1e-6] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 1e-6]
    best = (ts[0], -1.0)
    for t in ts:
        tpr = sum(1 for s in scores_pos if s >= t) / len(scores_pos)  # answered
        tnr = sum(1 for s in scores_neg if s < t) / len(scores_neg)  # refused
        bal = (tpr + tnr) / 2
        if bal > best[1] + 1e-12:
            best = (t, bal)
    return best
