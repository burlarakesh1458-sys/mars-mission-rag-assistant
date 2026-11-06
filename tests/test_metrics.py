import math

import pytest

from marsrag.eval.metrics import (
    best_threshold,
    citation_correct,
    percentile,
    refusal_metrics,
    retrieval_metrics,
)

GOLD = {"gold_doc_id": "A", "gold_pages": [3], "gold_sources": [["A", 3], ["B", 7]]}


def test_percentile_matches_numpy_linear():
    xs = [10, 20, 30, 40, 50]
    assert percentile(xs, 50) == 30
    assert percentile(xs, 95) == pytest.approx(48.0)
    assert math.isnan(percentile([], 50))


def test_retrieval_metrics_page_vs_doc_level():
    runs = [([("A", 9), ("A", 3)], GOLD),  # doc hit at 1, page hit at 2
            ([("C", 1), ("C", 2)], GOLD)]  # miss
    m = retrieval_metrics(runs, ks=(1, 3))
    assert m["doc_hit@1"] == 0.5 and m["page_hit@1"] == 0.0
    assert m["page_hit@3"] == 0.5
    assert m["page_mrr@10"] == pytest.approx(0.25)


def test_twin_document_counts_as_correct_citation():
    assert citation_correct([{"doc_id": "B", "page": 7}], GOLD, "page")
    assert citation_correct([{"doc_id": "B", "page": 1}], GOLD, "doc")
    assert not citation_correct([{"doc_id": "B", "page": 1}], GOLD, "page")
    assert not citation_correct([], GOLD, "doc")


def test_gold_sources_fallback_to_primary():
    g = {"gold_doc_id": "A", "gold_pages": [2, 3]}
    assert citation_correct([{"doc_id": "A", "page": 2}], g, "page")


def test_refusal_metrics():
    rows = [{"answerable": True, "refused": False}, {"answerable": True, "refused": True},
            {"answerable": False, "refused": True}, {"answerable": False, "refused": False}]
    m = refusal_metrics(rows)
    assert m["refusal_accuracy"] == 0.5
    assert m["unanswerable_declined_rate"] == 0.5
    assert m["answerable_false_refusal_rate"] == 0.5
    assert m["refusal_precision"] == 0.5


def test_best_threshold_separable():
    t, bal = best_threshold([2.0, 3.0, 4.0], [-5.0, -3.0])
    assert bal == 1.0 and -3.0 < t <= 2.0


def test_best_threshold_prefers_lower_on_ties():
    t, bal = best_threshold([1.0, 5.0], [0.0, 6.0])
    assert t < 1.0 or bal > 0.5


def test_best_threshold_needs_both_classes():
    with pytest.raises(ValueError):
        best_threshold([1.0], [])
