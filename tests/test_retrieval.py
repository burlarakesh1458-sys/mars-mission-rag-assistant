import pytest
from conftest import FakeDense, FakeReranker

from marsrag.retrieval import BM25Index, HybridRetriever, rrf_fuse, tokenize


def test_tokenize_drops_stopwords_and_keeps_numbers():
    toks = tokenize("What is the 4:1 ratio of H2 to CO2?")
    assert "the" not in toks and "what" not in toks
    assert "h2" in toks and "co2" in toks and "4" in toks


def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_between_retrievers():
    order = [cid for cid, _ in rrf_fuse([["x", "y", "z"], ["y", "x", "w"]])]
    assert order[:2] == ["x", "y"] or order[:2] == ["y", "x"]
    assert order.index("z") > 1 and order.index("w") > 1


def test_bm25_finds_lexical_match(chunks):
    idx = BM25Index(chunks)
    top = idx.search("Sabatier reaction hydrogen ratio", k=2)
    assert top[0][0] == "200:p3:c0"


def test_bm25_mode_without_dense(chunks):
    r = HybridRetriever(chunks, dense=None, reranker=None)
    hits = r.retrieve("storm shelter solar particle events", k=1, mode="bm25")
    assert hits[0].chunk.chunk_id == "100:p2:c0"
    assert hits[0].sources["bm25_rank"] == 1


def test_hybrid_mode_fuses_both(chunks):
    r = HybridRetriever(chunks, dense=FakeDense(chunks), reranker=None)
    hits = r.retrieve("supersonic retropropulsion human-scale payloads", k=3, mode="hybrid")
    assert hits[0].chunk.doc_id == "300"
    assert hits[0].sources["bm25_rank"] is not None and hits[0].sources["dense_rank"] is not None


def test_reranker_reorders_and_sets_scores(chunks):
    # reranker that always prefers the crew-isolation chunk
    rr = FakeReranker(lambda q, t: 10.0 if "isolation" in t else 0.0)
    r = HybridRetriever(chunks, dense=FakeDense(chunks), reranker=rr)
    hits = r.retrieve("radiation dose Mars transit", k=3, mode="hybrid", rerank=True)
    assert hits[0].chunk.doc_id == "400"
    assert all(h.rerank_score is not None for h in hits)


def test_rerank_can_be_disabled_per_call(chunks):
    r = HybridRetriever(chunks, dense=FakeDense(chunks), reranker=FakeReranker())
    hits = r.retrieve("radiation dose", k=2, mode="hybrid", rerank=False)
    assert all(h.rerank_score is None for h in hits)


def test_unknown_mode_raises(chunks):
    r = HybridRetriever(chunks, dense=None, reranker=None)
    with pytest.raises(ValueError):
        r.retrieve("x", mode="magic")


def test_rerank_fuse_mode_only_nudges_order(chunks):
    # a reranker that loves the isolation chunk cannot override strong BM25 + dense agreement
    rr = FakeReranker(lambda q, t: 10.0 if "isolation" in t else 0.0)
    r = HybridRetriever(chunks, dense=FakeDense(chunks), reranker=rr)
    replace = r.retrieve("Sabatier reaction hydrogen carbon dioxide", k=3, rerank=True, rerank_mode="replace")
    fused = r.retrieve("Sabatier reaction hydrogen carbon dioxide", k=3, rerank=True, rerank_mode="fuse")
    assert replace[0].chunk.doc_id == "400"
    assert fused[0].chunk.doc_id == "200"
    assert all(h.rerank_score is not None for h in fused)


def test_unknown_rerank_mode_raises(chunks):
    r = HybridRetriever(chunks, dense=FakeDense(chunks), reranker=FakeReranker())
    with pytest.raises(ValueError):
        r.retrieve("x y", rerank=True, rerank_mode="bogus")
