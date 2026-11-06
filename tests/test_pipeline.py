from conftest import FakeDense, FakeLLM, FakeReranker

from marsrag.config import Settings
from marsrag.pipeline import (
    REFUSAL_MESSAGE,
    REFUSAL_TOKEN,
    RAGPipeline,
    build_context,
    is_refusal,
    normalize_citations,
    parse_citations,
    strip_invalid,
)
from marsrag.retrieval import Hit, HybridRetriever


def make_pipe(chunks, llm, threshold=0.5, rerank=True):
    s = Settings()
    s.k_context = 3
    r = HybridRetriever(chunks, FakeDense(chunks), FakeReranker() if rerank else None)
    return RAGPipeline(settings=s, retriever=r, llm=llm, use_reranker=rerank, refusal_threshold=threshold)


def test_parse_citations_valid_invalid_and_dedup(chunks):
    hits = [Hit(c, 1.0) for c in chunks[:3]]
    cits, invalid = parse_citations("A [1]. B [3][1]. C [9].", hits)
    assert [c.n for c in cits] == [1, 3]
    assert cits[1].doc_id == "200" and cits[1].page == 3
    assert invalid == [9]


def test_gpt_oss_native_citation_markers_are_normalized(chunks):
    text = normalize_citations("Fact one【2】. Fact two【1†L2-L4】.")
    assert text == "Fact one[2]. Fact two[1]."
    cits, invalid = parse_citations(text, [Hit(c, 1.0) for c in chunks[:3]])
    assert [c.n for c in cits] == [2, 1] and invalid == []


def test_citation_label_has_title_page_and_link(chunks):
    cits, _ = parse_citations("x [1]", [Hit(chunks[2], 1.0)])
    assert cits[0].label == "Report 200, p. 3 (https://ntrs.nasa.gov/citations/200)"


def test_strip_invalid_removes_hallucinated_markers():
    assert strip_invalid("fact [1] other [7].", [7]) == "fact [1] other ."


def test_is_refusal():
    assert is_refusal(REFUSAL_TOKEN)
    assert is_refusal("  ")
    assert not is_refusal("The ratio is 4:1 [1].")


def test_build_context_numbers_sources(chunks):
    ctx = build_context([Hit(chunks[0], 1.0), Hit(chunks[2], 1.0)])
    assert ctx.startswith("[1] Report 100 (NTRS 100, page 1)")
    assert "[2] Report 200 (NTRS 200, page 3)" in ctx


def test_answer_with_citations(chunks):
    llm = FakeLLM("Hydrogen and CO2 combine in a 4:1 ratio [1].")
    a = make_pipe(chunks, llm, threshold=-100).answer("Sabatier reaction hydrogen carbon dioxide ratio")
    assert not a.refused and llm.calls == 1
    assert a.citations[0].doc_id == "200"
    assert set(a.timings_ms) == {"retrieve", "llm", "total"}


def test_low_relevance_gate_refuses_without_calling_llm(chunks):
    llm = FakeLLM()
    a = make_pipe(chunks, llm, threshold=5.0).answer("Who won the football world cup?")
    assert a.refused and a.answer == REFUSAL_MESSAGE
    assert a.refusal_reason.startswith("low_relevance")
    assert llm.calls == 0


def test_llm_decline_becomes_refusal(chunks):
    a = make_pipe(chunks, FakeLLM(REFUSAL_TOKEN), threshold=-100).answer("Sabatier hydrogen")
    assert a.refused and a.refusal_reason == "llm_declined" and a.citations == []


def test_no_reranker_means_no_gate(chunks):
    llm = FakeLLM("Shelter [1].")
    a = make_pipe(chunks, llm, threshold=1e9, rerank=False).answer("solar particle storm shelter")
    assert not a.refused and llm.calls == 1


def test_to_dict_is_json_ready(chunks):
    import json

    a = make_pipe(chunks, FakeLLM("x [1]"), threshold=-100).answer("radiation shielding")
    d = json.loads(json.dumps(a.to_dict()))
    assert d["citations"][0]["label"].startswith("Report")
