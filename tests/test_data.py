"""Integrity checks on the committed corpus manifest, chunks and evaluation sets."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def jl(name):
    return [json.loads(x) for x in (ROOT / name).read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def manifest():
    return jl("data/manifest.jsonl")


@pytest.fixture(scope="module")
def chunk_locs():
    return {(c["doc_id"], c["page"]) for c in jl("data/chunks.jsonl")}


def test_manifest_is_real_ntrs_and_unique(manifest):
    assert 40 <= len(manifest) <= 80
    ids = [m["ntrs_id"] for m in manifest]
    assert len(ids) == len(set(ids))
    for m in manifest:
        assert m["url"] == f"https://ntrs.nasa.gov/citations/{m['ntrs_id']}"
        assert m["pdf_url"].startswith("https://ntrs.nasa.gov/api/citations/")
        assert len(m["sha256"]) == 64 and m["pages"] > 0


def test_corpus_covers_all_topics(manifest):
    topics = {m["topic"] for m in manifest}
    assert {"radiation", "life_support", "spacecraft_design", "mission_planning", "edl"} <= topics


def test_golden_set_shape():
    g = jl("eval/golden.jsonl")
    assert len(g) == 50
    assert sum(not x["answerable"] for x in g) == 10
    assert len({x["id"] for x in g}) == 50
    assert any("Moon" in x["question"] for x in g if not x["answerable"])


def test_golden_gold_locations_exist_in_corpus(chunk_locs):
    for x in jl("eval/golden.jsonl"):
        if x["answerable"]:
            assert x["reference"]
            assert all((d, p) in chunk_locs for d, p in x["gold_sources"]), x["id"]


def test_calibration_disjoint_from_golden():
    gq = {x["question"] for x in jl("eval/golden.jsonl")}
    cq = {x["question"] for x in jl("eval/calibration.jsonl")}
    assert not gq & cq
