import pytest

from marsrag.ingest import chunk_document, clean_text, split_words, strip_running_lines


def test_split_short_text_is_single_chunk():
    assert split_words("a b c", 10, 2) == ["a b c"]


def test_split_empty_text():
    assert split_words("   ", 10, 2) == []


def test_split_overlap_and_coverage():
    words = [f"w{i}" for i in range(100)]
    parts = split_words(" ".join(words), 40, 10)
    assert parts[0].split()[0] == "w0"
    assert parts[1].split()[0] == "w30"  # step = size - overlap
    assert parts[-1].split()[-1] == "w99"  # nothing lost at the end
    assert all(len(p.split()) <= 40 for p in parts)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_split_rejects_bad_params(size, overlap):
    with pytest.raises(ValueError):
        split_words("a b c", size, overlap)


def test_clean_text_joins_hyphenated_words_and_drops_page_numbers():
    raw = "radia-\ntion shielding\n12\nis heavy."
    out = clean_text(raw)
    assert "radiation" in out
    assert "\n12\n" not in out and not out.startswith("12")


def test_strip_running_lines_removes_repeated_headers():
    pages = [f"NASA/TM-2020-1234\nUnique content number {i} about Mars." for i in range(6)]
    out = strip_running_lines(pages)
    assert all("NASA/TM-2020-1234" not in p for p in out)
    assert "Unique content number 3" in out[3]


def test_chunks_never_cross_pages_and_carry_metadata():
    meta = {"ntrs_id": "2020", "title": "Test Report", "url": "https://ntrs.nasa.gov/citations/2020", "topic": "edl"}
    pages = [" ".join(f"alpha{i}" for i in range(60)), "too short", " ".join(f"beta{i}" for i in range(300))]
    chunks = chunk_document(meta, pages, size=100, overlap=20, min_words=25)
    assert {c.page for c in chunks} == {1, 3}  # page 2 is below min_words
    assert all(("alpha" in c.text) != ("beta" in c.text) for c in chunks)
    assert chunks[0].chunk_id == "2020:p1:c0"
    assert all(c.url.endswith("2020") and c.title == "Test Report" for c in chunks)
