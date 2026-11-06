"""PDF -> cleaned page text -> page-bounded chunks with citation metadata.

Chunks never cross a page boundary, so every chunk carries exactly one page number and
an answer can cite "title, p. N" precisely. Running headers/footers (lines that repeat on
many pages of the same report) are stripped before chunking.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WS = re.compile(r"[ \t\u00a0]+")
_MANY_NL = re.compile(r"\n{2,}")
_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(of\s*\d+)?\s*$", re.I)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    page: int  # 1-based physical PDF page
    text: str
    topic: str = ""
    year: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pages(pdf_path: Path) -> list[str]:
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        return [p.get_text("text") for p in doc]


def strip_running_lines(pages: list[str], min_frac: float = 0.4) -> list[str]:
    """Remove lines that appear on >= min_frac of pages (headers, footers, distribution notices)."""
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({ln.strip() for ln in p.splitlines() if ln.strip()})
    limit = max(3, int(len(pages) * min_frac))
    common = {ln for ln, c in counts.items() if c >= limit and len(ln) < 120}
    out = []
    for p in pages:
        out.append("\n".join(ln for ln in p.splitlines() if ln.strip() not in common))
    return out


def clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [ln for ln in text.splitlines() if not _PAGE_NUM.match(ln)]
    text = "\n".join(lines)
    text = _WS.sub(" ", text)
    text = _MANY_NL.sub("\n", text)
    # join soft line wraps into prose
    text = re.sub(r"(?<![.:;!?])\n(?=[a-z(])", " ", text)
    return text.strip()


def split_words(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if start > 0 and len(piece) < overlap + 15:  # tail already covered by the overlap
            break
        out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


def chunk_document(meta: dict, pages: list[str], size: int = 220, overlap: int = 40,
                   min_words: int = 25) -> list[Chunk]:
    pages = strip_running_lines(pages)
    chunks: list[Chunk] = []
    for pno, raw in enumerate(pages, start=1):
        text = clean_text(raw)
        if len(text.split()) < min_words:
            continue
        for i, piece in enumerate(split_words(text, size, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{meta['ntrs_id']}:p{pno}:c{i}",
                doc_id=meta["ntrs_id"], title=meta["title"], url=meta["url"], page=pno,
                text=piece, topic=meta.get("topic", ""), year=meta.get("year"),
            ))
    return chunks


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_chunks(manifest_path: Path, pdf_dir: Path, out_path: Path, size: int, overlap: int) -> dict:
    manifest = load_manifest(manifest_path)
    n_chunks, n_docs, skipped = 0, 0, []
    with out_path.open("w") as f:
        for meta in manifest:
            pdf = pdf_dir / f"{meta['ntrs_id']}.pdf"
            if not pdf.exists():
                skipped.append(meta["ntrs_id"])
                continue
            chunks = chunk_document(meta, extract_pages(pdf), size, overlap)
            if not chunks:  # scanned PDF without a text layer
                skipped.append(meta["ntrs_id"])
                continue
            n_docs += 1
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
                n_chunks += 1
    return {"docs": n_docs, "chunks": n_chunks, "skipped_no_text": skipped}


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
