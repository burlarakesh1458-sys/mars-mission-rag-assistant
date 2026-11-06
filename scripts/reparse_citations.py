"""Re-run the citation validator over saved answers (no retrieval, no LLM call).

Used once after adding `normalize_citations` (gpt-oss models sometimes write 【2】 instead of
[2]): rows answered before the fix are re-parsed with exactly the code the pipeline now runs,
using the excerpts that were in the prompt (stored in each row's `contexts`). Timings, model
and answer text are kept; only `answer` markers and `citations` change.

    python scripts/reparse_citations.py reports/answers_rerank_fused.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marsrag.ingest import Chunk  # noqa: E402
from marsrag.pipeline import normalize_citations, parse_citations, strip_invalid  # noqa: E402
from marsrag.retrieval import Hit  # noqa: E402


def reparse(row: dict) -> tuple[dict, bool]:
    if row["refused"]:
        return row, False
    hits = []
    for c in row["contexts"]:
        ch = Chunk(chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c["title"], page=c["page"],
                   url=c["url"], text=c["text"])
        hits.append(Hit(ch, c.get("rerank_score") or 0.0))
    text = normalize_citations(row["answer"])
    cits, invalid = parse_citations(text, hits)
    new = dict(row, answer=strip_invalid(text, invalid),
               citations=[{"n": c.n, "doc_id": c.doc_id, "title": c.title, "page": c.page, "url": c.url,
                           "chunk_id": c.chunk_id, "label": c.label} for c in cits])
    return new, new["citations"] != row["citations"]


def main(path: str) -> int:
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    out, changed = [], []
    for r in rows:
        nr, ch = reparse(r)
        out.append(nr)
        if ch:
            changed.append(r["id"])
    p.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"re-parsed {len(rows)} rows; citations changed for: {changed or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
