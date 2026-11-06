"""Write docs/CORPUS.md: every NTRS report in the corpus with id, title, topic, year, pages and link."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(x) for x in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if x.strip()]
rows.sort(key=lambda r: (r["topic"], -(r["year"] or 0)))
out = ["# Corpus: NASA Technical Reports Server documents", "",
       f"{len(rows)} public PDFs, {sum(r['pages'] for r in rows)} pages, fetched with `scripts/fetch_corpus.py` "
       "from the NTRS search API (https://ntrs.nasa.gov/api/citations/search). SHA-256 of every file is in "
       "`data/manifest.jsonl`.", "",
       "| # | NTRS id | Title | Topic | Year | Type | Pages |", "|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    title = r["title"].replace("|", "/")
    out.append(f"| {i} | [{r['ntrs_id']}]({r['url']}) | {title} | {r['topic']} | {r['year'] or ''} | "
               f"{r['sti_type'] or ''} | {r['pages']} |")
(ROOT / "docs" / "CORPUS.md").write_text("\n".join(out) + "\n")
print(f"wrote docs/CORPUS.md ({len(rows)} rows)")
