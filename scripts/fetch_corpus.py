"""Download a focused corpus of public NASA Technical Reports Server (NTRS) PDFs.

Uses the public NTRS search API (https://ntrs.nasa.gov/api/citations/search). For each
topic query we keep only records that are public, disseminated with a document, and
have a PDF download. Every kept file is recorded in ``data/manifest.jsonl`` with its
NTRS id, title, citation URL, PDF URL, sha256 and page count, so the corpus is fully
reproducible and attributable.

Usage:  python scripts/fetch_corpus.py --target 60 --max-pages 80 --max-mb 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

API = "https://ntrs.nasa.gov/api/citations/search"
BASE = "https://ntrs.nasa.gov"

# topic -> list of queries. Order matters: we round-robin so every topic is represented.
TOPICS: dict[str, list[str]] = {
    "radiation": [
        "space radiation risk human Mars mission",
        "galactic cosmic ray shielding Mars transit crew",
        "solar particle event protection Mars crew",
    ],
    "life_support": [
        "life support system human Mars mission ECLSS",
        "closed loop life support water oxygen recovery Mars",
        "food system long duration Mars mission",
    ],
    "spacecraft_design": [
        "Mars transit habitat design",
        "Mars transit vehicle nuclear propulsion crew",
        "Mars ascent vehicle design",
    ],
    "mission_planning": [
        "human Mars mission architecture",
        "Mars design reference architecture",
        "Evolvable Mars Campaign",
        "Mars surface operations human exploration",
    ],
    "edl": [
        "human scale Mars entry descent landing",
        "supersonic retropropulsion Mars",
        "Mars aerocapture large payload",
    ],
    "crew_health": [
        "crew health performance long duration Mars mission",
        "behavioral health isolation Mars mission crew",
    ],
    "isru": [
        "in-situ resource utilization Mars propellant oxygen",
    ],
}


def search(client: httpx.Client, q: str, size: int = 25) -> list[dict]:
    r = client.get(API, params={"q": q, "page.size": size}, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])


def pdf_link(rec: dict) -> str | None:
    for d in rec.get("downloads") or []:
        link = (d.get("links") or {}).get("pdf") or (d.get("links") or {}).get("original")
        name = (d.get("name") or "").lower()
        if link and (name.endswith(".pdf") or link.lower().endswith(".pdf")):
            return BASE + link if link.startswith("/") else link
    return None


def eligible(rec: dict) -> bool:
    return (
        rec.get("distribution") == "PUBLIC"
        and rec.get("disseminated") == "DOCUMENT_AND_METADATA"
        and bool(rec.get("downloadsAvailable"))
        and pdf_link(rec) is not None
    )


def year(rec: dict) -> int | None:
    for p in rec.get("publications") or []:
        d = p.get("publicationDate") or ""
        if d[:4].isdigit():
            return int(d[:4])
    d = rec.get("created") or rec.get("submittedDate") or ""
    for tok in d.replace("-", " ").split():
        if len(tok) == 4 and tok.isdigit():
            return int(tok)
    return None


def page_count(path: Path) -> int:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        return doc.page_count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--per-query", type=int, default=6)
    ap.add_argument("--max-pages", type=int, default=80)
    ap.add_argument("--min-pages", type=int, default=3)
    ap.add_argument("--max-mb", type=float, default=12.0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    pdf_dir = out / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    existing = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["ntrs_id"]] = rec

    client = httpx.Client(headers={"User-Agent": "marsrag-portfolio/0.1 (research; contact burlarakesh1458@gmail.com)"},
                          follow_redirects=True)
    # gather candidates per topic
    candidates: dict[str, list[tuple[str, dict]]] = {}
    seen: set[int] = set()
    for topic, queries in TOPICS.items():
        lst = []
        for q in queries:
            try:
                res = search(client, q)
            except Exception as e:  # noqa: BLE001
                print(f"search failed {q!r}: {e}", file=sys.stderr)
                continue
            n = 0
            for rec in res:
                if rec["id"] in seen or not eligible(rec):
                    continue
                seen.add(rec["id"])
                lst.append((q, rec))
                n += 1
                if n >= args.per_query:
                    break
            time.sleep(0.5)
        candidates[topic] = lst
        print(f"{topic}: {len(lst)} candidates")

    kept = dict(existing)
    # round-robin across topics
    queues = {t: list(v) for t, v in candidates.items()}
    while len(kept) < args.target and any(queues.values()):
        for topic in list(queues):
            if not queues[topic] or len(kept) >= args.target:
                continue
            q, rec = queues[topic].pop(0)
            nid = str(rec["id"])
            if nid in kept:
                continue
            url = pdf_link(rec)
            dest = pdf_dir / f"{nid}.pdf"
            try:
                if not dest.exists():
                    with client.stream("GET", url, timeout=120) as r:
                        r.raise_for_status()
                        size = int(r.headers.get("content-length") or 0)
                        if size and size > args.max_mb * 1e6:
                            print(f"skip {nid}: {size/1e6:.1f} MB")
                            continue
                        data = b""
                        for chunk in r.iter_bytes():
                            data += chunk
                            if len(data) > args.max_mb * 1e6:
                                break
                        if len(data) > args.max_mb * 1e6 or not data.startswith(b"%PDF"):
                            print(f"skip {nid}: too big or not a PDF")
                            continue
                        dest.write_bytes(data)
                pages = page_count(dest)
            except Exception as e:  # noqa: BLE001
                print(f"skip {nid}: {e}")
                dest.unlink(missing_ok=True)
                continue
            if not (args.min_pages <= pages <= args.max_pages):
                print(f"skip {nid}: {pages} pages")
                dest.unlink(missing_ok=True)
                continue
            entry = {
                "ntrs_id": nid,
                "title": " ".join(rec["title"].split()),
                "url": f"{BASE}/citations/{nid}",
                "pdf_url": url,
                "topic": topic,
                "query": q,
                "sti_type": rec.get("stiType"),
                "year": year(rec),
                "center": (rec.get("center") or {}).get("name"),
                "authors": [a["meta"]["author"]["name"] for a in rec.get("authorAffiliations") or []
                            if a.get("meta", {}).get("author", {}).get("name")][:6],
                "pages": pages,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "bytes": dest.stat().st_size,
            }
            kept[nid] = entry
            print(f"[{len(kept):>3}] {topic:<18} {nid} {pages:>3}p {entry['title'][:70]}")
            time.sleep(0.3)

    with manifest_path.open("w") as f:
        for rec in kept.values():
            f.write(json.dumps(rec) + "\n")
    print(f"manifest: {len(kept)} docs -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
