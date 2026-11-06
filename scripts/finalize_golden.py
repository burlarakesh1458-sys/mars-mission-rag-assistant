"""Step 2 of the golden set: apply the manual review of eval/candidates.jsonl, add
hand-written questions, and write eval/golden.jsonl + eval/calibration.jsonl.

Gold locations: every (report, page) whose chunk text contains the evidence string is a
gold source (some reports exist twice in NTRS, e.g. a paper and its slide deck), so a
correct citation of the twin document is not counted as a miss. The candidate's own
source page is always a gold source.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marsrag.ingest import load_chunks  # noqa: E402

# --- review of LLM-drafted candidates (index in eval/candidates.jsonl) -----------------
# KEEP: used in the golden set (optionally with an edited question / reference answer).
KEEP: dict[int, dict] = {
    1: {}, 2: {}, 3: {}, 5: {},
    4: {"answer": "A 3-meter diameter metallic cylinder and a large 8-meter diameter softgoods inflatable compartment."}, 7: {}, 8: {}, 9: {}, 10: {}, 12: {}, 13: {}, 14: {}, 15: {},
    16: {}, 17: {}, 18: {}, 20: {}, 21: {}, 25: {}, 27: {}, 28: {}, 29: {}, 31: {}, 32: {}, 33: {},
    34: {}, 36: {}, 38: {}, 39: {}, 40: {}, 42: {}, 43: {},
}
# moved to the calibration set (keeps the golden set clean of anything used for tuning)
TO_CALIBRATION = [19, 23, 35, 41]
# all other candidates were model-flagged "skip" (reference lists, author bios, etc.)

# --- hand-written answerable questions (written by reading the cited page) -------------
HANDWRITTEN = [
    {"question": "In the DRA 5.0 power requirements study, what total power level is needed for the ISRU phase and what drives it?",
     "reference": "About 92 kWe, driven mainly by producing oxygen for the ascent stage (8 hours per sol for 300 sols).",
     "doc_id": "20120012929", "page": 4, "evidence": "The total power level of 92 kWe for this ISRU phase"},
    {"question": "How much water per year can the Sabatier Carbon Dioxide Reduction Assembly produce on the ISS?",
     "reference": "About 2000 lb of water per year (quoted as 2040 lb/year, equivalent to two shuttle flights).",
     "doc_id": "20100033195", "page": 1, "evidence": "the Sabatier CRA can produce as much a 2000 lb/year water"},
    {"question": "What four variants of the Common Habitat for Moon and Mars exploration came out of its trade study?",
     "reference": "Four Crew Horizontal, Four Crew Vertical, Eight Crew Horizontal and Eight Crew Vertical configurations.",
     "doc_id": "20205001678", "page": 1, "evidence": "Four Crew Horizontal Configuration, Four Crew Vertical Configuration"},
    {"question": "What ion thruster lifetime was assumed in the nuclear electric propulsion study for piloted Mars missions, and what did that imply?",
     "reference": "10,000 hours, so the thrusters (and propellant) had to be replaced after each piloted trip to Mars.",
     "doc_id": "19940024330", "page": 13, "evidence": "requiring that the thrusters (and propellant) replaced after each piloted trip to Mars"},
    {"question": "Under NASA planetary protection planning for human Mars missions in DRA 5.0, what are regions that have been demonstrated to be safe for humans called?",
     "reference": "Zones of Minimum Biological Risk (ZMBRs); astronauts will only be allowed in areas demonstrated to be safe.",
     "doc_id": "20090012109", "page": 28, "evidence": "Zones of Minimum Biological Risk"},
    {"question": "In the Phobos-Deimos mission study built on DRA 5.0, how many crew does the surface habitat lander carry and how long can it support them on the Mars surface?",
     "reference": "It transports 6 crew from Mars orbit to the surface and supports them for up to 550 days.",
     "doc_id": "20110005558", "page": 23, "evidence": "Supports the crew for up to 550 days on the surface of Mars"},
    {"question": "In the DRA 5.0 power requirements study, what speed and power range are estimated for the rover whose range is limited by suit power and walk-back distance?",
     "reference": "Around 10 km per hour, in the 1-2 kilowatt range.",
     "doc_id": "20120012929", "page": 4, "evidence": "around 10 km per hour and in the 1-2 kilowatt range"},
    {"question": "What class of mission and deployment strategy does the Mars Design Reference Architecture 5.0 flight sequence use?",
     "reference": "Conjunction-class (long-stay) missions with fast interplanetary transits, with a long surface stay and forward deployment: surface habitat, exploration gear and the Mars ascent vehicle are pre-deployed before the crew leaves Earth.",
     "doc_id": "20090012109", "page": 14, "evidence": "Conjunction class missions (long-stay) with fast inter"},
]

UNANSWERABLE = [
    ("When did someone last go to the Moon?",
     "Video's canonical decline example. The corpus mentions Apollo 17's launch date in passing but never states which crewed lunar mission was the last."),
    ("What is the price of a ticket for a SpaceX Starship flight to Mars?", "Commercial pricing is not in NASA reports."),
    ("Who has been selected as commander of the first crewed Mars landing?", "No such selection exists in the corpus."),
    ("What is the population of Mumbai?", "Off-topic."),
    ("How many goals did Lionel Messi score at the 2022 FIFA World Cup?", "Off-topic."),
    ("On what date did the Perseverance rover land on Mars?", "Perseverance is mentioned only by name; no landing date."),
    ("How much did the James Webb Space Telescope cost to build?", "Not in corpus."),
    ("What was Lockheed Martin's annual revenue in 2023?", "Company named only as an author affiliation/contractor."),
    ("What is the name of the first person born on Mars?", "False premise."),
    ("What is the recommended daily calorie intake for a pet dog?", "Off-topic."),
]

CAL_HANDWRITTEN = [
    {"question": "Under baseline assumptions, how much does loop closure at the Sabatier/Brine Drying level reduce the total cost of one human Mars mission?",
     "reference": "Approximately 4%.", "doc_id": "20205007682", "page": 6, "evidence": "approximately 4% reduction in the total cost"},
    {"question": "At what rate does the KSC Mars atmospheric ISRU (MARCO POLO) system collect and purify CO2?",
     "reference": "88 g CO2 per hour at >99% purity.", "doc_id": "20160005057", "page": 7, "evidence": "Collect and purify 88 g CO2/h"},
    {"question": "What mission length and crew size were the cis-lunar reusable transportation habitat mass estimates based on?",
     "reference": "An assumed 1050 day mission for four crew members.", "doc_id": "20160011470", "page": 8, "evidence": "assumed 1050 day mission for four crew members"},
    {"question": "What is the full-scale oxygen production goal for human Mars missions quoted for the KSC atmospheric ISRU work?",
     "reference": "0.75 kg O2 per hour.", "doc_id": "20160005057", "page": 7, "evidence": "0.75 kg O2/h"},
]
CAL_UNANSWERABLE = [
    "What is the capital of Australia?", "Who won the 2019 Cricket World Cup?", "What is the price of an iPhone 15?",
    "How tall is the Burj Khalifa?", "What is the orbital period of Neptune's moon Triton?",
    "Which streaming service released the movie The Martian?", "What is the current US Federal Reserve interest rate?",
    "Who wrote the novel Pride and Prejudice?",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def locate(evidence: str, chunks, doc_id: str, page: int) -> list[list]:
    srcs = {(doc_id, page)}
    ev = norm(evidence or "")
    if len(ev) >= 15:
        for c in chunks:
            if ev in norm(c.text):
                srcs.add((c.doc_id, c.page))
    return sorted([list(x) for x in srcs])


def record(qid, q, ref, doc_id, page, evidence, chunks, topic, origin):
    srcs = locate(evidence, chunks, doc_id, page)
    pages = sorted({p for d, p in srcs if d == doc_id})
    return {"id": qid, "question": q, "answerable": True, "reference": ref, "gold_doc_id": doc_id,
            "gold_pages": pages, "gold_sources": srcs, "evidence": evidence, "topic": topic, "origin": origin}


def main() -> int:
    chunks = load_chunks(ROOT / "data" / "chunks.jsonl")
    topic_of = {c.doc_id: c.topic for c in chunks}
    cands = [json.loads(line) for line in (ROOT / "eval" / "candidates.jsonl").read_text().splitlines() if line.strip()]
    golden, cal = [], []
    for i, edits in KEEP.items():
        c = {**cands[i], **edits}
        assert not c.get("skip"), i
        golden.append(record(f"q{len(golden) + 1:02d}", c["question"], str(c["answer"]), c["doc_id"], c["page"],
                             c.get("evidence", ""), chunks, c["topic"], f"llm_draft_reviewed#{i}"))
    for h in HANDWRITTEN:
        golden.append(record(f"q{len(golden) + 1:02d}", h["question"], h["reference"], h["doc_id"], h["page"],
                             h["evidence"], chunks, topic_of[h["doc_id"]], "handwritten"))
    for q, note in UNANSWERABLE:
        golden.append({"id": f"q{len(golden) + 1:02d}", "question": q, "answerable": False, "reference": None,
                       "gold_doc_id": None, "gold_pages": [], "gold_sources": [], "note": note,
                       "topic": "unanswerable", "origin": "handwritten"})
    for i in TO_CALIBRATION:
        c = cands[i]
        cal.append(record(f"c{len(cal) + 1:02d}", c["question"], str(c["answer"]), c["doc_id"], c["page"],
                          c.get("evidence", ""), chunks, c["topic"], f"llm_draft_reviewed#{i}"))
    for h in CAL_HANDWRITTEN:
        cal.append(record(f"c{len(cal) + 1:02d}", h["question"], h["reference"], h["doc_id"], h["page"],
                          h["evidence"], chunks, topic_of[h["doc_id"]], "handwritten"))
    for q in CAL_UNANSWERABLE:
        cal.append({"id": f"c{len(cal) + 1:02d}", "question": q, "answerable": False, "reference": None,
                    "gold_doc_id": None, "gold_pages": [], "gold_sources": [], "topic": "unanswerable",
                    "origin": "handwritten"})
    # sanity: evidence of hand-written items must be found verbatim on the stated page
    for g in golden + cal:
        if g["answerable"] and g["origin"] == "handwritten":
            assert [g["gold_doc_id"], g["gold_pages"][0]] in g["gold_sources"]
    for name, rows in (("golden", golden), ("calibration", cal)):
        with (ROOT / "eval" / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_ans = sum(r["answerable"] for r in rows)
        print(f"{name}: {len(rows)} questions ({n_ans} answerable, {len(rows) - n_ans} unanswerable)")
    multi = [g["id"] for g in golden if len(g["gold_sources"]) > 1]
    print("questions with >1 gold location:", multi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
