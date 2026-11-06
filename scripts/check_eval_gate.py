"""CI eval gate over the committed evaluation results.

Re-running the LLM in CI would need a paid/rate-limited API key, so CI instead checks that
the results committed in reports/ (produced by `make eval-*`) clear the floors in
eval/gate.json. A change that regresses quality must ship regenerated results that still
pass, which makes regressions visible in review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d[part]
    return d


def main() -> int:
    gate = json.loads((ROOT / "eval" / "gate.json").read_text())
    failures, checked = [], 0
    for file, rules in gate.items():
        path = ROOT / "reports" / file
        if not path.exists():
            failures.append(f"{file}: missing")
            continue
        data = json.loads(path.read_text())
        for key, rule in rules.items():
            val = get(data, key)
            checked += 1
            if "min" in rule and val < rule["min"]:
                failures.append(f"{file}:{key} = {val:.3f} < min {rule['min']}")
            if "max" in rule and val > rule["max"]:
                failures.append(f"{file}:{key} = {val:.3f} > max {rule['max']}")
            print(f"{'ok ' if not failures or not failures[-1].startswith(f'{file}:{key}') else 'BAD'} {file}:{key} = {val:.3f} {rule}")
    if failures:
        print("EVAL GATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"eval gate passed ({checked} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
