"""Run several heavy steps in ONE Python process so torch / models load once.

    python scripts/run_pipeline.py index calibrate retrieval e2e:rerank e2e:no_rerank

On a 2-vCPU sandbox on a network filesystem, importing torch + transformers alone can take
minutes, so chaining steps here instead of separate CLI calls saves most of the wall time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main(steps: list[str]) -> int:
    t0 = time.perf_counter()
    for step in steps:
        ts = time.perf_counter()
        print(f"=== {step} ===", flush=True)
        if step == "index":
            from marsrag.cli import cmd_index
            cmd_index(None)
        elif step == "calibrate":
            import calibrate
            calibrate.main()
        elif step == "retrieval":
            from marsrag.eval import run_retrieval
            run_retrieval.main([])
        elif step.startswith("e2e:"):
            from marsrag.eval import run_e2e
            run_e2e.main(["--config", step.split(":", 1)[1]])
        elif step == "summarize":
            from marsrag.eval import run_e2e
            run_e2e.main(["--summarize"])
        else:
            raise SystemExit(f"unknown step {step}")
        print(f"=== {step} done in {time.perf_counter() - ts:.0f}s (total {time.perf_counter() - t0:.0f}s) ===",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
