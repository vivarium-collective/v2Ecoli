"""The study's whole chain, in order, so no committed artifact is hand-carried.

    panel data  ->  card verdict  ->  figure  ->  acceptance criteria

Each stage reads only committed artifacts from the stage before it and is
deterministic, so re-running this reproduces every tracked file byte for byte.
That property is the study's own subject matter turned on itself: a screen whose
derived outputs cannot be regenerated cannot be audited.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SIMS = Path(__file__).resolve().parent
STAGES = ("make_synthetic_panel.py", "render_card.py",
          "render_panel.py", "check_acceptance.py")


def main() -> int:
    for stage in STAGES:
        print(f"\n=== {stage} ===")
        sys.argv = [str(SIMS / stage)]
        try:
            runpy.run_path(str(SIMS / stage), run_name="__main__")
        except SystemExit as e:
            if e.code:
                return int(e.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
