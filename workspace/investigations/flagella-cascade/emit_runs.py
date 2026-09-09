"""Run the flagella-cascade studies through the SQLite emitter so each lands
as a tracked simulation in the workspace Simulations DB
(``.pbg/composite-runs.db``), tagged with its study + investigation slug.

Unlike ``run_studies.py`` (which does in-memory runs and writes SVGs only),
this builds each study's declared baseline composite INSIDE a
``sqlite_emitter(...)`` context so the SQLiteEmitter step is wired at build
time and the run is persisted + grouped under the study in the workbench.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/emit_runs.py --seconds 600
    # smoke:
    ... emit_runs.py --seconds 30 --only flagella-01-overexpression-baseline
"""
import argparse
import os
import time

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.composites._helpers import sqlite_emitter
from pbg_emitters.sqlite_emitter import mark_simulation_finished

INVESTIGATION = "flagella-cascade"

# Each study's declared baseline (from its study.yaml `baseline:` block).
STUDIES = [
    {"slug": "flagella-01-overexpression-baseline",   "features": []},
    {"slug": "flagella-02-transcription-regulation",  "features": ["flagella_regulation"]},
]


def run_one(study, seconds, seed=0, cache_dir="out/cache"):
    slug = study["slug"]
    features = study["features"]
    tag = "regON" if features else "regOFF"
    name = f"{slug}__{tag}__seed{seed}__{seconds}s"
    print(f"\n=== {slug}: features={features or '[]'} seconds={seconds} ===", flush=True)

    enable_features(*features)
    try:
        with sqlite_emitter(name=name,
                            study_slug=slug,
                            investigation_slug=INVESTIGATION) as cfg:
            comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
            t0 = time.time()
            comp.update({}, float(seconds))
            elapsed = time.time() - t0
            # sqlite_emitter() stamps started_at but not completion; without this
            # the Simulations DB reports the run as perpetually "running".
            db_path = os.path.join(cfg["file_path"], cfg["db_file"])
            mark_simulation_finished(db_path, cfg["simulation_id"],
                                     elapsed_seconds=elapsed)
    finally:
        enable_features()  # reset the global feature set
    print(f"    done: {name}", flush=True)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="run just this study slug")
    ap.add_argument("--cache-dir", default="out/cache",
                    help="cache to build from. Feature-on studies need "
                         "out/cache_full (it carries the init_prob_override "
                         "promoter column; out/cache predates it).")
    args = ap.parse_args()

    todo = STUDIES if args.only is None else [s for s in STUDIES if s["slug"] == args.only]
    if not todo:
        raise SystemExit(f"no study matches --only {args.only!r}")

    names = [run_one(s, args.seconds, seed=args.seed, cache_dir=args.cache_dir)
             for s in todo]
    print("\nEmitted runs:")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
