"""Step 2 of NFSIM_WCM_WIRING_PLAN.md: does NFsim work seeded from the REAL
WCM bulk pool, with NO synthetic MonomerProduction feed at all?

Added 2026-08-12, part of Maya Abdalla's flagella-cascade investigation.

Builds a real ecoli_baseline composite, reads the ACTUAL bulk counts for
every real-bulk-ID species this model uses (generate_flagella_bngl.py's
real_bulk_ids(), step-1 renaming), seeds NFSimProcess's observables directly
from those real counts, and runs the reaction network with NO monomer
production process feeding it -- purely testing whether the real ambient
standing pool (whatever ParCa's own initial condition gives) supports
meaningful complexation on its own, chunk after chunk, carrying
scaffold_species forward the same way the (now-fixed) full-model runs do.

This does NOT yet write results back into a live composite's bulk store
(that -- plus real ongoing transcription/translation replenishment each
tick -- is step 3's wrapper Step job). This is a read-then-simulate-in-
isolation diagnostic: confirms real bulk IDs resolve correctly end-to-end
and characterizes what the real ambient pool alone can do, before building
the full two-way-coupled Step.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/diagnostic_real_bulk_seeding.py \
        --seconds 28800 --sample 1200 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os
import sys

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(STUDY_DIR, "models"))

import pkg_resources
import packaging as _packaging
if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = _packaging

import numpy as np
import generate_flagella_bngl as model
import v2ecoli
from v2ecoli.library.schema import bulk_name_to_idx
from pbg_nfsim.processes import NFSimProcess
from process_bigraph import allocate_core


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir, n_steps):
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]

    real_ids = sorted(model.real_bulk_ids())
    real_counts = {}
    for name in real_ids:
        idx = bulk_name_to_idx(name, bids)
        real_counts[name] = int(bulk["count"][idx])

    print("Real bulk counts read from live composite (n=%d):" % len(real_ids))
    for name in real_ids:
        print(f"  {name:45s} {real_counts[name]}")

    core = allocate_core()
    proc = NFSimProcess(
        config={"model_file": model.get_model_path(), "n_steps": n_steps},
        core=core,
    )

    # Seed observables directly from real counts, NO MonomerProduction.
    state = proc.initial_state()
    for name in real_ids:
        safe = model._safe_name(name)
        obs_name = f"Free_{safe}"
        if obs_name in state["observables"]:
            state["observables"][obs_name] = float(real_counts[name])

    track = ["CPLX0_7450_i", "CPLX0_7451_j", "FLAGELLAR_MOTOR_COMPLEX_j",
             "flagellar_hook", "flagella"]

    print("\n=== Running with ONLY the real ambient pool, no replenishment ===")
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        result = proc.update(state, chunk)
        for name, delta in result["observables"].items():
            state["observables"][name] = state["observables"].get(name, 0.0) + delta
        state["scaffold_species"] = result["scaffold_species"]
        total += chunk
        vals = {k: state["observables"].get(k, 0.0) for k in track}
        n_distinct_scaffolds = len(state["scaffold_species"])
        n_scaffold_instances = sum(state["scaffold_species"].values())
        print(f"t={total:7.0f}s  " + "  ".join(f"{k}={v:.0f}" for k, v in vals.items())
              + f"  distinct_scaffolds={n_distinct_scaffolds}  scaffold_instances={n_scaffold_instances:.0f}"
              + f"  Free_FLIF={state['observables'].get('Free_FLIF_FLAGELLAR_MS_RING_i', 0):.0f}")

    print("\nfinal:", {k: state["observables"].get(k, 0.0) for k in track})
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=28800)
    ap.add_argument("--sample", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--cache-dir", type=str, default="out/cache_full_flit_v11")
    args = ap.parse_args()
    run(args.seconds, args.sample, args.seed, args.cache_dir, args.n_steps)


if __name__ == "__main__":
    main()
