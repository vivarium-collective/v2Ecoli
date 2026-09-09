"""Step 3 of NFSIM_WCM_WIRING_PLAN.md: test FlagellaNFsimComplexation (the
new real-bulk-coupled v2ecoli Step) directly against a live composite's real
bulk store -- two-way coupling, not the read-only diagnostic from step 2.

Added 2026-08-12, part of Maya Abdalla's flagella-cascade investigation.

Manually drives FlagellaNFsimComplexation.update() in a loop against a real
ecoli_baseline composite's own bulk array and unique-molecule stores (same
manual-driving pattern used throughout this investigation's diagnostics),
mutating the REAL composite state in place each firing -- checking:
  (1) real bulk counts change and stay non-negative (mass conservation),
  (2) scaffold_species and internal_observables genuinely persist and grow
      across firings (not reset to zero each time),
  (3) 'flagella' completions create real nascent_flagellum unique molecules.

NOT yet wired into ecoli_baseline.py's execution layers (that's the
rollout's last step) -- this drives the Step directly, standalone.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/diagnostic_nfsim_step.py \
        --seconds 28800 --interval 1200 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.library.schema import bulk_name_to_idx
from v2ecoli.processes.flagella_nfsim_complexation import FlagellaNFsimComplexation


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, interval, seed, cache_dir, n_steps):
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    cell = comp.state["agents"]["0"]

    proc = FlagellaNFsimComplexation(
        parameters={"interval": interval, "n_steps": n_steps},
        core=comp.core,
    )

    scaffold_species = {}
    internal_observables = {}
    global_time = 0.0
    next_update_time = 0.0

    track_ids = ["CPLX0-7450[i]", "CPLX0-7451[j]", "FLAGELLAR-MOTOR-COMPLEX[j]"]
    bulk = _arr(cell["bulk"])
    track_idx = {name: bulk_name_to_idx(name, bulk["id"]) for name in track_ids}

    total = 0.0
    n_firings = 0
    while total < seconds:
        global_time = total
        states = {
            "bulk": _arr(cell["bulk"]),
            "nascent_flagellum": _arr(cell["unique"]["nascent_flagellum"]),
            "scaffold_species": scaffold_species,
            "internal_observables": internal_observables,
            "timestep": interval,
            "next_update_time": next_update_time,
            "global_time": global_time,
        }
        if not proc.update_condition(interval, states):
            total += interval
            continue

        result = proc.update(states, interval)
        n_firings += 1

        # Apply bulk deltas directly to the real composite state.
        for idx, deltas in result["bulk"]:
            bulk = _arr(cell["bulk"])
            bulk["count"][idx] += deltas

        # Apply nascent_flagellum creations directly.
        if "nascent_flagellum" in result:
            n_new = len(result["nascent_flagellum"]["add"]["filament_length"])
            nf = _arr(cell["unique"]["nascent_flagellum"])
            free_mask = ~nf["_entryState"].view(bool)
            free_rows = np.nonzero(free_mask)[0]
            if len(free_rows) < n_new:
                print(f"  WARNING: only {len(free_rows)} free nascent_flagellum "
                      f"rows, need {n_new} -- truncating")
                n_new = len(free_rows)
            for i in range(n_new):
                row = free_rows[i]
                nf["_entryState"][row] = True
                nf["filament_length"][row] = 0

        scaffold_species = result["scaffold_species"]
        internal_observables = result["internal_observables"]
        next_update_time = result["next_update_time"]
        total += interval

        bulk = _arr(cell["bulk"])
        vals = {name: int(bulk["count"][idx]) for name, idx in track_idx.items()}
        n_scaffold_distinct = len(scaffold_species)
        n_scaffold_instances = sum(scaffold_species.values())
        nf = _arr(cell["unique"]["nascent_flagellum"])
        n_nascent = int(nf["_entryState"].sum())
        print(f"firing {n_firings:3d}  t={total:7.0f}s  " +
              "  ".join(f"{k.split('[')[0]}={v}" for k, v in vals.items()) +
              f"  hook_internal={internal_observables.get('flagellar_hook', 0):.0f}" +
              f"  flagella_internal={internal_observables.get('flagella', 0):.0f}" +
              f"  scaffolds(distinct={n_scaffold_distinct},total={n_scaffold_instances:.0f})" +
              f"  nascent_flagellum_count={n_nascent}")

        any_negative = (bulk["count"] < 0).any()
        if any_negative:
            print("  *** ERROR: negative bulk count detected! ***")
            neg_idx = np.nonzero(bulk["count"] < 0)[0]
            for ni in neg_idx[:5]:
                print(f"    {bulk['id'][ni]}: {bulk['count'][ni]}")

    print(f"\nTotal firings: {n_firings}")
    print("Final internal_observables:", internal_observables)
    nf = _arr(cell["unique"]["nascent_flagellum"])
    print("Final nascent_flagellum count:", int(nf["_entryState"].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=28800)
    ap.add_argument("--interval", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--cache-dir", type=str, default="out/cache_full_flit_v11")
    args = ap.parse_args()
    run(args.seconds, args.interval, args.seed, args.cache_dir, args.n_steps)


if __name__ == "__main__":
    main()
