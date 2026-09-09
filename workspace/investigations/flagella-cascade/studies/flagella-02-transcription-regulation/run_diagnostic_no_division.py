"""Diagnostic ONLY (not a permanent model change): does a flagellum EVER
complete, given truly unconstrained time, once division is taken out of the
picture?

Added 2026-08-07, part of the FliC-supply / multi-gen investigation. Every
single-cell test this session hits division at t~2524-2528s (~42 min,
mass-threshold triggered under the default D-period division timer) and
crashes (comp.state["agents"]["0"] no longer exists -- replaced by two
daughters). That conflates two separate open questions into one:
  (1) can the current fliC-corrected calibration (~6/s measured, still
      ~7.6x short of the ~46/s Renault-implied peak demand) EVER complete a
      single filament given enough time, or does growth asymptotically
      stall because synthesis can't keep pace with several concurrently-
      growing filaments' combined demand (nucleation keeps firing every
      ~10 min, adding more competitors for a still-limited FliC pool)?
  (2) does DIVISION itself (interrupting construction, splitting in-progress
      flagella between daughters via divide_nascent_flagellum) prevent
      completion, independent of (1)?
This diagnostic isolates (1) by disabling division for one run -- so any
result here is about the synthesis-rate question alone, not about division/
inheritance dynamics (a separate, later question).

Mechanism: monkeypatches Division.next_update to a no-op ({}) for the
duration of this run only, fully reverted in a finally block -- same
try/finally-around-build_composite/run pattern as
run_diagnostic_no_fliA_autoreg.py in this same directory. Verified via
Explore-agent research: Division.next_update is the sole method both the
D-period path (default, self.d_period=True) and the legacy dry-mass-
threshold path dispatch through (division.py's Division.update just calls
self.next_update(1.0, state)), so patching this one method suffices. No
config_overrides path reaches this -- 'division' is not a key in the cache
bundle's configs dict, and d_period is hardcoded True in Division.initialize,
not sourced from any override-able config.

Same full-override initial condition as every other script in this study,
for direct comparability: 4 flagella, 0 motor, free FliA=500, FlgM=800 at t=0.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_diagnostic_no_division.py \
        --seconds 18000 --sample 120 --cache-dir out/cache_full_flit_v4
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx
from v2ecoli.steps.division import Division

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,                 # complete flagella
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,          # free FliA
    "G369-MONOMER[c]": 800,             # FlgM
}
READ_IDS = ["CPLX0-7452[j]", "EG10321-MONOMER[e]"]

_ORIG_NEXT_UPDATE = Division.next_update


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir):
    Division.next_update = lambda self, timestep, states: {}
    try:
        enable_features("flagella_regulation")
        comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
        enable_features()

        bulk = _arr(comp.state["agents"]["0"]["bulk"])
        bids = bulk["id"]
        for name, val in INIT.items():
            try:
                bulk["count"][bulk_name_to_idx(name, bids)] = val
            except Exception as e:
                print("  (skip IC", name, "->", e, ")")
        idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

        rec = {"t": [], "flag": [], "flic": [], "n_nascent": [], "mean_len": [],
               "max_len": [], "n_completed_ever": [], "dry_mass": []}
        completed_ever = [0]
        prev_flag = [None]

        def snap(t):
            cell = comp.state["agents"]["0"]
            b = _arr(cell["bulk"])
            nf = _arr(cell["unique"]["nascent_flagellum"])
            nf_mask = nf["_entryState"].view(bool)
            lengths = nf["filament_length"][nf_mask]
            flag = int(b["count"][idx["CPLX0-7452[j]"]])
            if prev_flag[0] is not None and flag > prev_flag[0]:
                completed_ever[0] += (flag - prev_flag[0])
            prev_flag[0] = flag
            dry_mass_raw = cell.get("listeners", {}).get("mass", {}).get("dry_mass", 0.0)
            dry_mass = float(getattr(dry_mass_raw, "magnitude", dry_mass_raw))

            rec["t"].append(t)
            rec["flag"].append(flag)
            rec["flic"].append(int(b["count"][idx["EG10321-MONOMER[e]"]]))
            rec["n_nascent"].append(int(len(lengths)))
            rec["mean_len"].append(float(lengths.mean()) if len(lengths) else 0.0)
            rec["max_len"].append(int(lengths.max()) if len(lengths) else 0)
            rec["n_completed_ever"].append(completed_ever[0])
            rec["dry_mass"].append(dry_mass)

        snap(0)
        total = 0.0
        while total < seconds:
            chunk = min(sample, seconds - total)
            comp.run(chunk)
            total += chunk
            snap(total)
            if int(total) % 1800 < sample:
                print(f"  t={total:.0f}s ({total/60:.0f}min)  n_nascent={rec['n_nascent'][-1]}  "
                      f"max_len={rec['max_len'][-1]}  free_flic={rec['flic'][-1]}  "
                      f"completed_ever={rec['n_completed_ever'][-1]}  "
                      f"dry_mass={rec['dry_mass'][-1]:.1f}fg")
        return {k: np.array(v) for k, v in rec.items()}
    finally:
        Division.next_update = _ORIG_NEXT_UPDATE


def figure(rec, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = rec["t"] / 60.0
    fig, axs = plt.subplots(1, 4, figsize=(23.0, 4.7))
    a, b, c, d = axs

    a.plot(t, rec["n_nascent"], "-o", ms=3, color="#8c564b")
    a.set_title("flagella under construction (n_nascent)")
    a.set_xlabel("time (min)"); a.set_ylabel("count")

    b.plot(t, rec["mean_len"], "-s", ms=3, color="#17becf", label="mean filament_length")
    b.plot(t, rec["max_len"], "--^", ms=3, color="#17becf", alpha=0.5, label="max filament_length")
    b.axhline(20000, color="gray", ls=":", lw=1, label="target (20,000)")
    b.set_title("filament construction progress (no division)")
    b.set_xlabel("time (min)"); b.set_ylabel("subunits"); b.legend(fontsize=8)

    c.plot(t, rec["flic"], "-o", ms=3, color="#bcbd22")
    c.set_title("free FliC monomer (supply pool)")
    c.set_xlabel("time (min)"); c.set_ylabel("count")

    d.plot(t, rec["flag"], "-o", ms=3, color="#9467bd", label="complete flagella (CPLX0-7452)")
    d.plot(t, rec["n_completed_ever"], "-s", ms=3, color="#d62728", label="cumulative completions")
    d.set_title("completions, undivided cell")
    d.set_xlabel("time (min)"); d.set_ylabel("count"); d.legend(fontsize=8)

    fig.suptitle(
        f"Diagnostic: division DISABLED, single cell, {seconds}s ({seconds/60:.0f} min) — "
        f"isolates whether fliC supply alone permits completion")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/09_diagnostic_no_division_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=18000)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    figure(rec, args.seconds)
    print(f"max_filament_length {rec['max_len'][0]}->{rec['max_len'][-1]} (target 20000)  "
          f"n_nascent {rec['n_nascent'][0]}->{rec['n_nascent'][-1]}  "
          f"cumulative_completions {rec['n_completed_ever'][-1]}  "
          f"free_flic {rec['flic'][0]}->{rec['flic'][-1]}  "
          f"final_dry_mass {rec['dry_mass'][-1]:.1f}fg")
    np.savez(f"{STUDY_DIR}/diagnostic_no_division_{args.seconds}s.npz", **rec)
    print(f"wrote {STUDY_DIR}/diagnostic_no_division_{args.seconds}s.npz")


if __name__ == "__main__":
    main()
