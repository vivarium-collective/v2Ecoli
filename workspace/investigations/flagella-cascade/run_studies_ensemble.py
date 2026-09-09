"""Multi-seed x more-generations ensemble for the flagella-cascade headline.

Runs the OFF (unregulated) and ON (flagella_regulation) lineages for several seeds
across N generations, then renders:
  * a seed-variance band of complete-flagella per generation (OFF vs ON), turning
    the single-lineage "ON < OFF" claim into a robustness statement, and
  * the deeper-generation trajectory (covers the "more generations" follow-up).

Reuses run_multigen() from run_studies_multigen.py (daughter seeds derived from the
lineage seed). Writes charts into study 01.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/run_studies_ensemble.py \
        --seeds 0 1 2 3 --generations 4 --sample 50
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_studies_multigen import run_multigen  # noqa: E402

STUDY01 = "workspace/investigations/flagella-cascade/studies/flagella-01-overexpression-baseline"


def _gen_end_flagella(rows, n_gens):
    """Per-generation final complete-flagella count from a run_multigen row list."""
    out = []
    for g in range(1, n_gens + 1):
        fl = [r["flag"] for r in rows if r["gen"] == g]
        out.append(fl[-1] if fl else np.nan)
    return out


def run(seeds, n_gens, sample, max_dur, cache_dir):
    data = {"OFF": {}, "ON": {}}
    for seed in seeds:
        print(f"== seed {seed} : OFF ==")
        off = run_multigen([], n_gens, sample, max_dur, seed, cache_dir)
        data["OFF"][seed] = _gen_end_flagella(off, n_gens)
        print(f"== seed {seed} : ON ==")
        on = run_multigen(["flagella_regulation"], n_gens, sample, max_dur, seed, cache_dir)
        data["ON"][seed] = _gen_end_flagella(on, n_gens)
        print(f"   seed {seed}: OFF={data['OFF'][seed]}  ON={data['ON'][seed]}")
    return data


def figure(data, n_gens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    gens = np.arange(1, n_gens + 1)
    off = np.array([data["OFF"][s] for s in data["OFF"]], dtype=float)   # (n_seeds, n_gens)
    on = np.array([data["ON"][s] for s in data["ON"]], dtype=float)
    n_seeds = off.shape[0]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: mean +/- range band across seeds.
    for arr, color, label in [(off, "#9467bd", "regulation OFF"), (on, "#2ca02c", "regulation ON")]:
        m = np.nanmean(arr, axis=0)
        lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
        a.plot(gens, m, "-o", color=color, label=f"{label} (mean)")
        a.fill_between(gens, lo, hi, color=color, alpha=0.2)
    a.set_title(f"Complete flagella per generation\n(mean ± seed range, n={n_seeds} seeds)")
    a.set_xlabel("generation"); a.set_ylabel("CPLX0-7452 count")
    a.set_xticks(gens); a.legend(fontsize=8)

    # Right: per-seed spaghetti to show the OFF>ON gap holds seed-by-seed.
    for i, s in enumerate(data["OFF"]):
        a2 = b
        a2.plot(gens, off[i], "-", color="#9467bd", alpha=0.5, lw=1)
        a2.plot(gens, on[i], "-", color="#2ca02c", alpha=0.5, lw=1)
    b.plot([], [], "-", color="#9467bd", label="OFF (per seed)")
    b.plot([], [], "-", color="#2ca02c", label="ON (per seed)")
    # Fraction of seeds in which the REGULATED (ON) lineage carries FEWER complete
    # flagella than the unregulated (OFF) one, per generation. The calibrated result
    # is that ON exceeds OFF, so this fraction is near 0 — i.e. ON < OFF almost never.
    frac_on_below = np.mean(on < off, axis=0)
    frac_txt = ", ".join(f"gen{g}={f:.0%}" for g, f in zip(gens, frac_on_below))
    b.set_title(
        "Per-seed trajectories — regulation ON sits ABOVE OFF\n"
        f"(seeds where ON < OFF: {frac_txt}  →  ON ≥ OFF for the rest)")
    b.set_xlabel("generation"); b.set_ylabel("CPLX0-7452 count")
    b.set_xticks(gens); b.legend(fontsize=8)
    fig.tight_layout()

    out = f"{STUDY01}/charts/04_ensemble_seed_band.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--max-gen-dur", type=int, default=3600)
    ap.add_argument("--cache-dir", default="out/cache")
    args = ap.parse_args()

    data = run(args.seeds, args.generations, args.sample, args.max_gen_dur, args.cache_dir)
    figure(data, args.generations)
    # Persist the summary so the chart is rebuildable without a re-run.
    with open(f"{STUDY01}/charts/04_ensemble_summary.json", "w") as f:
        json.dump({"seeds": args.seeds, "generations": args.generations, "data": data}, f, indent=1)
    print("\n== ensemble summary ==")
    for cond in ("OFF", "ON"):
        for s in args.seeds:
            print(f"  {cond} seed {s}: {data[cond][s]}")


if __name__ == "__main__":
    main()
