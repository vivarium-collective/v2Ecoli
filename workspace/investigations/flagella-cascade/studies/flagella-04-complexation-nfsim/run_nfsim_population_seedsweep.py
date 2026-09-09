"""Multi-seed replicate summary for the NFsim population test (added 2026-09-02).

Companion to run_nfsim_population_multigen.py, which produces one chart per
seed. That's the right tool for validating a fix (does seed N crash or not),
but it doesn't answer "how much does the population trajectory actually vary
seed-to-seed" -- each chart is a single stochastic realization, not a
distribution. This script runs the same population test across several
seeds, in-process (reusing run_population() directly, not via subprocess --
the per-seed timestep data is needed for aggregation and was never persisted
to disk by the single-seed script), and plots each headline metric as a
mean line with a shaded min-max band across seeds.

Common time grid: all seeds use the same fixed --sample interval starting at
t_cum=0, so seed series line up index-for-index; they differ only in how
many samples they run before hitting max_agents/target generation (division
timing is itself stochastic). Aggregation therefore just truncates every
seed's series to the shortest common length, rather than interpolating.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/run_nfsim_population_seedsweep.py \
        --generations 2 --sample 120 --cache-dir out/cache_full_flit_v12_flisflic_test2 --seeds 0,1,2,3,4,5
"""
import argparse
import os
import re

import numpy as np

from run_nfsim_population_multigen import run_population, TRACK_IDS, COLORS

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Headline metrics only (not all 22 single-seed panels) -- the point of this
# chart is "how much does the population-level story vary run to run," not a
# full per-species diagnostic dump. key -> (panel title, is_total_key).
# is_total_key=False means read the field directly off each row (n_agents),
# True means read row[f"{key}_total"].
HEADLINE_METRICS = [
    ("n_agents", "Live agent count (population size)", False),
    ("CPLX0-7452[j]", "Complete flagella, population total", True),
    ("FLIS-FLIC-CPLX[e]", "FLIS-FLIC-CPLX (protected FliC), population total", True),
    ("EG10321-MONOMER[e]", "Free FliC, population total", True),
    ("EG11355-MONOMER[c]", "FliA, population total", True),
    ("G369-MONOMER[c]", "FlgM, population total", True),
    ("flagella_internal_cumulative", "Hook-basal-body complete (cumulative), population total", True),
    ("n_nascent", "nascent_flagellum, population total", True),
]


def run_sweep(seeds, n_gens, sample, seconds_cap, cache_dir, max_agents, media="minimal"):
    all_rows = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        rows = run_population(n_gens, sample, seconds_cap, seed, cache_dir, max_agents,
                               media=media)
        all_rows.append(rows)
        print(f"  seed {seed}: {len(rows)} samples, final t_cum={rows[-1]['t_cum']:.0f}s")
    return all_rows


def _series(rows, key, is_total):
    if is_total:
        return np.array([r[f"{key}_total"] for r in rows])
    return np.array([r[key] for r in rows])


def figure(all_rows, seeds, n_gens, media="minimal"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    common_n = min(len(rows) for rows in all_rows)
    t = np.array([r["t_cum"] for r in all_rows[0][:common_n]]) / 60.0

    n_cols = 2
    n_panels = len(HEADLINE_METRICS)
    n_rows = -(-n_panels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.2 * n_rows), sharex=True)
    used_axes = list(np.atleast_1d(axes).flat)[:n_panels]

    for ax, (key, title, is_total) in zip(used_axes, HEADLINE_METRICS):
        # One row per seed, truncated to the common length so every seed
        # contributes a value at every plotted time point.
        stacked = np.array([_series(rows[:common_n], key, is_total) for rows in all_rows])
        mean = stacked.mean(axis=0)
        lo = stacked.min(axis=0)
        hi = stacked.max(axis=0)
        color = COLORS.get(key, "#1f77b4")
        ax.fill_between(t, lo, hi, color=color, alpha=0.2, label=f"min-max (n={len(seeds)} seeds)")
        ax.plot(t, mean, "-o", ms=2, color=color, label="mean")
        ax.set_ylabel("count")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)

    for ax in list(np.atleast_1d(axes).flat)[n_panels:]:
        fig.delaxes(ax)
    last_row = (n_panels - 1) // n_cols
    axes_2d = np.atleast_2d(axes)
    for col in range(n_cols):
        if last_row * n_cols + col < n_panels:
            axes_2d[last_row, col].set_xlabel("time (min)")

    fig.suptitle(f"NFsim population test, {n_gens}-generation target, media={media} -- "
                 f"mean ± min/max range across {len(seeds)} independent seeds "
                 f"({', '.join(str(s) for s in seeds)})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    charts_dir = f"{STUDY_DIR}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    existing = [int(m.group(1)) for f in os.listdir(charts_dir)
                if (m := re.match(r"^(\d+)_", f))]
    next_n = max(existing, default=0) + 1
    out = f"{charts_dir}/{next_n}_nfsim_population_seedsweep_{n_gens}gen_{media}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seconds-cap", type=int, default=7200)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v12_flisflic_test2")
    ap.add_argument("--max-agents", type=int, default=8)
    ap.add_argument("--media", type=str, default="minimal")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    all_rows = run_sweep(seeds, args.generations, args.sample, args.seconds_cap,
                          args.cache_dir, args.max_agents, media=args.media)
    figure(all_rows, seeds, args.generations, media=args.media)


if __name__ == "__main__":
    main()
