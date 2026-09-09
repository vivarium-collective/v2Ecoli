"""Diagnostic ONLY (not a permanent model change): with filament completion
made fast (target_length lowered from the real 20,000 to 2,000 subunits, via
config_overrides -- complexation_reactions_modified.tsv's real, cited
stoichiometry is UNTOUCHED), does the real mechanism (FliT checkpoint + real
motor/switch stoichiometry + nucleation/elongation) produce realistic,
self-limiting multi-generation flagella dynamics, ENTIRELY within the
currently-trustworthy ~3-generation (~2hr) window -- before the division/
mass-homeostasis limitation (investigation.yaml open_decisions,
division-mass-homeostasis-gap) can corrupt the result?

Added 2026-08-08. Rationale (Maya, 2026-08-08): the real 20,000-subunit
target requires >=133 min minimum even with unlimited FliC (see
CHANGES_2026-08-07.md's feasibility calculation) -- far longer than the
~3-generation window we can currently trust. Rather than trying to fix the
division bug (out of scope for now, logged as a known limitation) or wait
for it, this ROUTES AROUND it: temporarily lower the completion target so
real, synthesis-limited completions happen on a timescale of single-digit
minutes instead of hours, letting us directly test whether the FliT
checkpoint + physical assembly mechanisms alone (once completion itself
isn't the bottleneck) produce self-limiting flagella count dynamics -- the
original question this whole investigation arc has been chasing.

target_length=2000 is applied via config_overrides ('ecoli-flagella-
filament-elongation.target_length': 2000) at build_composite() call time --
a declarative, non-permanent override (see ecoli_baseline.py's
config_overrides mechanism). The real, cited stoichiometry in
complexation_reactions_modified.tsv (FliC ~20,000/filament, PMC7696725) is
NOT touched by this script and remains the calibrated value for any
non-diagnostic run.

Same "regulation ON" full-override initial condition as every other script
in this study: 4 flagella, 0 motor, free FliA=500, FlgM=800 at t=0.
Population-based (all daughters kept, same proven approach as
run_flit_checkpoint_multigen.py -- NOT single-lineage pruning, which this
session found no more reliable for surfacing the division bug than the
population approach anyway).

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_diagnostic_reduced_target.py \
        --seconds 7200 --sample 120 --target-length 2000 --cache-dir out/cache_full_flit_v4
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}
READ_IDS = ["CPLX0-7452[j]", "EG11355-MONOMER[c]", "G369-MONOMER[c]", "CPLX0-3930[c]", "FLIT-DIMER[c]"]


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir, target_length, nucleation_rate=None):
    # nucleation_rate override added 2026-08-08: the target-length-only
    # diagnostic (2000, then 500) showed IDENTICAL completion timing/counts
    # in both cases up through the first division -- revealing nucleation
    # rate (fixed ~1 event/600s), not target length, was pacing completions
    # in that regime (each new filament finishes well within one nucleation
    # interval regardless of target, so the next nucleation event is the
    # actual bottleneck). Bumping nucleation_rate too lets more filaments
    # exist/complete concurrently within the same clean pre-corruption
    # window. Still diagnostic-only -- nucleation_rate is itself already a
    # derived estimate (see flagella_filament_nucleation.py), not a
    # literature constant, so this is not overriding a "real" measured
    # value, just exploring a faster point on an already-uncertain rate.
    overrides = {"ecoli-flagella-filament-elongation.target_length": target_length}
    if nucleation_rate is not None:
        overrides["ecoli-flagella-filament-nucleation.nucleation_rate"] = nucleation_rate
    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite(
        "ecoli_baseline", cache_dir=cache_dir, seed=seed,
        config_overrides=overrides,
    )
    enable_features()

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    for name, val in INIT.items():
        try:
            bulk["count"][bulk_name_to_idx(name, bids)] = val
        except Exception as e:
            print("  (skip IC", name, "->", e, ")")
    idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

    rec = {"t": [], "n_agents": [], "total_flag": [], "mean_flag_per_cell": [],
           "total_nascent": [], "mean_len": [], "max_len": [], "n_completed_ever": [],
           "total_fliA": [], "total_flhdc": []}
    completed_ever = [0]
    prev_total_flag = [None]

    def snap(t):
        agents = comp.state["agents"]
        n_agents = len(agents)
        total_flag = 0
        total_nascent = 0
        all_lengths = []
        total_fliA = 0
        total_flhdc = 0
        for aid, cell in agents.items():
            b = _arr(cell["bulk"])
            total_flag += int(b["count"][idx["CPLX0-7452[j]"]])
            total_fliA += int(b["count"][idx["EG11355-MONOMER[c]"]])
            total_flhdc += int(b["count"][idx["CPLX0-3930[c]"]])
            nf = _arr(cell["unique"]["nascent_flagellum"])
            mask = nf["_entryState"].view(bool)
            lengths = nf["filament_length"][mask]
            total_nascent += len(lengths)
            all_lengths.extend(lengths.tolist())

        if prev_total_flag[0] is not None and total_flag > prev_total_flag[0]:
            completed_ever[0] += (total_flag - prev_total_flag[0])
        prev_total_flag[0] = total_flag

        rec["t"].append(t)
        rec["n_agents"].append(n_agents)
        rec["total_flag"].append(total_flag)
        rec["mean_flag_per_cell"].append(total_flag / n_agents if n_agents else 0.0)
        rec["total_nascent"].append(total_nascent)
        rec["mean_len"].append(float(np.mean(all_lengths)) if all_lengths else 0.0)
        rec["max_len"].append(int(max(all_lengths)) if all_lengths else 0)
        rec["n_completed_ever"].append(completed_ever[0])
        rec["total_fliA"].append(total_fliA)
        rec["total_flhdc"].append(total_flhdc)

    snap(0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        comp.run(chunk)
        total += chunk
        snap(total)
        if int(total) % 600 < sample:
            print(f"  t={total:.0f}s  n_agents={rec['n_agents'][-1]}  "
                  f"total_flag={rec['total_flag'][-1]}  total_nascent={rec['total_nascent'][-1]}  "
                  f"max_len={rec['max_len'][-1]}  completed_ever={rec['n_completed_ever'][-1]}  "
                  f"mean_flag/cell={rec['mean_flag_per_cell'][-1]:.2f}")
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec, seconds, target_length, nucleation_rate=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = rec["t"] / 60.0
    fig, axs = plt.subplots(1, 4, figsize=(23.0, 4.7))
    a, b, c, d = axs

    a.plot(t, rec["n_agents"], "-o", ms=3, color="#2ca02c", label="n_agents (population)")
    aa = a.twinx()
    aa.plot(t, rec["mean_flag_per_cell"], "-s", ms=3, color="#9467bd", label="mean flagella/cell")
    aa.set_ylabel("mean flagella/cell", color="#9467bd")
    aa.axhspan(2, 8, color="#9467bd", alpha=0.08)
    a.set_title("Population growth & flagella/cell (real range: 2-8)")
    a.set_xlabel("time (min)"); a.set_ylabel("n_agents", color="#2ca02c")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = aa.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    b.plot(t, rec["total_flag"], "-o", ms=3, color="#9467bd", label="total flagella (all cells)")
    b.plot(t, rec["n_completed_ever"], "-s", ms=3, color="#d62728", label="cumulative completions")
    b.set_title("Flagella count: total vs. cumulative completions")
    b.set_xlabel("time (min)"); b.set_ylabel("count"); b.legend(fontsize=8)

    c.plot(t, rec["total_nascent"], "-o", ms=3, color="#8c564b", label="total nascent (all cells)")
    c.set_title("Flagella under construction (population total)")
    c.set_xlabel("time (min)"); c.set_ylabel("n_nascent"); c.legend(fontsize=8)

    d.plot(t, rec["mean_len"], "-s", ms=3, color="#17becf", label="mean filament_length")
    d.plot(t, rec["max_len"], "--^", ms=3, color="#17becf", alpha=0.5, label="max filament_length")
    d.axhline(target_length, color="gray", ls=":", lw=1, label=f"target ({target_length})")
    d.set_title("Filament construction progress")
    d.set_xlabel("time (min)"); d.set_ylabel("subunits"); d.legend(fontsize=7)

    rate_note = f", nucleation_rate={nucleation_rate}" if nucleation_rate else ""
    fig.suptitle(f"DIAGNOSTIC (target_length={target_length}, real biology=20,000{rate_note}): "
                 f"self-limiting dynamics test, {seconds}s ({seconds/60:.0f} min)")
    fig.tight_layout()
    rate_tag = f"_rate{nucleation_rate}" if nucleation_rate else ""
    out = f"{STUDY_DIR}/charts/11_diagnostic_reduced_target_{target_length}{rate_tag}_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=7200)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-length", type=int, default=2000)
    ap.add_argument("--nucleation-rate", type=float, default=None)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v4")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir, args.target_length, args.nucleation_rate)
    figure(rec, args.seconds, args.target_length, args.nucleation_rate)
    print(f"\nn_agents {rec['n_agents'][0]}->{rec['n_agents'][-1]}  "
          f"total_flag {rec['total_flag'][0]}->{rec['total_flag'][-1]}  "
          f"mean_flag/cell {rec['mean_flag_per_cell'][0]:.1f}->{rec['mean_flag_per_cell'][-1]:.1f}  "
          f"total_nascent {rec['total_nascent'][0]}->{rec['total_nascent'][-1]}  "
          f"cumulative_completions {rec['n_completed_ever'][-1]}  "
          f"max_len {rec['max_len'][-1]}/{args.target_length}")
    rate_tag = f"_rate{args.nucleation_rate}" if args.nucleation_rate else ""
    npz_path = f"{STUDY_DIR}/diagnostic_reduced_target_{args.target_length}{rate_tag}_{args.seconds}s.npz"
    np.savez(npz_path, **rec)
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
