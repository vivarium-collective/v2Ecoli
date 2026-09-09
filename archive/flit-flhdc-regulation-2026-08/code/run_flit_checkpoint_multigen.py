"""FliT-checkpoint multi-generation test — does the real mechanism (correct
stoichiometry + FliT checkpoint + physical nucleation/elongation) produce a
realistic, self-limiting flagella count across multiple divisions?

Added 2026-08-06, after fixing two real bugs that blocked this test:
  (1) the division-boundary crash in flagella_transcription_regulation.py
      (features weren't threaded through daughter rebuilds -- see
      ecoli_baseline.py's baseline()/loader._features, _helpers.py's
      'division' branch, division.py's Division step)
  (2) nascent_flagellum being silently DUPLICATED (not split) to both
      daughters at division (division.py had no divider registered for it,
      falling through to divide_cell's "unknown type -> copy to both"
      catch-all) -- fixed with divide_nascent_flagellum in
      v2ecoli/library/division.py (binomial p=0.5 per-flagellum split).

Same full-override initial condition as run_flit_checkpoint_test.py (4
flagella, 0 motor, FliA=500, FlgM=800 at t=0), but tracks a DYNAMIC,
growing population of agents across divisions rather than a single cell --
at each snapshot, sums/aggregates flagella and nascent-flagella state
across every currently-alive agent (assumes a flat comp.state["agents"]
dict, confirmed empirically after a single division; if nesting appears at
deeper generations this will need revisiting).

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_flit_checkpoint_multigen.py \
        --seconds 10800 --sample 120 --cache-dir out/cache_full_flit_v3
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


def run(seconds, sample, seed, cache_dir):
    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    agent0 = comp.state["agents"]["0"]
    bulk = _arr(agent0["bulk"])
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
    completed_ever = [0]  # mutable closure cell

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

        # Track cumulative completions: total_flag can drop across a division
        # from binomial splitting alone (not a real "loss"), so only count
        # NET INCREASES beyond the running max as genuine new completions.
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
        if int(total) % 1200 < sample:
            print(f"  t={total:.0f}s  n_agents={rec['n_agents'][-1]}  "
                  f"total_flag={rec['total_flag'][-1]}  total_nascent={rec['total_nascent'][-1]}  "
                  f"max_len={rec['max_len'][-1]}  completed_ever={rec['n_completed_ever'][-1]}")
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec, seconds):
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
    d.axhline(20000, color="gray", ls=":", lw=1, label="target (20,000)")
    d.set_title("Filament construction progress")
    d.set_xlabel("time (min)"); d.set_ylabel("subunits"); d.legend(fontsize=7)

    fig.suptitle(f"Multi-generation test: real stoichiometry + FliT checkpoint, no cap, {seconds}s ({seconds/60:.0f} min)")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/07_multigen_no_cap_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=10800)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v3")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    figure(rec, args.seconds)
    print(f"n_agents {rec['n_agents'][0]}->{rec['n_agents'][-1]}  "
          f"total_flag {rec['total_flag'][0]}->{rec['total_flag'][-1]}  "
          f"mean_flag/cell {rec['mean_flag_per_cell'][0]:.1f}->{rec['mean_flag_per_cell'][-1]:.1f}  "
          f"total_nascent {rec['total_nascent'][0]}->{rec['total_nascent'][-1]}  "
          f"cumulative_completions {rec['n_completed_ever'][-1]}  "
          f"max_len {rec['max_len'][-1]}")
    np.savez(f"{STUDY_DIR}/flit_checkpoint_multigen_{args.seconds}s.npz", **rec)
    print(f"wrote {STUDY_DIR}/flit_checkpoint_multigen_{args.seconds}s.npz")


if __name__ == "__main__":
    main()
