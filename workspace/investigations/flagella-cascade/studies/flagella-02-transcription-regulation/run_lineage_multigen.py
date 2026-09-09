"""Single-followed-lineage, many-generation test — does inherited, in-progress
flagellar construction actually complete and converge to a realistic count
across REAL divisions?

Added 2026-08-07. The division-disabled diagnostic (run_diagnostic_no_
division.py) established that a single generation (~42 min) is far shorter
than even the best-case unconstrained filament-completion time (~180+ min),
so under current calibration no single cell can complete a flagellum within
its own lifetime -- completion, if it happens at all in the real (dividing)
model, can only come from a flagellum's in-progress construction (nascent_
flagellum's filament_length) being carried forward intact across several
real divisions via divide_nascent_flagellum's whole-unit binomial
inheritance (confirmed 2026-08-06 to survive division without loss).

METHODOLOGY NOTE (2026-08-07, important): the first version of this script
used run_studies_multigen.py's manual generation-chaining pattern -- call
divide_cell() directly, rebuild a fresh baseline() composite, hand-splice
bulk/unique/environment/boundary onto it. That approach has a REAL,
previously-undetected bug: dry_mass at division drifted sharply downward
generation over generation (706.7 -> 632.6 -> 553.5 -> ... -> 262.0 fg
by generation 7, vs. a normal ~700fg every time), eventually destabilizing
the metabolism FBA solver (floods of GLP_NOFEAS/numerical-instability
warnings from generation 3 onward) and driving complete-flagella count to
0. run_studies_multigen.py itself was never stress-tested past
--generations 2 (its only existing chart), so this compounding drift was
never caught before. Root cause not fully isolated, but plausibly a gap in
what the manual splice carries over for chromosome-replication-timer
continuity (D-period division is a fixed schedule set at a cell's "birth",
not re-derived from mass -- see division.py's MarkDPeriod).

FIX: this version uses the REAL Division step's own daughter-construction
machinery instead (the same path run_flit_checkpoint_multigen.py's
expanding-population approach uses, which showed no comparable drift --
706.7, 646.5, 598.9 fg across its real divisions, ordinary binomial
variance, not compounding decay) -- and achieves the SAME linear compute
cost as manual splicing by pruning to a single followed agent immediately
after each division (deleting the sibling from comp.state["agents"])
rather than keeping the whole expanding population. This is directly
analogous to real single-lineage "mother machine" tracking experiments
(Wang et al. 2010, Curr Biol) -- a real, standard technique, not an ad hoc
shortcut. The daughter to KEEP is chosen by a fixed, content-independent
rule (lexicographically lowest agent ID) so the choice never correlates
with which daughter happened to inherit more flagella/progress -- this
model has no old-pole/new-pole asymmetry mechanism (unlike real bacterial
aging, Stewart et al. 2005 PLoS Biology) that would make one daughter
systematically different, so an arbitrary fixed rule is unbiased here.

Same full-override initial condition as every other script in this study,
applied once at t=0: 4 flagella, 0 motor, free FliA=500, FlgM=800.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_lineage_multigen.py \
        --generations 10 --sample 120 --cache-dir out/cache_full_flit_v4
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_OVERRIDE = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _snap(comp, agent_id, idx, t_cum, gen, completed_ever, prev_flag):
    cell = comp.state["agents"][agent_id]
    b = _arr(cell["bulk"])
    nf = _arr(cell["unique"]["nascent_flagellum"])
    nf_mask = nf["_entryState"].view(bool)
    lengths = nf["filament_length"][nf_mask]
    flag = int(b["count"][idx["flag"]])
    flic = int(b["count"][idx["flic"]])
    dry_mass = fg_magnitude(cell["listeners"]["mass"].get("dry_mass", 0))
    # DIAGNOSTIC ONLY (2026-08-07): does the model's own size-correction
    # signal (mass-coupled replication initiation) behave as expected for a
    # pruned single lineage, or is it not kicking in? See
    # chromosome_replication.py:531-536. critical_mass_per_oriC ~1.0 means
    # "at the initiation threshold"; a daughter starting below-average
    # should show critical_mass_per_oriC climbing back toward 1.0 as it
    # grows, if the correction is working.
    repl = cell["listeners"].get("replication_data", {})
    crit_init_mass = fg_magnitude(repl.get("critical_initiation_mass", 0))
    crit_mass_per_oric = float(repl.get("critical_mass_per_oriC", 0) or 0)

    if prev_flag[0] is not None and flag > prev_flag[0]:
        completed_ever[0] += (flag - prev_flag[0])
    prev_flag[0] = flag

    return {
        "t_cum": t_cum, "gen": gen, "dry_mass": dry_mass,
        "flag": flag, "flic": flic,
        "n_nascent": int(len(lengths)),
        "mean_len": float(lengths.mean()) if len(lengths) else 0.0,
        "max_len": int(lengths.max()) if len(lengths) else 0,
        "completed_ever": completed_ever[0],
        "crit_init_mass": crit_init_mass,
        "crit_mass_per_oric": crit_mass_per_oric,
    }


def run_lineage(n_gens, sample, seconds_cap, seed, cache_dir):
    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    for name, val in FULL_OVERRIDE.items():
        bulk["count"][bulk_name_to_idx(name, bids)] = val
    idx = {
        "flag": bulk_name_to_idx("CPLX0-7452[j]", bids),
        "flic": bulk_name_to_idx("EG10321-MONOMER[e]", bids),
    }

    rows = []
    agent_id = "0"
    gen = 1
    completed_ever = [0]
    prev_flag = [None]

    rows.append(_snap(comp, agent_id, idx, 0.0, gen, completed_ever, prev_flag))

    total = 0.0
    while total < seconds_cap and gen <= n_gens:
        chunk = min(sample, seconds_cap - total)
        comp.run(chunk)
        total += chunk

        agents = comp.state.get("agents", {})
        if len(agents) > 1:
            # Real division happened. Keep exactly one daughter and discard
            # the sibling -- prunes population back to 1, so cost stays
            # linear in generation count.
            #
            # IMPORTANT (2026-08-07): the first version of this fix kept a
            # FIXED rule (lexicographically lowest agent ID) every time.
            # That produced a suspicious, monotonic dry_mass decline across
            # 4 consecutive real divisions (706.7->646.5->559.3->525.4 fg,
            # ~6% likely under pure chance) -- consistent with the fixed ID
            # rule being correlated with a smaller share of the binomial
            # split (e.g. if Division.next_update's d1/d2 assignment has
            # any consistent positional convention), NOT confirmed as a
            # real bias, but suspicious enough not to trust. divide_bulk
            # itself (library/division.py:58-74) IS unbiased in expectation
            # (rng.binomial(counts, 0.5)), so a genuinely random choice of
            # which daughter to keep -- decorrelated from ID string order
            # entirely -- removes any possible correlation with a fixed
            # convention, converting any residual risk into honest 50/50
            # chance at each division (the standard way to eliminate
            # deterministic-selection bias). Uses its own RNG stream (seeded
            # from the run seed + generation number, reproducible) so this
            # choice doesn't consume/perturb the simulation's own RNG state.
            #
            # MUST use comp.apply({"agents": {"_remove": [...]}}) here, NOT
            # raw `del comp.state["agents"][id]` -- confirmed via direct
            # testing (crashed on the very next comp.run() with
            # "AttributeError: 'NoneType' object has no attribute 'get'" in
            # process_bigraph's run_steps). Composite.apply() (process_bigraph/
            # composite.py:1783-1796) calls self.find_instance_paths(self.state)
            # afterward, which repopulates the internal process_paths/step_paths
            # registry -- exactly what raw dict deletion leaves stale. This is
            # the SAME "_remove" primitive division.py's own Division step uses
            # to remove the parent when installing both daughters
            # (division.py:439-447: {'agents': {'_remove': [...], '_add': [...]}}),
            # just without the paired "_add".
            candidate_ids = sorted(agents.keys())
            pick_rng = np.random.RandomState(seed=(seed * 1000 + gen) % (2**31 - 1))
            keep_id = candidate_ids[pick_rng.randint(len(candidate_ids))]
            discard_ids = [aid for aid in candidate_ids if aid != keep_id]
            comp.apply({"agents": {"_remove": discard_ids}})
            agent_id = keep_id
            gen += 1
            prev_flag[0] = None  # new generation: don't count the binomial-
            # split delta at this boundary as a "completion"
            _birth = _snap(comp, agent_id, idx, total, gen, completed_ever, prev_flag)
            print(f"  gen {gen} begins at t_cum={total:.0f}s, following agent '{agent_id}' "
                  f"-- BIRTH: dry_mass={_birth['dry_mass']:.1f}fg "
                  f"crit_init_mass={_birth['crit_init_mass']:.1f}fg "
                  f"crit_mass_per_oriC={_birth['crit_mass_per_oric']:.3f}")

        if agent_id not in comp.state.get("agents", {}):
            print(f"  agent '{agent_id}' vanished unexpectedly — stopping")
            break

        row = _snap(comp, agent_id, idx, total, gen, completed_ever, prev_flag)
        rows.append(row)
        if int(total) % 1800 < sample:
            print(f"    t_cum={total:.0f}s ({total/60:.0f}min) gen={gen} agent='{agent_id}' "
                  f"flag={row['flag']} n_nascent={row['n_nascent']} max_len={row['max_len']} "
                  f"free_flic={row['flic']} completed_ever={row['completed_ever']} "
                  f"dry_mass={row['dry_mass']:.1f}fg "
                  f"crit_init_mass={row['crit_init_mass']:.1f}fg "
                  f"crit_mass_per_oriC={row['crit_mass_per_oric']:.3f}")

    return rows


def _cols(rows, key):
    return np.array([r[key] for r in rows])


def _gen_bounds(rows):
    bounds = []
    for i in range(1, len(rows)):
        if rows[i]["gen"] != rows[i - 1]["gen"]:
            bounds.append(rows[i]["t_cum"] / 60.0)
    return bounds


def _shade(ax, rows):
    gens = sorted({r["gen"] for r in rows})
    colors = ["#eef4ff", "#fff4ee", "#eefff2", "#f7eeff"]
    for gi in gens:
        xs = [r["t_cum"] / 60.0 for r in rows if r["gen"] == gi]
        if xs:
            ax.axvspan(min(xs), max(xs), color=colors[(gi - 1) % len(colors)], alpha=0.5, zorder=0)
    for b in _gen_bounds(rows):
        ax.axvline(b, color="#c0392b", ls="--", lw=1, alpha=0.7)


def figure(rows, n_gens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = _cols(rows, "t_cum") / 60.0
    fig, axs = plt.subplots(1, 5, figsize=(28.0, 4.7))
    a, b, c, d, e = axs

    a.plot(t, _cols(rows, "flag"), "-o", ms=2, color="#9467bd", label="complete flagella (this lineage)")
    a.plot(t, _cols(rows, "completed_ever"), "-s", ms=2, color="#d62728", label="cumulative completions")
    _shade(a, rows)
    a.set_title("Complete flagella, followed lineage across generations")
    a.set_xlabel("time (min)"); a.set_ylabel("count"); a.legend(fontsize=8)

    b.plot(t, _cols(rows, "n_nascent"), "-o", ms=2, color="#8c564b")
    _shade(b, rows)
    b.set_title("flagella under construction (n_nascent)")
    b.set_xlabel("time (min)"); b.set_ylabel("count")

    c.plot(t, _cols(rows, "mean_len"), "-o", ms=2, color="#17becf", label="mean filament_length")
    c.plot(t, _cols(rows, "max_len"), "-s", ms=2, color="#17becf", alpha=0.5, label="max filament_length")
    c.axhline(20000, color="gray", ls=":", lw=1, label="target (20,000)")
    _shade(c, rows)
    c.set_title("filament construction progress")
    c.set_xlabel("time (min)"); c.set_ylabel("subunits"); c.legend(fontsize=7)

    d.plot(t, _cols(rows, "flic"), "-o", ms=2, color="#bcbd22")
    _shade(d, rows)
    d.set_title("free FliC monomer (supply pool)")
    d.set_xlabel("time (min)"); d.set_ylabel("count")

    e.plot(t, _cols(rows, "dry_mass"), "-o", ms=2, color="#1f77b4")
    _shade(e, rows)
    e.set_title("dry mass (sanity check: should NOT drift down across generations)")
    e.set_xlabel("time (min)"); e.set_ylabel("fg")

    fig.suptitle(f"Single-lineage, {n_gens}-generation test (real Division machinery, pruned to 1 "
                 f"followed agent) — does inherited construction converge? "
                 f"(shaded=generation, dashed=division)")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/10_lineage_multigen_{n_gens}gen.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seconds-cap", type=int, default=36000,
                     help="hard stop on total simulated time, as a safety ceiling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    args = ap.parse_args()
    rows = run_lineage(args.generations, args.sample, args.seconds_cap, args.seed, args.cache_dir)
    figure(rows, args.generations)
    last = rows[-1]
    print(f"\nFINAL (gen {last['gen']}, t_cum={last['t_cum']:.0f}s / {last['t_cum']/60:.0f}min): "
          f"flag={last['flag']}  completed_ever={last['completed_ever']}  "
          f"n_nascent={last['n_nascent']}  max_len={last['max_len']}  "
          f"free_flic={last['flic']}  dry_mass={last['dry_mass']:.1f}fg")
    np.savez(f"{STUDY_DIR}/lineage_multigen_{args.generations}gen.npz",
             **{k: _cols(rows, k) for k in rows[0].keys()})
    print(f"wrote {STUDY_DIR}/lineage_multigen_{args.generations}gen.npz")


if __name__ == "__main__":
    main()
