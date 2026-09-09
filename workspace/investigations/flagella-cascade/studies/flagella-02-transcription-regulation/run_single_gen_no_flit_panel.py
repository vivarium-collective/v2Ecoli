"""Single-generation panel, post-FliT-checkpoint-removal (2026-08-10).

Adapted from run_comprehensive_panel.py (2026-08-07), dropping the FliT/
FliD checkpoint-chemistry panel entirely (that mechanism was removed from
the WCM 2026-08-10 -- see archive/flit-flhdc-regulation-2026-08/) and
adding no replacement for it, since FlhD4C2 currently has no active
degradation/checkpoint pathway at all. Uses out/cache_full_flit_v11 (post
MS-ring-ordering fix + nucleation first-tick fix, 2026-08-11) -- NOT any
earlier vN cache, which pre-date one or more of this session's fixes.

12 data panels (4 rows x 3 cols, all filled) + a complexation dependency
flowchart spanning the full width at the bottom:
  Row 1 -- regulatory core
    (1,1) FlhD4C2 (CPLX0-3930) -- master regulator, NO degradation pathway
          anymore (expect unbounded growth, not suppression)
    (1,2) free FliA + FlgM, combined -- the anti-sigma sequestration pair
    (1,3) Class II vs Class III mean promoter override (init_prob_override)
  Row 2 -- flagella output
    (2,1) CPLX0-7452 -- complete flagella count (the headline output)
    (2,2) n_nascent -- flagella currently under construction
    (2,3) mean/max filament_length, with the 5,000-subunit target line
  Row 3 -- assembly supply chain, in real synthesis order
    (3,1) CPLX0-7450 -- motor switch/C-ring (feeds motor-complex assembly)
    (3,2) CPLX0-7451 -- export apparatus (the other motor-complex input;
          fixed 2026-08-11 -- see study.yaml. NOTE: FlhA was the bottleneck
          pre-fix (v8), but post-fix (v11) FlhA sits at 100-190, ample --
          the real bottleneck has shifted back to FliN, see Row 4 panels)
    (3,3) FLAGELLAR-MOTOR-COMPLEX -- full motor, consumed by nucleation
  Row 4 -- supply pools. CPLX0-7450/7451 themselves are structurally
    invisible to external sampling (same-tick producer-consumer chain --
    see study.yaml finding flagella-02-export-apparatus-ssa-race-fix), so
    panels (4,2)/(4,3) plot the raw MONOMER pools feeding them instead --
    these ARE visible (gradual translation supply, only partial per-tick
    consumption), added 2026-08-11 per Maya's request for real line plots
    of this dynamic.
    (4,1) free FliC monomer (the known supply-side bottleneck)
    (4,2) C-ring monomer supply: FliF/FliG/FliM/FliN
    (4,3) export-apparatus monomer supply: FlhA/FlhB/FliO/FliP/FliQ/FliR/
          FliH/FliI/FliJ (log scale -- FlhA sits single digits vs. others
          in the hundreds)
  Row 5 -- dependency flowchart (spans all 3 columns): the real
    complexation reaction chain from raw monomers through to a complete
    flagellum, with each reaction's current stoichiometry, cross-checked
    against complexation_reactions_added.tsv / complexation_reactions_modified.tsv
    and the individual process files' _REQUIREMENTS dicts.

Single generation: real Division fires at t~2520s (~42min) under this
IC/cache lineage (established across this whole investigation) -- default
2400s (40min) stays safely inside that window without needing to handle
division at all.

Same "regulation ON" override IC as every other script in this study:
4 flagella, 0 motor complex, free FliA=500, FlgM=800 at t=0.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_single_gen_no_flit_panel.py \
        --seconds 2400 --sample 30 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.core import load_cache_bundle
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,                 # complete flagella
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,          # free FliA
    "G369-MONOMER[c]": 800,             # FlgM
}

READ_IDS = [
    "CPLX0-7452[j]",              # complete flagella
    "CPLX0-7450[i]",              # motor switch/C-ring
    "CPLX0-7451[j]",              # export apparatus
    "FLAGELLAR-MOTOR-COMPLEX[j]", # full motor
    "EG11355-MONOMER[c]",         # free FliA
    "G369-MONOMER[c]",            # FlgM
    "CPLX0-3930[c]",              # FlhD4C2
    "EG10321-MONOMER[e]",         # free FliC monomer
    # Monomer supply pools feeding CPLX0-7450/7451 -- added 2026-08-11.
    # CPLX0-7450/7451 themselves are structurally invisible to external
    # sampling (same-tick producer-consumer chain, see study.yaml finding
    # flagella-02-export-apparatus-ssa-race-fix) -- these raw monomer pools
    # are NOT fully drained every tick (gradual translation supply, partial
    # consumption), so they show real, visible line-plot dynamics instead.
    "FLIF-FLAGELLAR-MS-RING[i]",       # FliF (MS-ring)
    "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]",  # FliG
    "FLIM-FLAGELLAR-C-RING-SWITCH[i]",   # FliM
    "FLIN-FLAGELLAR-C-RING-SWITCH[m]",   # FliN
    "G370-MONOMER[i]",             # FlhA
    "G7028-MONOMER[i]",            # FlhB
    "EG11224-MONOMER[j]",          # FliO
    "EG11975-MONOMER[i]",          # FliP
    "EG11976-MONOMER[j]",          # FliQ
    "EG11977-MONOMER[i]",          # FliR
    "EG11656-MONOMER[c]",          # FliH
    "G377-MONOMER[c]",             # FliI
    "G378-MONOMER[c]",             # FliJ
]


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    tu_II = set(rna_ids.index(r) for r in cfg["flg_classII_rnaids"])
    tu_III = set(rna_ids.index(r) for r in cfg["flg_classIII_rnaids"])

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
    idx = {}
    for name in READ_IDS:
        try:
            idx[name] = bulk_name_to_idx(name, bids)
        except Exception as e:
            print(f"  (skip READ_ID {name} -> {e})")

    rec = {
        "t": [], "flgM": [], "fliA": [], "flag": [], "flhdc": [], "flic": [],
        "cplx0_7450": [], "cplx0_7451": [], "motor_complex": [], "II": [], "III": [],
        "n_nascent": [], "mean_len": [], "max_len": [],
        "fliF": [], "fliG": [], "fliM": [], "fliN": [],
        "flhA": [], "flhB": [], "fliO": [], "fliP": [], "fliQ": [], "fliR": [],
        "fliH": [], "fliI": [], "fliJ": [],
    }

    def _count(name):
        return int(b["count"][idx[name]]) if name in idx else 0

    def snap(t):
        nonlocal b
        cell = comp.state["agents"]["0"]
        b = _arr(cell["bulk"])
        p = _arr(cell["unique"]["promoter"])
        m = p["_entryState"].view(bool)
        tu, ov = p["TU_index"][m], p["init_prob_override"][m]
        II = ov[np.isin(tu, list(tu_II))]
        III = ov[np.isin(tu, list(tu_III))]
        nf = _arr(cell["unique"]["nascent_flagellum"])
        nf_mask = nf["_entryState"].view(bool)
        lengths = nf["filament_length"][nf_mask]

        rec["t"].append(t)
        rec["flgM"].append(_count("G369-MONOMER[c]"))
        rec["fliA"].append(_count("EG11355-MONOMER[c]"))
        rec["flag"].append(_count("CPLX0-7452[j]"))
        rec["flhdc"].append(_count("CPLX0-3930[c]"))
        rec["flic"].append(_count("EG10321-MONOMER[e]"))
        rec["cplx0_7450"].append(_count("CPLX0-7450[i]"))
        rec["cplx0_7451"].append(_count("CPLX0-7451[j]"))
        rec["motor_complex"].append(_count("FLAGELLAR-MOTOR-COMPLEX[j]"))
        rec["II"].append(float(II.mean()) if len(II) else 0.0)
        rec["III"].append(float(III.mean()) if len(III) else 0.0)
        rec["n_nascent"].append(int(len(lengths)))
        rec["mean_len"].append(float(lengths.mean()) if len(lengths) else 0.0)
        rec["max_len"].append(int(lengths.max()) if len(lengths) else 0)
        rec["fliF"].append(_count("FLIF-FLAGELLAR-MS-RING[i]"))
        rec["fliG"].append(_count("FLIG-FLAGELLAR-SWITCH-PROTEIN[i]"))
        rec["fliM"].append(_count("FLIM-FLAGELLAR-C-RING-SWITCH[i]"))
        rec["fliN"].append(_count("FLIN-FLAGELLAR-C-RING-SWITCH[m]"))
        rec["flhA"].append(_count("G370-MONOMER[i]"))
        rec["flhB"].append(_count("G7028-MONOMER[i]"))
        rec["fliO"].append(_count("EG11224-MONOMER[j]"))
        rec["fliP"].append(_count("EG11975-MONOMER[i]"))
        rec["fliQ"].append(_count("EG11976-MONOMER[j]"))
        rec["fliR"].append(_count("EG11977-MONOMER[i]"))
        rec["fliH"].append(_count("EG11656-MONOMER[c]"))
        rec["fliI"].append(_count("G377-MONOMER[c]"))
        rec["fliJ"].append(_count("G378-MONOMER[c]"))

    b = None
    snap(0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        comp.run(chunk)
        total += chunk
        snap(total)
    return {k: np.array(v) for k, v in rec.items()}


def _box(ax, x, y, w, h, text, color, fontsize=7.5, weight="bold"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        fc=color, ec="black", lw=1.1, zorder=3,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, zorder=4)


def _arrow(ax, xy_from, xy_to, color="black", lw=1.3):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
        zorder=2,
    )


def _draw_flowchart(ax):
    """Complexation dependency chain, real stoichiometry, verified against
    complexation_reactions_added.tsv / complexation_reactions_modified.tsv
    and each process's _REQUIREMENTS dict.

    Rebuilt 2026-08-11 as a strictly SEQUENTIAL chain (was a two-branch
    merge before): real assembly order (Minamino & Namba 2008 Nature;
    Chevance & Hughes 2008 Nat Rev Microbiol) has the export apparatus
    insert into a PRE-FORMED C-ring, not assemble independently and merge
    downstream -- CPLX0-7451_RXN now consumes CPLX0-7450 directly. Also
    reflects moving CPLX0-7451_RXN out of ecoli-complexation (SSA) into its
    own deterministic Step (flagella_export_apparatus_assembly.py,
    2026-08-11) -- every stage below is now the same kind of Step, removing
    the cross-mechanism race that stalled motor-complex replenishment."""
    ax.set_xlim(0, 14.4)
    ax.set_ylim(-3.2, 3.2)
    ax.axis("off")
    ax.set_title(
        "Complexation dependency chain -- sequential, real stoichiometry (current cache v10)",
        fontsize=10.5, weight="bold", pad=4,
    )

    c_switch = "#c7e3f7"   # C-ring
    c_export = "#fde3c9"   # export apparatus
    c_motor = "#d7c7f7"    # motor complex
    c_flag = "#c9f7d1"     # flagellum assembly

    xs = [1.3, 4.1, 7.1, 10.1, 13.0]
    y = 0

    _box(ax, xs[0], y, 2.2, 1.1, "CPLX0-7450\n(C-ring)", c_switch, fontsize=7)
    _box(ax, xs[1], y, 2.4, 1.1, "CPLX0-7451\n(export apparatus)", c_export, fontsize=7)
    _box(ax, xs[2], y, 2.4, 1.3, "FLAGELLAR-\nMOTOR-COMPLEX", c_motor, fontsize=7)
    _box(ax, xs[3], y, 2.4, 1.3, "nascent_flagellum\n(+1 per event)", c_flag, fontsize=6.5)
    _box(ax, xs[4], y, 1.4, 1.3, "CPLX0-7452\n(complete)", c_flag, fontsize=6.5)

    # monomer inputs -> CPLX0-7450 (motor_switch_assembly._REQUIREMENTS,
    # deterministic Step)
    _arrow(ax, (0.05, y), (xs[0] - 1.15, y), color="#1f77b4")
    ax.text(0.05, y + 0.7, "FliG x34\nFliM x34\nFliN x111",
            fontsize=6.3, ha="left", va="bottom", color="#1f77b4")

    # CPLX0-7450 -> CPLX0-7451 (HIERARCHY FIX: real dependency, not a
    # parallel merge) + the rest of the export-gate monomers
    # (flagella_export_apparatus_assembly.py, deterministic Step, 2026-08-11)
    _arrow(ax, (xs[0] + 1.1, y), (xs[1] - 1.2, y), color="#1f77b4")
    ax.text((xs[0] + xs[1]) / 2, 0.28, "x1", fontsize=6.3, ha="center", color="#1f77b4")
    _arrow(ax, (xs[1], 1.55), (xs[1], 0.55), color="#d97706")
    ax.text(xs[1], 2.0,
            "FlhA x9  FlhB x1  FliO x1 (open Q)\nFliP x5  FliQ x4  FliR x1\nFliH x12  FliI x6  FliJ x1",
            fontsize=6.3, ha="center", va="bottom", color="#d97706")

    # CPLX0-7451 -> motor complex + other structural parts
    # (FLAGELLAR-MOTOR-COMPLEX_RXN, deterministic Step)
    _arrow(ax, (xs[1] + 1.2, y), (xs[2] - 1.2, y), color="#d97706")
    ax.text((xs[1] + xs[2]) / 2, 0.28, "x1", fontsize=6.3, ha="center", color="#d97706")
    _arrow(ax, (xs[2], -1.55), (xs[2], -0.68), color="#7b2ff7")
    ax.text(xs[2], -2.05,
            "FlgH x26  FlgI x26  MotA x55  MotB x22\n"
            "FlgB x5  FlgC x6  FlgF x5  FlgG x24\n"
            "FliF x34  FliL x2  FliE x6",
            fontsize=6.3, ha="center", va="top", color="#7b2ff7")

    # motor complex -> nascent flagellum (filament_nucleation)
    _arrow(ax, (xs[2] + 1.2, y), (xs[3] - 1.2, y), color="#2ca02c")
    ax.text((xs[2] + xs[3]) / 2, 0.28, "motor x1", fontsize=6.3, ha="center", color="#2ca02c")
    _arrow(ax, (xs[3], 1.55), (xs[3], 0.68), color="#2ca02c")
    ax.text(xs[3], 2.0, "FlgE x120\nFlgK x11  FlgL x11",
            fontsize=6.3, ha="center", va="bottom", color="#2ca02c")

    # nascent flagellum -> complete flagellum (elongation + CPLX0-7452_RXN)
    _arrow(ax, (xs[3] + 1.2, y), (xs[4] - 0.72, y), color="#c2410c")
    ax.text((xs[3] + xs[4]) / 2, -0.32,
            "FliC x5000\n(incremental,\nRenault et al. 2017)\nFliD x5",
            fontsize=6.0, ha="center", va="top", color="#c2410c")

    ax.text(0.05, -2.85,
            "all five reactions above now run as deterministic Steps (same tick order,\n"
            "left to right) -- none fire through the ordinary Gillespie SSA ecoli-complexation process",
            fontsize=6.5, style="italic", color="#555555")

    ax.text(0.05, 2.85,
            "arrow color = which lineage/reaction; box color = product complex",
            fontsize=6.5, style="italic", color="#555555")


def figure(rec, seconds, cache_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    t = rec["t"] / 60.0
    fig = plt.figure(figsize=(16.0, 19.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.5], hspace=0.6, wspace=0.32)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)])

    def panel(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (min)")

    ax = axs[0, 0]
    ax.plot(t, rec["flhdc"], "-o", ms=3, color="#1f77b4")
    panel(ax, "FlhD4C2 (CPLX0-3930) -- no checkpoint/degradation now")
    ax.set_ylabel("count")

    ax = axs[0, 1]
    ax.plot(t, rec["fliA"], "-o", ms=3, color="#2ca02c", label="free FliA")
    ax.plot(t, rec["flgM"], "-s", ms=3, color="#ff7f0e", label="FlgM")
    panel(ax, "free FliA + FlgM (anti-sigma pair)")
    ax.set_ylabel("count"); ax.legend(fontsize=7)

    ax = axs[0, 2]
    ax.plot(t, rec["II"], "-o", ms=3, color="#1f77b4", label="Class II <override>")
    ax.plot(t, rec["III"], "-s", ms=3, color="#d62728", label="Class III <override>")
    panel(ax, "Class II / III mean promoter override")
    ax.set_ylabel("init_prob_override"); ax.legend(fontsize=7)

    ax = axs[1, 0]
    ax.plot(t, rec["flag"], "-o", ms=3, color="#9467bd")
    panel(ax, "CPLX0-7452 -- complete flagella count")
    ax.set_ylabel("count")

    ax = axs[1, 1]
    ax.plot(t, rec["n_nascent"], "-o", ms=3, color="#8c564b")
    panel(ax, "flagella under construction (n_nascent)")
    ax.set_ylabel("count")

    ax = axs[1, 2]
    ax.plot(t, rec["mean_len"], "-o", ms=3, color="#17becf", label="mean filament_length")
    ax.plot(t, rec["max_len"], "--^", ms=3, color="#17becf", alpha=0.5, label="max filament_length")
    ax.axhline(10000, color="gray", ls=":", lw=1, label="target (10,000)")
    panel(ax, "filament construction progress")
    ax.set_ylabel("subunits"); ax.legend(fontsize=7)

    ax = axs[2, 0]
    ax.plot(t, rec["cplx0_7450"], "-o", ms=3, color="#1f77b4")
    panel(ax, "CPLX0-7450 -- motor switch/C-ring")
    ax.set_ylabel("count")

    ax = axs[2, 1]
    ax.plot(t, rec["cplx0_7451"], "-o", ms=3, color="#d97706")
    panel(ax, "CPLX0-7451 -- export apparatus")
    ax.set_ylabel("count")

    ax = axs[2, 2]
    ax.plot(t, rec["motor_complex"], "-o", ms=3, color="#7b2ff7")
    panel(ax, "FLAGELLAR-MOTOR-COMPLEX -- full motor")
    ax.set_ylabel("count")

    ax = axs[3, 0]
    ax.plot(t, rec["flic"], "-o", ms=3, color="#bcbd22")
    panel(ax, "free FliC monomer (supply pool)")
    ax.set_ylabel("count")

    # Monomer supply pools -- added 2026-08-11, per Maya's request for line
    # plots showing what CPLX0-7450/7451 dynamics look like, since the
    # complexes themselves are structurally invisible to external sampling
    # (same-tick producer-consumer chain -- see study.yaml finding
    # flagella-02-export-apparatus-ssa-race-fix). These raw monomers ARE
    # visible: gradual translation supply, only partial consumption per
    # tick, so they show real dynamics instead of a flat 0.
    ax = axs[3, 1]
    ax.plot(t, rec["fliF"], "-", lw=1.3, color="#1f77b4", label="FliF (x34)")
    ax.plot(t, rec["fliG"], "-", lw=1.3, color="#2ca02c", label="FliG (x34)")
    ax.plot(t, rec["fliM"], "-", lw=1.3, color="#ff7f0e", label="FliM (x34)")
    ax.plot(t, rec["fliN"], "-", lw=1.3, color="#d62728", label="FliN (x111)")
    panel(ax, "C-ring supply: FliF/FliG/FliM/FliN monomers")
    ax.set_ylabel("count"); ax.legend(fontsize=6.5, ncol=2)

    ax = axs[3, 2]
    ax.plot(t, rec["flhA"], "-", lw=1.6, color="#c2410c", label="FlhA (x9)")
    ax.plot(t, rec["flhB"], "-", lw=1.0, color="#7b2ff7", label="FlhB (x1)")
    ax.plot(t, rec["fliO"], "-", lw=1.0, color="#d97706", label="FliO (x1)")
    ax.plot(t, rec["fliP"], "-", lw=1.0, color="#1f77b4", label="FliP (x5)")
    ax.plot(t, rec["fliQ"], "-", lw=1.0, color="#2ca02c", label="FliQ (x4)")
    ax.plot(t, rec["fliR"], "-", lw=1.0, color="#17becf", label="FliR (x1)")
    ax.plot(t, rec["fliH"], "-", lw=1.0, color="#8c564b", label="FliH (x12)")
    ax.plot(t, rec["fliI"], "-", lw=1.0, color="#e377c2", label="FliI (x6)")
    ax.plot(t, rec["fliJ"], "-", lw=1.0, color="#bcbd22", label="FliJ (x1)")
    ax.set_yscale("log")
    panel(ax, "export-apparatus supply: FlhA/B, FliO/P/Q/R, FliH/I/J (log scale)")
    ax.set_ylabel("count (log)"); ax.legend(fontsize=5.5, ncol=2)

    ax_flow = fig.add_subplot(gs[4, :])
    _draw_flowchart(ax_flow)

    fig.suptitle(
        f"Single-generation panel, no FliT checkpoint -- {cache_dir}, "
        f"{seconds}s ({seconds/60:.0f} min)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{STUDY_DIR}/charts/12_single_gen_no_flit_panel_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=2400)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    path = figure(rec, args.seconds, args.cache_dir)
    print(f"flagella {rec['flag'][0]}->{rec['flag'][-1]}  "
          f"cplx0_7450 {rec['cplx0_7450'][0]}->{rec['cplx0_7450'][-1]}  "
          f"cplx0_7451 {rec['cplx0_7451'][0]}->{rec['cplx0_7451'][-1]}  "
          f"motor_complex {rec['motor_complex'][0]}->{rec['motor_complex'][-1]}  "
          f"FlhDC {rec['flhdc'][0]}->{rec['flhdc'][-1]}")
    print(f"free-FliA {rec['fliA'][0]}->{rec['fliA'][-1]}  "
          f"FlgM {rec['flgM'][0]}->{rec['flgM'][-1]}  "
          f"free-FliC {rec['flic'][0]}->{rec['flic'][-1]}")
    print(f"n_nascent {rec['n_nascent'][0]}->{rec['n_nascent'][-1]}  "
          f"max_filament_length {rec['max_len'][0]}->{rec['max_len'][-1]} (target 5000)")
    np.savez(f"{STUDY_DIR}/single_gen_no_flit_panel_{args.seconds}s.npz", **rec)
    print(f"wrote {STUDY_DIR}/single_gen_no_flit_panel_{args.seconds}s.npz")
    return path


if __name__ == "__main__":
    main()
