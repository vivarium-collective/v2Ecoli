"""Run the flagella-cascade studies and render rich per-study visualizations.

Two composite runs drive all three studies:
  * regulation OFF -> flagella-01-overexpression-baseline
  * regulation ON  -> flagella-02-transcription-regulation (sumgate + phase-portrait + flgm/flia charts)
    "Regulation ON" = the flagella_regulation feature (2 process Steps) PLUS
    Maya's full 4-value initial-state override (FULL_OVERRIDE below) — applied
    together, investigation-wide, whenever features is truthy.

Each study gets a multi-panel SVG written into its charts/ dir (referenced from
study.yaml visualizations) plus a headline regulated-vs-unregulated comparison.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/run_studies.py --seconds 600 --sample 20
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.core import load_cache_bundle
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx

INV = "workspace/investigations/flagella-cascade"
STUDIES = INV + "/studies"

CLASS_II_CISTRONS = ["EG10322", "EG11346", "EG11347", "G358", "G357", "G7028", "EG11355"]
CLASS_III_CISTRONS = ["EG10321", "EG10317", "EG11967", "EG11545",
                      "EG10601", "EG10602", "EG10146", "EG10149", "G369"]
SYMBOLS = {  # confident EcoCyc cistron -> gene symbol; others fall back to the ID
    "EG10322": "flhD", "EG11355": "fliA", "EG10321": "fliC", "G369": "flgM",
    "EG10841": "motA", "EG10601": "motA", "EG10602": "motB", "EG10146": "cheA",
    "EG10149": "cheW",
}

# Investigation-wide definition (Maya): "regulation ON" = the flagella_regulation
# feature (2 process Steps) PLUS this full 4-value initial-state override
# (her original vEcoli flagellum_initial_value.json), applied together whenever
# features is truthy. Applies uniformly across all studies now, not just study 01.
FULL_OVERRIDE = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}


def _arr(store):
    return store["_data"] if isinstance(store, dict) and "_data" in store else store


def run(features, seconds, sample, seed=0, cache_dir="out/cache"):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    tu_II = {rna_ids.index(r): SYMBOLS.get(c, c)
             for r, c in zip(cfg["flg_classII_rnaids"], CLASS_II_CISTRONS)}
    tu_III = {rna_ids.index(r): SYMBOLS.get(c, c)
              for r, c in zip(cfg["flg_classIII_rnaids"], CLASS_III_CISTRONS)}

    if features:
        enable_features(*features)
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]

    if features:
        for name, val in FULL_OVERRIDE.items():
            bulk["count"][bulk_name_to_idx(name, bids)] = val

    def idx(name):
        try:
            return bulk_name_to_idx(name, bids)
        except Exception:
            return None

    i_flhDC = idx("CPLX0-3930[c]")
    i_fliA = idx("EG11355-MONOMER[c]")
    i_flgM = idx("G369-MONOMER[c]")
    i_flag = idx("CPLX0-7452[j]")
    i_motor = idx("FLAGELLAR-MOTOR-COMPLEX[j]")
    i_flagellin = idx("G361-MONOMER[c]")

    def count(i):
        if i is None:
            return 0
        b = _arr(comp.state["agents"]["0"]["bulk"])
        return int(b["count"][i])

    def gene_overrides(tu_map):
        p = _arr(comp.state["agents"]["0"]["unique"]["promoter"])
        m = p["_entryState"].view(bool)
        tu, ov = p["TU_index"][m], p["init_prob_override"][m]
        out = {}
        for ti, label in tu_map.items():
            rows = ov[tu == ti]
            out[label] = float(rows.mean()) if len(rows) else 0.0
        return out

    def mass():
        try:
            return float(comp.state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
        except Exception:
            return float("nan")

    rec = {"t": [], "flhDC": [], "fliA": [], "flgM": [], "flag": [], "motor": [],
           "flagellin": [], "mass": [], "II": [], "III": [],
           "gene_II": {v: [] for v in tu_II.values()},
           "gene_III": {v: [] for v in tu_III.values()}}

    def snap(t):
        rec["t"].append(t)
        rec["flhDC"].append(count(i_flhDC)); rec["fliA"].append(count(i_fliA))
        rec["flgM"].append(count(i_flgM)); rec["flag"].append(count(i_flag))
        rec["motor"].append(count(i_motor)); rec["flagellin"].append(count(i_flagellin))
        rec["mass"].append(mass())
        gII, gIII = gene_overrides(tu_II), gene_overrides(tu_III)
        rec["II"].append(float(np.mean(list(gII.values()))) if gII else 0.0)
        rec["III"].append(float(np.mean(list(gIII.values()))) if gIII else 0.0)
        for k, v in gII.items():
            rec["gene_II"][k].append(v)
        for k, v in gIII.items():
            rec["gene_III"][k].append(v)

    snap(0)
    # Dense sampling for the first 120 s (every 5 s) then the coarser `sample`
    # grid after. Added 2026-07-28 to check whether the sharp FliA crash
    # visible in the phase portrait (fig_phase) at the old uniform 20 s grid
    # was a sampling-resolution artifact -- confirmed it is NOT: a 1 s check
    # shows free FliA is already at its post-equilibration value by t=5 s
    # under the full override, so the FLGM-FLIA-CPLX equilibrium reaction
    # genuinely completes within a single sub-5-second tick. This denser grid
    # is kept anyway since it still gives better resolution on the slower
    # climb that follows, even though it can't smooth the instant part.
    # Old uniform schedule (kept per standing preserve-old-code rule):
    # for t in range(sample, seconds + 1, sample):
    #     comp.run(sample)
    #     snap(t)
    t = 0.0
    while t < seconds:
        step = 5 if t < 120 else sample
        chunk = min(step, seconds - t)
        comp.run(chunk)
        t += chunk
        snap(t)
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in rec.items()}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                         "grid.alpha": 0.3, "font.size": 10})
    return plt


def save(fig, slug, name):
    d = f"{STUDIES}/{slug}/charts"
    os.makedirs(d, exist_ok=True)
    path = f"{d}/{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", path)


def fig_overexpression(off, on):
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    a.plot(off["t"], off["flag"], "-o", color="#9467bd", label="complete flagella CPLX0-7452")
    a.plot(off["t"], off["motor"], "-s", color="#8c564b", label="motor/HBB FLAGELLAR-MOTOR-COMPLEX")
    a.set_title("Unregulated baseline — flagellar components (feature OFF)")
    a.set_xlabel("time (s)"); a.set_ylabel("count"); a.legend(fontsize=8)

    # neutrality + comparison: flagella count off vs on
    b.plot(off["t"], off["flag"], "-o", color="#9467bd", label="regulation OFF")
    b.plot(on["t"], on["flag"], "-^", color="#2ca02c", label="regulation ON")
    b.set_title("Complete flagella: regulation OFF vs ON")
    b.set_xlabel("time (s)"); b.set_ylabel("CPLX0-7452 count"); b.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "flagella-01-overexpression-baseline", "01_overexpression_and_neutrality")


def fig_cascade(on):
    # Retired 2026-07-28: the ax1 mean-Class-II-vs-III line panel duplicated
    # chart 01's right panel (run_full_override_gate.py) -- same trace, same
    # override condition, just a 600s subset of chart 01's 900s window. Maya
    # asked to drop the duplicate here (chart 01 already covers it, longer)
    # and keep this chart to its more unique content, the per-gene heatmap.
    # Old two-panel version (kept per standing preserve-old-code rule):
    # plt = _mpl()
    # fig = plt.figure(figsize=(13, 5))
    # gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
    # ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    #
    # ax1.plot(on["t"], on["II"], "-o", color="#1f77b4", label="Class II ⟨override⟩")
    # ax1.plot(on["t"], on["III"], "-s", color="#d62728", label="Class III ⟨override⟩")
    # ax1.set_title("K&A SUM-gate: Class II vs Class III")
    # ax1.set_xlabel("time (s)"); ax1.set_ylabel("mean init_prob_override")
    # ax1.legend(fontsize=8)
    plt = _mpl()
    fig, ax2 = plt.subplots(figsize=(8.5, 5))

    # per-gene heatmap (Class II then Class III)
    genes = list(on["gene_II"].keys()) + list(on["gene_III"].keys())
    mat = np.array([on["gene_II"].get(g, on["gene_III"].get(g)) for g in genes])
    im = ax2.imshow(mat, aspect="auto", cmap="viridis",
                    extent=[on["t"][0], on["t"][-1], len(genes) - 0.5, -0.5])
    ax2.set_yticks(range(len(genes))); ax2.set_yticklabels(genes, fontsize=7)
    ax2.axhline(len(on["gene_II"]) - 0.5, color="w", lw=1.5)
    ax2.set_title("Per-gene init_prob_override (top: Class II | bottom: Class III)")
    ax2.set_xlabel("time (s)")
    fig.colorbar(im, ax=ax2, label="override", fraction=0.046)
    fig.tight_layout()
    save(fig, "flagella-02-transcription-regulation", "02_sumgate_classII_classIII")


def fig_phase(on, K_flhDC=50.0, K_fliA=600.0):
    """Phase portrait through SUM-gate input space (X=FlhDC act, Y=free-FliA act)."""
    plt = _mpl()
    flhDC = on["flhDC"].astype(float)
    fliA = on["fliA"].astype(float)
    X = flhDC / (K_flhDC + flhDC)
    Y = fliA / (K_fliA + fliA)
    t = on["t"] / 60.0
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(X, Y, "-", color="#bbbbbb", lw=1, zorder=1)
    sc = ax.scatter(X, Y, c=t, cmap="viridis", s=36, zorder=2, edgecolor="k", linewidth=0.3)
    ax.scatter([X[0]], [Y[0]], marker="o", s=120, facecolor="none", edgecolor="#1f77b4", lw=2, label="t=0")
    ax.scatter([X[-1]], [Y[-1]], marker="*", s=220, color="#d62728", label="end", zorder=3)
    # UPDATED 2026-07-28 (Maya): the full 0-1 axes buried every point in one
    # corner (X sits at 0.87-0.89, Y at 0.17-0.45, since FlhDC is saturated at
    # this count scale -- see chart's own description for why). Zoom to the
    # actual data range instead, so the shape of the trajectory is legible.
    # Old fixed 0-1 axes (kept per standing preserve-old-code rule):
    # ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def _lims(vals, frac=0.15, min_pad=0.01):
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * frac, min_pad)
        return max(0.0, lo - pad), min(1.0, hi + pad)
    ax.set_xlim(*_lims(X)); ax.set_ylim(*_lims(Y))
    ax.set_xlabel("X = FlhDC activity  [FlhDC]/(K+[FlhDC])   —  0 = no FlhDC, 1 = saturating")
    ax.set_ylabel("Y = free-FliA activity  [FliA]/(K+[FliA])   —  0 = fully sequestered, 1 = saturating")
    ax.set_title("Cascade trajectory through SUM-gate input space (zoomed to observed range)\n"
                 "(axes are the two normalized 0–1 activities the SUM-gate integrates, NOT time;\n"
                 "time is the color. ○ = birth, ★ = end)")
    fig.colorbar(sc, ax=ax, label="time (min)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "flagella-02-transcription-regulation", "04_phase_portrait_X_Y")


# Retired 2026-07-28: this 600s-window FlgM/FliA figure (chart 04) is a strict
# subset of chart 01 (run_full_override_gate.py, same override, 900s window —
# long enough to show FliA actually turn around and rise, which this 600s cut
# never reaches). Maya asked to retire it once chart 01 was promoted to lead
# figure. Left in place, commented out, per standing preserve-old-code rule.
# def fig_feedback(on):
#     plt = _mpl()
#     fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
#     a.plot(on["t"], on["flgM"], "-s", color="#ff7f0e", label="FlgM  G369-MONOMER[c]")
#     a.plot(on["t"], on["fliA"], "-o", color="#2ca02c", label="free FliA  EG11355-MONOMER[c]")
#     ab = a.twinx()
#     ab.plot(on["t"], on["flag"], "-^", color="#9467bd", alpha=0.7, label="complete flagella CPLX0-7452[j]")
#     ab.set_ylabel("complete flagella (count)", color="#9467bd")
#     # Flagella are whole molecules — force integer ticks so the right axis can't
#     # read a fractional "31.25 flagella" from matplotlib's autolocator.
#     from matplotlib.ticker import MaxNLocator
#     ab.yaxis.set_major_locator(MaxNLocator(integer=True))
#     a.set_title("FlgM secretion → FliA release  (regulation ON: feature + full override)")
#     a.set_xlabel("time (s)"); a.set_ylabel("molecule count")
#     h1, l1 = a.get_legend_handles_labels(); h2, l2 = ab.get_legend_handles_labels()
#     a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
#
#     fliA0 = on["fliA"][0] if on["fliA"][0] else 1
#     b.plot(on["t"], on["fliA"] / fliA0, "-o", color="#2ca02c")
#     b.axhline(1.0, color="#888", ls="--", lw=1, label="initial FliA")
#     b.set_title("Free FliA stays bounded (relative to initial)")
#     b.set_xlabel("time (s)"); b.set_ylabel("FliA / FliA(t=0)"); b.legend(fontsize=8)
#     fig.tight_layout()
#     save(fig, "flagella-02-transcription-regulation", "04_flgm_secretion_flia_release")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    args = ap.parse_args()

    print("== running feature OFF (study 01) ==")
    off = run([], args.seconds, args.sample, args.seed, args.cache_dir)
    print("== running feature ON (studies 02 + 03) ==")
    on = run(["flagella_regulation"], args.seconds, args.sample, args.seed, args.cache_dir)

    fig_overexpression(off, on)
    fig_cascade(on)
    fig_phase(on)
    # fig_feedback(on)  # retired 2026-07-28 — see commented-out fig_feedback above

    print("\nsummary @ t=%d:" % args.seconds)
    print(f"  OFF flagella={off['flag'][-1]}  ON flagella={on['flag'][-1]}")
    print(f"  ON Class II ⟨ov⟩={on['II'][-1]:.3e}  Class III ⟨ov⟩={on['III'][-1]:.3e}")
    print(f"  ON FlgM {on['flgM'][0]}->{on['flgM'][-1]}  free FliA {on['fliA'][0]}->{on['fliA'][-1]}")
    print(f"  OFF init_prob_override max = {max(off['II'].max(), off['III'].max()):.1e} (expect 0)")


if __name__ == "__main__":
    main()
