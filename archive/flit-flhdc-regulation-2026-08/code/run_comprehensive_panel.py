"""Comprehensive single-generation panel — every mechanism running together,
on the CURRENT, fully-fixed cache (real stoichiometry + FliT checkpoint +
physical nucleation/elongation + fliC 10x expression fix).

Added 2026-08-07, at Maya's request for a detailed, all-in-one view before
extending to a multi-generation version. Supersedes run_flit_checkpoint_
test.py's chart 06 as the reference single-cell view -- that one predates
the fliC fix and cache_full_flit_v4, and doesn't track free FliD, the FliT:
FliD bound complex, or the individual complexation-reaction products
(CPLX0-7450, FLAGELLAR-MOTOR-COMPLEX) that moved out of Gillespie this
session.

12 panels (4 rows x 3 cols):
  Row 1 -- master regulator + the two chaperone/anti-sigma binding pairs
    (1,1) FlhD4C2 (CPLX0-3930) -- the master regulator the checkpoint degrades
    (1,2) free FliA + FlgM, combined -- the anti-sigma sequestration pair
    (1,3) free FliT-dimer + free FliD + bound FliT:FliD complex, combined --
          the checkpoint's own signal chemistry (free/bound partition)
  Row 2 -- transcriptional override + supply
    (2,1) Class II vs Class III mean override (init_prob_override)
    (2,2) free FliC monomer (the supply-side bottleneck this session found)
    (2,3) fliDST operon's own override (TU0-14278 -- the specific TU shared
          by fliD/fliS/fliT; distinct from the generic Class III average)
  Row 3 -- the 3 reactions moved out of Gillespie complexation (staged pipeline)
    (3,1) CPLX0-7450 (motor switch/C-ring complex)
    (3,2) FLAGELLAR-MOTOR-COMPLEX (full motor)
    (3,3) CPLX0-7452 (complete flagellum) -- the headline output
  Row 4 -- filament construction detail + combined pipeline view
    (4,1) n_nascent (flagella currently under construction)
    (4,2) mean/max filament_length, with the 20,000-subunit target line
    (4,3) CPLX0-7450 vs FLAGELLAR-MOTOR-COMPLEX vs CPLX0-7452, combined --
          shows the staged lag between assembly steps directly

Design note on "separated or combined": FliA/FlgM and FliT/FliD/bound-complex
are shown as COMBINED overlays (1,2)/(1,3) rather than 4-5 fully separate
panels, because each pair's own dynamic (sequestration/release) is the
biologically meaningful thing to see, and each combined panel already shows
every individual trajectory overlaid -- not a reduction in information, just
grouped by what actually interacts. Flag if you want them split further.

Same "regulation ON" full-override initial condition as every other script
in this study, for direct comparability: 4 flagella, 0 motor complex,
free FliA=500, FlgM=800 at t=0.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_comprehensive_panel.py \
        --seconds 2400 --sample 30 --cache-dir out/cache_full_flit_v4
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
    "CPLX0-7452[j]",              # complete flagella (was CPLX0-7452_RXN)
    "CPLX0-7450[i]",              # motor switch/C-ring (was CPLX0-7450_RXN)
    "FLAGELLAR-MOTOR-COMPLEX[j]", # full motor (was FLAGELLAR-MOTOR-COMPLEX_RXN)
    "EG11355-MONOMER[c]",         # free FliA
    "G369-MONOMER[c]",            # FlgM
    "CPLX0-3930[c]",              # FlhD4C2
    "FLIT-DIMER[c]",              # free FliT-dimer
    "FLIT-FLID-CPLX[e]",          # bound FliT:FliD complex
    "EG10841-MONOMER[e]",         # free FliD monomer
    "EG10321-MONOMER[e]",         # free FliC monomer
]

FLIDST_TU_ID = "TU0-14278[c]"  # shared fliD/fliS/fliT operon


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    tu_II = set(rna_ids.index(r) for r in cfg["flg_classII_rnaids"])
    tu_III = set(rna_ids.index(r) for r in cfg["flg_classIII_rnaids"])
    tu_flidst = rna_ids.index(FLIDST_TU_ID) if FLIDST_TU_ID in rna_ids else None

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
        "t": [], "flgM": [], "fliA": [], "flag": [], "flhdc": [], "flit_dimer": [],
        "flit_flid_cplx": [], "flid": [], "flic": [], "cplx0_7450": [],
        "motor_complex": [], "II": [], "III": [], "flidst_ov": [],
        "n_nascent": [], "mean_len": [], "max_len": [],
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
        flidst = ov[tu == tu_flidst] if tu_flidst is not None else np.array([])
        nf = _arr(cell["unique"]["nascent_flagellum"])
        nf_mask = nf["_entryState"].view(bool)
        lengths = nf["filament_length"][nf_mask]

        rec["t"].append(t)
        rec["flgM"].append(_count("G369-MONOMER[c]"))
        rec["fliA"].append(_count("EG11355-MONOMER[c]"))
        rec["flag"].append(_count("CPLX0-7452[j]"))
        rec["flhdc"].append(_count("CPLX0-3930[c]"))
        rec["flit_dimer"].append(_count("FLIT-DIMER[c]"))
        rec["flit_flid_cplx"].append(_count("FLIT-FLID-CPLX[e]"))
        rec["flid"].append(_count("EG10841-MONOMER[e]"))
        rec["flic"].append(_count("EG10321-MONOMER[e]"))
        rec["cplx0_7450"].append(_count("CPLX0-7450[i]"))
        rec["motor_complex"].append(_count("FLAGELLAR-MOTOR-COMPLEX[j]"))
        rec["II"].append(float(II.mean()) if len(II) else 0.0)
        rec["III"].append(float(III.mean()) if len(III) else 0.0)
        rec["flidst_ov"].append(float(flidst.mean()) if len(flidst) else 0.0)
        rec["n_nascent"].append(int(len(lengths)))
        rec["mean_len"].append(float(lengths.mean()) if len(lengths) else 0.0)
        rec["max_len"].append(int(lengths.max()) if len(lengths) else 0)

    b = None
    snap(0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        comp.run(chunk)
        total += chunk
        snap(total)
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec, seconds, cache_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    t = rec["t"] / 60.0
    fig, axs = plt.subplots(4, 3, figsize=(19.0, 20.0))

    def panel(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (min)")

    # (1,1) FlhD4C2
    ax = axs[0, 0]
    ax.plot(t, rec["flhdc"], "-o", ms=3, color="#1f77b4")
    panel(ax, "FlhD4C2 (CPLX0-3930) — master regulator")
    ax.set_ylabel("count")

    # (1,2) FliA + FlgM combined
    ax = axs[0, 1]
    ax.plot(t, rec["fliA"], "-o", ms=3, color="#2ca02c", label="free FliA")
    ax.plot(t, rec["flgM"], "-s", ms=3, color="#ff7f0e", label="FlgM")
    panel(ax, "free FliA + FlgM (anti-sigma pair)")
    ax.set_ylabel("count"); ax.legend(fontsize=7)

    # (1,3) FliT-dimer + FliD + bound complex combined
    ax = axs[0, 2]
    ax.plot(t, rec["flit_dimer"], "-o", ms=3, color="#e377c2", label="free FliT-dimer")
    ax.plot(t, rec["flid"], "-s", ms=3, color="#8c564b", label="free FliD")
    ax.plot(t, rec["flit_flid_cplx"], "-^", ms=3, color="#7f7f7f", label="FliT:FliD (bound)")
    panel(ax, "FliT/FliD checkpoint chemistry (free + bound)")
    ax.set_ylabel("count"); ax.legend(fontsize=7)

    # (2,1) Class II vs III override
    ax = axs[1, 0]
    ax.plot(t, rec["II"], "-o", ms=3, color="#1f77b4", label="Class II ⟨override⟩")
    ax.plot(t, rec["III"], "-s", ms=3, color="#d62728", label="Class III ⟨override⟩")
    panel(ax, "Class II / III mean promoter override")
    ax.set_ylabel("init_prob_override"); ax.legend(fontsize=7)

    # (2,2) free FliC
    ax = axs[1, 1]
    ax.plot(t, rec["flic"], "-o", ms=3, color="#bcbd22")
    panel(ax, "free FliC monomer (supply pool)")
    ax.set_ylabel("count")

    # (2,3) fliDST-specific override
    ax = axs[1, 2]
    ax.plot(t, rec["flidst_ov"], "-o", ms=3, color="#9467bd")
    panel(ax, "fliDST operon override (TU0-14278 — fliD+fliS+fliT)")
    ax.set_ylabel("init_prob_override")

    # (3,1) CPLX0-7450
    ax = axs[2, 0]
    ax.plot(t, rec["cplx0_7450"], "-o", ms=3, color="#17becf")
    panel(ax, "CPLX0-7450 — motor switch/C-ring\n(was CPLX0-7450_RXN, Gillespie)")
    ax.set_ylabel("count")

    # (3,2) FLAGELLAR-MOTOR-COMPLEX
    ax = axs[2, 1]
    ax.plot(t, rec["motor_complex"], "-o", ms=3, color="#17becf")
    panel(ax, "FLAGELLAR-MOTOR-COMPLEX — full motor\n(was FLAGELLAR-MOTOR-COMPLEX_RXN, Gillespie)")
    ax.set_ylabel("count")

    # (3,3) complete flagella
    ax = axs[2, 2]
    ax.plot(t, rec["flag"], "-o", ms=3, color="#9467bd")
    panel(ax, "CPLX0-7452 — complete flagella\n(was CPLX0-7452_RXN, Gillespie)")
    ax.set_ylabel("count")

    # (4,1) n_nascent
    ax = axs[3, 0]
    ax.plot(t, rec["n_nascent"], "-o", ms=3, color="#8c564b")
    panel(ax, "flagella under construction (n_nascent)")
    ax.set_ylabel("count")

    # (4,2) filament length progress
    ax = axs[3, 1]
    ax.plot(t, rec["mean_len"], "-o", ms=3, color="#17becf", label="mean filament_length")
    ax.plot(t, rec["max_len"], "--^", ms=3, color="#17becf", alpha=0.5, label="max filament_length")
    ax.axhline(20000, color="gray", ls=":", lw=1, label="target (20,000)")
    panel(ax, "filament construction progress")
    ax.set_ylabel("subunits"); ax.legend(fontsize=7)

    # (4,3) combined pipeline view
    ax = axs[3, 2]
    ax.plot(t, rec["cplx0_7450"], "-o", ms=3, color="#1f77b4", label="CPLX0-7450 (switch)")
    ax.plot(t, rec["motor_complex"], "-s", ms=3, color="#2ca02c", label="motor complex")
    ax.plot(t, rec["flag"], "-^", ms=3, color="#9467bd", label="complete flagella")
    panel(ax, "assembly pipeline, combined (staged lag)")
    ax.set_ylabel("count"); ax.legend(fontsize=7)

    fig.suptitle(
        f"Comprehensive single-cell panel — full mechanism, {cache_dir}, "
        f"{seconds}s ({seconds/60:.0f} min)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{STUDY_DIR}/charts/08_comprehensive_panel_{seconds}s.svg"
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
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v4")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    path = figure(rec, args.seconds, args.cache_dir)
    print(f"flagella {rec['flag'][0]}->{rec['flag'][-1]}  "
          f"cplx0_7450 {rec['cplx0_7450'][0]}->{rec['cplx0_7450'][-1]}  "
          f"motor_complex {rec['motor_complex'][0]}->{rec['motor_complex'][-1]}  "
          f"FlhDC {rec['flhdc'][0]}->{rec['flhdc'][-1]}")
    print(f"free-FliT-dimer {rec['flit_dimer'][0]}->{rec['flit_dimer'][-1]}  "
          f"free-FliD {rec['flid'][0]}->{rec['flid'][-1]}  "
          f"FliT:FliD-bound {rec['flit_flid_cplx'][0]}->{rec['flit_flid_cplx'][-1]}  "
          f"free-FliC {rec['flic'][0]}->{rec['flic'][-1]}")
    print(f"n_nascent {rec['n_nascent'][0]}->{rec['n_nascent'][-1]}  "
          f"max_filament_length {rec['max_len'][0]}->{rec['max_len'][-1]} (target 20000)")
    np.savez(f"{STUDY_DIR}/comprehensive_panel_{args.seconds}s.npz", **rec)
    print(f"wrote {STUDY_DIR}/comprehensive_panel_{args.seconds}s.npz")


if __name__ == "__main__":
    main()
