"""FliT-checkpoint test (no artificial cap) — does the real mechanism alone
self-limit flagella count?

Added 2026-08-06. Runs the same "regulation ON" full-override initial
condition as run_full_override_gate.py (chart 01: 4 flagella, 0 motor, FliA=500,
FlgM=800 at t=0), but against out/cache_full_flit -- the fresh ParCa cache
built from the corrected reaction network (FliT homodimerization +
FliT:FliD equilibrium binding, CPLX0-7452_RXN unmodified) -- with the
ecoli-flagella-nucleation-cap REMOVED and ecoli-flit-flhdc-checkpoint (the
real Utsey & Keener 2020 FliT-mediated negative-feedback mechanism) as the
only thing standing between flagellum completion and flagella count.

Tracks, in addition to the original chart 01 panels (FlgM/FliA/flagella,
Class II/III override):
  - FlhD4C2 (CPLX0-3930[c]) — the master regulator the checkpoint degrades
  - free FliT-dimer (FLIT-DIMER[c]) — the checkpoint's own signal molecule
  - nascent_flagellum unique molecules (n_nascent, mean/max filament_length) —
    added 2026-08-06 after finding that CPLX0-7452 count staying flat can
    mean two very different things: nothing happening, or new flagella
    genuinely under construction but not yet complete (real FliC filament
    growth takes 180+ min per Renault et al. 2017 -- much longer than a
    single-generation test window). Without this, a flat flagella count is
    ambiguous between "the mechanism is broken" and "the mechanism is
    working exactly as slowly as real biology does."
so we can see directly whether flagellum completion is actually releasing
FliT and whether that FliT is actually suppressing FlhD4C2, not just
whether flagella count happens to plateau.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_flit_checkpoint_test.py \
        --seconds 900 --sample 30 --cache-dir out/cache_full_flit
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.core import load_cache_bundle
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Same full override as run_full_override_gate.py's chart 01.
INIT = {
    "CPLX0-7452[j]": 4,                 # complete flagella
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,          # free FliA
    "G369-MONOMER[c]": 800,             # FlgM
}
READ_IDS = [
    "CPLX0-7452[j]", "EG11355-MONOMER[c]", "G369-MONOMER[c]",
    "CPLX0-3930[c]",   # FlhD4C2
    "FLIT-DIMER[c]",   # free FliT-dimer
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
    idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

    rec = {"t": [], "flgM": [], "fliA": [], "flag": [], "flhdc": [], "flit": [],
           "II": [], "III": [], "n_nascent": [], "mean_len": [], "max_len": []}

    def snap(t):
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
        rec["flgM"].append(int(b["count"][idx["G369-MONOMER[c]"]]))
        rec["fliA"].append(int(b["count"][idx["EG11355-MONOMER[c]"]]))
        rec["flag"].append(int(b["count"][idx["CPLX0-7452[j]"]]))
        rec["flhdc"].append(int(b["count"][idx["CPLX0-3930[c]"]]))
        rec["flit"].append(int(b["count"][idx["FLIT-DIMER[c]"]]))
        rec["II"].append(float(II.mean()) if len(II) else 0.0)
        rec["III"].append(float(III.mean()) if len(III) else 0.0)
        rec["n_nascent"].append(int(len(lengths)))
        rec["mean_len"].append(float(lengths.mean()) if len(lengths) else 0.0)
        rec["max_len"].append(int(lengths.max()) if len(lengths) else 0)

    snap(0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        comp.run(chunk)
        total += chunk
        snap(total)
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = rec["t"] / 60.0
    fig, axs = plt.subplots(1, 4, figsize=(23.0, 4.7))
    a, b, c, d = axs

    a.plot(t, rec["flgM"], "-s", ms=3, color="#ff7f0e", label="FlgM G369-MONOMER[c]")
    a.plot(t, rec["fliA"], "-o", ms=3, color="#2ca02c", label="free FliA EG11355-MONOMER[c]")
    ab = a.twinx(); ab.plot(t, rec["flag"], "-^", ms=3, color="#9467bd", alpha=0.7, label="flagella")
    ab.set_ylabel("flagella", color="#9467bd")
    a.set_title("FlgM/FliA/flagella (no cap, FliT checkpoint only)")
    a.set_xlabel("time (min)"); a.set_ylabel("molecule count")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = ab.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    b.plot(t, rec["flhdc"], "-o", ms=3, color="#1f77b4", label="FlhD4C2 CPLX0-3930[c]")
    bb = b.twinx(); bb.plot(t, rec["flit"], "-s", ms=3, color="#e377c2", label="free FliT-dimer")
    bb.set_ylabel("free FliT-dimer", color="#e377c2")
    b.set_title("FliT checkpoint mechanism: does FliT rise & FlhDC fall?")
    b.set_xlabel("time (min)"); b.set_ylabel("FlhD4C2 count", color="#1f77b4")
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = bb.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    c.plot(t, rec["II"], "-o", ms=3, color="#1f77b4", label="Class II ⟨override⟩")
    c.plot(t, rec["III"], "-s", ms=3, color="#d62728", label="Class III ⟨override⟩")
    c.set_title("Class II/III promoter override")
    c.set_xlabel("time (min)"); c.set_ylabel("mean init_prob_override"); c.legend(fontsize=8)

    d.plot(t, rec["n_nascent"], "-o", ms=3, color="#8c564b", label="n_nascent (# under construction)")
    dd = d.twinx()
    dd.plot(t, rec["mean_len"], "-s", ms=3, color="#17becf", label="mean filament_length")
    dd.plot(t, rec["max_len"], "--^", ms=3, color="#17becf", alpha=0.5, label="max filament_length")
    dd.axhline(20000, color="gray", ls=":", lw=1, label="target (20,000)")
    dd.set_ylabel("filament_length (subunits)", color="#17becf")
    d.set_title("New flagella under construction (real ~180min build time)")
    d.set_xlabel("time (min)"); d.set_ylabel("n_nascent", color="#8c564b")
    h1, l1 = d.get_legend_handles_labels(); h2, l2 = dd.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    fig.suptitle(f"FliT-checkpoint test, no nucleation cap, {seconds}s ({seconds/60:.0f} min)")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/06_flit_checkpoint_no_cap_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    path = figure(rec, args.seconds)
    print(f"flagella {rec['flag'][0]}->{rec['flag'][-1]}  "
          f"FlhDC {rec['flhdc'][0]}->{rec['flhdc'][-1]}  "
          f"free-FliT-dimer {rec['flit'][0]}->{rec['flit'][-1]}  "
          f"FlgM {rec['flgM'][0]}->{rec['flgM'][-1]}  FliA {rec['fliA'][0]}->{rec['fliA'][-1]}  "
          f"ClassIII <ov> {rec['III'][0]:.2e}->{rec['III'][-1]:.2e}")
    print(f"n_nascent {rec['n_nascent'][0]}->{rec['n_nascent'][-1]}  "
          f"mean_filament_length {rec['mean_len'][0]:.0f}->{rec['mean_len'][-1]:.0f}  "
          f"max_filament_length {rec['max_len'][0]}->{rec['max_len'][-1]} (target 20000)")
    np.savez(f"{STUDY_DIR}/flit_checkpoint_test_{args.seconds}s.npz", **rec)
    print(f"wrote {STUDY_DIR}/flit_checkpoint_test_{args.seconds}s.npz")


if __name__ == "__main__":
    main()
