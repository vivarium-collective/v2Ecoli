"""Diagnostic ONLY (not a permanent model change): does zeroing fliA's own
Y-autoregulation term (beta_prime[6], its own promoter's FliA-dependence)
stop the flagella/FliA runaway?

Added 2026-08-06, part of the FlgM/FliA deep-dive investigation. Monkeypatches
CachedConfigLoader.get_config_by_name to intercept ONLY
'ecoli-flagella-transcription-regulation' and return a copy of its config
with beta_prime[6] (fliA's own Class II promoter's Y-coefficient) forced to
0.0 -- i.e. fliA's own transcription becomes driven ONLY by FlhDC (X), the
"basic" single-input feedforward architecture Kalir & Alon describe for
fliA's initial activation, with the SUM-gate (X+Y) reserved for the
downstream Class II operons as in their published model. This does NOT
modify sim_data.py or the cache bundle -- it's an in-memory, single-run
diagnostic to isolate whether fliA's own autoregulation coefficient is the
runaway's driver, before deciding whether/how to change it for real.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_diagnostic_no_fliA_autoreg.py \
        --seconds 2400 --sample 60 --cache-dir out/cache_full_flit_v11

NOTE (2026-08-11): --cache-dir default updated from out/cache_full_flit
(deleted during repo cleanup -- see study.yaml) to out/cache_full_flit_v11,
the current cache. This diagnostic tests fliA's own autoregulation
coefficient, unrelated to the FliT:FlhDC checkpoint removed 2026-08-10, so
it remains valid against the current reaction network.
"""
import argparse
import copy
import os

import numpy as np

import v2ecoli
from v2ecoli.core import load_cache_bundle
from v2ecoli.composites import _helpers
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}
READ_IDS = [
    "CPLX0-7452[j]", "EG11355-MONOMER[c]", "G369-MONOMER[c]",
    "CPLX0-3930[c]", "FLIT-DIMER[c]",
]


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


_ORIG_GET_CONFIG = _helpers.CachedConfigLoader.get_config_by_name


# sim_data.py's config-getter never overrides beta/beta_prime -- they come
# purely from FlagellaTranscriptionRegulation's own config_schema defaults
# (confirmed: the cached config dict has no "beta_prime" key at all). So the
# diagnostic must EXPLICITLY inject the full array (matching the Step's real
# default) with fliA's own entry (index 6) zeroed, rather than trying to
# modify a key that was never actually present in the cached config.
_BETA_PRIME_DEFAULT = [250, 350, 300, 450, 300, 350, 300]


def _patched_get_config(self, name):
    cfg = _ORIG_GET_CONFIG(self, name)
    if name == "ecoli-flagella-transcription-regulation":
        cfg = copy.deepcopy(cfg)
        bp = list(_BETA_PRIME_DEFAULT)
        # fliA is the LAST entry in classII_cistron_ids (index 6) -- see
        # sim_data.py get_flagella_transcription_regulation_config.
        print(f"  [diagnostic] beta_prime before: {bp}")
        bp[6] = 0.0
        cfg["beta_prime"] = bp
        print(f"  [diagnostic] beta_prime after (fliA autoreg OFF): {bp}")
    return cfg


def run(seconds, sample, seed, cache_dir):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    tu_II = set(rna_ids.index(r) for r in cfg["flg_classII_rnaids"])
    tu_III = set(rna_ids.index(r) for r in cfg["flg_classIII_rnaids"])

    _helpers.CachedConfigLoader.get_config_by_name = _patched_get_config
    try:
        enable_features("flagella_regulation")
        comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
        enable_features()
    finally:
        _helpers.CachedConfigLoader.get_config_by_name = _ORIG_GET_CONFIG

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    for name, val in INIT.items():
        try:
            bulk["count"][bulk_name_to_idx(name, bids)] = val
        except Exception as e:
            print("  (skip IC", name, "->", e, ")")
    idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

    rec = {"t": [], "flgM": [], "fliA": [], "flag": [], "flhdc": [], "flit": [],
           "II": [], "III": []}

    def snap(t):
        cell = comp.state["agents"]["0"]
        b = _arr(cell["bulk"])
        p = _arr(cell["unique"]["promoter"])
        m = p["_entryState"].view(bool)
        tu, ov = p["TU_index"][m], p["init_prob_override"][m]
        II = ov[np.isin(tu, list(tu_II))]
        III = ov[np.isin(tu, list(tu_III))]
        rec["t"].append(t)
        rec["flgM"].append(int(b["count"][idx["G369-MONOMER[c]"]]))
        rec["fliA"].append(int(b["count"][idx["EG11355-MONOMER[c]"]]))
        rec["flag"].append(int(b["count"][idx["CPLX0-7452[j]"]]))
        rec["flhdc"].append(int(b["count"][idx["CPLX0-3930[c]"]]))
        rec["flit"].append(int(b["count"][idx["FLIT-DIMER[c]"]]))
        rec["II"].append(float(II.mean()) if len(II) else 0.0)
        rec["III"].append(float(III.mean()) if len(III) else 0.0)

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
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 4.7))
    a, b, c = axs

    a.plot(t, rec["flgM"], "-s", ms=3, color="#ff7f0e", label="FlgM G369-MONOMER[c]")
    a.plot(t, rec["fliA"], "-o", ms=3, color="#2ca02c", label="free FliA EG11355-MONOMER[c]")
    ab = a.twinx(); ab.plot(t, rec["flag"], "-^", ms=3, color="#9467bd", alpha=0.7, label="flagella")
    ab.set_ylabel("flagella", color="#9467bd")
    a.set_title("FlgM/FliA/flagella (fliA autoreg OFF)")
    a.set_xlabel("time (min)"); a.set_ylabel("molecule count")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = ab.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    b.plot(t, rec["flhdc"], "-o", ms=3, color="#1f77b4", label="FlhD4C2 CPLX0-3930[c]")
    bb = b.twinx(); bb.plot(t, rec["flit"], "-s", ms=3, color="#e377c2", label="free FliT-dimer")
    bb.set_ylabel("free FliT-dimer", color="#e377c2")
    b.set_title("FlhDC / free FliT-dimer")
    b.set_xlabel("time (min)"); b.set_ylabel("FlhD4C2 count", color="#1f77b4")
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = bb.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    c.plot(t, rec["II"], "-o", ms=3, color="#1f77b4", label="Class II ⟨override⟩")
    c.plot(t, rec["III"], "-s", ms=3, color="#d62728", label="Class III ⟨override⟩")
    c.set_title("Class II/III promoter override")
    c.set_xlabel("time (min)"); c.set_ylabel("mean init_prob_override"); c.legend(fontsize=8)

    fig.suptitle(f"DIAGNOSTIC: fliA autoregulation forced OFF, {seconds}s ({seconds/60:.0f} min)")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/07_diagnostic_no_fliA_autoreg_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=2400)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    figure(rec, args.seconds)
    print(f"flagella {rec['flag'][0]}->{rec['flag'][-1]}  "
          f"FlhDC {rec['flhdc'][0]}->{rec['flhdc'][-1]}  "
          f"free-FliT-dimer {rec['flit'][0]}->{rec['flit'][-1]}  "
          f"FlgM {rec['flgM'][0]}->{rec['flgM'][-1]}  FliA {rec['fliA'][0]}->{rec['fliA'][-1]}  "
          f"ClassIII <ov> {rec['III'][0]:.2e}->{rec['III'][-1]:.2e}")
    np.savez(f"{STUDY_DIR}/diagnostic_no_fliA_autoreg_{args.seconds}s.npz", **rec)


if __name__ == "__main__":
    main()
