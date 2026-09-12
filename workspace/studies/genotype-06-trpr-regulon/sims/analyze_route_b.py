#!/usr/bin/env python
"""Reduce the route-B traces to the numbers the bands grade.

⚠ THE STATISTIC IS THE MEAN, AND THAT IS NOT A STYLE CHOICE. trp-operon repression
in this model is ON/OFF, not graded: TrpR is present at only 42-80 copies, so a
single complex binding and unbinding switches the operon between ~8e-12 and
~2.6e-4. A MEDIAN lands on whichever side of that split it happens to fall — the
same WT/KO pair yields 5.2e7, 1.98 and 4.5e7 across three generations of one run.
The MEAN is stable AND is the biologically right quantity, because transcription
integrates over time: mean = (fraction of time active) x (level when active).

Reads data/route_b_<arm>.json; writes data/route_b_summary.json. No simulation.

    python workspace/studies/genotype-06-trpr-regulon/sims/analyze_route_b.py
"""
from __future__ import annotations

import json
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA = STUDY_DIR / "data"
OUT = DATA / "route_b_summary.json"

# Four-plus orders below the derepressed level; the gap is ~7 orders, so the
# split is not sensitive to where in it this sits.
REPRESSED = 1e-8

PAIRS = {
    # The clean pair, and the one the study's claim now rests on: one nutrient
    # differs from basal (TRP), the doubling time is basal's own 44.0 min, and
    # condition_defs.tsv declares exactly one active TF -- CPLX-125, the TrpR
    # holorepressor. Anything that moves between these two arms is attributable to
    # trpR, which is not true of the aa_replete pair below.
    "plus_trp": ("wt_withtrp", "trpR_withtrp",
                 "TrpR is declared ACTIVE; TRP is the ONLY nutrient difference from "
                 "basal and the doubling time is unchanged at 44.0 min"),
    "aa_replete": ("wt_withaa", "trpR_withaa",
                   "TrpR is declared ACTIVE in this condition (condition_defs.tsv). "
                   "⚠ 30 species and the doubling time (44->25 min) also change, so "
                   "this pair cannot separate tryptophan from growth rate -- that is "
                   "what plus_trp is for"),
    "basal": ("wt", "trpR",
              "TrpR is declared INACTIVE in this condition"),
    "basal_double": ("wt", "trpRtnaA",
                     "trpR + tnaA, basal; same inactive-TrpR caveat"),
}


def window_values(trace: list[dict], gen_steps: int) -> list[float]:
    """Second half of each generation — the window the bands declare."""
    return [r["trp_tu_synth_prob"] for r in trace
            if (r["step"] - 1) % gen_steps > gen_steps / 2]


def arm_stats(arm: str) -> dict | None:
    path = DATA / f"route_b_{arm}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    v = window_values(d["trace"], d["gen_steps"])
    if not v:
        return None
    active = [x for x in v if x >= REPRESSED]
    trpr = [r["trpR_monomer"] for r in d["trace"] if r["trpR_monomer"] is not None]
    return {
        "arm": arm, "n_samples": len(v),
        "time_mean": sum(v) / len(v),
        "fraction_repressed": 1 - len(active) / len(v),
        "level_when_active": (sum(active) / len(active)) if active else None,
        "trpR_monomer_min": min(trpr) if trpr else None,
        "trpR_monomer_max": max(trpr) if trpr else None,
        "generations": d["generations"], "wall_seconds": d["wall_seconds"],
    }


# The tryptophan dose-response arms: one full-mode ParCa state, one condition
# (basal_with_trp -- 44.0 min, pPromoterBound[CPLX-125]=0.944322344818007 for every
# arm), one variable: the external tryptophan the run sees.
DOSE_ARMS = [
    ("wt_trpzero",   "WT,  TRP 0.0"),
    ("trpR_trpzero", "KO,  TRP 0.0"),
    ("wt_trplow",    "WT,  TRP 0.1 (= with_aa)"),
    ("wt_trpinf",    "WT,  TRP inf"),
    ("trpR_trpinf",  "KO,  TRP inf"),
]

# cell density used to turn counts into a concentration (g/L), and Avogadro.
CELL_DENSITY_G_PER_L = 1100.0
N_AVOGADRO = 6.02214076e23
# TrpR corepressor dissociation constant, read off the model's own equilibrium
# (rev/fwd rate for CPLX-125[c]); the metabolism homeostatic target for TRP[c] is
# the SAME number, which is what pins fitted occupancy at 0.5.
TRPR_KD_M = 2.17e-05


def dose_stats(arm: str) -> dict | None:
    """Per-arm second-half-window summary, keyed on what TrpR actually senses.

    ⚠ The external concentration is NOT the axis. A cell in `minimal` synthesises
    tryptophan de novo through the very operon under study, so external 0.0 does not
    mean the repressor has no corepressor. Intracellular TRP[c] is the quantity the
    holorepressor equilibrium binds, so it is converted to a concentration here
    (counts -> mol/L via cell mass and density) rather than reported as counts,
    which are not comparable across arms of differing cell size.
    """
    path = DATA / f"route_b_{arm}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    gs = d["gen_steps"]
    rows = [r for r in d["trace"] if (r["step"] - 1) % gs > gs / 2]
    if not rows:
        return None

    v = [r["trp_tu_synth_prob"] for r in rows]
    active = [x for x in v if x >= REPRESSED]

    conc = []
    for r in rows:
        trp, cm = r.get("trp"), r.get("cell_mass_fg")
        if trp is None or not cm:
            continue
        vol_L = (cm * 1e-15) / CELL_DENSITY_G_PER_L
        conc.append((trp / N_AVOGADRO) / vol_L)

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    # n_actual_bound ACCUMULATES; its slope is the binding rate. Differenced across
    # the window rather than pairwise so one missing sample cannot fake a negative.
    cum = [(r["step"], r["n_actual_bound_trpR_cumulative"]) for r in rows
           if r.get("n_actual_bound_trpR_cumulative") is not None]
    rate = None
    if len(cum) >= 2 and cum[-1][0] != cum[0][0]:
        rate = (cum[-1][1] - cum[0][1]) / (cum[-1][0] - cum[0][0])

    trp_mean = (sum(conc) / len(conc)) if conc else None
    return {
        "arm": arm, "n_samples": len(v),
        "time_mean_synth_prob": sum(v) / len(v),
        "fraction_repressed": 1 - len(active) / len(v),
        "trp_c_molar_mean": trp_mean,
        "trp_c_molar_min": min(conc) if conc else None,
        "trp_c_molar_max": max(conc) if conc else None,
        "implied_occupancy_at_mean": (trp_mean / (trp_mean + TRPR_KD_M)) if trp_mean else None,
        "mean_implied_occupancy": (sum(c / (c + TRPR_KD_M) for c in conc) / len(conc))
                                  if conc else None,
        "trpR_holo_free_mean": mean("trpR_holo"),
        "trpR_apo_free_mean": mean("trpR_apo"),
        "dry_mass_fg_mean": mean("dry_mass_fg"),
        "trpR_binding_events_per_s": rate,
    }


def main() -> int:
    out = {"statistic": "time_mean of trp TU (TU00067) synthesis probability, "
                        "second-half window; ratio KO/WT",
           "repressed_threshold": REPRESSED, "pairs": {}, "arms": {}}
    for key, (wt, ko, note) in PAIRS.items():
        w, k = arm_stats(wt), arm_stats(ko)
        if w is None or k is None:
            out["pairs"][key] = {"status": "missing arm", "wt": wt, "ko": ko}
            continue
        out["arms"][wt], out["arms"][ko] = w, k
        out["pairs"][key] = {
            "note": note, "wt": wt, "ko": ko,
            "ratio_ko_over_wt": k["time_mean"] / w["time_mean"],
        }
    dose = {}
    for arm, label in DOSE_ARMS:
        st = dose_stats(arm)
        if st is not None:
            st["label"] = label
            dose[arm] = st
    if dose:
        out["dose_response"] = {
            "design": "one full-mode ParCa state; one condition (basal_with_trp); the "
                      "only variable is the external tryptophan the run sees",
            "axis_note": "graded against INTRACELLULAR [TRP] (mol/L), not medium "
                         "composition -- a minimal-medium cell makes its own tryptophan "
                         "through the operon under study",
            "trpR_kd_M": TRPR_KD_M,
            "arms": dose,
        }

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}")
    for key, r in out["pairs"].items():
        if "ratio_ko_over_wt" not in r:
            print(f"  {key:<14} {r['status']}"); continue
        # pairs carry arm NAMES; the stats blobs live once in out["arms"].
        w, k = out["arms"][r["wt"]], out["arms"][r["ko"]]
        print(f"  {key:<14} KO/WT = {r['ratio_ko_over_wt']:.2f}x   "
              f"(WT repressed {w['fraction_repressed']:.0%} of the time, "
              f"KO {k['fraction_repressed']:.0%})")
    if dose:
        print("\n  --- tryptophan dose-response (one condition, one variable) ---")
        print(f"  {'arm':<14} {'label':<24} {'repr':>5} {'[TRP] uM mean':>14} "
              f"{'occ':>6} {'synth prob':>12}")
        for arm, st in dose.items():
            uM = f"{st['trp_c_molar_mean']*1e6:.2f}" if st['trp_c_molar_mean'] else "n/a"
            oc = f"{st['mean_implied_occupancy']:.3f}" if st['mean_implied_occupancy'] else "n/a"
            print(f"  {arm:<14} {st['label']:<24} {st['fraction_repressed']:>4.0%} "
                  f"{uM:>14} {oc:>6} {st['time_mean_synth_prob']:>12.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
