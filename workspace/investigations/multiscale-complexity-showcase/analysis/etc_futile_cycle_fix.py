#!/usr/bin/env python
"""mcs-11 ROOT-CAUSE fix — eliminate the free-energy futile proton cycle.

The ATP-supply defect's true root cause: a succinate/proton futile cycle generates
proton-motive force for FREE (thermodynamically impossible), which drives ATP synthase
synthesis without respiration.
  TRANS-RXN-300:   SUC[c] + 3 H+[c] -> SUC[p] + 3 H+[p]   (export, 3 H+ out)
  TRANS-RXN0-517:  SUC[p] + 2 H+[p] -> SUC[c] + 2 H+[c]   (import, 2 H+ in)
Run together (both ~18 flux) they leave succinate unchanged but pump 1 net H+
cytoplasm->periplasm per turn — a proton pump powered by nothing.

The principled fix: cap the futile export leg (TRANS-RXN-300) so the cell must build its
proton gradient by O2-coupled respiration. --cap sets its upper bound (0 = fully blocked).
Runs FREE (no other bounds) and reports yield / RQ / O2 / acetate / growth so we can see
whether the cell now respires at a physiological yield with preserved growth.

Usage: python etc_futile_cycle_fix.py --cache-dir out/cache --cap 0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metabolism_probe import find_indices, _agent_listeners, _q, M_GLC
from etc_fix_attempt import get_met

RXN = "TRANS-RXN-300"   # succinate-export proton-pumping leg of the futile cycle


def run(cache_dir, minutes, burn_in, cap, step_s=10.0):
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline", cache_dir=cache_dir)
    comp.update({}, step_s)
    met = get_met(comp.state)
    fba = met.model.fba
    rids = list(met.fba_reaction_ids)
    idx = find_indices(list(getattr(met, "externalMoleculeIDs", [])))
    have = RXN in rids
    orig_solve = fba.solve

    def patched_solve(*a, **k):
        if cap is not None and have:
            try:
                fba.setReactionFluxBounds([RXN], lowerBounds=[0.0], upperBounds=[float(cap)])
            except Exception:
                try:
                    fba.setReactionFluxBounds(RXN, lowerBounds=0.0, upperBounds=float(cap))
                except Exception:
                    pass
        return orig_solve(*a, **k)
    fba.solve = patched_solve

    recs, t, prev_dw, glc_g, bio_g, infeasible = [], step_s, None, 0.0, 0.0, 0
    for _ in range(int(minutes * 60 / step_s)):
        try:
            comp.update({}, step_s)
        except Exception:
            infeasible += 1; t += step_s; continue
        t += step_s
        L = _agent_listeners(comp.state)
        eef = L.get("fba_results", {}).get("external_exchange_fluxes")
        dw = _q(L.get("mass", {}).get("dry_mass", 0.0))
        if eef is None or len(eef) == 0 or dw <= 0:
            continue
        eef = np.asarray(eef, float)
        q = {kk: float(eef[i]) for kk, i in idx.items()}
        gr = _q(L.get("mass", {}).get("instantaneous_growth_rate", 0.0)) * 3600
        recs.append({"t_min": t/60, "dw": dw, "gr": gr, **{f"q_{kk}": vv for kk, vv in q.items()}})
        if t/60 >= burn_in and prev_dw is not None:
            glc_g += abs(q["glucose"]) * dw*1e-15 * (step_s/3600) * M_GLC
            bio_g += max(0.0, (dw - prev_dw)*1e-15)
        prev_dw = dw

    post = [r for r in recs if r.get("t_min", 0) >= burn_in and "q_glucose" in r]
    def mean(k):
        vs = [abs(r[k]) for r in post if k in r]
        return float(np.mean(vs)) if vs else None
    q_glc, q_o2, q_co2, q_ace = mean("q_glucose"), mean("q_o2"), mean("q_co2"), mean("q_acetate")
    return {
        "cache_dir": cache_dir, "capped_reaction": RXN, "cap": cap,
        "infeasible_ticks": infeasible, "n_post": len(post),
        "exchange_mmol_gDW_h": {"glucose": q_glc, "o2": q_o2, "co2": q_co2, "acetate": q_ace},
        "RQ": (q_co2/q_o2) if (q_o2 and q_o2 > 1e-9) else None,
        "O2_glucose": (q_o2/q_glc) if (q_glc and q_glc > 1e-9) else None,
        "biomass_yield_gDW_g_glucose": (bio_g/glc_g) if glc_g > 1e-30 else None,
        "growth_rate_per_h": float(np.mean([r["gr"] for r in post if r.get("gr", 0) > 0])) if post else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--minutes", type=float, default=16.0)
    ap.add_argument("--burn-in", type=float, default=6.0)
    ap.add_argument("--cap", type=float, default=0.0, help="upper bound on TRANS-RXN-300 (0 = block)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.cache_dir, a.minutes, a.burn_in, a.cap)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=2)
