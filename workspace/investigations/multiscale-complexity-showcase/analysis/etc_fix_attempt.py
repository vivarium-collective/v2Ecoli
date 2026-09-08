#!/usr/bin/env python
"""mcs-05 fix attempt — constrain ATP synthase to net-forward operation.

The mcs-05 diagnostic found ATP synthase runs in REVERSE (hydrolysis 7.48, synthesis 0),
so the cell makes no respiratory ATP. This attempts the flagged fix: force the reverse
reaction to 0 (ATP synthase net-forward only) by wrapping the FBA solve each tick, then
re-measures biomass yield / RQ / O2:glucose / growth via the same estimator as
metabolism_probe. If the FBA goes infeasible, the run will show it (zero/garbage fluxes).
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metabolism_probe import MOL_IDS, find_indices, _agent_listeners, _q, M_GLC

REV_RXN = "ATPSYN-RXN (reverse)"


def get_met(state):
    for aid, cell in state.get("agents", {}).items():
        if isinstance(cell, dict):
            for k, v in cell.items():
                if isinstance(v, dict) and "instance" in v and "metabol" in str(k).lower():
                    return v["instance"]
    return None


def run(cache_dir, minutes, burn_in, step_s=10.0, apply_fix=True, reverse_max=0.0):
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline", cache_dir=cache_dir)
    # build one step to instantiate the FBA, then patch solve
    comp.update({}, step_s)
    met = get_met(comp.state)
    fba = met.model.fba
    ex_ids = list(getattr(met, "externalMoleculeIDs", []))
    idx = find_indices(ex_ids)
    applied = False
    if apply_fix and REV_RXN in list(met.fba_reaction_ids):
        orig_solve = fba.solve
        def patched_solve(*a, **k):
            # Cap the reverse (ATPase / hydrolysis) direction of ATP synthase to a small
            # physiological maximum instead of letting it dissipate the ATP surplus freely.
            try:
                fba.setReactionFluxBounds([REV_RXN], lowerBounds=[0.0], upperBounds=[reverse_max])
            except Exception:
                fba.setReactionFluxBounds(REV_RXN, lowerBounds=0.0, upperBounds=reverse_max)
            return orig_solve(*a, **k)
        fba.solve = patched_solve
        applied = True

    recs = []
    t = step_s
    prev_dw = None
    glc_g = 0.0
    bio_g = 0.0
    infeasible_ticks = 0
    for _ in range(int(minutes * 60 / step_s)):
        try:
            comp.update({}, step_s)
        except Exception as e:
            infeasible_ticks += 1
            recs.append({"t_min": t/60, "error": str(e)[:60]})
            t += step_s
            continue
        t += step_s
        L = _agent_listeners(comp.state)
        fbares = L.get("fba_results", {})
        eef = fbares.get("external_exchange_fluxes")
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
    q_glc, q_o2, q_co2 = mean("q_glucose"), mean("q_o2"), mean("q_co2")
    return {
        "cache_dir": cache_dir, "fix_applied": applied, "fix_reaction": REV_RXN,
        "reverse_max": reverse_max,
        "infeasible_ticks": infeasible_ticks, "n_post": len(post),
        "exchange_mmol_gDW_h": {"glucose": q_glc, "o2": q_o2, "co2": q_co2},
        "RQ": (q_co2/q_o2) if (q_o2 and q_o2 > 1e-9) else None,
        "O2_glucose": (q_o2/q_glc) if (q_glc and q_glc > 1e-9) else None,
        "biomass_yield_gDW_g_glucose": (bio_g/glc_g) if glc_g > 1e-30 else None,
        "growth_rate_per_h": float(np.mean([r["gr"] for r in post if r.get("gr", 0) > 0])) if post else None,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--burn-in", type=float, default=8.0)
    ap.add_argument("--no-fix", action="store_true")
    ap.add_argument("--reverse-max", type=float, default=0.0,
                    help="upper bound on the reverse (ATPase) ATP-synthase flux")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.cache_dir, a.minutes, a.burn_in, apply_fix=not a.no_fix, reverse_max=a.reverse_max)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=2)
