#!/usr/bin/env python
"""Test the chemiosmotic loop-law constraint before building the process module.

Root cause (mcs-11): transport reactions (TRANS-RXN*) form free-energy-generating proton
futile cycles that build proton-motive force for free, so the cell never respires.

Principled constraint: net proton translocation to the periplasm BY TRANSPORT must be <= 0
(transport CONSUMES the gradient; only the electron-transport chain may net-pump protons).
Implemented via a pseudo-metabolite PMF_TRANS[c] added to every TRANS-RXN* proton-translocating
reaction with coefficient = its PROTON[p] stoichiometry, balanced by a source reaction with
flux >= 0 (which forces sum of transport net-H+[p] <= 0). The ETC (NADH-DEHYDROG*, cytochrome
oxidases) and ATP synthase are untouched, so respiration can still pump/consume protons.

Usage: python loopless_test.py --cache-dir out/cache
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metabolism_probe import find_indices, _agent_listeners, _q, M_GLC
from etc_fix_attempt import get_met

PSEUDO = "PMF_TRANS[c]"
RELIEF = "PMF_TRANS_RELIEF"


def patch_reaction_stoich(rs: dict) -> int:
    """Add the PMF_TRANS pseudo-metabolite to TRANS-RXN* proton translocators + a relief source."""
    touched = 0
    for rid in list(rs.keys()):
        st = rs[rid]
        if rid.startswith("TRANS-RXN") and isinstance(st, dict) and "PROTON[p]" in st:
            st = dict(st)
            st[PSEUDO] = st["PROTON[p]"]     # this reaction's net periplasmic-proton contribution
            rs[rid] = st
            touched += 1
    rs[RELIEF] = {PSEUDO: 1}                  # source: flux>=0 => sum(TRANS net H+[p]) <= 0
    return touched


def run(cache_dir, minutes, burn_in, apply_fix, step_s=10.0):
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    import copy
    from v2ecoli.core import load_cache_bundle, build_core
    from v2ecoli.composites.ecoli_baseline import baseline as baseline_doc
    from process_bigraph import Composite

    core = build_core()
    bundle = dict(load_cache_bundle(cache_dir))
    configs = {k: (copy.deepcopy(v) if isinstance(v, dict) else v) for k, v in bundle["configs"].items()}
    met_cfg = configs["ecoli-metabolism"]
    touched = 0
    if apply_fix:
        touched = patch_reaction_stoich(met_cfg["reaction_stoich"])
        # register the relief reaction in the base-reaction bookkeeping so the
        # metabolism process's flux-aggregation init does not KeyError on it
        m = dict(met_cfg["fba_reaction_ids_to_base_reaction_ids"]); m[RELIEF] = RELIEF
        met_cfg["fba_reaction_ids_to_base_reaction_ids"] = m
        bri = list(met_cfg["base_reaction_ids"])
        if RELIEF not in bri:
            bri.append(RELIEF)
        met_cfg["base_reaction_ids"] = bri
    bundle["configs"] = configs
    doc = baseline_doc(core=core, seed=0, cache_dir=cache_dir, bundle=bundle)
    comp = Composite(doc, core=core)
    comp.update({}, step_s)
    met = get_met(comp.state)
    rids = list(met.fba_reaction_ids)
    idx = find_indices(list(getattr(met, "externalMoleculeIDs", [])))

    # ensure the relief reaction can only run forward (flux >= 0) so the constraint bites
    fba = met.model.fba
    orig_solve = fba.solve
    have_relief = RELIEF in rids

    def patched_solve(*a, **k):
        if apply_fix and have_relief:
            try:
                fba.setReactionFluxBounds([RELIEF], lowerBounds=[0.0], upperBounds=[1e6])
            except Exception:
                try:
                    fba.setReactionFluxBounds(RELIEF, lowerBounds=0.0, upperBounds=1e6)
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
        "apply_fix": apply_fix, "trans_reactions_constrained": touched,
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
    ap.add_argument("--no-fix", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.cache_dir, a.minutes, a.burn_in, apply_fix=not a.no_fix)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=2)
