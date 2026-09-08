#!/usr/bin/env python
"""mcs-11 follow-up — reformulate the ETC fix to FORCE forward respiration.

mcs-11 showed the reverse-cap (upper-bound on ATPSYN-RXN (reverse)) is not robust:
it PERMITS but does not FORCE respiration, so the FBA is degenerate (yield 0.395 or
0.819 at the same cap). This tests the principled reformulation: lower-bound the
FORWARD reaction (ATPSYN-RXN) to a positive value so a non-respiring optimum
(forward = 0) is INFEASIBLE by construction — respiration is mandatory, not optional.

Reports biomass yield / RQ / O2 / acetate AND the solved ATPSYN forward & reverse
fluxes, so robustness (forward > 0 always) is visible. --forward-min sweeps the floor;
--reverse-max optionally also caps the reverse (defaults to leaving it free).

Usage:
  python etc_fix_v2.py --cache-dir out/cache --forward-min 12 --minutes 20 --burn-in 6
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metabolism_probe import find_indices, _agent_listeners, _q, M_GLC
from etc_fix_attempt import get_met

FWD_RXN = "ATPSYN-RXN"
REV_RXN = "ATPSYN-RXN (reverse)"


def run(cache_dir, minutes, burn_in, step_s=10.0, forward_min=None, reverse_max=None):
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline", cache_dir=cache_dir)
    comp.update({}, step_s)                       # instantiate the FBA
    met = get_met(comp.state)
    fba = met.model.fba
    rids = list(met.fba_reaction_ids)
    idx = find_indices(list(getattr(met, "externalMoleculeIDs", [])))

    def _set(rid, lo, hi):
        try:
            fba.setReactionFluxBounds([rid], lowerBounds=[lo], upperBounds=[hi])
        except Exception:
            fba.setReactionFluxBounds(rid, lowerBounds=lo, upperBounds=hi)

    orig_solve = fba.solve
    have_fwd = FWD_RXN in rids
    have_rev = REV_RXN in rids

    def patched_solve(*a, **k):
        # FORCE forward respiration: lower-bound the forward reaction so forward=0
        # (the degenerate non-respiring optimum) is infeasible.
        if forward_min is not None and have_fwd:
            _set(FWD_RXN, float(forward_min), np.inf)
        if reverse_max is not None and have_rev:
            _set(REV_RXN, 0.0, float(reverse_max))
        return orig_solve(*a, **k)
    fba.solve = patched_solve

    recs, t, prev_dw, glc_g, bio_g, infeasible = [], step_s, None, 0.0, 0.0, 0
    fwd_fluxes, rev_fluxes = [], []
    for _ in range(int(minutes * 60 / step_s)):
        try:
            comp.update({}, step_s)
        except Exception as e:
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
        rec = {"t_min": t/60, "dw": dw, "gr": gr, **{f"q_{kk}": vv for kk, vv in q.items()}}
        recs.append(rec)
        if t/60 >= burn_in:
            try:
                fl = list(fba.getReactionFluxes())
                if have_fwd: fwd_fluxes.append(fl[rids.index(FWD_RXN)])
                if have_rev: rev_fluxes.append(fl[rids.index(REV_RXN)])
            except Exception:
                pass
            if prev_dw is not None:
                glc_g += abs(q["glucose"]) * dw*1e-15 * (step_s/3600) * M_GLC
                bio_g += max(0.0, (dw - prev_dw)*1e-15)
        prev_dw = dw

    post = [r for r in recs if r.get("t_min", 0) >= burn_in and "q_glucose" in r]
    def mean(k):
        vs = [abs(r[k]) for r in post if k in r]
        return float(np.mean(vs)) if vs else None
    q_glc, q_o2, q_co2, q_ace = mean("q_glucose"), mean("q_o2"), mean("q_co2"), mean("q_acetate")
    return {
        "cache_dir": cache_dir, "forward_min": forward_min, "reverse_max": reverse_max,
        "infeasible_ticks": infeasible, "n_post": len(post),
        "atpsyn_forward_flux": float(np.mean(fwd_fluxes)) if fwd_fluxes else None,
        "atpsyn_reverse_flux": float(np.mean(rev_fluxes)) if rev_fluxes else None,
        "exchange_mmol_gDW_h": {"glucose": q_glc, "o2": q_o2, "co2": q_co2, "acetate": q_ace},
        "RQ": (q_co2/q_o2) if (q_o2 and q_o2 > 1e-9) else None,
        "biomass_yield_gDW_g_glucose": (bio_g/glc_g) if glc_g > 1e-30 else None,
        "growth_rate_per_h": float(np.mean([r["gr"] for r in post if r.get("gr", 0) > 0])) if post else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--burn-in", type=float, default=6.0)
    ap.add_argument("--forward-min", type=float, default=None,
                    help="lower bound on the FORWARD ATPSYN-RXN flux (forces respiration)")
    ap.add_argument("--reverse-max", type=float, default=None,
                    help="optional upper bound on the reverse (ATPase) flux")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.cache_dir, a.minutes, a.burn_in, forward_min=a.forward_min, reverse_max=a.reverse_max)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=2)
