#!/usr/bin/env python
"""mcs-11 stoichiometric correction — change the ATP-synthase H+/ATP ratio.

mcs-11 localized the ATP-supply defect as STOICHIOMETRIC (no flux bound works). This
patches the oxidative-phosphorylation stoichiometry in the cache bundle and rebuilds:
the ATP-synthase reactions (synthesis "ATPSYN-RXN (reverse)" and hydrolysis "ATPSYN-RXN")
carry a transmembrane PROTON[p] coefficient of 4 (H+/ATP = 4); E. coli's c10 ring is
~3.3. --hplusper sets the new transmembrane H+/ATP; the cell is then run FREE (no flux
bounds) so we see whether the corrected stoichiometry makes it respire at an in-band yield
with preserved growth. Reports yield / RQ / O2 / acetate / growth + true ATPSYN fluxes.

Usage: python etc_stoich_fix.py --cache-dir out/cache --hplusper 3
"""
from __future__ import annotations
import argparse, json, os, sys, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metabolism_probe import find_indices, _agent_listeners, _q, M_GLC
from etc_fix_attempt import get_met

SYN = "ATPSYN-RXN (reverse)"   # NB: this is the SYNTHESIS direction (makes ATP)
HYD = "ATPSYN-RXN"             # this is the HYDROLYSIS direction (consumes ATP)


def run(cache_dir, minutes, burn_in, hplusper, step_s=10.0):
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    from v2ecoli.core import load_cache_bundle, build_core
    from v2ecoli.composites.ecoli_baseline import baseline as baseline_doc
    from process_bigraph import Composite

    core = build_core()
    bundle = dict(load_cache_bundle(cache_dir))
    configs = {k: (copy.deepcopy(v) if isinstance(v, dict) else v) for k, v in bundle["configs"].items()}
    met_cfg = configs["ecoli-metabolism"]
    rs = met_cfg["reaction_stoich"]
    applied = {}
    if hplusper is not None and SYN in rs and HYD in rs:
        h = int(hplusper)
        # synthesis: ADP + Pi + h H+[p] -> ATP + H2O + (h-1) H+[c]
        rs[SYN] = dict(rs[SYN]); rs[SYN]["PROTON[p]"] = -h; rs[SYN]["PROTON[c]"] = h - 1
        # hydrolysis: mirror image
        rs[HYD] = dict(rs[HYD]); rs[HYD]["PROTON[p]"] = h; rs[HYD]["PROTON[c]"] = -(h - 1)
        applied = {"hplusper": h, "synthesis": rs[SYN]}
    met_cfg["reaction_stoich"] = rs
    bundle["configs"] = configs

    doc = baseline_doc(core=core, seed=0, cache_dir=cache_dir, bundle=bundle)
    comp = Composite(doc, core=core)
    comp.update({}, step_s)
    met = get_met(comp.state)
    rids = list(met.fba_reaction_ids)
    idx = find_indices(list(getattr(met, "externalMoleculeIDs", [])))

    recs, t, prev_dw, glc_g, bio_g, infeasible = [], step_s, None, 0.0, 0.0, 0
    syn_f, hyd_f = [], []
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
        if t/60 >= burn_in:
            try:
                fl = list(met.model.fba.getReactionFluxes())
                if SYN in rids: syn_f.append(fl[rids.index(SYN)])
                if HYD in rids: hyd_f.append(fl[rids.index(HYD)])
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
        "cache_dir": cache_dir, "hplusper": hplusper, "applied": applied,
        "infeasible_ticks": infeasible, "n_post": len(post),
        "atpsyn_synthesis_flux": float(np.mean(syn_f)) if syn_f else None,
        "atpsyn_hydrolysis_flux": float(np.mean(hyd_f)) if hyd_f else None,
        "exchange_mmol_gDW_h": {"glucose": q_glc, "o2": q_o2, "co2": q_co2, "acetate": q_ace},
        "RQ": (q_co2/q_o2) if (q_o2 and q_o2 > 1e-9) else None,
        "O2_glucose": (q_o2/q_glc) if (q_glc and q_glc > 1e-9) else None,
        "biomass_yield_gDW_g_glucose": (bio_g/glc_g) if glc_g > 1e-30 else None,
        "growth_rate_per_h": float(np.mean([r["gr"] for r in post if r.get("gr", 0) > 0])) if post else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--minutes", type=float, default=18.0)
    ap.add_argument("--burn-in", type=float, default=6.0)
    ap.add_argument("--hplusper", type=float, default=None,
                    help="new transmembrane H+/ATP at ATP synthase (default 4 = unchanged)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.cache_dir, a.minutes, a.burn_in, a.hplusper)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=2)
