#!/usr/bin/env python
"""Arc 2 instrument — in-process metabolism physiology probe.

On current main, array-valued listener leaves (external_exchange_fluxes) are NOT
persisted by the emitter (OOM-safety, #752/#776/#777). This probe reads them
LIVE from the composite store (agents.0.listeners.fba_results.*) while stepping a
single cell, and computes the energy-balance observables the biomass-yield
investigation defined:

  biomass_yield (Yxs)  gDW / g glucose   (integrated mass balance)
  RQ                   CO2 / O2          (~1 for full glucose oxidation)
  O2:glucose           respiratory ratio
  growth_rate          1/h               (held-out check)

Fluxes are specific (mmol/gDW/h). Molecule indices are derived BY NAME from the
metabolism process's exchange-molecule ordering, so the probe is robust to media.

Usage:
  python metabolism_probe.py --cache-dir out/cache --minutes 40 --burn-in 8 \
      --label baseline --out arc2_baseline.json
"""
from __future__ import annotations
import argparse, json, os, re, uuid
import numpy as np

# canonical E. coli exchange-molecule ids, matched EXACTLY against the
# metabolism process's externalMoleculeIDs (the order the FBA emits
# external_exchange_fluxes in). Verified indices on basal: GLC 36, O2 65,
# CO2 10, ACET 2 (== the diagnostic's 1-indexed EX_IDX minus 1).
MOL_IDS = {
    "glucose": "GLC[p]",
    "o2": "OXYGEN-MOLECULE[p]",
    "co2": "CARBON-DIOXIDE[p]",
    "acetate": "ACET[p]",
}
M_GLC = 0.18015  # g per mmol glucose


def _agent_listeners(state):
    ag = state.get("agents") or {}
    # follow the (single) live agent
    for k, v in ag.items():
        if isinstance(v, dict) and "listeners" in v:
            return v["listeners"]
    return state.get("listeners", {})


def _q(x):
    """pint Quantity or plain -> float magnitude."""
    return float(getattr(x, "magnitude", x))


def find_indices(exchange_ids):
    ids = list(map(str, exchange_ids))
    idx = {}
    for name, mid in MOL_IDS.items():
        if mid in ids:
            idx[name] = ids.index(mid)
    return idx


def _build(cache_dir, dark_atp_scale, ngam_scale):
    """Build ecoli_baseline, optionally scaling the GAM (dark_atp) / NGAM
    maintenance-ATP config values via a patched cache bundle (no ParCa rerun)."""
    # We read fluxes in-process, so disable the persisting emitter entirely —
    # avoids parallel probes colliding on the default .pbg/parquet-runs/baseline/
    # hive path (the composite's default decl hardcodes experiment_id=baseline).
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    from v2ecoli import build_composite
    if dark_atp_scale == 1.0 and ngam_scale == 1.0:
        return build_composite("ecoli_baseline", cache_dir=cache_dir), {}
    from v2ecoli.core import load_cache_bundle, build_core
    from v2ecoli.composites.ecoli_baseline import baseline as baseline_doc
    from process_bigraph import Composite
    core = build_core()
    bundle = dict(load_cache_bundle(cache_dir))
    configs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in bundle["configs"].items()}
    met = dict(configs["ecoli-metabolism"])
    applied = {}
    if dark_atp_scale != 1.0:
        met["dark_atp"] = met["dark_atp"] * dark_atp_scale
        applied["dark_atp"] = str(met["dark_atp"])
    if ngam_scale != 1.0:
        met["ngam"] = met["ngam"] * ngam_scale
        applied["ngam"] = str(met["ngam"])
    configs["ecoli-metabolism"] = met
    bundle["configs"] = configs
    doc = baseline_doc(core=core, seed=0, cache_dir=cache_dir, bundle=bundle)
    return Composite(doc, core=core), applied


def run(cache_dir: str, minutes: float, burn_in: float, step_s: float = 10.0,
        dark_atp_scale: float = 1.0, ngam_scale: float = 1.0):
    comp, applied = _build(cache_dir, dark_atp_scale, ngam_scale)
    # exchange-molecule ordering (len == external_exchange_fluxes)
    met = None
    def find_proc(state, needle):
        for k, v in state.items():
            if isinstance(v, dict):
                if "instance" in v and needle in str(k).lower():
                    return v["instance"]
                r = find_proc(v, needle)
                if r is not None:
                    return r
        return None
    met = find_proc(comp.state, "metabol")
    # external_exchange_fluxes is emitted in externalMoleculeIDs order.
    ex_ids = list(getattr(met, "externalMoleculeIDs", []))
    idx = find_indices(ex_ids)
    if set(idx) < {"glucose", "o2", "co2"}:
        raise SystemExit(f"could not map molecules; found {idx} of {len(ex_ids)} ids; "
                         f"sample={ex_ids[:5]}")

    records = []
    t = 0.0
    prev_dw = None
    glc_consumed_g = 0.0
    biomass_made_g = 0.0
    n_steps = int(minutes * 60.0 / step_s)
    for _ in range(n_steps):
        comp.update({}, step_s)
        t += step_s
        lst = _agent_listeners(comp.state)
        fba = lst.get("fba_results", {})
        eef = fba.get("external_exchange_fluxes")
        mass = lst.get("mass", {})
        dw = _q(mass.get("dry_mass", 0.0))  # fg
        if eef is None or len(eef) == 0 or dw <= 0:
            continue
        eef = np.asarray(eef, dtype=float)
        q = {k: float(eef[i]) for k, i in idx.items()}  # mmol/gDW/h, signed
        gr = _q(mass.get("instantaneous_growth_rate", 0.0)) * 3600.0  # 1/h
        rec = {"t_min": t / 60.0, "dry_mass_fg": dw, "growth_per_h": gr, **{f"q_{k}": v for k, v in q.items()}}
        records.append(rec)
        # integrate mass balance over the post-burn-in window
        if t / 60.0 >= burn_in and prev_dw is not None:
            dw_g = dw * 1e-15
            dt_h = step_s / 3600.0
            glc_consumed_g += abs(q["glucose"]) * dw_g * dt_h * M_GLC
            biomass_made_g += max(0.0, (dw - prev_dw) * 1e-15)
        prev_dw = dw

    post = [r for r in records if r["t_min"] >= burn_in]
    def mean(key):
        vals = [abs(r[key]) for r in post if key in r]
        return float(np.mean(vals)) if vals else None
    q_glc, q_o2, q_co2 = mean("q_glucose"), mean("q_o2"), mean("q_co2")
    q_ace = mean("q_acetate") if "acetate" in idx else 0.0
    rq = (q_co2 / q_o2) if (q_o2 and q_o2 > 1e-9) else None
    o2_glc = (q_o2 / q_glc) if (q_glc and q_glc > 1e-9) else None
    yxs = (biomass_made_g / glc_consumed_g) if glc_consumed_g > 1e-30 else None
    growth = float(np.mean([r["growth_per_h"] for r in post if r["growth_per_h"] > 0])) if post else None
    return {
        "cache_dir": cache_dir, "minutes": minutes, "burn_in_min": burn_in,
        "n_steps_post": len(post),
        "overrides": {"dark_atp_scale": dark_atp_scale, "ngam_scale": ngam_scale, "applied": applied},
        "exchange_indices": idx,
        "exchange_mmol_gDW_h": {"glucose": q_glc, "o2": q_o2, "co2": q_co2, "acetate": q_ace},
        "RQ": rq, "O2_glucose": o2_glc,
        "biomass_yield_gDW_g_glucose": yxs,
        "growth_rate_per_h": growth,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--burn-in", type=float, default=8.0)
    ap.add_argument("--step-s", type=float, default=10.0)
    ap.add_argument("--dark-atp-scale", type=float, default=1.0,
                    help="scale factor on GAM (dark_atp) maintenance ATP")
    ap.add_argument("--ngam-scale", type=float, default=1.0,
                    help="scale factor on NGAM maintenance ATP")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.cache_dir, args.minutes, args.burn_in, args.step_s,
              dark_atp_scale=args.dark_atp_scale, ngam_scale=args.ngam_scale)
    if args.label:
        res["label"] = args.label
    print(json.dumps(res, indent=2))
    if args.out:
        here = os.path.dirname(os.path.abspath(__file__))
        path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
        json.dump(res, open(path, "w"), indent=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
