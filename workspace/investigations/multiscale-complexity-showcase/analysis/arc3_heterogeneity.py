#!/usr/bin/env python
"""Arc 3 — single-cell heterogeneity from a seed ensemble.

Reads the per-seed parquet hives (out/arc3_seed*/arc3_seed*) and, per cell
(generation partition, gen >= burn-in), extracts:
  division_time (tau)   generation duration from global_time range (min)
  birth_mass            first dry_mass in the generation (fg)
  division_mass         last dry_mass in the generation (fg)
  added_mass            division_mass - birth_mass  (adder test)

Then reports the ensemble interdivision-time CV and the adder slope
(regression of added_mass on birth_mass; slope ~0 == adder).

Usage: python arc3_heterogeneity.py --glob 'out/arc3_seed*' --gen-lb 2 --json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import pyarrow.parquet as pq

T = "global_time"
DW = "listeners__mass__dry_mass"


def cell_records(hive_glob: str, gen_lb: int):
    recs = []
    for seed_root in sorted(glob.glob(hive_glob)):
        # hive is <root>/<experiment_id>/history/...
        for gendir in sorted(glob.glob(os.path.join(seed_root, "**", "agent_id=*"), recursive=True)):
            if "history" not in gendir:
                continue
            gen = int(re.search(r"generation=(\d+)", gendir).group(1))
            agent = re.search(r"agent_id=(\w+)", gendir).group(1)
            # Follow ONE lineage only: the mother-line agent_id is all zeros
            # ("0","00",...). The sibling daughter (…1) and terminal daughters
            # emit ~1-min stubs — exclude them.
            if set(agent) != {"0"}:
                continue
            if gen < gen_lb:
                continue
            files = sorted(glob.glob(os.path.join(gendir, "**", "*.pq"), recursive=True),
                           key=lambda p: int(re.search(r"(\d+)\.pq$", p).group(1)) if re.search(r"(\d+)\.pq$", p) else 0)
            if not files:
                continue
            ts, dws = [], []
            for f in files:
                cols = set(pq.ParquetFile(f).schema.names)
                use = [c for c in (T, DW) if c in cols]
                d = pq.read_table(f, columns=use).to_pydict()
                ts.extend(d.get(T, []))
                dws.extend(d.get(DW, []))
            if len(ts) < 2 or len(dws) < 2:
                continue
            ts = np.asarray(ts, float); dws = np.asarray(dws, float)
            order = np.argsort(ts)
            ts, dws = ts[order], dws[order]
            tau_min = (ts[-1] - ts[0]) / 60.0
            # Exclude terminal-daughter stubs (a full basal cycle is ~40-56 min;
            # stubs are ~1 min tails emitted at the last division).
            if tau_min < 20.0:
                continue
            recs.append({"seed_root": os.path.basename(seed_root), "gen": gen,
                         "division_time_min": float(tau_min),
                         "birth_mass_fg": float(dws[0]),
                         "division_mass_fg": float(dws[-1]),
                         "added_mass_fg": float(dws[-1] - dws[0])})
    return recs


def summarize(recs):
    taus = np.array([r["division_time_min"] for r in recs], float)
    birth = np.array([r["birth_mass_fg"] for r in recs], float)
    added = np.array([r["added_mass_fg"] for r in recs], float)
    out = {"n_cells": len(recs)}
    if len(taus):
        out["division_time"] = {"mean_min": float(taus.mean()),
                                 "sd_min": float(taus.std(ddof=1)) if len(taus) > 1 else 0.0,
                                 "cv": float(taus.std(ddof=1) / taus.mean()) if len(taus) > 1 and taus.mean() else 0.0}
    if len(birth) > 2:
        slope, intercept = np.polyfit(birth, added, 1)
        out["adder"] = {"slope": float(slope), "intercept_fg": float(intercept),
                        "interpretation": "adder (slope~0)" if abs(slope) < 0.3 else
                                          ("timer (slope~ -1)" if slope < -0.7 else
                                           "sizer (slope>0)" if slope > 0.3 else "intermediate")}
    out["cells"] = recs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/arc3_seed*")
    ap.add_argument("--gen-lb", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # resolve relative to workspace root (two up from analysis dir) if needed
    recs = cell_records(args.glob, args.gen_lb)
    if not recs:
        # try workspace-root-relative
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        recs = cell_records(os.path.join(root, args.glob), args.gen_lb)
    s = summarize(recs)
    if args.out:
        here = os.path.dirname(os.path.abspath(__file__))
        json.dump(s, open(os.path.join(here, args.out), "w"), indent=2)
    if args.json:
        print(json.dumps({k: v for k, v in s.items() if k != "cells"}))
    else:
        print(f"n_cells={s['n_cells']}")
        if "division_time" in s:
            d = s["division_time"]; print(f"  division_time: mean {d['mean_min']:.1f} min  CV {d['cv']:.3f}")
        if "adder" in s:
            a = s["adder"]; print(f"  adder slope {a['slope']:.3f} ({a['interpretation']})")


if __name__ == "__main__":
    main()
