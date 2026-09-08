#!/usr/bin/env python
"""Cell-level observable aggregation for the multiscale-complexity-showcase.

Reads a parquet hive produced by scripts/run_condition_multigen_parquet.py and
aggregates the discovery observables with the standard cell-level discipline:
time-average WITHIN each cell over the post-burn-in window, then take the
ensemble mean/quantities ACROSS cells. Per-cell values are retained for
distribution views (Arc 3).

Observables (native listener columns):
  origins_per_cell   listeners__replication_data__number_of_oric
  growth_rate        listeners__mass__instantaneous_growth_rate   (1/s -> 1/h)
  ppgpp_conc         listeners__growth_limits__ppgpp_conc
  ribosome_conc      listeners__growth_limits__ribosome_conc
  rela_conc          listeners__growth_limits__rela_conc
  cell_mass          listeners__mass__cell_mass  (fg)
  dry_mass           listeners__mass__dry_mass   (fg)

Usage:
  python observables.py --hive out/arc1_basal/arc1_basal --gen-lb 1 --label basal
  python observables.py --hive out/arc1_basal/arc1_basal --json   # machine-readable
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import pyarrow.parquet as pq

COLS = {
    "origins_per_cell": "listeners__replication_data__number_of_oric",
    "growth_rate_per_s": "listeners__mass__instantaneous_growth_rate",
    "ppgpp_conc": "listeners__growth_limits__ppgpp_conc",
    "ribosome_conc": "listeners__growth_limits__ribosome_conc",
    "rela_conc": "listeners__growth_limits__rela_conc",
    "cell_mass": "listeners__mass__cell_mass",
    "dry_mass": "listeners__mass__dry_mass",
}


def _cell_partitions(hive: str):
    """Yield (generation:int, list-of-parquet-files) per cell partition."""
    hist = os.path.join(hive, "history")
    for gendir in sorted(glob.glob(os.path.join(hist, "**", "generation=*"), recursive=True)):
        m = re.search(r"generation=(\d+)", gendir)
        gen = int(m.group(1)) if m else 0
        files = sorted(glob.glob(os.path.join(gendir, "**", "*.pq"), recursive=True))
        if files:
            yield gen, files


def _read_cols(files, names):
    keep = None
    tables = []
    for f in files:
        avail = set(pq.ParquetFile(f).schema.names)
        want = [n for n in names if n in avail]
        keep = want if keep is None else keep
        tables.append(pq.read_table(f, columns=want).to_pydict())
    out = {}
    for n in keep:
        vals = []
        for t in tables:
            vals.extend(t.get(n, []))
        out[n] = np.asarray(vals, dtype=float)
    return out


def aggregate(hive: str, gen_lb: int = 1):
    """Per-cell time-averages (gen >= gen_lb), then ensemble summary."""
    per_cell = {k: [] for k in COLS}
    n_cells = 0
    for gen, files in _cell_partitions(hive):
        if gen < gen_lb:
            continue
        data = _read_cols(files, list(COLS.values()))
        if not data:
            continue
        n_cells += 1
        for k, col in COLS.items():
            arr = data.get(col)
            per_cell[k].append(float(np.nanmean(arr)) if arr is not None and arr.size else np.nan)
    summary = {"n_cells": n_cells, "gen_lb": gen_lb}
    for k, vals in per_cell.items():
        a = np.asarray(vals, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            summary[k] = None
            continue
        summary[k] = {
            "mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "cv": float(np.std(a, ddof=1) / np.mean(a)) if a.size > 1 and np.mean(a) else 0.0,
            "values": [float(x) for x in a],
        }
    # Convenience derived: growth per hour, initiation mass (cell_mass/origins)
    gr = summary.get("growth_rate_per_s")
    if gr:
        summary["growth_rate_per_h"] = {"mean": gr["mean"] * 3600.0,
                                         "sd": gr["sd"] * 3600.0}
    cm, orig = summary.get("cell_mass"), summary.get("origins_per_cell")
    if cm and orig and orig["mean"]:
        summary["initiation_mass_fg"] = {"mean": cm["mean"] / orig["mean"]}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hive", required=True, help="path to <out>/<experiment_id> hive root")
    ap.add_argument("--gen-lb", type=int, default=1, help="lowest generation to include (burn-in)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = aggregate(args.hive, args.gen_lb)
    if args.label:
        s["label"] = args.label
    if args.json:
        print(json.dumps(s))
        return
    print(f"[{args.label or args.hive}] n_cells={s['n_cells']} (gen>={s['gen_lb']})")
    for k in ("origins_per_cell", "growth_rate_per_h", "ppgpp_conc", "ribosome_conc",
              "rela_conc", "initiation_mass_fg", "cell_mass"):
        v = s.get(k)
        if isinstance(v, dict):
            extra = f"  CV={v['cv']:.3f}" if "cv" in v else ""
            print(f"  {k:20s} mean={v['mean']:.4g}{extra}")


if __name__ == "__main__":
    main()
