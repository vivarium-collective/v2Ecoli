#!/usr/bin/env python
"""Macromolecular composition growth law (Bremer-Dennis / Scott-Hwa):
does RNA/protein mass ratio rise with growth rate? Reads arc1 parquet hives."""
import glob, json, os, re
import numpy as np, pyarrow.parquet as pq

COLS = {"protein": "listeners__mass__protein_mass", "rna": "listeners__mass__rna_mass",
        "rrna": "listeners__mass__rRna_mass", "dna": "listeners__mass__dna_mass",
        "dry": "listeners__mass__dry_mass", "gr": "listeners__mass__instantaneous_growth_rate"}

def cond_means(hive, gen_lb=1):
    per = {k: [] for k in COLS}
    for gendir in sorted(glob.glob(os.path.join(hive, "**", "generation=*"), recursive=True)):
        if "history" not in gendir: continue
        g = int(re.search(r"generation=(\d+)", gendir).group(1))
        if g < gen_lb: continue
        files = sorted(glob.glob(os.path.join(gendir, "**", "*.pq"), recursive=True))
        if not files: continue
        acc = {k: [] for k in COLS}
        for f in files:
            avail = set(pq.ParquetFile(f).schema.names)
            d = pq.read_table(f, columns=[c for c in COLS.values() if c in avail]).to_pydict()
            for k, c in COLS.items():
                if c in d: acc[k].extend([float(x) for x in d[c]])
        if acc["dry"] and np.mean(acc["dry"]) > 100:  # real cell (skip stubs)
            for k in COLS:
                if acc[k]: per[k].append(float(np.nanmean(acc[k])))
    out = {k: (float(np.mean(v)) if v else None) for k, v in per.items()}
    out["n_cells"] = len(per["dry"])
    out["growth_per_h"] = out["gr"] * 3600 if out["gr"] else None
    out["rna_protein_ratio"] = out["rna"] / out["protein"] if out["protein"] else None
    out["dna_dry_ratio"] = out["dna"] / out["dry"] if out["dry"] else None
    return out

res = {"minimal": cond_means("out/arc1_basal/arc1_basal", 1),
       "rich": cond_means("out/arc1_withaa/arc1_withaa", 1)}
# Bremer-Dennis: RNA/protein rises with growth rate. slope sign check.
m, r = res["minimal"], res["rich"]
slope = (r["rna_protein_ratio"] - m["rna_protein_ratio"]) / (r["growth_per_h"] - m["growth_per_h"])
res["rna_protein_slope_per_h"] = slope
res["reproduces_positive_slope"] = slope > 0
here = os.path.dirname(os.path.abspath(__file__))
json.dump(res, open(os.path.join(here, "composition.json"), "w"), indent=2)
print(json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in res.items()}, indent=2))
