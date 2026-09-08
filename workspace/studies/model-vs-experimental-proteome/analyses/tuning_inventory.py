"""Enumerate every flat-file entry that tunes the model toward a desired outcome.

These molecules cannot serve as independent validation: for them the model was
adjusted toward the right answer, often with a comment saying so. They are
excluded from the headline model-vs-experiment comparison and scored separately
as a positive control.

Swept: EVERY tsv under flat/ including subdirectories (131 files), not just the
_modified/_added/_removed set. A first pass using a narrow keyword list missed
both fold_changes_removed.tsv and complexation_reactions_removed.tsv, so the
pattern here is deliberately broad and every hit is printed for inspection
rather than silently counted.

Run:  .venv/bin/python3 <this>
"""
from __future__ import annotations
import csv, json, pathlib, re, sys

D = pathlib.Path(".venv/lib/python3.12/site-packages/ecoli_sources/data/flat")
INTENT = re.compile(
    r"adjust|tune[d]?|to (get|match|increase|decrease|prevent|fix|improve|obtain)|"
    r"underexpress|overexpress|fit_sim_data|so that|correct(ed)? to|arbitrar|"
    r"deplet|causes issues|for the .* condition", re.I)
cl = lambda s: (s or "").strip('"')  # noqa: E731

def rows(p):
    with open(p, errors="replace") as fh:
        return list(csv.reader((l for l in fh if not l.startswith("#")), delimiter="\t"))

def ids_from(cell):
    v = cl(cell)
    if v.startswith("["):
        try:
            return [str(x) for x in json.loads(v)]
        except Exception:
            return []
    return [v] if v else []

# --- direct parameter adjustments -------------------------------------------
DIRECT = ["adjustments/rna_expression_adjustments.tsv",
          "adjustments/translation_efficiencies_adjustments.tsv",
          "adjustments/balanced_translation_efficiencies.tsv",
          "adjustments/rna_deg_rates_adjustments.tsv",
          "adjustments/protein_deg_rates_adjustments.tsv"]
direct = {}
for name in DIRECT:
    p = D / name
    if not p.exists():
        continue
    r = rows(p)
    got = [i for x in r[1:] for i in ids_from(x[0])]
    direct[name] = got
    print(f"  {name:52s} {len(got):3d} molecules")

# --- structural removals made to protect pools or fix expression ------------
STRUCTURAL = ["fold_changes_removed.tsv", "complexation_reactions_removed.tsv",
              "equilibrium_reactions_removed.tsv"]
structural = {}
print()
for name in STRUCTURAL:
    p = D / name
    r = rows(p)
    hits = [x for x in r[1:] if any(INTENT.search(c or "") for c in x)]
    structural[name] = hits
    print(f"  {name:52s} {len(hits):3d} of {len(r)-1} rows carry tuning intent")
    for x in hits[:3]:
        print(f"      {cl(x[0])[:34]:36s} {cl(x[-1])[:96]}")

# --- anything else across the whole tree ------------------------------------
print("\n  full sweep — any other TSV with tuning-intent comments:")
known = set(DIRECT) | set(STRUCTURAL)
for p in sorted(D.rglob("*.tsv")):
    rel = str(p.relative_to(D))
    if rel in known:
        continue
    try:
        txt = p.read_text(errors="replace")
    except Exception:
        continue
    # only look at comment-ish short fields, not sequence columns
    hits = [l for l in txt.splitlines()
            if INTENT.search(l) and not re.search(r"\t\"?[ACDEFGHIKLMNPQRSTVWY]{40,}", l)]
    if hits:
        print(f"    {rel:50s} {len(hits)} line(s)")
        for h in hits[:2]:
            print(f"        {h[:150]}")

mols = {re.sub(r"\[.\]$", "", m) for v in direct.values() for m in v}
print(f"\n  DIRECT parameter adjustments : {len(mols)} distinct molecules")
print(f"  STRUCTURAL removals          : "
      f"{sum(len(v) for v in structural.values())} rows across {len(structural)} files")
print("  NOTE: this is DATA-level only. Several _comments name fit_sim_data_1.py,")
print("        so code-level tuning exists and is not enumerated here (req-3).")
