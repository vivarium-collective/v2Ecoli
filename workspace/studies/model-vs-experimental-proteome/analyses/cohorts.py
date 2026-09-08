"""Build the four cohorts and measure Schmidt 2016 coverage (the gate axis).

Cohorts:
  underdetermined-tu       genes on a rank-deficient NNLS block
  contradicted-regulation  genes on a TU whose regulation EcoCyc places elsewhere
  hand-adjusted            EXCLUSION — molecules tuned in flat/ (tuning_inventory.py)
  baseline                 every other gene with a measurement

Run:  .venv/bin/python3 <this>
"""
from __future__ import annotations
import csv, gzip, json, pickle, re
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

D = ".venv/lib/python3.12/site-packages/ecoli_sources/data/flat/"
cl = lambda s: (s or "").strip('"')  # noqa: E731

rows = list(csv.reader((l for l in open(D + "genes.tsv") if not l.startswith("#")), delimiter="\t"))
h = [cl(x) for x in rows[0]]
id2sym = {cl(r[h.index("id")]): cl(r[h.index("symbol")]) for r in rows[1:] if len(r) > 2}

st = pickle.load(gzip.open("models/parca/parca_state.pkl.gz", "rb"))
tr = st["process"]["transcription"]
md = st["process"]["translation"].monomer_data
mono_ids = [str(x) for x in md["id"]]
cis = [str(x) for x in md["cistron_id"]]
mono2sym = {m: id2sym[re.sub(r"_RNA$", "", c)] for m, c in zip(mono_ids, cis)
            if re.sub(r"_RNA$", "", c) in id2sym}

# --- underdetermined TUs -> their monomers ---------------------------------
M = sp.csc_matrix(tr.cistron_tu_mapping_matrix).astype(float)
nC, nT = M.shape
big = sp.bmat([[None, M], [M.T, None]]).tocsr()
_, lab = connected_components(big, directed=False)
bl = {}
for j in range(nT):
    bl.setdefault(lab[nC + j], ([], []))[1].append(j)
for i in range(nC):
    bl.setdefault(lab[i], ([], []))[0].append(i)
und_tu = set()
for r_, c_ in bl.values():
    if len(c_) < 2 or not r_:
        continue
    if np.linalg.matrix_rank(M[r_][:, c_].toarray()) < len(c_):
        und_tu |= set(c_)
cid = [str(x) for x in tr.cistron_data["id"]]
und_cistrons = {cid[i] for tu in und_tu for i in M[:, tu].nonzero()[0]}
und_mono = {m for m, c in zip(mono_ids, cis) if c in und_cistrons}

# --- hand-adjusted exclusion ------------------------------------------------
adj = set()
for name in ("rna_expression_adjustments", "translation_efficiencies_adjustments",
             "balanced_translation_efficiencies", "rna_deg_rates_adjustments",
             "protein_deg_rates_adjustments"):
    p = D + "adjustments/" + name + ".tsv"
    try:
        rr = list(csv.reader((l for l in open(p) if not l.startswith("#")), delimiter="\t"))
    except FileNotFoundError:
        continue
    for x in rr[1:]:
        v = cl(x[0])
        vals = json.loads(v) if v.startswith("[") else [v]
        for i in vals:
            adj.add(re.sub(r"\[.\]$", "", str(i)))

schmidt = {r["gene"]: r for r in csv.DictReader(open("references/proteomics/schmidt_2016.tsv"), delimiter="\t")}

def cov(mons, label):
    syms = {mono2sym[m] for m in mons if m in mono2sym}
    have = {s for s in syms if s in schmidt}
    frac = len(have) / len(syms) if syms else float("nan")
    print(f"  {label:26s} genes={len(syms):5d}  with Schmidt={len(have):5d}  coverage={frac:.4f}")
    return frac, syms, have

print("=== cohort coverage against Schmidt 2016 (gate band [0.5, 1.0]) ===")
f_und, s_und, _ = cov(und_mono, "underdetermined-tu")
f_adj, s_adj, _ = cov({m for m in mono_ids if m in adj or re.sub(r"\[.\]$", "", m) in adj}, "hand-adjusted")
base = set(mono_ids) - und_mono - adj
f_base, s_base, _ = cov(base, "baseline")
print(f"\n  minimum cohort coverage = {min(f_und, f_base):.4f}")
print("  NOTE contradicted-regulation cohort needs the EcoCyc join and is measured separately.")
