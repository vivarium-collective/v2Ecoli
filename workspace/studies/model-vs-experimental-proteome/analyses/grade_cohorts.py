"""Grade the cohort axes: model protein counts against Schmidt 2016.

Model side  : shipped out/cache, DEFAULT config (mechanistic_replisome False),
              generation means over generations 4-12, median across 3 seeds.
Experiment  : Schmidt 2016 Table S6, copies/cell section, Glucose condition.
Cohorts     : underdetermined-tu, contradicted-regulation, hand-adjusted
              (EXCLUDED from the headline and scored separately), baseline.

Every number is a median |log10(model/measured)|. Cohort axes are the ratio of a
cohort's median to baseline's: 1.0 means indistinguishable.

Run:  .venv/bin/python3 <this>
"""
from __future__ import annotations
import csv, gzip, json, pickle, re
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

D = ".venv/lib/python3.12/site-packages/ecoli_sources/data/flat/"
MC = "listeners__monomer_counts"
GENS = range(4, 13)
cl = lambda s: (s or "").strip('"')  # noqa: E731

# ---- model counts ----------------------------------------------------------
per_seed = []
for s in (0, 1, 2):
    d = json.load(open(f"out/model-vs-experimental-proteome-distilled/shipped-default/seed{s}/per_generation.json"))
    g = [np.asarray(d[str(k)][MC], float) for k in GENS if str(k) in d and MC in d[str(k)]]
    if g:
        per_seed.append(np.mean(np.vstack(g), axis=0))
model = np.median(np.vstack(per_seed), axis=0)
print(f"  model counts: {len(per_seed)} seeds x generations {GENS.start}-{GENS.stop-1} -> {len(model)} monomers")

# ---- ids ------------------------------------------------------------------
rows = list(csv.reader((l for l in open(D + "genes.tsv") if not l.startswith("#")), delimiter="\t"))
h = [cl(x) for x in rows[0]]
id2sym = {cl(r[h.index("id")]): cl(r[h.index("symbol")]) for r in rows[1:] if len(r) > 2}
st = pickle.load(gzip.open("models/parca/parca_state.pkl.gz", "rb"))
tr = st["process"]["transcription"]
md = st["process"]["translation"].monomer_data
mono = [str(x) for x in md["id"]]
cis = [str(x) for x in md["cistron_id"]]
sym = [id2sym.get(re.sub(r"_RNA$", "", c), "") for c in cis]

# ---- cohorts ---------------------------------------------------------------
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
    if len(c_) > 1 and r_ and np.linalg.matrix_rank(M[r_][:, c_].toarray()) < len(c_):
        und_tu |= set(c_)
cid = [str(x) for x in tr.cistron_data["id"]]
und_cis = {cid[i] for tu in und_tu for i in M[:, tu].nonzero()[0]}

# contradicted-regulation: genes on a TU whose regulation EcoCyc places elsewhere
contra_tu = set()
try:
    d = json.load(open("workspace/studies/promoter-specific-regulation/analyses/edge_evidence_split.json"))
    contra_tu = {c[0][:-3] if c[0].endswith("[c]") else c[0] for c in d["contradicted"]}
except FileNotFoundError:
    pass
rna_ids = [str(x) for x in tr.rna_data["id"]]
contra_idx = {i for i, r in enumerate(rna_ids) if (r[:-3] if r.endswith("[c]") else r) in contra_tu}
contra_cis = {cid[i] for tu in contra_idx for i in M[:, tu].nonzero()[0]}

adj = set()
for name in ("rna_expression_adjustments", "translation_efficiencies_adjustments",
             "balanced_translation_efficiencies", "rna_deg_rates_adjustments",
             "protein_deg_rates_adjustments"):
    try:
        rr = list(csv.reader((l for l in open(D + "adjustments/" + name + ".tsv") if not l.startswith("#")), delimiter="\t"))
    except FileNotFoundError:
        continue
    for x in rr[1:]:
        v = cl(x[0])
        for i in (json.loads(v) if v.startswith("[") else [v]):
            adj.add(re.sub(r"\[.\]$", "", str(i)))

sch = {}
for r in csv.DictReader(open("references/proteomics/schmidt_2016.tsv"), delimiter="\t"):
    try:
        v = float(r["Glucose"])
    except (TypeError, ValueError):
        continue
    if v > 0:
        sch[r["gene"]] = v

def errs(idxs):
    out = []
    for i in idxs:
        s, m = sym[i], model[i]
        if s in sch and m > 0:
            out.append(abs(np.log10(m / sch[s])))
    return np.array(out)

is_adj = [i for i, m in enumerate(mono) if re.sub(r"\[.\]$", "", m) in adj]
is_und = [i for i, c in enumerate(cis) if c in und_cis and i not in is_adj]
is_con = [i for i, c in enumerate(cis) if c in contra_cis and i not in is_adj]
base = [i for i in range(len(mono)) if i not in set(is_und) | set(is_con) | set(is_adj)]

e_b, e_u, e_c, e_a = errs(base), errs(is_und), errs(is_con), errs(is_adj)
print(f"\n  cohort sizes (with a Schmidt Glucose value): baseline {len(e_b)}, "
      f"underdetermined {len(e_u)}, contradicted {len(e_c)}, hand-adjusted {len(e_a)}")
mb = float(np.median(e_b))
print(f"\n=== model-wide-accuracy-baseline (pin <= 2.0) ===")
print(f"  median |log10(model/Schmidt)| = {mb:.4f} dex ({10**mb:.2f}x)   IQR "
      f"{np.percentile(e_b,25):.3f}-{np.percentile(e_b,75):.3f}")
print(f"  within 2x: {(e_b<np.log10(2)).mean():.1%}   within 10x: {(e_b<1).mean():.1%}")
for nm, e, band in (("underdetermined-cohort-is-worse", e_u, (0.7, 1.3)),
                    ("contradicted-regulation-cohort-is-worse", e_c, (0.7, 1.3))):
    if len(e) == 0:
        print(f"\n=== {nm} === NO GENES — cannot grade"); continue
    r = float(np.median(e)) / mb
    print(f"\n=== {nm}  (band {band}) ===")
    print(f"  cohort median {np.median(e):.4f} dex / baseline {mb:.4f} = {r:.4f}   n={len(e)}")
print(f"\n=== hand-adjusted-cohort-is-better  (band <= 0.8) ===")
if len(e_a):
    print(f"  cohort median {np.median(e_a):.4f} / baseline {mb:.4f} = {np.median(e_a)/mb:.4f}   n={len(e_a)}")
