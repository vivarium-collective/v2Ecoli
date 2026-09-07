"""Corrected fidelity join: use tf_to_fold_change, the mapping ParCa actually fit.

Supersedes measure.py, which had two defects found on 2026-09-07:

  1. it read fold_changes.tsv ONLY. There is a second source,
     fold_changes_nca.tsv, carrying lexA->dnaG at log2FC -1.71 and lexA->rpsU at
     -1.89. Roughly half the model's regulatory input was outside the join.
  2. it paired every cistron in a TU with that TU's delta. rpoD shares TU00352
     with dnaG, so the lexA->rpoD row (-0.65) was credited with a delta that
     derives from dnaG's own edge. rpoD is not in tf_to_fold_change at all.

Joining against sim_data.tf_to_fold_change removes both: it is keyed on the TF
ids and cistron RNA ids ParCa fit, its values are already retained-fractions,
and it reflects whatever fold_changes_removed.tsv removed.

Run:  .venv/bin/python3 <this> [parca_state.pkl.gz] [cache_dir]
"""
from __future__ import annotations
import gzip, json, pickle, sys
import numpy as np
import scipy.sparse as sp

STATE = sys.argv[1] if len(sys.argv) > 1 else "models/parca/parca_state.pkl.gz"
CACHE = sys.argv[2] if len(sys.argv) > 2 else "out/cache"

opener = gzip.open if STATE.endswith(".gz") else open
st = pickle.load(opener(STATE, "rb"))
tr = st["process"]["transcription"]
fc = st["tf_to_fold_change"]

import dill
sd = dill.load(open(f"{CACHE}/sim_data_cache.dill", "rb"))
ti = sd["configs"]["ecoli-transcript-initiation"]
tf_ids = [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]]
tu_ids = [str(x) for x in ti["rna_data"]["id"]]
bp = np.asarray(ti["basal_prob"], dtype=float)
# deltaV, in PROBABILITY units — NOT delta_prob_matrix, which holds the
# relative (ppgpp-adjusted) delta. The retained-fraction formula
# 1 + deltaV/basal_prob is only defined on the former; using the matrix gives
# nonsense like -8244 for TU00352.
dp = ti["delta_prob"]
D = sp.csr_matrix((dp["deltaV"], (dp["deltaI"], dp["deltaJ"])), shape=dp["shape"]).toarray()
C = sp.csc_matrix(tr.cistron_tu_mapping_matrix)
cis = [str(x) for x in tr.cistron_data["id"]]
cidx = {c: i for i, c in enumerate(cis)}
tfidx = {t: i for i, t in enumerate(tf_ids)}

recs, unjoined = [], 0
for tf, targets in fc.items():
    j = tfidx.get(tf)
    if j is None:
        continue
    for cistron, expected in targets.items():
        i = cidx.get(cistron)
        if i is None:
            unjoined += 1
            continue
        tus = C[i, :].nonzero()[1] if C.shape[0] == len(cis) else C[:, i].nonzero()[0]
        for tu in np.atleast_1d(tus):
            v = float(D[tu, j])
            if v == 0.0 or bp[tu] <= 0:
                continue
            recs.append({"tf": tf, "cistron": cistron, "tu": tu_ids[tu],
                         "expected_retained": float(expected),
                         "implied_retained": 1.0 + v / bp[tu],
                         "n_tu_for_cistron": int(np.atleast_1d(tus).size)})

print(f"state {STATE}   cache {CACHE}")
print(f"joined delta entries: {len(recs)}   (measure.py joined from fold_changes.tsv alone)")
ratio = np.array([r["implied_retained"] / r["expected_retained"] for r in recs])
rep = np.array([r["expected_retained"] < 1 for r in recs])
act = ~rep
def med(mask, label):
    v = ratio[mask]
    print(f"  {label:32s} n={mask.sum():4d}   median implied/expected = {np.median(v):.4f}")
med(np.ones(len(recs), bool), "ALL entries")
med(rep, "repressors")
med(act, "activators")
neg = sum(1 for r in recs if r["implied_retained"] < 0)
print(f"  entries violating the probability bound (implied < 0): {neg}")

print("\n--- the dnaG locus, per cistron (this is what the old lexa-rpod pin got wrong)")
for r in recs:
    if r["tf"] == "PC00010" and r["tu"].startswith(("TU00352", "TU00434", "TU00435")):
        print(f"  {r['cistron']:14s} on {r['tu']:14s} expected {r['expected_retained']:.5f}  "
              f"implied {r['implied_retained']:.6e}  discrepancy "
              f"{r['expected_retained']/r['implied_retained']:.4f}x")
json.dump(recs, open("workspace/studies/foldchange-to-delta-fidelity/analyses/fidelity_v2.json", "w"))
print("\nwrote analyses/fidelity_v2.json")
