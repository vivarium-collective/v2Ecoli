"""How much of EcoCyc the model represents, and how much of the model EcoCyc covers.

Scope facts for every claim in this investigation that leans on the EcoCyc join.
Both directions matter and they are lopsided in OPPOSITE directions: the model
holds nearly all of EcoCyc's transcription units but only a small curated slice
of its regulators.

Reads references/ecocyc-30.0/ (gitignored, licensed — regenerate with
pathway-tools -lisp, (so 'ECOLI), (create-flat-files-for-current-kb)) plus the
model's own flat inputs and cache.

Run:  .venv/bin/python3 <this> [cache_dir]
"""
from __future__ import annotations
import collections, csv, gzip, json, os, pickle, sys
import dill

CACHE = sys.argv[1] if len(sys.argv) > 1 else "out/cache"
R = "references/ecocyc-30.0/"
D = ".venv/lib/python3.12/site-packages/ecoli_sources/data/flat/"
ALIAS = {"FNR-4FE-4S-CPLX": "CPLX0-7797",   # the model's id does not exist in EcoCyc 30.0
         "PHOSPHO-DCUR": "CPLX0-7721",
         "PUTA-CPLX": "PUTA-MONOMER"}
cl = lambda s: (s or "").strip('"')  # noqa: E731

def records(path):
    rec = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("//"):
            if rec:
                yield dict(rec)
            rec = collections.defaultdict(list)
            continue
        if line.startswith("#") or " - " not in line:
            continue
        k, v = line.split(" - ", 1)
        rec[k].append(v)
    if rec:
        yield dict(rec)

if not os.path.isdir(R):
    sys.exit(f"{R} not found — regenerate the PGDB flatfiles first.")

# ---- transcription units ---------------------------------------------------
eco_tus = {r["UNIQUE-ID"][0] for r in records(R + "transunits.dat")}
rows = list(csv.reader((l for l in open(D + "transcription_units.tsv") if not l.startswith("#")),
                       delimiter="\t"))
gi = [cl(x) for x in rows[0]].index("genes")
src = {cl(r[0]): r for r in rows[1:]}
removed = {cl(r[0]) for r in csv.reader(
    (l for l in open(D + "transcription_units_removed.tsv") if not l.startswith("#")),
    delimiter="\t")} - {"id"}

st = pickle.load(gzip.open("models/parca/parca_state.pkl.gz", "rb"))
ids = [str(x) for x in st["process"]["transcription"].rna_data["id"]]
kept_tu = {i[:-3] for i in ids if i.startswith("TU")}
renamed = sum(1 for i in ids if not i.startswith("TU"))

def dedup_casualties(pool):
    g = collections.defaultdict(list)
    for tu in pool:
        try:
            genes = tuple(sorted(json.loads(src[tu][gi])))
        except Exception:
            genes = ()
        if genes:
            g[genes].append(tu)
    return sum(len(v) - 1 for v in g.values() if len(v) > 1)

print("TRANSCRIPTION UNITS")
print(f"  EcoCyc 30.0 transunits.dat                 {len(eco_tus):5d}")
print(f"  model input transcription_units.tsv        {len(src):5d}")
print(f"  model rna_data entries                     {len(ids):5d}")
print(f"  source TUs that exist in EcoCyc            {len(set(src) & eco_tus):5d}"
      f"  ({len(set(src) & eco_tus)/len(src):.1%}) — the input is essentially complete")
print(f"  EcoCyc TUs absent from the input            {len(eco_tus - set(src)):5d}")
after = set(src) - removed
print(f"\n  accounting for the input -> model loss:")
print(f"    explicitly removed (unsupported types)   {len(removed & set(src)):5d}")
print(f"    not kept thereafter                      {len(after - kept_tu):5d}")
print(f"    -> total absent from the model           {len(removed & set(src)) + len(after - kept_tu):5d}"
      f"   (tu-coverage-loss records 760)")
print(f"    kept with a TU* id                       {len(kept_tu & after):5d}")
print(f"    single-gene TUs renamed to gene-RNA ids  {renamed:5d}   (relabelled, not lost)")
print(f"\n  dedup casualties over ALL source TUs      {dedup_casualties(set(src)):5d}"
      f"   (tu-coverage-loss records 637)")
print(f"  dedup casualties excluding removed TUs     {dedup_casualties(after):5d}"
      f"   — same phenomenon, different denominator; 13 TUs are in both sets")

# ---- transcription factors -------------------------------------------------
regs = collections.Counter()
for r in records(R + "regulation.dat"):
    if "Transcription-Factor-Binding" in r.get("TYPES", []) and r.get("REGULATOR"):
        regs[r["REGULATOR"][0]] += 1
sd = dill.load(open(f"{CACHE}/sim_data_cache.dill", "rb"))
model_tfs = [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]]
resolved = [ALIAS.get(t, t) for t in model_tfs]
hit = [t for t, e in zip(model_tfs, resolved) if e in regs]
edges = len(sd["configs"]["ecoli-transcript-initiation"]["delta_prob"]["deltaV"])

print("\nTRANSCRIPTION FACTORS")
print(f"  distinct regulators in EcoCyc              {len(regs):5d}")
print(f"  transcription factors in the model         {len(model_tfs):5d}")
print(f"  model TFs found in EcoCyc, with aliases    {len(hit):5d} of {len(model_tfs)}")
print(f"  model TFs found WITHOUT the 3 aliases      {sum(1 for t in model_tfs if t in regs):5d} of {len(model_tfs)}")
print(f"  EcoCyc regulators the model represents     {len(hit):5d} of {len(regs)}"
      f"  ({len(hit)/len(regs):.1%}) — a curated subset by design")
tot = sum(regs.values())
inm = sum(regs[e] for e in resolved if e in regs)
print(f"\n  EcoCyc TF-binding records                  {tot:5d}")
print(f"  ...belonging to the model's TFs            {inm:5d}  ({inm/tot:.1%})")
print(f"  model's declared (TU, TF) edges            {edges:5d}")
