"""Extract the proteomics workbooks into auditable TSVs with provenance.

The xlsx sources live outside the repo (~/Documents/Proteomics data/) and are
not redistributed. What lands in references/proteomics/ is a small, tidy,
inspectable table per dataset, plus a MANIFEST recording exactly which sheet,
header row and columns each came from — so the join can be audited without
reopening a 16 MB spreadsheet.

Schmidt 2016 is the PRIMARY validation set. Mori 2021 and Maeck 2015 are
comparison sets whose value is disagreement: the spread between them is the
noise floor for any model-vs-experiment claim. Li 2014 is deliberately NOT
extracted as validation — it is a ParCa input via translation_efficiency.tsv.

Run:  .venv/bin/python3 scripts/extract_proteomics.py
"""
from __future__ import annotations
import csv, json, pathlib, sys

import openpyxl

SRC = pathlib.Path.home() / "Documents" / "Proteomics data"
OUT = pathlib.Path("references/proteomics")
OUT.mkdir(parents=True, exist_ok=True)


def sheet_rows(path, sheet, header_row):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[header_row]]
    return header, rows[header_row + 1:]


def write(name, header, records, meta):
    p = OUT / f"{name}.tsv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(records)
    print(f"  wrote {p}  {len(records)} rows x {len(header)} cols")
    return {**meta, "rows": len(records), "columns": header}


manifest = {}

# ---- Schmidt 2016 — PRIMARY -------------------------------------------------
p = SRC / "schmidt et al 2016" / "41587_2016_BFnbt3418_MOESM18_ESM (1).xlsx"
if p.exists():
    header, rows = sheet_rows(p, "Table S6", header_row=2)
    gi = header.index("Gene")
    ui = header.index("Uniprot Accession")
    # every condition column: everything after the fixed metadata block
    meta_cols = {"Uniprot Accession", "Description", "Gene", "Peptides.used.for.quantitation",
                 "Confidence.score", "Molecular weight (Da)", "Dataset"}
    cond = [(i, h) for i, h in enumerate(header) if h and h not in meta_cols]
    out_header = ["gene", "uniprot"] + [h for _, h in cond]
    recs = []
    for r in rows:
        if r is None or gi >= len(r) or not r[gi]:
            continue
        recs.append([str(r[gi]).strip(), str(r[ui]).strip() if r[ui] else ""]
                    + [r[i] if i < len(r) else "" for i, _ in cond])
    manifest["schmidt_2016"] = write(
        "schmidt_2016", out_header, recs,
        {"role": "PRIMARY validation", "independent_of_model": True,
         "source_file": str(p), "sheet": "Table S6", "header_row_index": 2,
         "units": "protein copies/cell",
         "note": ("Combined absolute abundance from both datasets. NOT a ParCa input: v2ecoli protein "
                  "counts emerge from RNA-seq expression x Li 2014 translation efficiency / SILAC "
                  "degradation, so agreement is prediction. Use the condition column matching the "
                  "simulated medium and state which.")})
else:
    print(f"  MISSING: {p}", file=sys.stderr)

(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"\n  manifest: {OUT/'MANIFEST.json'}")
