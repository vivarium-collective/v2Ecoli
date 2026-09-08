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
    # Table S6 has THREE sections with IDENTICAL column names — row 1 labels them:
    #   col  7  "Protein copies/cell"                 <- the one we want
    #   col 29  "Protein Mass (fg) / Cell"
    #   col 51  "Coefficient of Variance (%)"
    # Selecting by column NAME silently returns the LAST match (the CV block),
    # which reads a coefficient of variation as if it were a copy number. Select
    # the section by POSITION from the row-1 labels instead.
    wb_ = openpyxl.load_workbook(p, read_only=True, data_only=True)
    ws_ = wb_["Table S6"]
    head_rows = list(ws_.iter_rows(min_row=1, max_row=3, values_only=True))
    section = [(i, str(c)) for i, c in enumerate(head_rows[1]) if c]
    start = next(i for i, lab in section if lab.strip().startswith("Protein copies/cell"))
    ends = [i for i, _ in section if i > start]
    end = ends[0] if ends else len(head_rows[2])
    header, rows = sheet_rows(p, "Table S6", header_row=2)
    gi = header.index("Gene")
    ui = header.index("Uniprot Accession")
    cond = [(i, header[i]) for i in range(start, end) if header[i]]
    print(f"  Schmidt: copies/cell section = columns {start}..{end - 1} "
          f"({len(cond)} conditions); other sections ignored")
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

# ---- Maeck 2015 — comparison, SAME UNITS as Schmidt --------------------------
p = SRC / "maeck et al 2015" / "Data Sheet 1" / "Supplementary_Tables S1-S9.xlsx"
if p.exists():
    header, rows = sheet_rows(p, "S2", header_row=0)
    gi = header.index("Gene Names")
    pi = header.index("Protein IDs")
    cn = [(i, h) for i, h in enumerate(header) if h.startswith("CN (")]
    out_header = ["gene", "uniprot", "copies_per_cell_median", "n_estimates"]
    recs = []
    for r in rows:
        if r is None or gi >= len(r) or not r[gi]:
            continue
        # "b2206;JW2194;napA;yojC" — the gene symbol is the first non-b/JW token
        # "b2206;JW2194;napA;yojC" — drop locus tags (b####, JW####, ECDH10B_####)
        # and take the first real symbol. The earlier predicate was inverted and
        # selected JW tags, which then matched nothing in Schmidt.
        import re as _re
        toks = [t.strip() for t in str(r[gi]).split(";") if t.strip()]
        real = [t for t in toks
                if not _re.fullmatch(r"(b\d+|JW\d+[A-Za-z]?|ECDH10B_\d+|[A-Z]\d{4,})", t)]
        sym = real[0] if real else toks[0]
        vals = []
        for i, _ in cn:
            if i < len(r):
                try:
                    v = float(r[i])
                    if v > 0:
                        vals.append(v)
                except (TypeError, ValueError):
                    pass          # '#VALUE!' cells are Excel errors, not zeros
        if not vals:
            continue
        vals.sort()
        med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        recs.append([sym, str(r[pi]).split(";")[0] if r[pi] else "", med, len(vals)])
    manifest["maeck_2015"] = write(
        "maeck_2015", out_header, recs,
        {"role": "comparison", "independent_of_model": True,
         "source_file": str(p), "sheet": "S2", "header_row_index": 0,
         "units": "protein copies/cell",
         "note": ("Copy-number estimates during growth in M9 minimal — the same units as Schmidt and a "
                  "comparable medium, so this is the direct noise-floor partner. Median taken over the "
                  "CN (TP3/TP5/...) replicate-timepoint columns; '#VALUE!' cells are Excel errors and "
                  "are dropped, NOT read as zero. Gene symbol parsed from the first non-locus token of "
                  "the semicolon-joined Gene Names field.")})
else:
    print(f"  MISSING: {p}", file=sys.stderr)

# ---- Mori 2021 — comparison, DIFFERENT UNITS ---------------------------------
p = SRC / "mori et al 2021" / "44320_2021_BFMSB20209536_MOESM9_ESM.xlsx"
if p.exists():
    header, rows = sheet_rows(p, "EV8-AbsoluteMassFractions-1", header_row=0)
    gi = header.index("Gene name")
    pi = header.index("Protein ID")
    libs = [(i, h) for i, h in enumerate(header) if str(h).startswith("Lib-")]
    recs = []
    for r in rows:
        if r is None or gi >= len(r) or not r[gi]:
            continue
        vals = []
        for i, _ in libs:
            if i < len(r):
                try:
                    v = float(r[i])
                    if v > 0:
                        vals.append(v)
                except (TypeError, ValueError):
                    pass
        if not vals:
            continue
        vals.sort()
        med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        recs.append([str(r[gi]).strip(), str(r[pi]).strip() if r[pi] else "", med, len(vals)])
    manifest["mori_2021"] = write(
        "mori_2021", ["gene", "uniprot", "mass_fraction_median", "n_libraries"], recs,
        {"role": "comparison", "independent_of_model": True,
         "source_file": str(p), "sheet": "EV8-AbsoluteMassFractions-1", "header_row_index": 0,
         "units": "MASS FRACTION — not copies/cell",
         "note": ("DIFFERENT UNITS from Schmidt and Maeck. Usable for rank/relative comparisons and as a "
                  "third opinion on which genes are abundant, but converting to copies/cell needs total "
                  "protein mass and per-protein molecular weights, which introduces assumptions this "
                  "study has not made. Median over the Lib-NN columns; zeros dropped as non-detections.")})
else:
    print(f"  MISSING: {p}", file=sys.stderr)

(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"\n  manifest: {OUT/'MANIFEST.json'}")
