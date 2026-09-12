#!/usr/bin/env python
"""Does a chromosome-level trpR deletion prune trpR's own regulon? Measure it.

THE QUESTION. `discovery_implications` records that ParCa keeps fitting a deleted
regulator's regulon -- the deletion transform touches genes, transcription_units,
dna_sites and the genome sequence, and never fold_changes. That is right in kind. The
COUNT it shipped with ("all 12 rows survive") was reasoned, not measured, and it is
wrong: one of the twelve rows targets trpR ITSELF (autoregulation), and the loader's
one guard on this path keys on the TARGET, not on the TF:

    simulation_data.py:184   rna_ids_with_coordinates = {... if left_end_pos is not
                                                          None and right_end_pos ...}
    simulation_data.py:219   if target not in rna_ids_with_coordinates: continue

A knockout TOMBSTONES its gene (row retained, coordinates nulled), so the autoregulation
row loses its target and drops while the other eleven do not. ⚠ The distinction matters
more than one row: it says the surviving eleven are exactly the rows whose targets are
still on the chromosome, i.e. every row that could still mis-parameterize a downstream
gene. The one row the edit removes is the one that had no downstream consequence left.

HOW IT IS MEASURED. By calling the REAL loader
(`SimulationDataEcoli._add_condition_data`) on two raw_data objects -- wild type, and a
trpR knockout bundle generated through `genotype_build.make_knockout_bundle` -- and
reading `tf_to_fold_change[CPLX-125]` off each. Reimplementing the filter chain here
would measure this file's idea of the loader rather than the loader.

⚠ AND WITH A POSITIVE CONTROL, because the interesting answer is a NEGATIVE one.
"Eleven rows survived" is only evidence that the guard spares them if the guard can be
shown to fire at all. So the same comparison is run for a gene that IS a fold_changes
target of some other TF: deleting it must take its incoming edges to zero. If that
control does not fire, the guard is inert and the trpR number means nothing -- so the
script reports the control alongside the result rather than in a comment.

Reads only; writes data/regulon_survival.json. No ParCa fit, no simulation; it does
generate KO bundles, which is why it is not part of extract_evidence.py (that script's
contract is "flat tables only, seconds to run").

Run from the workspace root (the canonical_runs contract):
    python workspace/studies/genotype-06-trpr-regulon/sims/measure_regulon_survival.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[4]
OUT = STUDY_DIR / "data" / "regulon_survival.json"

sys.path.insert(0, str(WS_ROOT))

TRPR_GENE = "EG11029"

# Candidates for the positive control, in preference order. Each must be a
# fold_changes TARGET of some TF in the wild-type build, or it cannot demonstrate
# anything. The script picks the first that qualifies and records which, so a reader
# can see the control was real rather than nominal.
CONTROL_CANDIDATES = ("lacZ", "lacY", "araB", "malE", "galK", "tnaA")


def _condition_data(raw_data):
    """Run the real loader against a bare carrier and return it."""
    from v2ecoli.processes.parca.reconstruction.ecoli.simulation_data import (
        SimulationDataEcoli,
    )

    class _Shim:
        pass

    shim = _Shim()
    SimulationDataEcoli._add_condition_data(shim, raw_data)
    return shim


def _trpr_active_id(raw_data) -> str:
    """trpR's ACTIVE form is the key tf_to_fold_change is addressed by, not 'trpR'."""
    m = {x["TF"]: x["activeId"].split(", ")
         for x in raw_data.transcription_factors if len(x["activeId"]) > 0}
    return m["trpR"][0]


def _rna_to_symbol(raw_data) -> dict:
    out = {}
    for gene in raw_data.genes:
        for rna_id in gene["rna_ids"]:
            out[rna_id] = gene["symbol"]
    return out


def _regulon(raw_data) -> dict:
    """{target symbol -> 2**log2FC} for trpR, as the loader stores it."""
    shim = _condition_data(raw_data)
    symbol = _rna_to_symbol(raw_data)
    fc = shim.tf_to_fold_change.get(_trpr_active_id(raw_data), {})
    return {symbol.get(k, k): round(v, 6) for k, v in sorted(fc.items())}


def _incoming_edges(raw_data, gene_symbol: str) -> int:
    """How many TFs carry a surviving fold-change edge ONTO this gene."""
    shim = _condition_data(raw_data)
    symbol = _rna_to_symbol(raw_data)
    return sum(1 for targets in shim.tf_to_fold_change.values()
               for t in targets if symbol.get(t) == gene_symbol)


def main() -> int:
    from v2ecoli.library import genotype_build as gb

    wt_raw = gb.resolve_raw_data(None)
    wt = _regulon(wt_raw)
    print(f"WT trpR regulon: {len(wt)} rows -> {sorted(wt)}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        manifest, genotype, _ = gb.make_knockout_bundle([TRPR_GENE], Path(td) / "trpR")
        ko = _regulon(gb.resolve_raw_data(manifest))
        print(f"trpR-KO regulon: {len(ko)} rows -> {sorted(ko)}", flush=True)

        # --- positive control ------------------------------------------------
        gene_id_of = {g["symbol"]: g["id"] for g in wt_raw.genes}
        control = {"status": "no qualifying candidate", "candidates": list(CONTROL_CANDIDATES)}
        for symbol in CONTROL_CANDIDATES:
            if symbol not in gene_id_of:
                continue
            before = _incoming_edges(wt_raw, symbol)
            if before == 0:
                continue  # not a target: it could not demonstrate the guard firing
            c_manifest, _, _ = gb.make_knockout_bundle([gene_id_of[symbol]], Path(td) / "ctl")
            after = _incoming_edges(gb.resolve_raw_data(c_manifest), symbol)
            control = {"gene": symbol, "frame_id": gene_id_of[symbol],
                       "incoming_edges_wt": before, "incoming_edges_after_deletion": after,
                       "guard_fires": after < before}
            print(f"control {symbol}: {before} incoming edges -> {after} after deletion "
                  f"({'guard FIRES' if after < before else '⛔ guard INERT'})", flush=True)
            break

    dropped = sorted(set(wt) - set(ko))
    result = {
        "question": "does a chromosome-level trpR deletion prune trpR's fold_changes rows?",
        "method": "SimulationDataEcoli._add_condition_data run against WT and KO raw_data; "
                  "tf_to_fold_change[<trpR activeId>] read off each",
        "trpr_gene_id": TRPR_GENE,
        "trpr_active_id": _trpr_active_id(wt_raw),
        "genotype_id": genotype,
        "wt_rows": len(wt), "ko_rows": len(ko),
        "wt_regulon": wt, "ko_regulon": ko,
        "dropped_targets": dropped,
        "mechanism": "the Target-keyed coordinate guard at simulation_data.py:219 "
                     "(rna_ids_with_coordinates); a knockout nulls its gene's coordinates",
        "positive_control": control,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(wt)} -> {len(ko)} rows; dropped {dropped or 'nothing'}")
    print(f"wrote {OUT.relative_to(WS_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
