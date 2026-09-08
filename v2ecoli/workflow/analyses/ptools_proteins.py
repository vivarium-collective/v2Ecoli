"""Native port of vEcoli ``ecoli/analysis/single/ptools_proteins.py``.

Produces a protein-monomer × timepoint proteomics TSV (PathwayTools-compatible
format).  Registered in ANALYSIS_REGISTRY as ``"ptools_proteins"``
(scale: ``"single"``).

Four v2ecoli parquet shims are applied (see _shims.py):
  - bulk__id + bulk__count  → bulk_count_matrix()            (Shim A)
  - list_sum(n_ribosomes_per_transcript) → active_ribosome   (Shim B)
  - listeners__replication_data__number_of_oric AS oriC      (Shim C)
  - len(active_rnap_unique_indexes) AS active_RNAP           (Shim D)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import ptools_heatmap_view, available_columns
from v2ecoli.workflow.analyses._shims import (
    bulk_count_matrix,
    ACTIVE_RIBOSOME_SQL,
    ORIC_SQL,
    ACTIVE_RNAP_SQL,
)
# Re-use pure-computation helpers already defined in ptools_rna to avoid
# duplication (same algorithm, identical to the vEcoli originals).
from v2ecoli.workflow.analyses.ptools_rna import (
    get_bulk_ids,
    build_bulk2monomers_matrix,
    consolidate_timepoints,
    _groupby_time_keep_generation,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def build_query(columns, history_sql, include_generation=False):
    """Generate SQL query for user-specified parquet columns.

    ``include_generation`` carries the ``generation`` partition column for
    per-generation consolidation; callers detect its presence first.
    """
    gen = ", generation" if include_generation else ""
    query_sql = f"""
        SELECT {",".join(columns)}, global_time AS time{gen}
        FROM ({history_sql})
        ORDER BY time
    """
    return query_sql


def read_outputs(
    history_sql: str,
    conn: DuckDBPyConnection,
    columns=None,
):
    """Retrieve specific columns from parquet outputs and return a DataFrame."""
    if columns is None:
        columns = [
            "bulk__id",
            "bulk__count",
            ORIC_SQL,
            ACTIVE_RNAP_SQL,
            ACTIVE_RIBOSOME_SQL,
        ]
    incl_gen = "generation" in available_columns(conn, history_sql)
    query_sql = build_query(columns, history_sql, incl_gen)
    outputs_df = conn.sql(query_sql).df()
    return _groupby_time_keep_generation(outputs_df)


# ---------------------------------------------------------------------------
# Analysis subclass
# ---------------------------------------------------------------------------

class PtoolsProteins(Analysis):
    """Protein-monomer × timepoint proteomics table (PathwayTools-compatible TSV)."""

    name = "ptools_proteins"
    scale = "single"
    config_schema = {
        "n_tp": "integer",
        "time_unit": "string",
        "per_generation": "boolean",
        "skip_n_gens": "integer",
    }

    def _do_read_outputs(
        self,
        history_sql: str,
        conn: DuckDBPyConnection,
        columns=None,
    ):
        """Delegate to module-level read_outputs (overridable by mixins)."""
        return read_outputs(history_sql, conn, columns)

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        params = dict(variant_metadata or {})
        params.setdefault("n_tp", 8)
        params.setdefault("time_unit", "minutes")

        if params["time_unit"] not in ("minutes", "seconds"):
            params["time_unit"] = "minutes"

        bulk_ids = get_bulk_ids(sim_data)

        output_columns = [
            "bulk__id",
            "bulk__count",
            ORIC_SQL,            # Shim C: oriC count (number_of_oric scalar)
            ACTIVE_RNAP_SQL,     # Shim D: active RNAP (len of unique_indexes list)
            ACTIVE_RIBOSOME_SQL, # Shim B: active ribosome (sum of per-transcript list)
        ]

        output_df = self._do_read_outputs(history_sql, conn, output_columns)

        # Shim A: reorder bulk__count columns to sim_data order
        bulk_mtx = bulk_count_matrix(output_df, sim_data)

        translation_module = sim_data.process.translation.monomer_data.fullArray()

        replisome_monomer_subunits = sim_data.molecule_groups.replisome_monomer_subunits
        replisome_trimer_subunits = sim_data.molecule_groups.replisome_trimer_subunits
        riboproteins = [
            sim_data.molecule_ids.s30_full_complex,
            sim_data.molecule_ids.s50_full_complex,
        ]
        rnap_id = sim_data.molecule_ids.full_RNAP

        # Distribute replisome-bound proteins back into bulk counts.
        # Each oriC fork has 2 copies of monomer subunits and 6 of trimer subunits
        # (matching vEcoli's unique-molecule → bulk decomplexation logic).
        for bulk_id in replisome_monomer_subunits:
            unique_complex = output_df["oriC"].values  # Shim C
            add_bulk = unique_complex * 2
            bulk_idx = bulk_ids.index(bulk_id)
            bulk_mtx[:, bulk_idx] = bulk_mtx[:, bulk_idx] + add_bulk

        for bulk_id in replisome_trimer_subunits:
            unique_complex = output_df["oriC"].values  # Shim C
            add_bulk = unique_complex * 6
            bulk_idx = bulk_ids.index(bulk_id)
            bulk_mtx[:, bulk_idx] = bulk_mtx[:, bulk_idx] + add_bulk

        for bulk_id in riboproteins:
            unique_complex = output_df["active_ribosome"].values  # Shim B
            add_bulk = unique_complex
            bulk_idx = bulk_ids.index(bulk_id)
            bulk_mtx[:, bulk_idx] = bulk_mtx[:, bulk_idx] + add_bulk

        rnap_counts = output_df["active_RNAP"].values  # Shim D
        rnap_idx = bulk_ids.index(rnap_id)
        bulk_mtx[:, rnap_idx] = bulk_mtx[:, rnap_idx] + rnap_counts

        bulk2monomers, all_monomers = build_bulk2monomers_matrix(sim_data)

        protein_monomers = translation_module["id"]

        protein_monomer_idxs = np.array(
            [all_monomers.index(protein) for protein in protein_monomers]
        )

        bulk2protein_monomers = bulk2monomers[:, protein_monomer_idxs]

        protein_labels = [protein[:-3] for protein in protein_monomers]

        proteomics = np.matmul(bulk_mtx, bulk2protein_monomers)

        n_tp = int(params["n_tp"])
        gens = (
            output_df["generation"].values
            if params.get("per_generation") and "generation" in output_df.columns
            else None
        )

        proteomics_bulksum, tp_idx = consolidate_timepoints(
            proteomics, n_tp, normalized=True, generations=gens
        )

        tp_checkpoints = output_df["time"].values[tp_idx]

        if params["time_unit"] == "minutes":
            tp_checkpoints = tp_checkpoints / 60
            tp_checkpoints = [round(x) for x in tp_checkpoints]

        tp_columns = [str(i) + params["time_unit"][0] for i in tp_checkpoints]

        ptools_proteins_df = pd.DataFrame(
            data=proteomics_bulksum.transpose(),
            index=protein_labels,
            columns=tp_columns,
        )
        ptools_proteins_df.index.name = "$"

        tsv = ptools_proteins_df.to_csv(
            sep="\t", index=True, header=True, float_format="%.4f"
        )
        view = ptools_heatmap_view(
            ptools_proteins_df, "Protein counts (monomer × timepoint)"
        )
        return {"data": {"filename": "ptools_proteins.tsv", "tsv": tsv}, "view": view}
