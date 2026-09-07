"""Native port of vEcoli ``ecoli/analysis/single/ptools_rxns.py``.

Produces a reaction × timepoint flux TSV (PathwayTools-compatible format).
Registered in ANALYSIS_REGISTRY as ``"ptools_rxns"`` (scale: ``"single"``).

No bulk or ribosome shims required: only
``listeners__fba_results__base_reaction_fluxes`` is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import ptools_heatmap_view
from v2ecoli.workflow.analyses.ptools_rna import (
    consolidate_timepoints,
    _groupby_time_keep_generation,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def build_query(columns, history_sql):
    """Generate SQL query for user-specified parquet columns."""
    query_sql = f"""
        SELECT {",".join(columns)}, global_time AS time, generation
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
        columns = ["listeners__fba_results__base_reaction_fluxes"]
    query_sql = build_query(columns, history_sql)
    outputs_df = conn.sql(query_sql).df()
    return _groupby_time_keep_generation(outputs_df)


# ---------------------------------------------------------------------------
# Analysis subclass
# ---------------------------------------------------------------------------

class PtoolsRxns(Analysis):
    """Reaction × timepoint flux table (PathwayTools-compatible TSV)."""

    name = "ptools_rxns"
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

        output_columns = ["listeners__fba_results__base_reaction_fluxes"]
        output_df = self._do_read_outputs(history_sql, conn, output_columns)

        # Drop timestep-0 row where FBA hasn't run yet (flux array is empty at
        # t=0 because metabolism hasn't been called).  v2ecoli deviation: vEcoli
        # keeps t=0 in the raw table; we drop it so the first column is the
        # first real FBA timepoint.
        flux_col = "listeners__fba_results__base_reaction_fluxes"
        output_df = output_df[
            output_df[flux_col].apply(len) > 0
        ].reset_index(drop=True)

        rxn_mtx = np.stack(output_df[flux_col].values)

        rxn_ids_base = sim_data.process.metabolism.base_reaction_ids

        # v2ecoli's metabolism listener emits
        #   base_reaction_fluxes = reaction_mapping_matrix.dot(...)
        # 1:1 positional with base_reaction_ids: flux column i corresponds to
        # base_reaction_ids[i].
        #
        # Two width cases matter, and they are NOT symmetric:
        #
        #  * flux NARROWER than base_reaction_ids (flux_width < n_ids): the caller
        #    passed a sim_data that does not pair with this parquet (e.g.
        #    out/kb/simData.cPickle vs out/workflow/simData.cPickle). Mapping would
        #    silently drop/mislabel reactions via truncation — raise loudly.
        #
        #  * flux WIDER than base_reaction_ids (flux_width > n_ids): the sim's
        #    metabolism was BUILT with reactions that are not in the pickled
        #    base_reaction_ids — a heterologous pathway injected at build time
        #    (e.g. include_violacein_reactions appends a violacein reaction). Those
        #    reactions are appended AFTER the base set, so base_reaction_ids[i]
        #    still pairs with flux column i for i < n_ids; only the trailing
        #    columns are the injected reactions. Label those explicitly and keep
        #    their flux (dropping it would hide the exact readout an engineered
        #    strain exists to show) instead of raising. The pickled sim_data does
        #    not carry the injected ids, so a positional "injected-reaction-k"
        #    label is used; EcoCyc's Omics Viewer ignores frame IDs it doesn't
        #    know, so an unmapped heterologous reaction is harmless on the map
        #    while every base E. coli reaction still paints.
        n_ids = len(rxn_ids_base)
        flux_width = rxn_mtx.shape[1]
        if flux_width < n_ids:
            raise ValueError(
                f"base_reaction_ids ({n_ids}) > flux width ({flux_width}); "
                "sim_data does not pair with this parquet (flux is narrower than "
                "the reaction id list — wrong sim_data)."
            )
        rxn_ids = list(rxn_ids_base)
        if flux_width > n_ids:
            extra = flux_width - n_ids
            print(
                f"[ptools_rxns] flux width ({flux_width}) exceeds "
                f"base_reaction_ids ({n_ids}) by {extra}; labeling the trailing "
                f"{extra} column(s) as injected reaction(s) (e.g. a heterologous "
                f"pathway added at build time)."
            )
            rxn_ids = rxn_ids + [f"injected-reaction-{k}" for k in range(extra)]

        n_tp = int(params["n_tp"])
        gens = (
            output_df["generation"].values
            if params.get("per_generation") and "generation" in output_df.columns
            else None
        )

        rxn_blocksum, tp_idx = consolidate_timepoints(
            rxn_mtx, n_tp, normalized=True, generations=gens
        )

        tp_checkpoints = output_df["time"].values[tp_idx]

        if params["time_unit"] == "minutes":
            tp_checkpoints = tp_checkpoints / 60
            tp_checkpoints = [round(x) for x in tp_checkpoints]

        tp_columns = [str(i) + params["time_unit"][0] for i in tp_checkpoints]

        ptools_rxns_df = pd.DataFrame(
            data=np.abs(rxn_blocksum.transpose()),
            index=rxn_ids,
            columns=tp_columns,
        )
        ptools_rxns_df.index.name = "$"

        tsv = ptools_rxns_df.to_csv(
            sep="\t", index=True, header=True, float_format="%.4f"
        )
        # Flux magnitudes are extremely heavy-tailed: a handful of central-
        # carbon reactions carry O(10) flux while most of the ~hundreds of
        # active reactions carry <0.1 (and ~3/4 of all base reactions carry
        # exactly 0).  On a linear color scale the few large reactions saturate
        # the range and the entire matrix renders as a flat ~0 field with no
        # visible band structure.  Render on a log10 color scale and sort
        # reactions by descending magnitude so the decade-spanning structure is
        # legible across all reactions.
        view = ptools_heatmap_view(
            ptools_rxns_df,
            "Reaction fluxes (reaction × timepoint)",
            log_color=True,
            sort_rows=True,
            color_label="|flux| (mmol/gDCW/h)",
        )
        return {"data": {"filename": "ptools_rxns.tsv", "tsv": tsv}, "view": view}
