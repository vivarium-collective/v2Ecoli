"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_higher_order_properties.py``.

Per-cell whole-organism properties — cell mass, cell volume, DNA and RNA mass
fractions, and glycogen content — as a wide TSV: one row per property, one
column per cell, plus the across-cell mean and standard deviation.  Registered
as ``"cd1_higher_order_properties"`` (scale: ``"multiseed"``).

v2ecoli adaptations
-------------------
* **Shim A (bulk molecules).**  The glycogen count comes from
  ``bulk__count`` indexed via the parquet bulk ordering
  (:func:`~v2ecoli.workflow.analyses._helpers.bulk_field_ids`) rather than from
  vEcoli's single ``bulk`` column plus ``field_metadata(conn, config_sql,
  "bulk")``.
* A missing ``glycogen-monomer[c]`` raises rather than surfacing as an opaque
  ``ValueError`` from ``list.index`` — the substance of the original, with the
  molecule named in the message.
* The TSV is returned as ``data["tsv"]`` instead of being written to ``outdir``
  (the runner places it under the sweep's ``ptools/`` dir).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection
from scipy.constants import N_A
from unum import Unum
from wholecell.utils import units

from v2ecoli.workflow.analyses._helpers import (
    DEFAULT_CD1_CHUNK_SIZE,
    bulk_field_ids,
    cd1_filter_clause,
    read_stacked_columns,
    run_chunked,
    with_cross_cell_stats,
)
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]
_GLYCOGEN_ID = "glycogen-monomer[c]"


class Cd1HigherOrderProperties(Analysis):
    name = "cd1_higher_order_properties"
    scale = "multiseed"
    config_schema = {
        "generation_lower_bound": "integer",
        "time_lower_bound": "float",
        "chunk_size": {"_type": "integer", "_default": DEFAULT_CD1_CHUNK_SIZE},
    }

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        params = {**(self.config or {}), **(variant_metadata or {})}
        filter_clause = cd1_filter_clause(params)
        chunk_size = int(params.get("chunk_size", DEFAULT_CD1_CHUNK_SIZE))

        # Shim A: parquet bulk ordering, the equivalent of field_metadata("bulk")
        bulk_ids = bulk_field_ids(conn, history_sql)
        try:
            bulk_idx_glycogen = bulk_ids.index(_GLYCOGEN_ID)
        except ValueError:
            raise ValueError(
                f"{_GLYCOGEN_ID!r} not in the parquet bulk ordering "
                f"({len(bulk_ids)} molecules) — cannot compute glycogen content"
            ) from None
        glycogen_sql = f"bulk__count[{bulk_idx_glycogen + 1}] AS glycogen_raw"

        history_subquery = read_stacked_columns(
            history_sql,
            [
                "listeners__mass__cell_mass",
                "listeners__mass__dry_mass",
                "listeners__mass__volume",
                "listeners__mass__dna_mass",
                "listeners__mass__rna_mass",
                glycogen_sql,
            ],
            order_results=False,
        )

        # (count / fg dry mass) * (1 mol / N_A count) * (1e3 mmol / mol)
        #   = mmol / g dry mass
        glycogen_scale = Unum.asNumber(units.g / units.fg) / N_A * 1e3
        # (fg) * (1 mg / 1e12 fg) / (1e9 cells) = mg per 1e9 cells
        mass_scale = Unum.asNumber(units.mg / units.fg) * 10**-9

        id_cols = ", ".join(_ID_COLS)
        # Chunked per cell (sim 742: this module alone needed 44.8 GiB at 10 x 8
        # -- `bulk__count[i]` still decodes every row's ~16k-wide list). The
        # aggregate is GROUP BY the cell columns, so batching by cell changes
        # nothing but peak memory; rows are re-sorted below to keep the
        # ORDER BY the single query had.
        filtered_sql = f"""
            SELECT * FROM ({history_subquery})
            {filter_clause}
        """

        def _batch_sql(cell_filter: str) -> str:
            return f"""
            SELECT
                {id_cols},
                AVG(listeners__mass__cell_mass / {mass_scale}) AS mass_converted,
                AVG(listeners__mass__volume) AS cell_volume,
                AVG(listeners__mass__dna_mass
                    / NULLIF(listeners__mass__dry_mass, 0)) AS dna_converted,
                AVG(listeners__mass__rna_mass
                    / NULLIF(listeners__mass__dry_mass, 0)) AS rna_converted,
                AVG(glycogen_raw * {glycogen_scale}
                    / NULLIF(listeners__mass__dry_mass, 0)) AS glycogen_converted
            FROM ({filtered_sql})
            WHERE {cell_filter}
            GROUP BY {id_cols}
            """

        aggregated = run_chunked(
            conn, filtered_sql, _batch_sql, id_cols=_ID_COLS, chunk_size=chunk_size
        )
        if not aggregated.is_empty():
            aggregated = aggregated.sort(_ID_COLS)

        if aggregated.is_empty():
            empty = pl.DataFrame({"Properties": [], "mean": [], "std": []})
            return {"data": {"filename": "higher_order_properties.tsv",
                             "tsv": empty.write_csv(separator="\t"),
                             "n_properties": 0, "n_cells": 0}}

        label_map = {
            "mass_converted": "Cell mass (mg/10^9 cells)",
            "cell_volume": "Cell volume (um^3)",
            "dna_converted": "Genetic material - DNA (g DNA/g dry weight)",
            "rna_converted": "Genetic material - RNA (g RNA/g dry weight)",
            "glycogen_converted":
                "Metabolism - Glycogen (mmol glycosyl units/g dry weight)",
        }
        aggregated = aggregated.rename(label_map)
        value_labels = list(label_map.values())
        aggregated = aggregated.with_columns(
            pl.format(
                "Cell: {}_{}_{}", pl.col("lineage_seed"), pl.col("generation"),
                pl.col("agent_id")
            ).alias("cell_id")
        )
        output_final = (
            aggregated.select(["cell_id", *value_labels])
            .unpivot(
                on=value_labels,
                index="cell_id",
                variable_name="Properties",
                value_name="value",
            )
            .pivot(
                values="value",
                index="Properties",
                on="cell_id",
                aggregate_function="first",
                sort_columns=True,
            )
        )
        n_cells = output_final.width - 1
        output_final = with_cross_cell_stats(output_final, "Properties")
        return {"data": {"filename": "higher_order_properties.tsv",
                         "tsv": output_final.write_csv(separator="\t"),
                         "n_properties": output_final.height, "n_cells": n_cells}}
