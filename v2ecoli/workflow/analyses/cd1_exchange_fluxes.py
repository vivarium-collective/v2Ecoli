"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_exchange_fluxes.py``.

Per-cell mean external exchange flux for every exchanged metabolite, plus the
mean growth rate, as a wide TSV: one row per compound (and one for
``growth_rate_h``), one column per cell, plus the across-cell mean and standard
deviation.  Registered as ``"cd1_exchange_fluxes"`` (scale: ``"multiseed"``).

The vEcoli original carries a stale ``TODO: Implement`` banner, but its
``plot()`` is complete and is what this ports.

v2ecoli adaptations
-------------------
* **Shim E (exchange flux ordering).**  vEcoli explodes the exchange fluxes
  into one column per molecule and discovers them by ``fnmatch`` on
  ``listeners__fba_results__external_exchange_fluxes__*``.  v2ecoli emits a
  single 87-wide list column with no per-element names, so the port indexes it
  by :func:`~v2ecoli.workflow.analyses._shims.external_exchange_molecule_ids`
  (the sorted external-exchange molecule ids) — see ``_shims`` for why that is
  the emitted order and how it was verified.
* Each cell's FIRST emitted row carries a zero-length flux list (emitted before
  FBA has solved).  Indexing an empty list yields NULL, which ``AVG`` skips, so
  those rows drop out on their own.
* The TSV is returned as ``data["tsv"]`` instead of being written to ``outdir``
  (the runner places it under the sweep's ``ptools/`` dir).
* **Redux binding.**  ``ecoli-metabolism-redux`` never emits
  ``external_exchange_fluxes``; it emits the LP-raw exchange per molecule as
  ``listeners__fba_results__estimated_exchange_dmdt__<MOL>[c|p]`` — molecule
  COUNTS per tick, UPTAKE POSITIVE (the LP sign; redux negates its own
  ``environment.exchange`` to the canonical convention but keeps this leaf raw).
  Measured on a real J3 sweep (CD2 Run 2, sim 666): a 2×1 with the swap wrote 65
  ``fba_results`` columns and none of them was ``external_exchange_fluxes``.
  When the classic column is absent, this analysis binds those columns instead
  and converts each one exactly as the ``ExchangeFluxListener`` does —
  ``counts_to_gdcw_rate`` — then flips the sign, so the TSV is the same quantity
  in the same units and convention (mmol/gDCW/h, uptake negative) either way.
  Verified against that listener's own ``listeners__exchange_flux__glucose_exchange``
  on sim 666: median ratio −1.0000 over 136 rows. Two things a narrower binding
  would get wrong (@cplong90): violacein is the ONE exchange secreted from ``[c]``
  (``metabolism_redux.py`` adds ``VIOLACEIN[c]`` explicitly), so the match is on
  the prefix, compartment-agnostic; and the sign must flip, or a yield computed
  as ``product / abs(glucose)`` silently reads backwards (v2ecoli#86).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    available_columns,
    cd1_filter_clause,
    read_stacked_columns,
    with_cross_cell_stats,
)
from v2ecoli.workflow.analyses._shims import external_exchange_molecule_ids
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]
_FLUX_COL = "listeners__fba_results__external_exchange_fluxes"
#: Redux's per-molecule LP-raw exchange leaf, flattened by the ParquetEmitter.
_REDUX_PREFIX = "listeners__fba_results__estimated_exchange_dmdt__"
_DRY_MASS_COL = (
    "listeners__mass__dry_mass"  # femtograms, as the mass listener reports it
)
_N_AVOGADRO = 6.02214076e23  # keep equal to exchange_flux_listener._N_AVOGADRO (tested)


def _strip_compartment(mol: str) -> str:
    """``"GLC[p]" -> "GLC"``, the way the vEcoli original's column-name split did."""
    return mol[:-3] if len(mol) > 3 and mol.endswith("]") and mol[-3] == "[" else mol


def redux_flux_sql(col: str, dt_expr: str) -> str:
    """SQL for one redux exchange column -> mmol/gDCW/h, uptake NEGATIVE.

    Mirrors :func:`v2ecoli.steps.derivers.exchange_flux_listener.counts_to_gdcw_rate`
    term for term — ``(counts / N_A * 1e3) / (dry_mass_fg * 1e-15) / (dt_s / 3600)`` —
    with the LP-raw sign flipped. A test pins the two against each other.
    """
    return (
        f'-(("{col}" / {_N_AVOGADRO}) * 1e3) '
        f'/ ("{_DRY_MASS_COL}" * 1e-15) / (({dt_expr}) / 3600.0)'
    )


class Cd1ExchangeFluxes(Analysis):
    name = "cd1_exchange_fluxes"
    scale = "multiseed"
    config_schema = {
        "generation_lower_bound": "integer",
        "time_lower_bound": "float",
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

        present = available_columns(conn, history_sql)
        selects, avg_fluxes = [], []
        seen: set[str] = set()

        def _name(mol: str) -> str:
            name = _strip_compartment(mol)
            if name in seen:  # keep names unique; compartment-stripping can collide
                name = mol
            seen.add(name)
            return name.replace('"', '""')

        if _FLUX_COL in present:
            # Classic metabolism. Shim E: name each element of the flux list by
            # its sorted molecule id, then strip the compartment suffix.
            for i, mol in enumerate(external_exchange_molecule_ids(sim_data)):
                escaped = _name(mol)
                selects.append(f'{_FLUX_COL}[{i + 1}] AS "{escaped}"')
                avg_fluxes.append(f'AVG("{escaped}") AS "{escaped}"')
        else:
            # Redux: one column per molecule, counts/tick, LP-raw sign. Convert
            # per row with that row's own tick length; the first row of each
            # cell has no predecessor, so its dt is NULL and AVG skips it --
            # the same first-row drop the classic branch gets from the empty
            # pre-FBA flux list.
            dmdt_cols = sorted(c for c in present if c.startswith(_REDUX_PREFIX))
            if not dmdt_cols:
                raise ValueError(
                    f"{self.name}: history has neither {_FLUX_COL!r} (classic metabolism) "
                    f"nor any {_REDUX_PREFIX}* column (metabolism-redux); nothing to bind"
                )
            if _DRY_MASS_COL not in present:
                raise ValueError(
                    f"{self.name}: redux binding needs {_DRY_MASS_COL!r} to convert "
                    "counts/tick to mmol/gDCW/h, and the history does not carry it"
                )
            partition = ", ".join(_ID_COLS)
            dt_expr = (
                f"global_time - LAG(global_time) OVER "
                f"(PARTITION BY {partition} ORDER BY global_time)"
            )
            for col in dmdt_cols:
                escaped = _name(col[len(_REDUX_PREFIX) :])
                selects.append(f'{redux_flux_sql(col, dt_expr)} AS "{escaped}"')
                avg_fluxes.append(f'AVG("{escaped}") AS "{escaped}"')

        columns = [
            "listeners__mass__instantaneous_growth_rate * 3600 AS growth_rate_h",
            *selects,
        ]
        flux_subquery = read_stacked_columns(history_sql, columns, order_results=False)
        id_cols = ", ".join(_ID_COLS)

        flux_data = conn.sql(
            f"""
            SELECT {", ".join(avg_fluxes)},
                avg(growth_rate_h) AS growth_rate_h,
                concat('Cell: ', lineage_seed, '_', generation, '_', agent_id) AS cell_id
            FROM ({flux_subquery})
            {filter_clause}
            GROUP BY {id_cols}
            """
        ).pl()

        if flux_data.is_empty():
            empty = pl.DataFrame({"EcoCyc Compound ID": [], "mean": [], "std": []})
            return {
                "data": {
                    "filename": "exchange_fluxes.tsv",
                    "tsv": empty.write_csv(separator="\t"),
                    "n_compounds": 0,
                    "n_cells": 0,
                }
            }

        # Transpose: cell_ids become columns; metabolites + growth_rate_h rows.
        cell_ids = flux_data["cell_id"].to_list()
        wide_table = flux_data.drop("cell_id").transpose(
            include_header=True,
            header_name="EcoCyc Compound ID",
            column_names=cell_ids,
        )
        n_cells = len(cell_ids)
        wide_table = with_cross_cell_stats(wide_table, "EcoCyc Compound ID")
        return {
            "data": {
                "filename": "exchange_fluxes.tsv",
                "tsv": wide_table.write_csv(separator="\t"),
                "n_compounds": wide_table.height,
                "n_cells": n_cells,
            }
        }
