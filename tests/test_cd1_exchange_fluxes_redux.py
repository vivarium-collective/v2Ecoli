"""cd1_exchange_fluxes on metabolism-redux history (the binding that Run 2 needs).

Real J3 sweeps (CD2 Run 2) run the ``ecoli-metabolism-redux`` swap, which never
emits ``external_exchange_fluxes``; the analysis used to fail with a DuckDB
binder error on every one of them (sims 574, 679; MNP 666 would have too).
"""

from __future__ import annotations

import polars as pl
import pytest


def _history(tmp_path, *, dt: float = 1.0, with_classic: bool = False) -> str:
    """A tiny hive-partitioned history: one cell, four ticks, redux exchange leaves."""
    n = 4
    frame = pl.DataFrame(
        {
            "experiment_id": ["e"] * n,
            "variant": [0] * n,
            "lineage_seed": [0] * n,
            "generation": [0] * n,
            "agent_id": ["0"] * n,
            "global_time": [i * dt for i in range(n)],
            "listeners__mass__dry_mass": [640.0] * n,  # femtograms
            "listeners__mass__instantaneous_growth_rate": [1e-4] * n,
            # LP-raw: uptake POSITIVE, secretion NEGATIVE (counts per tick)
            "listeners__fba_results__estimated_exchange_dmdt__GLC[p]": [6.0e5] * n,
            "listeners__fba_results__estimated_exchange_dmdt__OXYGEN-MOLECULE[p]": [
                1.2e6
            ]
            * n,
            "listeners__fba_results__estimated_exchange_dmdt__VIOLACEIN[c]": [-1.5e4]
            * n,
        }
    )
    if with_classic:
        frame = frame.with_columns(
            pl.Series(
                "listeners__fba_results__external_exchange_fluxes", [[-6.0, 2.0]] * n
            )
        )
    path = tmp_path / "history.parquet"
    frame.write_parquet(path)
    return f"SELECT * FROM read_parquet('{path}')"


def _run(history_sql, sim_data=None):
    import v2ecoli.workflow.analyses  # noqa: F401  (register ports)
    from v2ecoli.library.parquet_emitter import create_duckdb_conn
    from v2ecoli.workflow.analyses.cd1_exchange_fluxes import Cd1ExchangeFluxes
    from bigraph_schema import allocate_core

    conn = create_duckdb_conn(temp_dir=None)
    step = Cd1ExchangeFluxes({}, core=allocate_core())
    out = step.analyze(conn=conn, history_sql=history_sql, sim_data=sim_data)
    tsv = out["data"]["tsv"]
    rows = {line.split("\t")[0]: line.split("\t") for line in tsv.splitlines()[1:]}
    header = tsv.splitlines()[0].split("\t")
    return header, rows


def test_redux_history_binds_without_the_classic_column(tmp_path) -> None:
    header, rows = _run(_history(tmp_path))
    assert "GLC" in rows and "OXYGEN-MOLECULE" in rows and "growth_rate_h" in rows
    # the ONE cytoplasmic exchange must survive a compartment-agnostic match
    assert "VIOLACEIN" in rows, list(rows)


def test_redux_sign_is_flipped_to_uptake_negative_and_units_match_the_listener(
    tmp_path,
) -> None:
    from v2ecoli.steps.derivers.exchange_flux_listener import counts_to_gdcw_rate

    header, rows = _run(_history(tmp_path, dt=1.0))
    cell_col = next(i for i, h in enumerate(header) if h.startswith("Cell:"))
    glc = float(rows["GLC"][cell_col])
    vio = float(rows["VIOLACEIN"][cell_col])
    # uptake-positive LP counts -> canonical uptake-NEGATIVE flux; secretion positive
    assert glc < 0 and vio > 0
    expected_glc = -counts_to_gdcw_rate(6.0e5, 640.0, 1.0)
    assert glc == pytest.approx(expected_glc, rel=1e-9)
    assert vio == pytest.approx(-counts_to_gdcw_rate(-1.5e4, 640.0, 1.0), rel=1e-9)


def test_redux_uses_each_ticks_own_length(tmp_path) -> None:
    """counts/tick over a 2 s tick is half the rate of the same counts over 1 s."""
    from v2ecoli.steps.derivers.exchange_flux_listener import counts_to_gdcw_rate

    header, rows = _run(_history(tmp_path, dt=2.0))
    cell_col = next(i for i, h in enumerate(header) if h.startswith("Cell:"))
    assert float(rows["GLC"][cell_col]) == pytest.approx(
        -counts_to_gdcw_rate(6.0e5, 640.0, 2.0), rel=1e-9
    )


def test_sql_formula_and_python_helper_are_one_conversion() -> None:
    """The SQL in redux_flux_sql must stay term-for-term equal to the listener's
    counts_to_gdcw_rate, or the two paths drift apart silently."""
    from v2ecoli.library.parquet_emitter import create_duckdb_conn
    from v2ecoli.steps.derivers import exchange_flux_listener as efl
    from v2ecoli.workflow.analyses.cd1_exchange_fluxes import (
        _N_AVOGADRO,
        redux_flux_sql,
    )

    assert _N_AVOGADRO == efl._N_AVOGADRO
    conn = create_duckdb_conn(temp_dir=None)
    sql = redux_flux_sql("x", "dt").replace('"listeners__mass__dry_mass"', "m")
    got = conn.sql(
        f"SELECT {sql} AS f FROM (SELECT 6.0e5 AS x, 640.0 AS m, 1.0 AS dt)"
    ).fetchone()[0]
    assert got == pytest.approx(-efl.counts_to_gdcw_rate(6.0e5, 640.0, 1.0), rel=1e-12)


def test_classic_column_still_wins_when_present(tmp_path, monkeypatch) -> None:
    from v2ecoli.workflow.analyses import cd1_exchange_fluxes as mod

    monkeypatch.setattr(
        mod, "external_exchange_molecule_ids", lambda sim_data: ["GLC[p]", "ACET[p]"]
    )
    header, rows = _run(_history(tmp_path, with_classic=True), sim_data=object())
    cell_col = next(i for i, h in enumerate(header) if h.startswith("Cell:"))
    # classic values are used verbatim (-6.0 for GLC), not re-derived from the redux leaves
    assert float(rows["GLC"][cell_col]) == pytest.approx(-6.0)
    assert "ACET" in rows and "VIOLACEIN" not in rows
