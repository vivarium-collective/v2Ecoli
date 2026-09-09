"""Regression tests for item 38's cd1_* query-restructuring fix.

The real-AWS re-test (item 38) showed the ``cd1_*`` multiseed analyses
OOM-kill the analysis DAG node: each of ``cd1_proteomics``/``cd1_fluxomics``/
``cd1_metabolomics``/``cd1_transcriptomics`` explodes every cell's full
per-timepoint entity-count list into individual rows (``unnest`` +
``generate_subscripts``) BEFORE its ``GROUP BY`` collapses them back down —
materializing the whole sweep's arrays in memory at once. The fix
(``_helpers.run_chunked``/``distinct_cell_filters``) restructures each
module's query to process one cell at a time and concatenate results,
bounding peak memory to a single cell's arrays. Every module's ``GROUP BY``
already partitions by the same cell-identity columns (never combines rows
across cells), so this is provably lossless — these tests confirm it
numerically, not just structurally.

Nothing about the aggregation math, filters, unit conversions, or biological
assumptions changes — only how the query executes.
"""

from __future__ import annotations

import types

import duckdb
import numpy as np
import polars as pl
import pytest
from bigraph_schema import allocate_core


# ---------------------------------------------------------------------------
# Generic chunking-primitive tests (_helpers.distinct_cell_filters/run_chunked)
# ---------------------------------------------------------------------------


def _synthetic_conn():
    conn = duckdb.connect()
    df = pl.DataFrame(
        {
            "cell": ["a", "a", "b", "b", "c", "c"],
            "value": [1.0, 3.0, 10.0, 20.0, 100.0, 300.0],
        }
    )
    conn.register("t", df)
    return conn


def test_distinct_cell_filters_chunk_size_one_yields_one_fragment_per_cell():
    from v2ecoli.workflow.analyses._helpers import distinct_cell_filters

    conn = _synthetic_conn()
    fragments = distinct_cell_filters(conn, "SELECT * FROM t", id_cols=["cell"], chunk_size=1)
    assert len(fragments) == 3
    assert all("(cell) IN (" in f for f in fragments)


def test_distinct_cell_filters_batches_by_chunk_size():
    from v2ecoli.workflow.analyses._helpers import distinct_cell_filters

    conn = _synthetic_conn()
    fragments = distinct_cell_filters(conn, "SELECT * FROM t", id_cols=["cell"], chunk_size=2)
    assert len(fragments) == 2  # 3 cells, batches of 2 -> [2, 1]


def test_default_chunk_size_matches_documented_constant_for_both_entry_points():
    """Regression guard (item 77): every OTHER test above passes an explicit
    chunk_size=1/2, so none of them would catch a future silent regression
    back to the old chunk_size=1 default. Confirm both `distinct_cell_filters`
    and `run_chunked` batch by DEFAULT_CD1_CHUNK_SIZE (8 since sim 742) when chunk_size is
    omitted entirely, for an n_cells that spans multiple chunks at that size.
    """
    import math

    from v2ecoli.workflow.analyses._helpers import (
        DEFAULT_CD1_CHUNK_SIZE,
        distinct_cell_filters,
        run_chunked,
    )

    assert DEFAULT_CD1_CHUNK_SIZE == 8

    n_cells = 250  # spans many chunks at the default size (250 -> 32)
    conn = duckdb.connect()
    conn.register(
        "many_cells",
        pl.DataFrame(
            {
                "cell": [f"cell_{i}" for i in range(n_cells)],
                "value": list(range(n_cells)),
            }
        ),
    )
    expected_chunks = math.ceil(n_cells / DEFAULT_CD1_CHUNK_SIZE)

    # distinct_cell_filters: fragment count directly reflects chunk_size.
    fragments = distinct_cell_filters(conn, "SELECT * FROM many_cells", id_cols=["cell"])
    assert len(fragments) == expected_chunks

    # run_chunked: count the chunk-batches it actually issues by instrumenting
    # build_batch_sql, with chunk_size omitted (not just fragments un-tested).
    calls = []

    def _batch_sql(cell_filter: str) -> str:
        calls.append(cell_filter)
        return f"SELECT cell, value FROM many_cells WHERE {cell_filter}"

    result = run_chunked(conn, "SELECT * FROM many_cells", _batch_sql, id_cols=["cell"])
    assert len(calls) == expected_chunks
    assert result.height == n_cells


def test_run_chunked_matches_unchunked_equivalent():
    """The chunked path must produce the exact same AVG/GROUP BY result as a
    single unchunked query over the whole table — the core correctness claim
    every cd1_* module's fix depends on."""
    from v2ecoli.workflow.analyses._helpers import run_chunked

    conn = _synthetic_conn()

    unchunked = conn.sql(
        "SELECT cell, AVG(value) AS avg_value FROM t GROUP BY cell ORDER BY cell"
    ).pl()

    def _batch_sql(cell_filter: str) -> str:
        return f"""
            SELECT cell, AVG(value) AS avg_value FROM t WHERE {cell_filter}
            GROUP BY cell
            """

    chunked = run_chunked(conn, "SELECT * FROM t", _batch_sql, id_cols=["cell"], chunk_size=1)
    chunked = chunked.sort("cell")
    assert chunked.to_dicts() == unchunked.to_dicts()


def test_run_chunked_empty_input_yields_empty_dataframe():
    from v2ecoli.workflow.analyses._helpers import run_chunked

    conn = duckdb.connect()
    conn.register("empty_t", pl.DataFrame({"cell": [], "value": []}))
    result = run_chunked(
        conn, "SELECT * FROM empty_t", lambda f: "SELECT 1", id_cols=["cell"]
    )
    assert result.is_empty()


# ---------------------------------------------------------------------------
# End-to-end parity tests: each exploding cd1_* module against hand-computed
# expected values, run through the real Analysis Step (registration + config
# + analyze()), exactly as analysis_runner.py invokes them in production.
# ---------------------------------------------------------------------------


def _history_conn(columns: dict) -> tuple[duckdb.DuckDBPyConnection, str]:
    conn = duckdb.connect()
    df = pl.DataFrame(columns)
    conn.register("history_tbl", df)
    return conn, "SELECT * FROM history_tbl"


_BASE_ID_COLS = {
    "experiment_id": ["e"] * 4,
    "variant": [0] * 4,
    "lineage_seed": [0, 0, 1, 1],
    "generation": [0, 0, 0, 0],
    "agent_id": ["0"] * 4,
    "global_time": [0.0, 1.0, 0.0, 1.0],
}


def _run_step(step_cls, sim_data, conn, history_sql, config=None):
    step = step_cls(config or {}, core=allocate_core())
    return step.update(
        {
            "conn": conn,
            "history_sql": history_sql,
            "config_sql": "",
            "success_sql": "",
            "sim_data": sim_data,
            "validation_data": None,
            "variant_metadata": {},
        }
    )


def test_cd1_proteomics_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_proteomics import Cd1Proteomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "listeners__monomer_counts": [
                [10.0, 20.0], [20.0, 30.0], [100.0, 200.0], [200.0, 300.0],
            ],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            translation=types.SimpleNamespace(monomer_data={"id": ["M1[c]", "M2[c]"]})
        )
    )
    out = _run_step(Cd1Proteomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    assert rows["M1"][3:5] == ["15.0", "150.0"]  # Cell: 0_0, Cell: 1_0
    assert rows["M2"][3:5] == ["25.0", "250.0"]


def test_cd1_fluxomics_chunked_matches_hand_computed_unit_converted_means():
    from wholecell.utils import units

    from v2ecoli.workflow.analyses.cd1_fluxomics import Cd1Fluxomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "listeners__fba_results__base_reaction_fluxes": [
                [1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0],
            ],
            "listeners__mass__cell_mass": [20.0] * 4,
            "listeners__mass__dry_mass": [10.0] * 4,
        }
    )
    cell_density = 1100.0
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            metabolism=types.SimpleNamespace(base_reaction_ids=["R1", "R2"])
        ),
        constants=types.SimpleNamespace(cell_density=cell_density * units.g / units.L),
    )
    out = _run_step(Cd1Fluxomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}

    def expected(fluxes, dry, cell_mass):
        coeff = dry / cell_mass * cell_density
        avg = sum(f / coeff for f in fluxes) / len(fluxes)
        return avg * 3600.0  # mmol/g/s -> mmol/g/h, matches the module's own unit conversion

    assert float(rows["R1"][3]) == pytest.approx(expected([1.0, 3.0], 10.0, 20.0))
    assert float(rows["R1"][4]) == pytest.approx(expected([10.0, 30.0], 10.0, 20.0))
    assert float(rows["R2"][3]) == pytest.approx(expected([2.0, 4.0], 10.0, 20.0))
    assert float(rows["R2"][4]) == pytest.approx(expected([20.0, 40.0], 10.0, 20.0))


def test_cd1_metabolomics_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_metabolomics import Cd1Metabolomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "bulk__id": [["A[c]", "B[c]", "C[c]"]] * 4,
            "bulk__count": [[10, 20, 30], [20, 30, 40], [100, 200, 300], [200, 300, 400]],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            metabolism=types.SimpleNamespace(conc_dict={"A[c]": 1, "C[c]": 1})
        )
    )
    out = _run_step(Cd1Metabolomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    assert rows["A"][3:5] == ["15.0", "150.0"]
    assert rows["C"][3:5] == ["35.0", "350.0"]


def test_cd1_transcriptomics_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_transcriptomics import Cd1Transcriptomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "listeners__rna_counts__mRNA_cistron_counts": [
                [5.0, 6.0], [7.0, 8.0], [50.0, 60.0], [70.0, 80.0],
            ],
        }
    )
    cistron_data = {
        "id": np.array(["G1_RNA", "G2_RNA"]),
        "is_mRNA": np.array([True, True]),
    }
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            transcription=types.SimpleNamespace(cistron_data=cistron_data)
        )
    )
    out = _run_step(Cd1Transcriptomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    assert rows["G1"][3:5] == ["6.0", "60.0"]
    assert rows["G2"][3:5] == ["7.0", "70.0"]


# ---------------------------------------------------------------------------
# Multi-generation cell-identity regression (item 71 verification): before
# this fix, cell_id was built from only (lineage_seed, agent_id) — generation
# omitted. A batch with >1 generation per seed produced duplicate cell_id
# values (e.g. "Cell: 0_0" for BOTH seed 0's generation 0 and generation 1),
# which crashed cd1_proteomics/cd1_fluxomics/cd1_transcriptomics/
# cd1_exchange_fluxes outright (DuplicateError / ComputeError on the
# duplicate-keyed pivot/transpose) and silently produced WRONG output for
# cd1_metabolomics (summed two generations' values together) and
# cd1_higher_order_properties (silently kept only the first generation's
# value). Confirmed against a real production run (item 71 Phase 4 pilot
# 218) hitting exactly this. Fixed by including `generation` in the cell_id
# label (v2ecoli's per-generation runner mode deliberately keeps `agent_id`
# constant across generations — see division.py — so generation is the only
# column that actually distinguishes these cells; vEcoli-private's own
# mechanism, a binary-encoded `agent_id`, is generation-unique by
# construction there instead, but v2ecoli's runner mode doesn't use that
# scheme, so the equivalent information has to come from the separate
# `generation` field it already tracks).
#
# 2 seeds x 2 generations, 1 timepoint per cell (4 cells, 4 rows) — every
# module below must produce exactly 4 distinct cell columns with correctly
# separated values, not 2 (collapsed-by-seed) and not a crash.
# ---------------------------------------------------------------------------

_MULTIGEN_ID_COLS = {
    "experiment_id": ["e"] * 4,
    "variant": [0] * 4,
    "lineage_seed": [0, 0, 1, 1],
    "generation": [0, 1, 0, 1],
    "agent_id": ["0"] * 4,  # constant across generations — v2ecoli's runner mode
    "global_time": [0.0, 0.0, 0.0, 0.0],
}


def test_cd1_proteomics_separates_generations_not_just_seeds():
    from v2ecoli.workflow.analyses.cd1_proteomics import Cd1Proteomics

    conn, history_sql = _history_conn(
        {
            **_MULTIGEN_ID_COLS,
            "listeners__monomer_counts": [
                [10.0], [20.0], [100.0], [200.0],
            ],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            translation=types.SimpleNamespace(monomer_data={"id": ["M1[c]"]})
        )
    )
    out = _run_step(Cd1Proteomics, sim_data, conn, history_sql)  # must not raise
    header, rows = out["data"]["tsv"].splitlines()[0].split("\t"), out["data"]["tsv"].splitlines()[1:]
    assert len(header) - 3 == 4, f"expected 4 distinct cell columns, got header {header}"
    values = {h: v for h, v in zip(header[3:], rows[0].split("\t")[3:])}
    assert values["Cell: 0_0_0"] == "10.0"
    assert values["Cell: 0_1_0"] == "20.0"
    assert values["Cell: 1_0_0"] == "100.0"
    assert values["Cell: 1_1_0"] == "200.0"


def test_cd1_fluxomics_separates_generations_not_just_seeds():
    from wholecell.utils import units

    from v2ecoli.workflow.analyses.cd1_fluxomics import Cd1Fluxomics

    conn, history_sql = _history_conn(
        {
            **_MULTIGEN_ID_COLS,
            "listeners__fba_results__base_reaction_fluxes": [
                [1.0], [2.0], [3.0], [4.0],
            ],
            "listeners__mass__cell_mass": [20.0] * 4,
            "listeners__mass__dry_mass": [10.0] * 4,
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            metabolism=types.SimpleNamespace(base_reaction_ids=["R1"])
        ),
        constants=types.SimpleNamespace(cell_density=1100.0 * units.g / units.L),
    )
    out = _run_step(Cd1Fluxomics, sim_data, conn, history_sql)  # must not raise
    header = out["data"]["tsv"].splitlines()[0].split("\t")
    assert len(header) - 3 == 4, f"expected 4 distinct cell columns, got header {header}"
    assert set(header[3:]) == {
        "Cell: 0_0_0", "Cell: 0_1_0", "Cell: 1_0_0", "Cell: 1_1_0",
    }


def test_cd1_transcriptomics_separates_generations_not_just_seeds():
    from v2ecoli.workflow.analyses.cd1_transcriptomics import Cd1Transcriptomics

    conn, history_sql = _history_conn(
        {
            **_MULTIGEN_ID_COLS,
            "listeners__rna_counts__mRNA_cistron_counts": [
                [5.0], [6.0], [7.0], [8.0],
            ],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            transcription=types.SimpleNamespace(cistron_data={
                "id": np.array(["G1_RNA"]), "is_mRNA": np.array([True]),
            })
        )
    )
    out = _run_step(Cd1Transcriptomics, sim_data, conn, history_sql)  # must not raise
    header, rows = out["data"]["tsv"].splitlines()[0].split("\t"), out["data"]["tsv"].splitlines()[1:]
    assert len(header) - 3 == 4, f"expected 4 distinct cell columns, got header {header}"
    values = {h: v for h, v in zip(header[3:], rows[0].split("\t")[3:])}
    assert values["Cell: 0_0_0"] == "5.0"
    assert values["Cell: 0_1_0"] == "6.0"
    assert values["Cell: 1_0_0"] == "7.0"
    assert values["Cell: 1_1_0"] == "8.0"


def test_cd1_metabolomics_separates_generations_not_just_seeds():
    """Before the fix, this module didn't crash — it silently SUMMED two
    generations' values together (aggregate_function="sum", originally added
    for an unrelated reason: metabolites sharing a name once the location
    suffix is stripped). Confirm each generation's value now survives
    distinctly instead of being silently folded into its sibling generation."""
    from v2ecoli.workflow.analyses.cd1_metabolomics import Cd1Metabolomics

    conn, history_sql = _history_conn(
        {
            **_MULTIGEN_ID_COLS,
            "bulk__id": [["A[c]"]] * 4,
            "bulk__count": [[10], [20], [100], [200]],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            metabolism=types.SimpleNamespace(conc_dict={"A[c]": 1})
        )
    )
    out = _run_step(Cd1Metabolomics, sim_data, conn, history_sql)
    header, rows = out["data"]["tsv"].splitlines()[0].split("\t"), out["data"]["tsv"].splitlines()[1:]
    assert len(header) - 3 == 4, f"expected 4 distinct cell columns, got header {header}"
    values = {h: v for h, v in zip(header[3:], rows[0].split("\t")[3:])}
    assert values["Cell: 0_0_0"] == "10.0"  # previously silently summed to 30.0
    assert values["Cell: 0_1_0"] == "20.0"


def test_cd1_higher_order_properties_separates_generations_not_just_seeds():
    """Before the fix, this module didn't crash — it silently kept only the
    FIRST generation's value (aggregate_function="first"), dropping the
    second generation entirely. Confirm both generations now survive."""
    from v2ecoli.workflow.analyses.cd1_higher_order_properties import (
        Cd1HigherOrderProperties,
    )

    conn, history_sql = _history_conn(
        {
            **_MULTIGEN_ID_COLS,
            "listeners__mass__cell_mass": [20.0, 40.0, 20.0, 40.0],
            "listeners__mass__dry_mass": [10.0, 20.0, 10.0, 20.0],
            "listeners__mass__volume": [1.0, 2.0, 1.0, 2.0],
            "listeners__mass__dna_mass": [1.0, 2.0, 1.0, 2.0],
            "listeners__mass__rna_mass": [1.0, 2.0, 1.0, 2.0],
            "bulk__id": [["glycogen-monomer[c]"]] * 4,
            "bulk__count": [[0], [0], [0], [0]],
        }
    )
    sim_data = None  # not read directly; glycogen comes from the parquet bulk ordering
    out = _run_step(Cd1HigherOrderProperties, sim_data, conn, history_sql)
    header = out["data"]["tsv"].splitlines()[0].split("\t")
    assert len(header) - 3 == 4, (
        f"expected 4 distinct cell columns (both generations survive), got "
        f"header {header} — a length of 2 here means generation 1 was "
        f"silently dropped, exactly the pre-fix bug"
    )
    assert set(header[3:]) == {
        "Cell: 0_0_0", "Cell: 0_1_0", "Cell: 1_0_0", "Cell: 1_1_0",
    }


def test_cd1_exchange_fluxes_separates_generations_not_just_seeds():
    from v2ecoli.workflow.analyses.cd1_exchange_fluxes import Cd1ExchangeFluxes

    conn, history_sql = _history_conn(
        {
            **_MULTIGEN_ID_COLS,
            "listeners__mass__instantaneous_growth_rate": [0.001, 0.002, 0.003, 0.004],
            "listeners__fba_results__external_exchange_fluxes": [
                [1.0], [2.0], [3.0], [4.0],
            ],
        }
    )
    sim_data = types.SimpleNamespace(
        external_state=types.SimpleNamespace(
            all_external_exchange_molecules=["GLC[p]"]
        )
    )
    out = _run_step(Cd1ExchangeFluxes, sim_data, conn, history_sql)  # must not raise
    header = out["data"]["tsv"].splitlines()[0].split("\t")
    assert len(header) - 3 == 4, f"expected 4 distinct cell columns, got header {header}"
    assert set(header[3:]) == {
        "Cell: 0_0_0", "Cell: 0_1_0", "Cell: 1_0_0", "Cell: 1_1_0",
    }


# ---------------------------------------------------------------------------
# cd1_higher_order_properties went through run_chunked in the sim-742 memory
# work (it was one of the two 44.8 GiB modules). Chunking must not change a
# single value: every aggregate is GROUP BY cell and a cell never spans two
# batches. Five cells so that chunk_size=2 yields uneven batches [2, 2, 1].
# ---------------------------------------------------------------------------


def _hop_history():
    n_cells, per_cell = 5, 3
    rows = n_cells * per_cell
    cols = {
        "experiment_id": ["e"] * rows,
        "variant": [0] * rows,
        "lineage_seed": [i // per_cell % 3 for i in range(rows)],
        "generation": [i // per_cell // 3 for i in range(rows)],
        "agent_id": ["0"] * rows,
        "global_time": [float(i % per_cell) for i in range(rows)],
        "listeners__mass__cell_mass": [1000.0 + 37.0 * i for i in range(rows)],
        "listeners__mass__dry_mass": [300.0 + 11.0 * i for i in range(rows)],
        "listeners__mass__volume": [1.0 + 0.05 * i for i in range(rows)],
        "listeners__mass__dna_mass": [10.0 + 0.5 * i for i in range(rows)],
        "listeners__mass__rna_mass": [60.0 + 1.5 * i for i in range(rows)],
        "bulk__id": [["ATP[c]", "glycogen-monomer[c]", "GLC[p]"]] * rows,
        "bulk__count": [[5, 1000 + 13 * i, 7] for i in range(rows)],
    }
    return _history_conn(cols)


@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_cd1_higher_order_properties_chunked_is_identical_to_unchunked(chunk_size):
    from v2ecoli.workflow.analyses.cd1_higher_order_properties import (
        Cd1HigherOrderProperties,
    )

    conn, history_sql = _hop_history()
    sim_data = types.SimpleNamespace()
    unchunked = _run_step(Cd1HigherOrderProperties, sim_data, conn, history_sql, {"chunk_size": 1000})
    chunked = _run_step(Cd1HigherOrderProperties, sim_data, conn, history_sql, {"chunk_size": chunk_size})
    assert chunked["data"]["n_cells"] == 5
    assert chunked["data"]["tsv"] == unchunked["data"]["tsv"]
    header = chunked["data"]["tsv"].splitlines()[0].split("\t")
    # the column order the single ORDER BY query produced, preserved by the re-sort
    # ORDER BY the id columns: lineage_seed before generation
    assert header[3:] == ["Cell: 0_0_0", "Cell: 0_1_0", "Cell: 1_0_0", "Cell: 1_1_0", "Cell: 2_0_0"]


def test_cd1_higher_order_properties_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_higher_order_properties import (
        Cd1HigherOrderProperties,
    )

    conn, history_sql = _hop_history()
    out = _run_step(Cd1HigherOrderProperties, types.SimpleNamespace(), conn, history_sql, {"chunk_size": 2})
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    # first cell = rows 0..2: volume mean of 1.0, 1.05, 1.10
    vol = [r for r in rows if r.startswith("Cell volume")][0]
    assert abs(float(rows[vol][3]) - 1.05) < 1e-12
    # last column = Cell: 2_0_0 = rows 6..8: volume mean of 1.30, 1.35, 1.40
    assert abs(float(rows[vol][7]) - 1.35) < 1e-12
