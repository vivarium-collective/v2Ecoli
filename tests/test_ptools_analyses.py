"""Fidelity tests for the ptools_rna/rxns/proteins native Analysis ports."""

import glob as _glob
import os
import pytest

FIX = "/Users/eranagmon/code/sms-api/tests/fixtures/analysis_data"

_REF_HISTORY = "out/compare_harness/v2_sim/parquet/two_generations/history"
_PAIRED_SIMDATA = "out/workflow/simData.cPickle"

_HAS_SWEEP = bool(
    _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    if os.path.isdir(_REF_HISTORY) else []
) and os.path.isfile(_PAIRED_SIMDATA)


def _frame_ids(tsv_text):
    rows = [r for r in tsv_text.strip().splitlines() if r]
    return {r.split("\t")[0] for r in rows[1:]}  # skip header ($ row)


# ---------------------------------------------------------------------------
# Registration tests — always run (no fixture needed)
# ---------------------------------------------------------------------------

def test_ptools_rna_registered():
    from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rna"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def test_ptools_rxns_registered():
    from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rxns"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def test_ptools_proteins_registered():
    from v2ecoli.workflow.analyses import ptools_proteins  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_proteins"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
    from bigraph_schema import allocate_core
    assert cls({}, core=allocate_core()).outputs() == {"view": "string", "data": "map"}


# ---------------------------------------------------------------------------
# Per-generation windowing + drop-first-generation (pure, no fixture)
# ---------------------------------------------------------------------------

def test_consolidate_timepoints_per_generation_one_window_per_gen():
    """generations= collapses to ONE window per generation (mean/sum by flag)."""
    import numpy as np
    from v2ecoli.workflow.analyses.ptools_rna import consolidate_timepoints

    # 5 time rows, 2 features; generations [0,0,1,1,1].
    mtx = np.array(
        [[2.0, 10.0], [4.0, 20.0], [6.0, 30.0], [8.0, 40.0], [10.0, 50.0]]
    )
    gens = np.array([0, 0, 1, 1, 1])

    # normalized=True -> per-generation MEAN, one column per generation.
    blocks, idx = consolidate_timepoints(mtx, n_tp=99, normalized=True, generations=gens)
    assert blocks.shape == (2, 2)  # exactly one window per generation (n_tp ignored)
    np.testing.assert_allclose(blocks[0], [3.0, 15.0])  # mean of gen-0 rows 0,1
    np.testing.assert_allclose(blocks[1], [8.0, 40.0])  # mean of gen-1 rows 2,3,4
    assert list(idx) == [0, 2]  # first-row index of each generation (for time labels)

    # normalized=False -> per-generation SUM.
    bsum, _ = consolidate_timepoints(mtx, n_tp=99, normalized=False, generations=gens)
    np.testing.assert_allclose(bsum[0], [6.0, 30.0])
    np.testing.assert_allclose(bsum[1], [24.0, 120.0])


def test_groupby_time_keeps_generation_label_not_summed():
    """generation is a label carried through as min-per-time, never summed."""
    import pandas as pd
    from v2ecoli.workflow.analyses.ptools_rna import _groupby_time_keep_generation

    # Two rows share each time; data column sums, generation must NOT.
    df = pd.DataFrame(
        {
            "active_ribosome": [10, 20, 5, 7],
            "time": [0, 0, 60, 60],
            "generation": [0, 0, 1, 1],
        }
    )
    out = _groupby_time_keep_generation(df).sort_values("time").reset_index(drop=True)
    assert list(out["time"]) == [0, 60]
    assert list(out["active_ribosome"]) == [30, 12]  # data summed
    assert list(out["generation"]) == [0, 1]  # min-preserved (summing would give 0, 2)


def test_drop_leading_generations_relative_to_min():
    """skip=1 drops just the first generation, whatever its (0-indexed) start."""
    import duckdb
    from v2ecoli.workflow.analyses.ptools_multiscale import drop_leading_generations

    conn = duckdb.connect()
    conn.sql(
        "CREATE TABLE h AS SELECT * FROM (VALUES "
        "(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)) AS t(generation, v)"
    )
    base = "SELECT * FROM h"

    # skip=1 -> generation >= 1 (drops the first, gen 0).
    kept = conn.sql(
        f"SELECT DISTINCT generation FROM ({drop_leading_generations(conn, base, 1)}) "
        "ORDER BY generation"
    ).df()["generation"].tolist()
    assert kept == [1, 2, 3]

    # skip=0 -> unchanged (all generations).
    kept0 = conn.sql(
        f"SELECT DISTINCT generation FROM ({drop_leading_generations(conn, base, 0)}) "
        "ORDER BY generation"
    ).df()["generation"].tolist()
    assert kept0 == [0, 1, 2, 3]

    # Guard: skip >= number of generations -> unchanged (never empties the table).
    unchanged = drop_leading_generations(conn, base, 4)
    assert conn.sql(f"SELECT COUNT(*) FROM ({unchanged})").fetchone()[0] == 4

    # Relative to a non-zero minimum: gens {5,6,7}, skip=1 -> keep {6,7}.
    conn.sql(
        "CREATE TABLE h2 AS SELECT * FROM (VALUES "
        "(5, 1.0), (6, 2.0), (7, 3.0)) AS t(generation, v)"
    )
    kept_rel = conn.sql(
        "SELECT DISTINCT generation FROM ("
        f"{drop_leading_generations(conn, 'SELECT * FROM h2', 1)}) ORDER BY generation"
    ).df()["generation"].tolist()
    assert kept_rel == [6, 7]


def _fake_sim_data(bulk_ids):
    import numpy as np
    from types import SimpleNamespace
    return SimpleNamespace(
        internal_state=SimpleNamespace(
            bulk_molecules=SimpleNamespace(bulk_data={"id": np.array(bulk_ids)})
        )
    )


def test_bulk_count_matrix_tolerates_sim_id_absent_from_parquet():
    """A sim_data bulk molecule missing from the emitted bulk__id list (e.g. an
    injected NG-GFP-MONOMER[c]) gets a zero column, not a KeyError — the crash
    that took out ptools_rna/proteins on the Run-4 GFP-reporter genotypes."""
    import numpy as np
    import pandas as pd
    from v2ecoli.workflow.analyses._shims import bulk_count_matrix

    sim_ids = ["ATP[c]", "NG-GFP-MONOMER[c]", "GDP[c]"]
    # Parquet emitted only two of the three ids, in a different order.
    pq_ids = ["GDP[c]", "ATP[c]"]
    df = pd.DataFrame({
        "bulk__id": [pq_ids, pq_ids],
        "bulk__count": [np.array([7, 3]), np.array([8, 4])],
    })
    with pytest.warns(UserWarning, match="absent from the emitted"):
        mtx = bulk_count_matrix(df, _fake_sim_data(sim_ids))

    # (n_tp=2, n_bulk=3), columns in sim_data order.
    assert mtx.shape == (2, 3)
    np.testing.assert_array_equal(mtx[:, 0], [3, 4])   # ATP[c] <- pq col 1
    np.testing.assert_array_equal(mtx[:, 1], [0, 0])   # NG-GFP-MONOMER[c] zero-filled
    np.testing.assert_array_equal(mtx[:, 2], [7, 8])   # GDP[c] <- pq col 0


def test_bulk_count_matrix_all_present_unchanged_and_silent():
    """When every sim_data id is emitted, the reorder is exact and no warning."""
    import warnings
    import numpy as np
    import pandas as pd
    from v2ecoli.workflow.analyses._shims import bulk_count_matrix

    sim_ids = ["ATP[c]", "GDP[c]"]
    pq_ids = ["GDP[c]", "ATP[c]"]
    df = pd.DataFrame({
        "bulk__id": [pq_ids, pq_ids],
        "bulk__count": [np.array([7, 3]), np.array([8, 4])],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        mtx = bulk_count_matrix(df, _fake_sim_data(sim_ids))
    assert mtx.shape == (2, 2)
    np.testing.assert_array_equal(mtx[:, 0], [3, 4])   # ATP[c]
    np.testing.assert_array_equal(mtx[:, 1], [7, 8])   # GDP[c]


def test_build_tu_mrna_dict_zero_fills_unemitted_trailing_mrna():
    """sim_data carries an mRNA (a tail-appended GFP new-gene) the run didn't
    emit a full_mRNA_counts column for → zero trace, no IndexError overrun."""
    import numpy as np
    from v2ecoli.workflow.analyses.ptools_rna import build_tu_mrna_dict

    # 2 timepoints, 2 emitted mRNA columns; sim_data lists a 3rd (GFP) at the tail.
    mrna_mtx = np.array([[10, 20], [11, 21]])
    mrna_tu_ids = ["b0001_RNA", "b0002_RNA", "NG-GFP-RNA"]
    with pytest.warns(UserWarning, match="unemitted trailing"):
        d = build_tu_mrna_dict(mrna_mtx, mrna_tu_ids)
    assert list(d.keys()) == mrna_tu_ids
    np.testing.assert_array_equal(d["b0001_RNA"], [10, 11])
    np.testing.assert_array_equal(d["b0002_RNA"], [20, 21])
    np.testing.assert_array_equal(d["NG-GFP-RNA"], [0, 0])  # zero-filled


def test_build_tu_mrna_dict_exact_width_silent():
    """Widths match → exact positional map, no warning."""
    import warnings
    import numpy as np
    from v2ecoli.workflow.analyses.ptools_rna import build_tu_mrna_dict

    mrna_mtx = np.array([[10, 20], [11, 21]])
    mrna_tu_ids = ["b0001_RNA", "b0002_RNA"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = build_tu_mrna_dict(mrna_mtx, mrna_tu_ids)
    np.testing.assert_array_equal(d["b0001_RNA"], [10, 11])
    np.testing.assert_array_equal(d["b0002_RNA"], [20, 21])


# ---------------------------------------------------------------------------
# Oracle shape tests (sms-api fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rna_output_shape_matches_oracle():
    oracle = open(os.path.join(FIX, "ptools_rna.txt")).read()
    header = oracle.strip().splitlines()[0].split("\t")
    assert header[0] == "$"
    assert len(_frame_ids(oracle)) > 0


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rxns_oracle_shape():
    oracle = open(os.path.join(FIX, "ptools_rxns.txt")).read()
    assert oracle.strip().splitlines()[0].split("\t")[0] == "$"
    assert len(_frame_ids(oracle)) > 0


# ---------------------------------------------------------------------------
# data["tsv"] + view present — require actual sweep parquet
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_rna_has_tsv_and_view():
    """analyze() must return both a non-empty tsv string and a non-empty view."""
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsRna({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "ptools_rna: data['tsv'] is empty"
    assert out.get("view"), "ptools_rna: view is absent or empty"
    view = out["view"]
    assert "vega" in view.lower() or "plotly" in view.lower() or "<svg" in view, (
        "ptools_rna: view does not contain a recognizable plot"
    )


@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_rxns_has_tsv_and_view():
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsRxns({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "ptools_rxns: data['tsv'] is empty"
    assert out.get("view"), "ptools_rxns: view is absent or empty"
    view = out["view"]
    assert "vega" in view.lower() or "plotly" in view.lower() or "<svg" in view


@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_proteins_has_tsv_and_view():
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsProteins({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "ptools_proteins: data['tsv'] is empty"
    assert out.get("view"), "ptools_proteins: view is absent or empty"
    view = out["view"]
    assert "vega" in view.lower() or "plotly" in view.lower() or "<svg" in view
