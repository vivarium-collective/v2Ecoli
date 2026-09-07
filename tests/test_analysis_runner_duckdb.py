from v2ecoli.workflow.analysis_runner import scale_history_sql

_FROM = "read_parquet(['x.pq'], hive_partitioning=true)"


def test_single_sql_filters_full_cell():
    sql = scale_history_sql("single", _FROM, (0, 1, 2, "00"))
    assert "variant = 0" in sql and "lineage_seed = 1" in sql
    assert "generation = 2" in sql and "agent_id = '00'" in sql


def test_multiseed_sql_filters_variant_and_canonical_lineage():
    sql = scale_history_sql("multiseed", _FROM, (3,))
    assert "variant = 3" in sql
    assert "lineage_seed" not in sql
    # lineage-collapsing scale: restrict to the all-zeros single-daughter chain
    assert "NOT LIKE '%1%'" in sql


def test_multivariant_sql_filters_canonical_lineage_only():
    sql = scale_history_sql("multivariant", _FROM, ())
    # No variant/seed key, but still restrict to the canonical all-zeros lineage
    assert "NOT LIKE '%1%'" in sql
    assert _FROM in sql


def test_multigeneration_sql_excludes_birth_stubs():
    sql = scale_history_sql("multigeneration", _FROM, (0, 1))
    assert "variant = 0" in sql and "lineage_seed = 1" in sql
    # the d?1 birth-stub partitions (agent_id containing '1') must be excluded
    # so they don't contaminate the cross-generation aggregation
    assert "NOT LIKE '%1%'" in sql


def test_multidaughter_keeps_sisters_not_allzeros_filter():
    # multidaughter deliberately keeps sister daughters — must NOT get the
    # all-zeros lineage filter that the collapsing scales use.
    sql = scale_history_sql("multidaughter", _FROM, (0, 1, 2, "00"))
    assert "agent_id LIKE '00_'" in sql
    assert "NOT LIKE '%1%'" not in sql


def test_all_connection_sites_use_configured_factory_not_bare_connect():
    """Item 38 regression guard: sim 214's real analysis job OOM-killed
    (exitCode 137) after every ``duckdb.connect()`` call site in this module
    went unconfigured — no ``temp_directory``, so DuckDB couldn't spill large
    intermediate query state to disk before the container's memory ceiling hit.

    The fix ports vEcoli-private's own real pattern (``create_duckdb_conn`` in
    ``ecoli/library/parquet_emitter.py``, already re-exported and installed via
    ``v2ecoli.library.parquet_emitter`` / ``viva_emitters`` — no new dependency)
    instead of a bare, unconfigured connection. This is a real source-content
    assertion, not a mock of DuckDB's own behavior (which ``viva_emitters``
    already owns and tests) — it only guards that the analysis DuckDB path keeps
    calling the configured factory at every site, so a future edit can't silently
    reintroduce a bare ``duckdb.connect()``.

    v2ecoli splits the connection sites across two modules (upstream vEcoli-private
    kept them in one): ``sweep_io`` owns the S3/glob helpers (``history_files``,
    ``connect_for``) and ``analysis_runner`` owns the record builders
    (``build_cell_records``, ``run_analyses._analysis_ctx``). Both are scanned.
    """
    import inspect

    import v2ecoli.library.sweep_io as sio
    import v2ecoli.workflow.analysis_runner as ar

    for mod in (ar, sio):
        source = inspect.getsource(mod)
        assert "duckdb.connect()" not in source, (
            f"found a bare duckdb.connect() in {mod.__name__} — every connection "
            "site must go through create_duckdb_conn(temp_dir=...) so DuckDB can "
            "spill to disk instead of OOM-killing the container (item 38)"
        )
    connect_call_count = sum(
        inspect.getsource(mod).count("create_duckdb_conn(temp_dir=")
        for mod in (ar, sio)
    )
    assert connect_call_count == 5, (
        "expected all 5 known connection sites (sweep_io.history_files, "
        "sweep_io.connect_for, analysis_runner.build_cell_records, "
        "analysis_runner.run_analyses._analysis_ctx, and "
        "analysis_runner._run_identity_candidates -- the S3 run_identity.json "
        "glob) to use "
        f"create_duckdb_conn(temp_dir=...), found {connect_call_count}"
    )


def test_create_duckdb_conn_actually_configures_spill_and_memory_pragmas(tmp_path):
    """Real behavioral check (no mocking) that the factory this module now
    calls genuinely sets the memory-safety pragmas — not just that the right
    function name is called, but that it does what item 38's fix needs.
    """
    from v2ecoli.library.parquet_emitter import create_duckdb_conn

    conn = create_duckdb_conn(temp_dir=str(tmp_path))
    try:
        row = conn.sql(
            "SELECT current_setting('temp_directory') AS td, "
            "current_setting('preserve_insertion_order') AS pio, "
            "current_setting('parquet_metadata_cache') AS pmc, "
            "current_setting('enable_external_file_cache') AS efc"
        ).fetchone()
        (
            temp_directory,
            preserve_insertion_order,
            parquet_metadata_cache,
            enable_external_file_cache,
        ) = row
        assert temp_directory == str(tmp_path)
        assert preserve_insertion_order is False
        assert parquet_metadata_cache is False
        assert enable_external_file_cache is False
    finally:
        conn.close()


import glob as _glob
import os as _os
import pytest as _pytest


def _ref_history_dir():
    d = "out/compare_harness/v2_sim/parquet/two_generations/history"
    return d if _glob.glob(_os.path.join(d, "**", "*.pq"), recursive=True) else None


_PAIRED_SIMDATA = "out/workflow/simData.cPickle"


@_pytest.mark.skipif(_ref_history_dir() is None or not _os.path.isfile(_PAIRED_SIMDATA),
                     reason="reference sweep parquet or paired sim_data absent")
def test_proving_set_end_to_end(tmp_path):
    # Build a self-contained sweep dir: history parquet (symlinked) + the PAIRED
    # sim_data at the sweep root, so resolve_sim_data finds the correct one.
    import v2ecoli.workflow.analyses  # noqa: F401  (register ports)
    from v2ecoli.workflow.analysis_runner import run_analyses
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "history").symlink_to(_os.path.abspath(_ref_history_dir()))
    (sweep / "simData.cPickle").symlink_to(_os.path.abspath(_PAIRED_SIMDATA))

    opts = {
        "single": {"ptools_rna": {"n_tp": 8}, "ptools_rxns": {"n_tp": 8}},
        "multiseed": {"central_carbon_metabolism_scatter": {}},
    }
    res = run_analyses(str(sweep), opts)

    # data product present (not a recorded error) for a single-scale data analysis
    rna = res["single"]["ptools_rna"]
    assert rna, "ptools_rna produced no per-group result"
    first = next(iter(rna.values()))
    assert "error" not in first, f"ptools_rna errored: {first}"
    assert first.get("filename") == "ptools_rna.tsv"

    # reaction analysis did NOT trip the pairing assertion (correct sim_data)
    rxns = res["single"]["ptools_rxns"]
    first_rx = next(iter(rxns.values()))
    assert "error" not in first_rx, f"ptools_rxns errored (pairing?): {first_rx}"

    # ptools TSV files written to sweep/ptools/
    rna_tsvs = _glob.glob(str(sweep / "ptools" / "ptools_rna__*.tsv"))
    assert rna_tsvs, "no ptools_rna TSV written under sweep/ptools/"
    rxns_tsvs = _glob.glob(str(sweep / "ptools" / "ptools_rxns__*.tsv"))
    assert rxns_tsvs, "no ptools_rxns TSV written under sweep/ptools/"
    # TSV content sanity: first non-comment row should start with "$"
    with open(rna_tsvs[0]) as f:
        header = f.readline().rstrip("\n")
    assert header.startswith("$"), f"ptools_rna TSV header does not start with '$': {header!r}"

    # ptools view htmls written to sweep/viz/
    rna_htmls = _glob.glob(str(sweep / "viz" / "ptools_rna__*.html"))
    assert rna_htmls, "no ptools_rna view HTML written under sweep/viz/"

    # multiseed VIEW analysis wrote a viz html
    assert res["multiseed"]["central_carbon_metabolism_scatter"]
    viz = _glob.glob(str(sweep / "viz" / "central_carbon_metabolism_scatter*.html"))
    assert viz, "no viz html written for ccm_scatter"

    # analysis.json written
    assert _os.path.isfile(str(sweep / "analysis.json"))


@_pytest.mark.skipif(_ref_history_dir() is None or not _os.path.isfile(_PAIRED_SIMDATA),
                     reason="reference sweep parquet or paired sim_data absent")
def test_explicit_sim_data_path(tmp_path):
    """run_analyses with an explicit sim_data_path uses that pickle (no glob).

    The sweep directory has the parquet history symlinked in but NO sim_data
    pickle at the sweep root — so resolve_sim_data(sweep_dir) would raise
    FileNotFoundError.  Providing sim_data_path bypasses the glob entirely.
    """
    import v2ecoli.workflow.analyses  # noqa: F401  (register ports)
    from v2ecoli.workflow.analysis_runner import run_analyses

    sweep = tmp_path / "sweep_explicit_simdata"
    sweep.mkdir()
    # Link in the history parquet but deliberately omit any sim_data pickle
    # at the sweep root — this proves the explicit path is used, not the glob.
    (sweep / "history").symlink_to(_os.path.abspath(_ref_history_dir()))

    opts = {
        "single": {"ptools_rna": {"n_tp": 8}},
    }
    # Pass the sim_data path explicitly; the sweep dir has no co-located pickle.
    res = run_analyses(str(sweep), opts, sim_data_path=_PAIRED_SIMDATA)

    rna = res["single"]["ptools_rna"]
    assert rna, "ptools_rna produced no per-group result"
    first = next(iter(rna.values()))
    assert "error" not in first, f"ptools_rna errored with explicit sim_data_path: {first}"
    assert first.get("filename") == "ptools_rna.tsv"

    # TSV written via the explicit sim_data path
    rna_tsvs = _glob.glob(str(sweep / "ptools" / "ptools_rna__*.tsv"))
    assert rna_tsvs, "no ptools_rna TSV written when using explicit sim_data_path"

    # analysis.json present
    assert _os.path.isfile(str(sweep / "analysis.json"))


@_pytest.mark.skipif(_ref_history_dir() is None or not _os.path.isfile(_PAIRED_SIMDATA),
                     reason="reference sweep parquet or paired sim_data absent")
def test_s3_secret_refreshed_per_module_not_once(tmp_path, monkeypatch):
    """The DuckDB S3 SECRET must be re-issued before every analysis module.

    Regression test for item 71's b4 finding: configure_duckdb_s3() used to run
    only on the FIRST call to the shared _analysis_ctx() (guarded by `if not
    _ctx`), so a long multi-module sweep kept using the credential snapshot
    taken at the very start. Any STS session that expired before the sweep
    finished made every subsequent S3 read fail identically with ExpiredToken,
    even though the job itself reported success. configure_duckdb_s3 must now
    run before EVERY module (boto3 already knows how to refresh IRSA
    credentials -- it only needs to be asked again).

    Uses a real local sweep (no real AWS credentials needed): configure_duckdb_s3
    is replaced with a counting stub, and is_s3_uri is forced True for this
    sweep path only, so the S3-refresh branch executes without touching
    anything DuckDB actually reads.
    """
    import v2ecoli.workflow.analyses  # noqa: F401  (register ports)
    from v2ecoli.workflow import analysis_runner
    from unittest.mock import MagicMock

    sweep = tmp_path / "sweep_s3_refresh"
    sweep.mkdir()
    (sweep / "history").symlink_to(_os.path.abspath(_ref_history_dir()))
    (sweep / "simData.cPickle").symlink_to(_os.path.abspath(_PAIRED_SIMDATA))
    sweep_str = str(sweep)

    real_is_s3_uri = analysis_runner.is_s3_uri
    monkeypatch.setattr(
        analysis_runner, "is_s3_uri",
        lambda p: True if p == sweep_str else real_is_s3_uri(p))
    stub = MagicMock()
    monkeypatch.setattr(analysis_runner, "configure_duckdb_s3", stub)

    opts = {
        "single": {"ptools_rna": {"n_tp": 8}, "ptools_rxns": {"n_tp": 8}},
        "multiseed": {"central_carbon_metabolism_scatter": {}},
    }
    analysis_runner.run_analyses(sweep_str, opts)

    # 3 modules requested (ptools_rna, ptools_rxns, central_carbon_metabolism_scatter)
    # -- one refresh per module, NOT one refresh total for the whole sweep.
    assert stub.call_count == 3, (
        f"configure_duckdb_s3 called {stub.call_count} time(s) for 3 modules -- "
        "expected exactly 3 (once per module). A call_count of 1 means the old "
        "one-time-provisioning bug (frozen credentials shared across the whole "
        "sweep) has regressed.")
