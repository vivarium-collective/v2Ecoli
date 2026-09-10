"""scripts/run_standalone_analysis.py must build real rows from summary.json
files and drive the actual ANALYSIS_REGISTRY -- the standalone K8s analysis
path this replaces silently pulled a nonexistent image for every Ray-backend
simulation (never worked), so this entrypoint carries the whole burden of
proof that analysis genuinely executes for this dispatch shape."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_standalone_analysis import build_multiseed_rows, resolve_modules, run


def _fake_aws_cp(monkeypatch, seed_summaries: dict[int, dict], written: dict[str, str]):
    """Stub _aws_cp: 'download' pre-seeded summaries by writing them to the
    requested local dest; 'upload' by recording dest -> local content."""
    import scripts.run_standalone_analysis as mod

    def fake(src: str, dst: str) -> None:
        if src.startswith("s3://"):
            seed = int(src.rsplit("seed_", 1)[1].split("/", 1)[0])
            Path(dst).write_text(json.dumps(seed_summaries[seed]))
        else:
            written[dst] = Path(src).read_text()

    monkeypatch.setattr(mod, "_aws_cp", fake)


def test_build_multiseed_rows_reflects_no_division(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {
        0: {"seed": 0, "dry_mass_fg": 220.5},
        1: {"seed": 1, "dry_mass_fg": 221.1},
    }, written)

    rows = build_multiseed_rows("s3://bucket/exp", 2, tmp_path)

    assert len(rows) == 2
    assert all(r["divided"] is False and r["division_time"] == 0.0 for r in rows)
    assert [r["final_dry_mass"] for r in rows] == [220.5, 221.1]


def test_run_writes_real_doubling_time_result_and_manifest(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {
        0: {"seed": 0, "dry_mass_fg": 200.0},
        1: {"seed": 1, "dry_mass_fg": 240.0},
    }, written)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=2,
        modules={"multiseed": {"doubling_time_distribution": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    assert manifest["written"] == ["s3://bucket/exp/analyses/test-analysis/doubling_time_distribution.json"]
    result = json.loads(written["s3://bucket/exp/analyses/test-analysis/doubling_time_distribution.json"])
    assert result["n_cells"] == 2
    assert result["n_divided"] == 0  # this dispatch never simulates division
    assert result["final_dry_mass_mean"] == 220.0
    manifest_written = json.loads(written["s3://bucket/exp/analyses/test-analysis/_manifest.json"])
    assert manifest_written == manifest


def test_run_records_error_for_unknown_module(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {0: {"seed": 0, "dry_mass_fg": 200.0}}, written)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=1,
        modules={"multiseed": {"not_a_real_module": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert "not_a_real_module" in manifest["errors"][0]["name"]


def test_run_records_error_for_missing_seed_summary(tmp_path, monkeypatch):
    import scripts.run_standalone_analysis as mod

    def always_fail(src: str, _dst: str) -> None:
        import subprocess
        if src.startswith("s3://"):  # only the seed-summary download should fail
            raise subprocess.CalledProcessError(1, ["aws"], stderr=b"NoSuchKey")

    monkeypatch.setattr(mod, "_aws_cp", always_fail)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=1,
        modules={"multiseed": {"doubling_time_distribution": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert "no summary.json" in manifest["errors"][0]["error"]


def test_run_routes_duckdb_analysis_name_to_run_analyses(tmp_path, monkeypatch):
    """cd1_metabolomics is a real, registered Analysis (DuckDB) subclass at
    scale=multiseed -- the same scale doubling_time_distribution (an
    AnalysisStep) uses. Requesting it must route to run_analyses(), not the
    row-building path (which would TypeError against Analysis.analyze's
    conn/history_sql/sim_data signature)."""
    import scripts.run_standalone_analysis as mod

    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {}, written)

    calls: list[dict] = []

    def fake_run_analyses(*, sweep_dir, analysis_options, out_dir):
        calls.append({"sweep_dir": sweep_dir, "analysis_options": analysis_options, "out_dir": out_dir})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "analysis.json").write_text(json.dumps({"multiseed": {"cd1_metabolomics": {}}}))

    synced: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "_aws_sync", lambda src, dst: synced.append((src, dst)))

    import v2ecoli.workflow.analysis_runner as analysis_runner_mod
    monkeypatch.setattr(analysis_runner_mod, "run_analyses", fake_run_analyses)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=1,
        modules={"multiseed": {"cd1_metabolomics": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    assert manifest["written"] == ["s3://bucket/exp/analyses/test-analysis/analysis.json"]
    assert len(calls) == 1
    assert calls[0]["sweep_dir"] == "s3://bucket/exp"
    assert calls[0]["analysis_options"] == {"multiseed": {"cd1_metabolomics": {}}}
    assert len(synced) == 1
    assert synced[0][1] == "s3://bucket/exp/analyses/test-analysis"


def test_run_splits_mixed_scale_between_both_families(tmp_path, monkeypatch):
    """Same scale (multiseed), one AnalysisStep name + one Analysis (DuckDB)
    name in the same request -- both must run, via their own paths."""
    import scripts.run_standalone_analysis as mod

    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {0: {"seed": 0, "dry_mass_fg": 200.0}}, written)

    duckdb_calls: list[dict] = []

    def fake_run_analyses(*, sweep_dir, analysis_options, out_dir):
        duckdb_calls.append(analysis_options)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "analysis.json").write_text("{}")

    monkeypatch.setattr(mod, "_aws_sync", lambda src, dst: None)
    import v2ecoli.workflow.analysis_runner as analysis_runner_mod
    monkeypatch.setattr(analysis_runner_mod, "run_analyses", fake_run_analyses)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=1,
        modules={"multiseed": {"doubling_time_distribution": {}, "cd1_metabolomics": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    assert duckdb_calls == [{"multiseed": {"cd1_metabolomics": {}}}]  # DuckDB name only
    assert "s3://bucket/exp/analyses/test-analysis/doubling_time_distribution.json" in written  # AnalysisStep name


def test_full_script_subprocess_reaches_final_manifest_write_not_import_crash(tmp_path):
    """The real, load-bearing regression test: run the actual script as a
    subprocess (not an in-process import, which can mask sys.path[0] behavior)
    with the exact CLI shape viva-api's chain-dispatch analysis auto-trigger
    sends. Pre-fix, this dies inside comparison_summary.py's import with
    ModuleNotFoundError. Post-fix, it must run the ENTIRE analysis pipeline
    (proving the "applicable" resolution + full ANALYSIS_REGISTRY import both
    genuinely work end-to-end) and only fail at the very last step -- the
    deliberately-fake S3 upload this test gives it, not the import."""
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable, "scripts/run_standalone_analysis.py",
            "--out-uri", f"s3://fake-bucket-does-not-exist-{tmp_path.name}/x",
            "--n-seeds", "2", "--n-generations", "2",
            "--modules", "applicable", "--analysis-name", "test-repro",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    # Must reach the real final step (uploading the manifest) -- proof the
    # entire pipeline, including the previously-crashing import chain, ran.
    assert "_run_aws" in result.stderr and "_manifest.json" in result.stderr


def test_resolve_modules_passes_explicit_dict_through_verbatim():
    explicit = {"multiseed": {"doubling_time_distribution": {}}}
    assert resolve_modules(explicit, n_seeds=2, n_generations=2) is explicit


def test_resolve_modules_parses_explicit_json_string():
    assert resolve_modules(
        '{"multiseed": {"doubling_time_distribution": {}}}', n_seeds=1, n_generations=1,
    ) == {"multiseed": {"doubling_time_distribution": {}}}


def test_resolve_modules_applicable_keyword_delegates_to_build_analysis_options():
    """This is the exact real bug (item 60): viva-api's chain-dispatch analysis
    auto-trigger sends --modules applicable --n-generations <N>, and this
    entrypoint had no support for the keyword at all -- json.loads("applicable")
    would itself raise before generation count ever mattered. Delegates to
    v2ecoli's own build_analysis_options (the same resolution the composite's
    inline flush already uses) rather than re-deriving scale selection here."""
    single_gen = resolve_modules("applicable", n_seeds=1, n_generations=1)
    multi_gen = resolve_modules("applicable", n_seeds=2, n_generations=3)

    # single-generation, single-seed: only the "single" scale applies
    assert "multigeneration" not in single_gen
    assert "multiseed" not in single_gen
    # multi-generation, multi-seed: both scales must now be present and real
    assert "multigeneration" in multi_gen
    assert "multiseed" in multi_gen
    assert multi_gen["multigeneration"]  # non-empty -- real registered analyses, not a stub
    # keyword is case/whitespace-insensitive, matching build_analysis_options's own contract
    assert resolve_modules("  Applicable  ", n_seeds=2, n_generations=3) == multi_gen


def test_main_accepts_n_generations_and_applicable_end_to_end(tmp_path, monkeypatch):
    """Full reproduction of item 60's real failure: the actual CLI invocation
    viva-api's simulation_service_ray.py emits
    (--out-uri ... --n-seeds N --n-generations N --modules applicable
    --analysis-name ...) must parse and resolve to a real, non-empty analysis
    run, not die on 'unrecognized arguments: --n-generations'."""
    import scripts.run_standalone_analysis as mod

    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {
        0: {"seed": 0, "dry_mass_fg": 200.0},
        1: {"seed": 1, "dry_mass_fg": 240.0},
    }, written)

    duckdb_calls: list[dict] = []

    def fake_run_analyses(*, sweep_dir, analysis_options, out_dir):
        duckdb_calls.append(analysis_options)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "analysis.json").write_text("{}")

    monkeypatch.setattr(mod, "_aws_sync", lambda src, dst: None)
    import v2ecoli.workflow.analysis_runner as analysis_runner_mod
    monkeypatch.setattr(analysis_runner_mod, "run_analyses", fake_run_analyses)

    monkeypatch.setattr(sys, "argv", [
        "run_standalone_analysis.py",
        "--out-uri", "s3://bucket/exp",
        "--n-seeds", "2",
        "--n-generations", "2",
        "--modules", "applicable",
        "--analysis-name", "test-analysis",
    ])

    mod.main()  # must not raise SystemExit from argparse or json.loads("applicable")

    assert len(duckdb_calls) >= 1  # at least one real DuckDB-family analysis actually ran
    assert any("multigeneration" in opts for opts in duckdb_calls)


def test_main_reads_config_file(tmp_path, monkeypatch, capsys):
    """The K8s job template writes args to a ConfigMap-mounted JSON file
    rather than embedding the modules JSON in a shell command string."""
    import scripts.run_standalone_analysis as mod

    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {0: {"seed": 0, "dry_mass_fg": 200.0}}, written)

    config_file = tmp_path / "params.json"
    config_file.write_text(json.dumps({
        "out_uri": "s3://bucket/exp", "n_seeds": 1,
        "modules": {"multiseed": {"doubling_time_distribution": {}}},
        "analysis_name": "test-analysis",
    }))
    monkeypatch.setattr(sys, "argv", ["run_standalone_analysis.py", "--config-file", str(config_file)])

    mod.main()

    assert "s3://bucket/exp/analyses/test-analysis/doubling_time_distribution.json" in written
    assert '"status": "done"' in capsys.readouterr().out


# --- V2ECOLI_EXTRA_ANALYSIS_MODULES hook -------------------------------------
# Downstream packages (e.g. sms-ecoli's sms_modules) register Analysis ports
# into the same ANALYSIS_REGISTRY as an import side effect; this script never
# imported them, so their names resolved to "unknown analysis". The env-var
# hook imports caller-named modules for that side effect, best-effort.

def test_extra_modules_unset_is_noop(monkeypatch):
    import scripts.run_standalone_analysis as mod
    monkeypatch.delenv("V2ECOLI_EXTRA_ANALYSIS_MODULES", raising=False)
    assert mod._register_extra_analysis_modules() == []


def test_extra_modules_bad_name_reported_not_fatal(monkeypatch):
    import scripts.run_standalone_analysis as mod
    monkeypatch.setenv("V2ECOLI_EXTRA_ANALYSIS_MODULES", "definitely_not_a_real_module_xyz")
    errs = mod._register_extra_analysis_modules()
    assert len(errs) == 1
    assert errs[0]["module"] == "definitely_not_a_real_module_xyz"
    assert "import failed" in errs[0]["error"]


def test_extra_modules_comma_and_space_separated(monkeypatch):
    """Both separators work; modules that import fine produce no error, so only
    the bogus one is reported (order preserved)."""
    import scripts.run_standalone_analysis as mod
    monkeypatch.setenv("V2ECOLI_EXTRA_ANALYSIS_MODULES", "json, os bogus_xyz")
    errs = mod._register_extra_analysis_modules()
    assert [e["module"] for e in errs] == ["bogus_xyz"]


def test_extra_modules_import_side_effect_runs(tmp_path, monkeypatch):
    """A named module is actually imported -- an Analysis port registers this
    way. Proven via an observable import side effect."""
    import sys as _sys
    import scripts.run_standalone_analysis as mod
    (tmp_path / "fake_extra_analyses.py").write_text(
        "import sys\nsys.modules.setdefault('_extra_import_marker', object())\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _sys.modules.pop("_extra_import_marker", None)
    _sys.modules.pop("fake_extra_analyses", None)
    monkeypatch.setenv("V2ECOLI_EXTRA_ANALYSIS_MODULES", "fake_extra_analyses")
    assert mod._register_extra_analysis_modules() == []
    assert "_extra_import_marker" in _sys.modules
