"""scripts/run_multi_node_analysis.py -- backlog item 88's "Analysis flush"
entrypoint for a completed multi-node process-bigraph composite dispatch
(e.g. a colony composite spread across N Ray-cluster nodes). Proves the
download -> run_flush() -> upload pipeline against the REAL run_flush
mechanism (not a mock of it -- the whole point of routing through it instead
of a hardcoded per-composite renderer), and the manifest contract
``GET /analyses/{id}/status`` actually reads (``written``/``errors``)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_multi_node_analysis import run  # noqa: E402


def _fake_aws_cp(monkeypatch: pytest.MonkeyPatch, downloads: dict[str, str], written: dict[str, str]) -> None:
    """Stub _aws_cp: 'download' (src is s3://) writes pre-seeded content to
    the requested local dest; 'upload' (dst is s3://) records dst -> local
    file content. Mirrors run_standalone_analysis.py's own identical pattern
    for the sibling entrypoint script."""
    import scripts.run_multi_node_analysis as mod

    def fake(src: str, dst: str) -> None:
        if src.startswith("s3://"):
            if src not in downloads:
                raise subprocess.CalledProcessError(1, ["aws"], stderr=b"NoSuchKey")
            Path(dst).write_text(downloads[src])
        else:
            written[dst] = Path(src).read_text()

    monkeypatch.setattr(mod, "_aws_cp", fake)


def test_run_writes_report_and_manifest_when_history_and_final_state_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real pipeline end-to-end, through the REAL run_flush (not mocked) and
    the REAL new EmitterHistorySummary step it discovers."""
    written: dict[str, str] = {}
    emitter_history = {"emitter": [[0.0, {"x": 1}], [60.0, {"x": 2}]]}
    _fake_aws_cp(
        monkeypatch,
        {
            "s3://bucket/exp/emitter_history.json": json.dumps(emitter_history),
            "s3://bucket/exp/final_state.json": json.dumps({"cells": {"a": {}}}),
        },
        written,
    )

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    assert manifest["errors"] == []
    # Containment, not equality: POST_SIM_REGISTRY is process-wide and additive
    # (same convention as workflow.analyses/report_cards) -- other Visualization
    # subclasses collected elsewhere in this same process (e.g. tests/
    # test_visualization_base.py's _DemoViz) legitimately also apply generically
    # to this out_dir and may co-appear here; this test only owns asserting its
    # OWN step's output landed.
    assert "s3://bucket/exp/analyses/test-analysis/emitter_history_summary.html" in manifest["written"]
    report_html = written["s3://bucket/exp/analyses/test-analysis/emitter_history_summary.html"]
    assert "<h2>Emitter history summary</h2>" in report_html
    assert "cells" in report_html  # a final_state.json top-level key, real content not a stub
    manifest_written = json.loads(written["s3://bucket/exp/analyses/test-analysis/_manifest.json"])
    assert manifest_written == manifest


def test_run_writes_report_from_final_state_only_when_no_emitter_history_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No emitter_history.json under history_uri (e.g. _persist_emitter_history's
    best-effort gather found nothing) is a legitimate, honest case -- must still
    produce a real report from final_state.json alone, not fail."""
    written: dict[str, str] = {}
    _fake_aws_cp(
        monkeypatch,
        {"s3://bucket/exp/final_state.json": json.dumps({"agents": {"a": {}, "b": {}}})},
        written,
    )

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    # Containment, not equality -- see the identical note in the test above.
    assert "s3://bucket/exp/analyses/test-analysis/emitter_history_summary.html" in manifest["written"]
    report_html = written["s3://bucket/exp/analyses/test-analysis/emitter_history_summary.html"]
    assert "No in-memory emitter history was captured" in report_html


def test_run_records_error_when_neither_history_nor_final_state_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {}, written)  # every download 404s

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert "neither emitter_history.json nor final_state.json" in manifest["errors"][0]["error"]
    # the manifest itself must still be uploaded even on a hard failure, so
    # GET /analyses/{id}/status has something authoritative to read.
    assert "s3://bucket/exp/analyses/test-analysis/_manifest.json" in written


def test_run_reports_failed_status_when_uploading_the_placed_output_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real, distinct failure mode from "nothing to analyze": run_flush
    succeeded and produced real output, but shipping it to S3 failed. Must be
    surfaced as a real error, not silently dropped."""
    import scripts.run_multi_node_analysis as mod

    downloads = {"s3://bucket/exp/final_state.json": json.dumps({"x": 1})}
    written: dict[str, str] = {}

    def fake(src: str, dst: str) -> None:
        # Every placed-output upload fails, regardless of how many steps
        # POST_SIM_REGISTRY happens to hold in this process (see the
        # containment-vs-equality note above) -- guarantees written == []
        # deterministically, which is what this test is actually about.
        # The manifest itself must still upload even on a hard failure, so
        # GET /analyses/{id}/status has something authoritative to read.
        if src.startswith("s3://"):
            if src not in downloads:
                raise subprocess.CalledProcessError(1, ["aws"], stderr=b"NoSuchKey")
            Path(dst).write_text(downloads[src])
        elif dst.endswith("_manifest.json"):
            written[dst] = Path(src).read_text()
        else:
            raise subprocess.CalledProcessError(1, ["aws"], stderr=b"AccessDenied: upload failed")

    monkeypatch.setattr(mod, "_aws_cp", fake)

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert any("upload failed" in e["error"] for e in manifest["errors"])


def test_run_degrades_when_run_flush_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A best-effort guard around the run_flush call itself: an environment
    where v2ecoli.workflow.flush can't be imported (or raises for any other
    reason) must be surfaced in the manifest, not crash the whole script."""
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {"s3://bucket/exp/final_state.json": json.dumps({"x": 1})}, written)
    monkeypatch.setitem(sys.modules, "v2ecoli.workflow.flush", None)  # forces ImportError

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert "run_flush failed" in manifest["errors"][0]["error"]


def test_run_routes_to_hive_parquet_path_when_n_seeds_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """item 109's own fix: a lineage_ray_batch-shaped dispatch (n_seeds given)
    must route to the SAME DuckDB-httpfs mechanism run_standalone_analysis.py
    already uses, reading history_uri as a hive-parquet sweep -- not attempt
    the flat-file download path at all (real GovCloud dispatches 282/283
    confirmed empirically that this shape never produces those flat files)."""
    import scripts.run_multi_node_analysis as mod

    calls: list[dict] = []

    def fake_run_analyses(*, sweep_dir: str, analysis_options: dict, out_dir: str) -> None:
        calls.append({"sweep_dir": sweep_dir, "analysis_options": analysis_options})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "analysis.json").write_text(json.dumps({"multiseed": {"doubling_time_distribution": {}}}))

    import v2ecoli.workflow.analysis_runner as analysis_runner_mod

    monkeypatch.setattr(analysis_runner_mod, "run_analyses", fake_run_analyses)

    import scripts.run_standalone_analysis as standalone_mod

    synced: list[tuple[str, str]] = []
    monkeypatch.setattr(standalone_mod, "_aws_sync", lambda src, dst: synced.append((src, dst)))

    # A real download of either flat file (the Path-2 fallback's own signature
    # move) is a test failure -- this dispatch shape must never even attempt
    # it once Path 1 succeeds. The manifest upload itself (a legitimate,
    # expected _aws_cp call in BOTH paths) must still be allowed through.
    flat_files = {"s3://bucket/exp/emitter_history.json", "s3://bucket/exp/final_state.json"}
    uploaded: dict[str, str] = {}

    def guard_aws_cp(src: str, dst: str) -> None:
        if src in flat_files:
            raise AssertionError(f"flat-file path attempted despite n_seeds given: {src} -> {dst}")
        uploaded[dst] = Path(src).read_text()

    monkeypatch.setattr(mod, "_aws_cp", guard_aws_cp)

    manifest = run(
        composite_id="v2ecoli.composites.lineage_ray_batch",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
        n_seeds=10,
        n_generations=10,
        modules={"multiseed": {"doubling_time_distribution": {}}},
    )

    assert manifest["status"] == "done"
    assert manifest["analysis_kind"] == "multi-node-composite-hive-parquet"
    assert manifest["written"] == ["s3://bucket/exp/analyses/test-analysis/analysis.json"]
    assert len(calls) == 1
    assert calls[0]["sweep_dir"] == "s3://bucket/exp"  # history_uri, read in place -- no download
    assert calls[0]["analysis_options"] == {"multiseed": {"doubling_time_distribution": {}}}
    assert len(synced) == 1
    assert synced[0][1] == "s3://bucket/exp/analyses/test-analysis"


def test_run_falls_through_to_flat_file_path_when_hive_parquet_path_finds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_seeds given but the DuckDB read produces nothing real (e.g. an empty
    modules resolution) -- must still fall through to the original flat-file
    path rather than report a hard failure, exactly like colony's own shape."""
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {"s3://bucket/exp/final_state.json": json.dumps({"x": 1})}, written)

    manifest = run(
        composite_id="v2ecoli.composites.lineage_ray_batch",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
        n_seeds=10,
        modules={},  # resolves to nothing -- Path 1 produces zero written entries
    )

    assert manifest["status"] == "done"
    assert manifest["analysis_kind"] == "multi-node-composite"  # Path 2's own manifest shape
    assert "s3://bucket/exp/analyses/test-analysis/emitter_history_summary.html" in manifest["written"]


def test_run_does_not_import_the_removed_colony_visualization_module(tmp_path: Path) -> None:
    """Regression guard for the exact bug this rewrite fixes: the old draft
    imported v2ecoli.visualizations.colony.ColonyVisualization directly (dead
    code since v2ecoli PR #414 dropped it from the composite -- it renders
    '?' placeholders, its `history` input is never supplied by any current
    composite). Nothing in the current module should reference it."""
    import inspect

    import scripts.run_multi_node_analysis as mod

    source = inspect.getsource(mod)
    assert "ColonyVisualization" not in source
    assert "run_flush" in source
