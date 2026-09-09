"""analysis.json carries a per-module `runtime` block (wall time, peak RSS, DuckDB
memory/temp after the query) so gather memory can be compared before/after
changes with numbers. Results and summaries are untouched."""

from __future__ import annotations

import time

import duckdb


def test_snapshot_reports_time_rss_and_duckdb_memory() -> None:
    from v2ecoli.workflow.analysis_runner import _runtime_snapshot

    conn = duckdb.connect()
    t0 = time.perf_counter()
    conn.execute("CREATE TABLE t AS SELECT range AS i, random() AS x FROM range(200000)")
    snap = _runtime_snapshot(conn.cursor(), t0)
    assert snap["elapsed_s"] >= 0 and snap["process_peak_rss_mb"] > 10
    assert "duckdb_memory_mb_after" in snap and snap["duckdb_memory_mb_after"] >= 0
    assert "duckdb_temp_mb_after" in snap and "duckdb_memory_error" not in snap


def test_snapshot_never_raises_on_a_broken_cursor() -> None:
    from v2ecoli.workflow.analysis_runner import _runtime_snapshot

    class Broken:
        def execute(self, q):
            raise RuntimeError("no duckdb here")

    snap = _runtime_snapshot(Broken(), time.perf_counter())
    assert snap["elapsed_s"] >= 0 and "duckdb_memory_error" in snap
