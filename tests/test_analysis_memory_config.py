"""DuckDB analysis memory budget: the container-OOM fix.

vEcoli's create_duckdb_conn spills to a temp_directory but sets no explicit
memory_limit, so DuckDB budgets ~80% of DETECTED (host) RAM. In a container that
reads the host's RAM, not the cgroup limit -- a full-hive ptools ORDER BY then
overshoots and dies ("failed to pin block ... NGiB/NGiB") instead of spilling.
These cover the explicit-budget helpers and the memory-aware worker cap.
"""
from __future__ import annotations

import pytest

from v2ecoli.library.sweep_io import (
    _parse_size,
    analysis_memory_budget_bytes,
    analysis_memory_limit,
    apply_analysis_duckdb_config,
)

pytestmark = pytest.mark.fast


def test_parse_size_units():
    assert _parse_size("8GB") == 8 * 10**9
    assert _parse_size("512MB") == 512 * 10**6
    assert _parse_size("4GiB") == 4 * 2**30
    assert _parse_size("1024") == 1024  # bare bytes
    assert _parse_size("nonsense") is None
    assert _parse_size("") is None


def test_env_memory_limit_takes_precedence(monkeypatch):
    monkeypatch.setenv("V2E_ANALYSIS_MEMORY_LIMIT", "6GB")
    assert analysis_memory_limit() == "6GB"
    assert analysis_memory_budget_bytes() == 6 * 10**9


def test_no_env_no_cgroup_is_none(monkeypatch):
    """Unknown budget (no env, no cgroup) leaves DuckDB on its default."""
    monkeypatch.delenv("V2E_ANALYSIS_MEMORY_LIMIT", raising=False)
    from v2ecoli.library import sweep_io
    monkeypatch.setattr(sweep_io, "_container_memory_limit_bytes", lambda: None)
    assert analysis_memory_limit() is None
    assert analysis_memory_budget_bytes() is None


def test_cgroup_limit_used_when_present(monkeypatch):
    monkeypatch.delenv("V2E_ANALYSIS_MEMORY_LIMIT", raising=False)
    from v2ecoli.library import sweep_io
    monkeypatch.setattr(sweep_io, "_container_memory_limit_bytes", lambda: 10 * 2**30)
    assert analysis_memory_limit() == f"{int(10 * 2**30 * 0.7) // (1000 * 1000)}MB"
    assert analysis_memory_budget_bytes() == int(10 * 2**30 * 0.7)


def test_apply_sets_memory_limit_on_a_real_connection(monkeypatch):
    import duckdb
    monkeypatch.setenv("V2E_ANALYSIS_MEMORY_LIMIT", "512MB")
    conn = duckdb.connect()
    apply_analysis_duckdb_config(conn)
    val = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    num, unit = val.split()
    # DuckDB normalizes 512MB (512e6 B) to ~488.3 MiB.
    assert unit == "MiB" and 480 <= float(num) <= 520


def test_apply_is_noop_when_budget_unknown(monkeypatch):
    import duckdb
    monkeypatch.delenv("V2E_ANALYSIS_MEMORY_LIMIT", raising=False)
    from v2ecoli.library import sweep_io
    monkeypatch.setattr(sweep_io, "_container_memory_limit_bytes", lambda: None)
    conn = duckdb.connect()
    before = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    apply_analysis_duckdb_config(conn)  # must not raise
    after = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert before == after


# --- worker cap (imports analysis_runner, which needs a current viva_superpowers
# to load; CI verifies these) -------------------------------------------------

def test_default_workers_env_override(monkeypatch):
    monkeypatch.setenv("V2E_ANALYSIS_MAX_WORKERS", "2")
    from v2ecoli.workflow.analysis_runner import _default_analysis_workers
    assert _default_analysis_workers() == 2


def test_default_workers_capped_by_memory_budget(monkeypatch):
    monkeypatch.delenv("V2E_ANALYSIS_MAX_WORKERS", raising=False)
    monkeypatch.setenv("V2E_ANALYSIS_MEMORY_LIMIT", "6GB")  # ~1 worker per 4GiB
    from v2ecoli.workflow.analysis_runner import _default_analysis_workers
    assert _default_analysis_workers() == 1
