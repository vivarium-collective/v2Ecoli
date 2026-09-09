"""The gather's resource knobs (sim 683: five multiseed modules OOM'd together on a
32 GB task) ride in the analyze config's ``runner`` block and reach DuckDB."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


class _Conn:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, q: str) -> None:
        self.sql.append(q)


def test_apply_config_sets_threads_and_temp_cap_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from v2ecoli.library import sweep_io

    monkeypatch.setattr(sweep_io, "analysis_memory_limit", lambda: "20GB")
    c = _Conn()
    sweep_io.apply_analysis_duckdb_config(c)
    assert c.sql == ["SET memory_limit = '20GB'"]
    c = _Conn()
    sweep_io.apply_analysis_duckdb_config(c, threads=2, max_temp_directory_size="150GB")
    assert c.sql == ["SET memory_limit = '20GB'", "SET threads = 2", "SET max_temp_directory_size = '150GB'"]


def test_temp_dir_precedence_env_then_preferred_then_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v2ecoli.library.sweep_io import analysis_temp_dir

    monkeypatch.delenv("V2E_DUCKDB_TEMP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert analysis_temp_dir("duckdb_tmp") == "duckdb_tmp" and (tmp_path / "duckdb_tmp").is_dir()
    monkeypatch.setenv("V2E_DUCKDB_TEMP_DIR", str(tmp_path / "env_tmp"))
    assert analysis_temp_dir("duckdb_tmp") == str(tmp_path / "env_tmp") and (tmp_path / "env_tmp").is_dir()
    monkeypatch.delenv("V2E_DUCKDB_TEMP_DIR")
    assert os.path.isdir(analysis_temp_dir(None))


def test_cli_threads_the_runner_block_through_to_run_analyses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v2ecoli.workflow import analysis_runner as ar

    cfg = tmp_path / "analysis.config.json"
    cfg.write_text(json.dumps({
        "analysis_options": {"single": {"mass_fraction_summary": {}}},
        "out_dir": "analysis_v3",
        "runner": {"max_workers": 1, "duckdb": {"threads": 2, "temp_dir": "duckdb_tmp"}},
    }))
    seen: dict = {}

    def fake_run(sweep_dir, analysis_options, out_dir=None, max_workers=None, duckdb=None, **kw):
        seen.update(sweep_dir=sweep_dir, out_dir=out_dir, max_workers=max_workers, duckdb=duckdb)
        return {}

    monkeypatch.setattr(ar, "run_analyses", fake_run)
    monkeypatch.setattr("sys.argv", ["v2ecoli-analyze", str(tmp_path), "--config", str(cfg)])
    ar.main()
    assert seen == {"sweep_dir": str(tmp_path), "out_dir": "analysis_v3", "max_workers": 1,
                    "duckdb": {"threads": 2, "temp_dir": "duckdb_tmp"}}


def test_cli_without_a_runner_block_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v2ecoli.workflow import analysis_runner as ar

    cfg = tmp_path / "c.json"; cfg.write_text(json.dumps({"analysis_options": {"single": {"x": {}}}}))
    seen: dict = {}
    monkeypatch.setattr(ar, "run_analyses", lambda *a, **k: seen.update(k) or {})
    monkeypatch.setattr("sys.argv", ["v2ecoli-analyze", str(tmp_path), "--config", str(cfg)])
    ar.main()
    assert seen == {"out_dir": None, "max_workers": None, "duckdb": None}
