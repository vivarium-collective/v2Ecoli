"""run_mbp_tracked's parquet output root is overridable via V2E_STUDIES_ROOT.

A remote multi-node entrypoint syncs exactly one directory to S3 (sms-api's Ray
entrypoint uploads {V2ECOLI_DIR}/.pbg/runs/phase0-xarray and nothing else). A run
whose parquet lands under REPO_ROOT/studies/ is therefore computed but never
uploaded — dispatch 322 (reactor_bird_coupled) ran for hours and nothing reached
S3 for exactly this reason. V2E_STUDIES_ROOT lets the entrypoint point the output
under the synced directory; unset, it keeps REPO_ROOT/studies so the local
dashboard still discovers runs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_mbp_tracked.py"


def _load():
    spec = importlib.util.spec_from_file_location("_mbp_tracked_studies_root", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_studies_root_is_repo_studies(monkeypatch):
    monkeypatch.delenv("V2E_STUDIES_ROOT", raising=False)
    mod = _load()
    assert mod.STUDIES_ROOT == mod.REPO_ROOT / "studies"
    assert mod._parquet_root_for("mbp-03") == mod.REPO_ROOT / "studies" / "mbp-03" / "parquet-runs"


def test_env_override_redirects_output_root(tmp_path, monkeypatch):
    synced = tmp_path / ".pbg" / "runs" / "phase0-xarray" / "studies"
    monkeypatch.setenv("V2E_STUDIES_ROOT", str(synced))
    mod = _load()
    assert mod.STUDIES_ROOT == synced
    # the per-study parquet root (what run_multigen_parquet writes into) now lands
    # under the synced tree, so the entrypoint's single upload path covers it.
    assert mod._parquet_root_for("mbp-03") == synced / "mbp-03" / "parquet-runs"


def test_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("V2E_STUDIES_ROOT", "")
    mod = _load()
    assert mod.STUDIES_ROOT == mod.REPO_ROOT / "studies"
