"""Standalone analysis resolves sim_data from the sweep's own provenance.

Running a sim first and analysing the sweep later must work like the all-at-once
study: the sweep records a resolvable sim_data pointer (run_provenance) and the
analysis reads it (analysis_runner.resolve_sim_data), so no manual
$V2ECOLI_SIM_DATA hunting is needed.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


def test_sim_data_ref_precedence(tmp_path, monkeypatch):
    from v2ecoli.library.run_provenance import sim_data_ref
    for v in ("V2ECOLI_SIM_DATA", "RAY_STAGE_S3", "CONTAINER_STAGE_S3"):
        monkeypatch.delenv(v, raising=False)

    # 1. explicit uri wins over everything
    assert sim_data_ref(cache_dir="/c", sim_data_uri="s3://b/k/simData.cPickle") == {
        "uri": "s3://b/k/simData.cPickle", "source": "explicit"}

    # 2. $V2ECOLI_SIM_DATA
    monkeypatch.setenv("V2ECOLI_SIM_DATA", "/env/simData.cPickle")
    assert sim_data_ref(cache_dir="/c") == {
        "uri": "/env/simData.cPickle", "source": "V2ECOLI_SIM_DATA"}
    monkeypatch.delenv("V2ECOLI_SIM_DATA")

    # 3. a remote dispatch's staged S3 cache
    monkeypatch.setenv("RAY_STAGE_S3", "s3://b/cache/")
    assert sim_data_ref(cache_dir="/c") == {
        "uri": "s3://b/cache/simData.cPickle", "source": "stage_s3"}
    monkeypatch.delenv("RAY_STAGE_S3")

    # 4. a local sim's cache_dir -- a real path a later local analysis can read
    sd = tmp_path / "simData.cPickle"
    sd.write_text("x")
    r = sim_data_ref(cache_dir=str(tmp_path))
    assert r == {"uri": str(sd), "source": "cache_dir", "exists": True}
    # missing local build is recorded as exists:False, not silently dropped
    assert sim_data_ref(cache_dir=str(tmp_path / "gone"))["exists"] is False
    # nothing to go on
    assert sim_data_ref()["uri"] is None


def test_build_run_identity_records_sim_data(monkeypatch):
    from v2ecoli.library.run_provenance import build_run_identity
    monkeypatch.setenv("V2ECOLI_SIM_DATA", "/env/simData.cPickle")
    rec = build_run_identity(cache_dir=None, design={"experiment_id": "x"})
    assert rec["sim_data"]["uri"] == "/env/simData.cPickle"
    assert "code" in rec and "cache_version" in rec  # unchanged fields


def test_resolve_reads_sim_data_from_run_identity(tmp_path):
    from v2ecoli.library.run_provenance import write_run_identity_record
    from v2ecoli.workflow.analysis_runner import _sim_data_uri_from_identity

    sd = tmp_path / "cache" / "simData.cPickle"
    sd.parent.mkdir()
    sd.write_text("x")
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    write_run_identity_record(str(sweep), {
        "code": {}, "cache_version": {}, "design": {},
        "sim_data": {"uri": str(sd), "source": "cache_dir", "exists": True}})

    assert _sim_data_uri_from_identity(str(sweep)) == str(sd)
    # a sweep with no sidecar -> None (falls through to the fallback/clear error)
    assert _sim_data_uri_from_identity(str(tmp_path / "no_sweep")) is None
    # a sidecar with no sim_data block -> None
    bare = tmp_path / "bare"
    bare.mkdir()
    write_run_identity_record(str(bare), {"code": {}, "design": {}})
    assert _sim_data_uri_from_identity(str(bare)) is None
