"""derived_from provenance chain in cache_version.json (schema 3).

A schema-3 cache records the artifacts it was DERIVED FROM — founder →
sim_data → chassis — hashing each source's bytes, embedding its
``*.provenance.json`` sidecar verbatim, and nesting a source bundle's own
``cache_version.json`` under ``parent_cache_version``. A stable projection
``[{layer,source_sha256,commit,dirty}]`` is folded into ``inputs_hash`` so
swapping the chassis a cache was built on changes the fingerprint.

Hermetic: no ParCa build, no S3. Projection tests use a one-file INPUT set
against the source root so ``compute_cache_version`` doesn't re-hash the 38 MB
ParCa fixture on every call; verify tests build a genuine full-fingerprint
cache once so build-time and verify-time hashes match.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import v2ecoli
from v2ecoli.library.cache_version import (
    CacheVersion,
    StaleCacheError,
    _audit_main,
    compute_cache_version,
    verify_cache_version,
    write_cache_version,
)

REPO_ROOT = str(Path(v2ecoli.__file__).resolve().parents[1])
# One small, always-present INPUT_FILES entry — keeps projection tests fast.
FAST_FILES = ("v2ecoli/library/cache_version.py",)


def _fast_cv(**kw):
    return compute_cache_version(repo_root=REPO_ROOT, files=FAST_FILES, **kw)


def _entry(commit="c1", dirty=False, sha="a" * 64, layer="chassis",
           created_at="t1"):
    return {
        "layer": layer,
        "path": f"/x/{layer}.pkl",
        "source_sha256": sha,
        "provenance": {
            "schema": "chassis-provenance/1",
            "layer": layer,
            "code": {"v2ecoli": {"commit": commit, "dirty": dirty,
                                 "source": "git"}},
            "created_at": created_at,
        },
    }


# --------------------------------------------------------------------------
# compute_cache_version(sources=...) — resolution + nesting
# --------------------------------------------------------------------------

def test_sources_embeds_sidecar_and_hashes_bytes(tmp_path):
    """A file source records the sha256 of its exact bytes and embeds a
    ``*.provenance.json`` sidecar (stem-swap name) verbatim under provenance."""
    import hashlib

    from v2ecoli.library.run_provenance import write_chassis_provenance

    pkl = tmp_path / "parca_state.pkl"
    pkl.write_bytes(b"chassis-bytes")
    write_chassis_provenance(pkl, build={"mode": "full"}, repo_root=REPO_ROOT,
                             workspace_root=None)

    cv = _fast_cv(sources=[{"layer": "chassis", "path": str(pkl)}])
    assert len(cv.derived_from) == 1
    entry = cv.derived_from[0]
    assert entry["layer"] == "chassis"
    assert entry["source_sha256"] == hashlib.sha256(b"chassis-bytes").hexdigest()
    assert entry["provenance"]["schema"] == "chassis-provenance/1"
    assert entry["provenance"]["build"] == {"mode": "full"}


def test_sources_missing_sidecar_is_honest_null(tmp_path):
    pkl = tmp_path / "parca_state.pkl"
    pkl.write_bytes(b"no-sidecar")
    cv = _fast_cv(sources=[{"layer": "chassis", "path": str(pkl)}])
    prov = cv.derived_from[0]["provenance"]
    assert prov["available"] is False
    assert "no provenance sidecar" in prov["reason"]


def test_bundle_dir_source_nests_parent_cache_version(tmp_path):
    """A source that is itself a bundle dir embeds that bundle's
    ``cache_version.json`` under ``parent_cache_version`` and uses its
    inputs_hash as the source identity, so the chain nests."""
    bundle = tmp_path / "sim_data_bundle"
    parent = CacheVersion(schema_version="3", inputs_hash="parentHASH123",
                          per_file_hashes={}, derived_from=[_entry()])
    write_cache_version(str(bundle), version=parent)

    cv = _fast_cv(sources=[{"layer": "sim_data", "path": str(bundle)}])
    entry = cv.derived_from[0]
    assert entry["parent_cache_version"]["inputs_hash"] == "parentHASH123"
    assert entry["source_sha256"] == "parentHASH123"


# --------------------------------------------------------------------------
# The core regression: the chain is folded into inputs_hash.
# --------------------------------------------------------------------------

def test_source_bytes_change_inputs_hash(tmp_path):
    a = tmp_path / "a.pkl"
    a.write_bytes(b"chassis-A")
    b = tmp_path / "b.pkl"
    b.write_bytes(b"chassis-B-different")
    cv_a = _fast_cv(sources=[{"layer": "chassis", "path": str(a)}])
    cv_b = _fast_cv(sources=[{"layer": "chassis", "path": str(b)}])
    assert cv_a.inputs_hash != cv_b.inputs_hash


def test_chassis_commit_changes_inputs_hash_unrelated_field_does_not():
    """The fold projection is [{layer,source_sha256,commit,dirty}]: the chassis
    commit and source_sha256 move inputs_hash; an unrelated sidecar field
    (created_at) does not."""
    base = _fast_cv(derived_from=[_entry(commit="c1", created_at="t1")])

    # Unrelated field (created_at) — inputs_hash unchanged.
    unrelated = _fast_cv(derived_from=[_entry(commit="c1", created_at="t2")])
    assert unrelated.inputs_hash == base.inputs_hash

    # Chassis commit changed — inputs_hash moves.
    diff_commit = _fast_cv(derived_from=[_entry(commit="c2", created_at="t1")])
    assert diff_commit.inputs_hash != base.inputs_hash

    # source_sha256 changed — inputs_hash moves.
    diff_sha = _fast_cv(derived_from=[_entry(commit="c1", sha="b" * 64)])
    assert diff_sha.inputs_hash != base.inputs_hash

    # dirty flag changed — inputs_hash moves.
    diff_dirty = _fast_cv(derived_from=[_entry(commit="c1", dirty=True)])
    assert diff_dirty.inputs_hash != base.inputs_hash


def test_to_from_dict_roundtrip_preserves_derived_from():
    cv = _fast_cv(derived_from=[_entry()])
    assert CacheVersion.from_dict(cv.to_dict()) == cv


# --------------------------------------------------------------------------
# verify_cache_version guards
# --------------------------------------------------------------------------

def _write_stored(cache_dir, derived_from, inputs_hash="deadbeef", schema="3"):
    """Hand-write a cache_version.json for the guards that fire BEFORE the
    inputs_hash check (they don't need a matching fingerprint)."""
    version = CacheVersion(schema_version=schema, inputs_hash=inputs_hash,
                           per_file_hashes={}, derived_from=derived_from)
    write_cache_version(str(cache_dir), version=version)


def test_missing_chain_on_schema3_raises(tmp_path):
    _write_stored(tmp_path, derived_from=[])
    with pytest.raises(StaleCacheError, match="derived_from"):
        verify_cache_version(str(tmp_path), repo_root=REPO_ROOT)


def test_dirty_chassis_warns_by_default_and_raises_when_required(tmp_path):
    _write_stored(tmp_path, derived_from=[_entry(commit="c1", dirty=True)])

    with pytest.warns(UserWarning, match="untrustworthy"):
        # Fires the dirty warning, then raises on the inputs_hash mismatch
        # (hand-written stored hash != recompute) — the warning is what we
        # assert here.
        with pytest.raises(StaleCacheError):
            verify_cache_version(str(tmp_path), repo_root=REPO_ROOT)

    with pytest.raises(StaleCacheError, match="REFUSING"):
        verify_cache_version(str(tmp_path), repo_root=REPO_ROOT,
                             require_clean_chain=True)


def test_dirty_chassis_env_var_forces_hard_fail(tmp_path, monkeypatch):
    _write_stored(tmp_path, derived_from=[_entry(commit="c1", dirty=True)])
    monkeypatch.setenv("V2E_REQUIRE_CLEAN_CHAIN", "1")
    with pytest.raises(StaleCacheError, match="REFUSING"):
        verify_cache_version(str(tmp_path), repo_root=REPO_ROOT)


def test_null_commit_layer_is_flagged(tmp_path):
    _write_stored(tmp_path, derived_from=[_entry(commit=None, dirty=False)])
    with pytest.raises(StaleCacheError, match="REFUSING"):
        verify_cache_version(str(tmp_path), repo_root=REPO_ROOT,
                             require_clean_chain=True)


def test_expected_chassis_mismatch_raises(tmp_path):
    _write_stored(tmp_path, derived_from=[_entry(commit="c1")])
    with pytest.raises(StaleCacheError, match="chassis commit mismatch"):
        verify_cache_version(str(tmp_path), repo_root=REPO_ROOT,
                             expected_chassis={"commit": "c2"})


def test_clean_chain_passes(tmp_path):
    """A clean chain (committed chassis, dirty=False) with a genuine
    fingerprint verifies silently, including under require_clean_chain and a
    matching expected_chassis."""
    pkl = tmp_path / "parca_state.pkl"
    pkl.write_bytes(b"clean-chassis")
    # Controlled sidecar: committed, not dirty.
    (tmp_path / "parca_state.provenance.json").write_text(json.dumps({
        "schema": "chassis-provenance/1", "layer": "chassis",
        "code": {"v2ecoli": {"commit": "abc123", "dirty": False,
                             "source": "git"}},
    }))

    cache_dir = tmp_path / "cache"
    cv = compute_cache_version(
        repo_root=REPO_ROOT,
        sources=[{"layer": "chassis", "path": str(pkl)}])
    write_cache_version(str(cache_dir), version=cv)

    # No raise, no warning.
    verify_cache_version(str(cache_dir), repo_root=REPO_ROOT)
    verify_cache_version(str(cache_dir), repo_root=REPO_ROOT,
                         require_clean_chain=True)
    verify_cache_version(str(cache_dir), repo_root=REPO_ROOT,
                         expected_chassis={"commit": "abc123"})


# --------------------------------------------------------------------------
# schema-2 (pre-chain) — loud fail + audit CLI PRE-CHAIN message
# --------------------------------------------------------------------------

def test_schema2_cache_fails_on_schema_check(tmp_path):
    """A pre-chain schema-2 cache trips the existing schema_version bust."""
    version = CacheVersion(schema_version="2", inputs_hash="x",
                           per_file_hashes={})
    write_cache_version(str(tmp_path), version=version)
    with pytest.raises(StaleCacheError, match="schema_version mismatch"):
        verify_cache_version(str(tmp_path), repo_root=REPO_ROOT)


def test_audit_cli_prechain_message_for_schema2(tmp_path, capsys):
    version = CacheVersion(schema_version="2", inputs_hash="x" * 64,
                           per_file_hashes={})
    write_cache_version(str(tmp_path), version=version)

    rc = _audit_main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRE-CHAIN cache (schema 2)" in out
    assert "claim, not a fact" in out


def test_audit_cli_prints_chain_for_schema3(tmp_path, capsys):
    version = CacheVersion(schema_version="3", inputs_hash="y" * 64,
                           per_file_hashes={},
                           derived_from=[_entry(commit="abc123def456")])
    write_cache_version(str(tmp_path), version=version)

    rc = _audit_main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "derived_from chain" in out
    assert "layer='chassis'" in out
    assert "abc123def456"[:12] in out
