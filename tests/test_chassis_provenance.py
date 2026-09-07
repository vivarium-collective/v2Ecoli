"""Chassis provenance sidecar (PR 1 of the chassis-provenance chain).

``chassis_provenance`` / ``write_chassis_provenance`` pin a ParCa
``parca_state.pkl`` chassis: its exact bytes, the code that produced it
(git, or the installed-dependency ``direct_url.json`` fallback, or an honest
null), and the caller-supplied build inputs. The sidecar lands beside the
pickle as ``parca_state.provenance.json`` and is embedded verbatim into a
downstream cache's ``derived_from`` chain.

Hermetic: no ParCa build. A fake pickle stands in for ``parca_state.pkl``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import v2ecoli
from v2ecoli.library.run_provenance import (
    CHASSIS_PROVENANCE_SCHEMA,
    chassis_provenance,
    chassis_provenance_path,
    read_chassis_provenance,
    write_chassis_provenance,
)

REPO_ROOT = Path(v2ecoli.__file__).resolve().parents[1]


def _fake_pkl(tmp_path: Path, payload: bytes = b"fake parca state") -> Path:
    pkl = tmp_path / "parca_state.pkl"
    pkl.write_bytes(payload)
    return pkl


def test_sidecar_path_is_stem_swap():
    """parca_state.pkl -> parca_state.provenance.json (beside the pickle)."""
    p = chassis_provenance_path("/a/b/parca_state.pkl")
    assert p == Path("/a/b/parca_state.provenance.json")


def test_write_chassis_provenance_git_path_shape_and_artifact(tmp_path):
    """The git path: artifact sha256/bytes are exact, the v2ecoli code block
    is git-sourced with a real commit, and the build dict + schema/layer/
    created_at fields are all present."""
    payload = b"chassis-bytes-\x00\x01\x02"
    pkl = _fake_pkl(tmp_path, payload)
    build = {"mode": "full", "new_genes": "off", "argv": ["v2ecoli-parca"]}

    record = write_chassis_provenance(
        pkl, build=build, repo_root=REPO_ROOT, workspace_root=None)

    # Sidecar written at the stem-swap path, and round-trips.
    sidecar = tmp_path / "parca_state.provenance.json"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text()) == record
    assert read_chassis_provenance(pkl) == record

    assert record["schema"] == CHASSIS_PROVENANCE_SCHEMA
    assert record["layer"] == "chassis"
    assert record["artifact"] == {
        "file": "parca_state.pkl",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    assert record["build"] == build
    assert record["created_at"].endswith("Z")

    v2 = record["code"]["v2ecoli"]
    assert v2["source"] == "git"
    assert v2["commit"]  # this worktree is always gittable
    assert set(v2) >= {"commit", "dirty", "diff_sha256", "untracked", "source"}


def test_direct_url_fallback_when_no_git(tmp_path, monkeypatch):
    """A repo with no ``.git`` falls back to the installed package's
    ``direct_url.json`` vcs_info.commit_id (PEP 610)."""
    import importlib.metadata as metadata

    class _FakeDist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({
                "url": "https://github.com/vivarium-collective/v2ecoli.git",
                "vcs_info": {"vcs": "git", "commit_id": "cafef00d" * 5},
            })

    monkeypatch.setattr(metadata, "distribution", lambda name: _FakeDist())

    non_git = tmp_path / "no_git_root"
    non_git.mkdir()
    pkl = _fake_pkl(tmp_path)

    record = chassis_provenance(pkl, repo_root=non_git, workspace_root=None)
    assert record["code"]["v2ecoli"] == {
        "commit": "cafef00d" * 5, "dirty": None, "source": "direct_url"}


def test_total_failure_null_when_no_git_and_no_dist(tmp_path, monkeypatch):
    """No ``.git`` AND no resolvable dist-info -> honest null with a reason,
    mirroring code_provenance's convention (never a guessed value)."""
    import importlib.metadata as metadata

    def _boom(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", _boom)

    non_git = tmp_path / "no_git_root"
    non_git.mkdir()
    pkl = _fake_pkl(tmp_path)

    record = chassis_provenance(pkl, repo_root=non_git, workspace_root=None)
    v2 = record["code"]["v2ecoli"]
    assert v2["commit"] is None
    assert v2["source"] is None
    assert "reason" in v2 and v2["reason"]


def test_workspace_block_honest_null_without_a_workspace(tmp_path):
    """No workspace root resolved -> a null workspace block, not a crash."""
    pkl = _fake_pkl(tmp_path)
    record = chassis_provenance(pkl, repo_root=REPO_ROOT, workspace_root=None)
    ws = record["code"]["workspace"]
    assert ws["repo"] is None
    assert ws["commit"] is None
    assert ws["source"] is None


def test_read_chassis_provenance_accepts_append_form(tmp_path):
    """The reader also finds a literal-append ``<path>.provenance.json``."""
    pkl = _fake_pkl(tmp_path)
    record = {"schema": CHASSIS_PROVENANCE_SCHEMA, "layer": "chassis"}
    (tmp_path / "parca_state.pkl.provenance.json").write_text(
        json.dumps(record))
    assert read_chassis_provenance(pkl) == record


def test_read_chassis_provenance_none_when_absent(tmp_path):
    assert read_chassis_provenance(_fake_pkl(tmp_path)) is None
