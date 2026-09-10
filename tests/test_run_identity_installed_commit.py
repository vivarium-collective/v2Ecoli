"""``run_identity.json`` must record a commit for INSTALLED deployments.

``build_run_identity`` used to call the git-only ``code_provenance`` directly,
so every deployed run (container: v2ecoli is an installed git dependency, no
``.git`` tree) wrote ``code.commit: null`` — "which code ran" could not be
recovered from the S3 artifact, even though pip had recorded the resolved
commit in ``direct_url.json`` (PEP 610) at install time. The fallback chain
(``_repo_code_provenance``) already existed for chassis provenance; run
identity now uses the same one.
"""
from __future__ import annotations

import json
from pathlib import Path

from v2ecoli.library.run_provenance import build_run_identity

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_git_checkout_still_records_git_commit():
    """The dev-checkout path is unchanged: git commit, source 'git'."""
    record = build_run_identity(repo_root=REPO_ROOT)
    code = record["code"]
    assert code["source"] == "git"
    assert code["commit"]  # this worktree is always gittable
    assert set(code) >= {"commit", "dirty", "diff_sha256", "untracked", "source"}


def test_installed_deployment_records_direct_url_commit(tmp_path, monkeypatch):
    """No ``.git`` at repo_root -> the installed package's resolved commit."""
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

    record = build_run_identity(repo_root=non_git)
    assert record["code"] == {
        "commit": "cafef00d" * 5, "dirty": None, "source": "direct_url"}


def test_total_failure_is_honest_null_with_reason(tmp_path, monkeypatch):
    """No ``.git`` AND no dist-info -> null with a reason, never a guess."""
    import importlib.metadata as metadata

    def _boom(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", _boom)
    non_git = tmp_path / "no_git_root"
    non_git.mkdir()

    record = build_run_identity(repo_root=non_git)
    code = record["code"]
    assert code["commit"] is None
    assert code["source"] is None
    assert "reason" in code and code["reason"]


def test_simulator_identity_recorded():
    """The record names the engine so a mixed-source sweep is attributable."""
    record = build_run_identity(repo_root=REPO_ROOT)
    assert record["simulator"]["id"] == "v2ecoli"
    # version is whatever the installed dist reports; None only when the
    # package metadata is unavailable (e.g. a raw source tree on sys.path).
    assert "version" in record["simulator"]
