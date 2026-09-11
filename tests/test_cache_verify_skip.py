"""``V2ECOLI_SKIP_CACHE_VERIFY`` must make ``verify_cache_version`` skip at the
single verification chokepoint, so EVERY caller honours the advertised bypass --
not just ``core.load_cache_bundle`` (which has its own pre-check). The staging
path (the ray-batch entrypoint's cache-verify, via run_pbg) calls
``verify_cache_version`` directly; before this the env var, documented as the
bypass, had no effect there, so a legitimately-copied cross-commit cache failed
at staging despite the flag being set.
"""
from __future__ import annotations

import pytest

from v2ecoli.library import cache_version
from v2ecoli.library.cache_version import StaleCacheError, verify_cache_version


def test_missing_cache_version_raises_without_flag(tmp_path, monkeypatch):
    """Baseline: a dir with no cache_version.json is stale and raises."""
    monkeypatch.delenv(cache_version.SKIP_CACHE_VERIFY_ENV, raising=False)
    with pytest.raises(StaleCacheError):
        verify_cache_version(str(tmp_path))


def test_flag_skips_verification(tmp_path, monkeypatch):
    """With the flag set, the SAME would-be-stale dir skips and returns None."""
    monkeypatch.setattr(cache_version, "_skip_verify_warned", False)
    monkeypatch.setenv(cache_version.SKIP_CACHE_VERIFY_ENV, "1")
    assert verify_cache_version(str(tmp_path)) is None


def test_flag_warns_once_per_process(tmp_path, monkeypatch):
    """The skip is never silent, but warns at most once per process."""
    monkeypatch.setattr(cache_version, "_skip_verify_warned", False)
    monkeypatch.setenv(cache_version.SKIP_CACHE_VERIFY_ENV, "1")
    with pytest.warns(UserWarning, match=cache_version.SKIP_CACHE_VERIFY_ENV):
        verify_cache_version(str(tmp_path))
    # second call: still skips, but does not warn again
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning would raise
        assert verify_cache_version(str(tmp_path)) is None
