"""Task A2: the production cache loader must verify cache_version.json.

``v2ecoli.core.load_cache_bundle`` (and the ``lru_cache``-memoized
``_load_cache_bundle_cached`` it wraps) previously loaded
``out/cache``-style bundles without ever calling
``v2ecoli.library.cache_version.verify_cache_version`` — the only caller in
the repo was ``tests/conftest.py``'s ``sim_data_cache`` fixture. A cache
built from an older commit (different ParCa fixture, sim_data.py,
unit-bridge, or composite document shape) would load silently and
mis-calibrate a simulation instead of raising ``StaleCacheError``.

These tests build tiny, self-contained cache dirs (no real ParCa fixture
needed) and exercise the load path directly through ``load_cache_bundle``.
"""

import json

import dill
import pytest

from v2ecoli.library.cache_version import CacheVersion, write_cache_version


def _write_minimal_cache_dir(cache_dir, version=None):
    """Write a minimal but structurally valid cache bundle into ``cache_dir``.

    ``load_cache_bundle`` needs ``initial_state.json`` (loadable via
    ``v2ecoli.cache.load_initial_state``) and ``sim_data_cache.dill`` (a
    dict, walked by ``rebind_cache_quantities``); an empty dict/state is
    sufficient for both — this test only cares about the cache_version.json
    gate, not the payload contents.

    ``version=None`` writes the CURRENT fingerprint (a "fresh" cache) with a
    clean chassis chain (schema 3 requires a ``derived_from``); pass an
    explicit stale ``CacheVersion`` to simulate a mismatched cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "initial_state.json").write_text("{}")
    with open(cache_dir / "sim_data_cache.dill", "wb") as f:
        dill.dump({"configs": {}}, f)
    sources = None
    if version is None:
        chassis = cache_dir / "parca_state.pkl"
        chassis.write_bytes(b"fake-chassis")
        (cache_dir / "parca_state.provenance.json").write_text(json.dumps({
            "schema": "chassis-provenance/1", "layer": "chassis",
            "code": {"v2ecoli": {"commit": "abc123", "dirty": False,
                                 "source": "git"}},
        }))
        sources = [{"layer": "chassis", "path": str(chassis)}]
    write_cache_version(str(cache_dir), version=version, sources=sources)


def _stale_version():
    return CacheVersion(
        schema_version="0",
        inputs_hash="0" * 64,
        per_file_hashes={},
    )


@pytest.mark.fast
def test_load_cache_bundle_verifies(tmp_path):
    """A cache whose cache_version.json mismatches the current fingerprint
    raises StaleCacheError when loaded via load_cache_bundle.

    This is the regression test for A2: before wiring
    verify_cache_version into load_cache_bundle, this cache loaded
    silently instead of raising.
    """
    from v2ecoli.core import StaleCacheError, load_cache_bundle

    cache_dir = tmp_path / "stale_cache"
    _write_minimal_cache_dir(cache_dir, version=_stale_version())

    with pytest.raises(StaleCacheError):
        load_cache_bundle(str(cache_dir))


@pytest.mark.fast
def test_skip_cache_verify_env_bypasses(tmp_path, monkeypatch):
    """V2ECOLI_SKIP_CACHE_VERIFY=1 loads the same stale cache without
    raising — the escape hatch for deliberate cross-version work."""
    from v2ecoli.core import load_cache_bundle

    cache_dir = tmp_path / "stale_cache_skip"
    _write_minimal_cache_dir(cache_dir, version=_stale_version())

    monkeypatch.setenv("V2ECOLI_SKIP_CACHE_VERIFY", "1")
    bundle = load_cache_bundle(str(cache_dir))
    assert "initial_state" in bundle


@pytest.mark.fast
def test_fresh_cache_loads_clean(tmp_path):
    """A cache whose cache_version.json matches the current fingerprint
    loads without raising."""
    from v2ecoli.core import load_cache_bundle

    cache_dir = tmp_path / "fresh_cache"
    _write_minimal_cache_dir(cache_dir, version=None)

    bundle = load_cache_bundle(str(cache_dir))
    assert "initial_state" in bundle
