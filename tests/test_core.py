"""Unit tests for v2ecoli.core."""

import os

import pytest


@pytest.mark.fast
def test_build_core_returns_core_with_ecoli_types():
    from v2ecoli.core import build_core
    from v2ecoli.types import ECOLI_TYPES
    core = build_core()
    # Spot-check that ECOLI_TYPES is registered — pick a known custom type
    # name from ECOLI_TYPES.
    sample_type = next(iter(ECOLI_TYPES))
    assert sample_type in core.registry


@pytest.mark.fast
def test_build_core_registers_ray_lineage_without_touching_ray():
    """item 101/109 N3: build_core() must make a raw ray:LineageProcess address
    resolvable (for /compose/v1/run-document, which has no --composite-id and so
    never runs apply_core_extensions) WITHOUT eagerly calling ray.init() -- that
    would break every other caller of build_core (chain-dispatch, local scripts,
    tests). Confirmed directly: RayProtocolRuntime.__init__ calls ray.init()
    whenever Ray isn't already running (process_bigraph/protocols/ray.py:528-536),
    so build_core must register the process class WITHOUT calling
    prewarm_lineage_pool / get_or_create_runtime."""
    import ray
    from process_bigraph.protocols import ray as ray_protocol
    from v2ecoli.core import build_core
    from v2ecoli.workflow.lineage import LineageProcess

    build_core()
    assert ray_protocol.get_registry()["LineageProcess"] is LineageProcess
    assert not ray.is_initialized()


@pytest.mark.sim
def test_load_cache_bundle_returns_expected_keys(tmp_path):
    """If the cache exists at out/cache, load_cache_bundle returns a dict
    with the expected keys. Marked ``sim`` because it depends on the
    ParCa cache being materialized — locally via ``python scripts/build_cache.py``
    and in CI by the behavior-tests job (the fast-tests job does NOT build
    the cache)."""
    from v2ecoli.core import load_cache_bundle
    cache_dir = "out/cache"
    if not os.path.isdir(cache_dir) and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    bundle = load_cache_bundle(cache_dir)
    assert isinstance(bundle, dict), f"expected dict, got {type(bundle).__name__}"
    assert "initial_state" in bundle, f"expected 'initial_state' key, got {sorted(bundle)}"
    # The dill cache should contribute at least one of these keys.
    assert any(k in bundle for k in ("configs", "unique_names", "dry_mass_inc_dict")), \
        f"expected at least one cache key in bundle, got {sorted(bundle)}"


@pytest.mark.fast
def test_save_cache_writes_files(tmp_path, monkeypatch):
    """save_cache writes a sim_data_cache.dill + cache_version.json
    inside the target dir. Use the committed ParCa fixture as the source.

    The parca_state.pkl.gz is a gzipped parca state dict, not a plain
    simData pickle. We hydrate it into a plain dill file first, which is
    what save_cache (and LoadSimData) expect.
    """
    import dill
    from v2ecoli.core import save_cache
    fixture_path = "models/parca/parca_state.pkl.gz"
    if not os.path.exists(fixture_path):
        pytest.skip("ParCa fixture missing")
    # Hydrate the gzipped parca state into a plain simData pickle
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state,
        load_parca_state,
    )
    state = load_parca_state(fixture_path)
    sim_data = hydrate_sim_data_from_state(state)
    sd_path = str(tmp_path / "simData.cPickle")
    with open(sd_path, "wb") as f:
        dill.dump(sim_data, f)
    target = tmp_path / "cache"
    save_cache(sim_data_path=sd_path, cache_dir=str(target), seed=0)
    assert (target / "sim_data_cache.dill").exists()
    assert (target / "cache_version.json").exists()
