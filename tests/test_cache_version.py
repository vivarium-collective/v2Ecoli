"""Guards that the ParCa-derived cache stays loadable under the current code.

Two classes of regression we specifically want to catch here, both of which
slipped past the existing test suite during the unum→pint migration:

1. Unit-type boundary: every ``Quantity`` field in ``sim_data_cache.dill``
   should be a pint Quantity.  If a LoadSimData config-getter ever emits
   a raw ``Unum`` object, downstream ``.to(...)`` calls explode three
   frames deep.  This test asserts the types directly on the cache, so
   the failure mode is ``assert isinstance(x, pint.Quantity)`` instead
   of ``AttributeError: 'Unum' object has no attribute 'to'``.

2. Cache-version guard: ``cache_version.json`` must exist and match the
   current inputs hash.  If someone removes ``write_cache_version`` from
   ``save_cache`` or the verification call from ``build_composite``, the
   stale-cache footgun returns.  We test the guard both ways: a fresh
   cache passes, a tampered fixture field fails with ``StaleCacheError``.
"""
from __future__ import annotations

import json
import os
import shutil

import dill
import pint
import pytest

# Side-effect import: registers `nucleotide` / `amino_acid` / `count` on the
# shared pint registry so dill.load of the cache can resolve those names.
import v2ecoli.library.unit_bridge  # noqa: F401
from v2ecoli.library.cache_version import (
    CACHE_VERSION_FILENAME,
    StaleCacheError,
    compute_cache_version,
    read_cache_version,
    verify_cache_version,
)


pytestmark = pytest.mark.sim  # needs sim_data_cache fixture


def test_mass_listener_config_doubling_times_are_pint(sim_data_cache):
    """The specific field that broke in the unum→pint migration.

    Regression target: v2ecoli/steps/listeners/mass_listener.py:203 calls
    ``.to(units.s)`` on condition_to_doubling_time[condition]. Under a
    stale cache this was a Unum object → AttributeError deep in sim build.
    """
    with open(os.path.join(sim_data_cache, 'sim_data_cache.dill'), 'rb') as f:
        cache = dill.load(f)

    configs = cache.get('configs', {})
    mass_cfg = configs.get('ecoli-mass-listener')
    assert mass_cfg is not None, (
        'cache has no ecoli-mass-listener config — rebuild with '
        'scripts/build_cache.py'
    )
    ctdt = mass_cfg['condition_to_doubling_time']
    assert ctdt, 'condition_to_doubling_time is empty'
    for condition, value in ctdt.items():
        assert isinstance(value, pint.Quantity), (
            f'condition_to_doubling_time[{condition!r}] is '
            f'{type(value).__name__}, expected pint.Quantity. '
            f'A LoadSimData config-getter failed to call unum_to_pint.'
        )


# ---------------------------------------------------------------------------
# cache_version.json: exists, matches, and verify_cache_version enforces it.
# ---------------------------------------------------------------------------

def test_cache_version_file_exists_and_matches(sim_data_cache):
    """inputs_hash folds in build_params (PARCA_REVIEW A7) — condition/
    seed/media/n_seeds identity of *this artifact* — which a bare
    ``compute_cache_version()`` can't independently re-derive (it defaults
    build_params to :data:`DEFAULT_BUILD_PARAMS`, all-None). A real built
    cache legitimately carries non-default build_params (e.g. condition=
    basal, seed=0), so comparing against the all-None default would fail
    even on a freshly built, fully current cache.

    Mirror ``verify_cache_version``'s own echo of ``stored.build_params``
    back into the recompute (see that function's docstring for why: build_
    params describe *which artifact* the cache is, not something "current
    code" can re-derive) so this stays a genuine input-files/context
    freshness check rather than a false positive on build_params identity.
    ``context`` is still re-probed fresh by ``compute_cache_version``, so a
    real staleness (changed INPUT_FILES content, or a dep upgrade between
    build and test run) still moves ``current.inputs_hash`` and fails here,
    same as before.

    Complementary to ``test_verify_cache_version_passes_on_current_cache``
    below: that one is a smoke test of the production call path (no raise);
    this one does the explicit hash comparison so a genuine mismatch names
    the stored vs. current hash prefixes directly in the assertion message.
    """
    stored = read_cache_version(sim_data_cache)
    assert stored is not None, (
        f'{sim_data_cache}/{CACHE_VERSION_FILENAME} missing — rebuild cache'
    )
    # Echo BOTH build_params (A7) and derived_from (schema 3): each describes
    # *which artifact* the cache is (its strain identity, and the ParCa chassis
    # it was built on) — neither is re-derivable by a bare recompute, so
    # verify_cache_version folds both back in; mirror that here so this stays a
    # genuine input-files/context freshness check, not a false positive on a
    # legitimately-chained cache.
    current = compute_cache_version(build_params=stored.build_params,
                                    derived_from=stored.derived_from)
    assert stored.inputs_hash == current.inputs_hash, (
        f'inputs_hash mismatch; stored={stored.inputs_hash[:16]}, '
        f'current={current.inputs_hash[:16]}. Rebuild cache.'
    )


def test_verify_cache_version_passes_on_current_cache(sim_data_cache):
    """Smoke test: the verify call that build_composite makes should be
    silent on a just-built cache."""
    verify_cache_version(sim_data_cache)  # no raise


def test_verify_cache_version_detects_missing_file(tmp_path, sim_data_cache):
    """Copy the cache data without cache_version.json and verify we
    raise StaleCacheError, not some obscure downstream error."""
    # Copy only the two data files — leave cache_version.json out.
    for name in ('initial_state.json', 'sim_data_cache.dill', 'metadata.json'):
        src = os.path.join(sim_data_cache, name)
        if os.path.exists(src):
            shutil.copy2(src, tmp_path / name)

    with pytest.raises(StaleCacheError, match='missing'):
        verify_cache_version(str(tmp_path))


def test_verify_cache_version_detects_hash_mismatch(tmp_path, sim_data_cache):
    """Write a cache_version.json with a deliberately wrong hash; verify
    raises StaleCacheError and lists the mismatched file."""
    # Copy data files.
    for name in ('initial_state.json', 'sim_data_cache.dill', 'metadata.json'):
        src = os.path.join(sim_data_cache, name)
        if os.path.exists(src):
            shutil.copy2(src, tmp_path / name)

    # Write a tampered version file — valid schema, wrong hash.
    current = compute_cache_version()
    tampered = {
        'schema_version': current.schema_version,
        'inputs_hash': 'deadbeef' * 8,
        'per_file_hashes': {
            # Flip one file's hash — verify should call it out by path.
            k: ('0' * 64 if k == 'v2ecoli/library/sim_data.py' else v)
            for k, v in current.per_file_hashes.items()
        },
    }
    (tmp_path / CACHE_VERSION_FILENAME).write_text(json.dumps(tampered))

    with pytest.raises(StaleCacheError, match='sim_data.py'):
        verify_cache_version(str(tmp_path))
