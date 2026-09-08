"""P1-6: a wrong-strain ParCa cache must be REJECTED, not verified clean.

The silent failure this closes: two strains that differed only in
``new_genes`` / ``bundle_overrides`` / a baked ``perturbations`` edit produced
byte-identical ``cache_version.json`` fingerprints, and ``verify_cache_version``
echoed the *stored* build_params back into its "current" recompute — so it
compared a bundle against itself and never noticed a strain-B run pointed at a
strain-A cache. The sim then ran on the wrong genome.

The fix has two halves, both exercised here:
  1. strain-defining content is folded into ``inputs_hash`` (so the fingerprints
     diverge, ``compute_cache_version`` half);
  2. ``verify_cache_version`` takes ``expected_build_params`` and does a real
     requested-vs-stored comparison that raises on divergence.

These tests are hermetic — no ParCa fixture / fit required.
"""
from __future__ import annotations

import json

import dill
import pytest

from v2ecoli.library.cache_version import (
    StaleCacheError,
    compute_cache_version,
    verify_cache_version,
    write_cache_version,
)

STRAIN_A = {"new_genes": "vioABCDE_MG1655_v2"}
STRAIN_B = {"new_genes": "gfp_MG1655_v2"}


def _write_strain_cache(cache_dir, build_params):
    """Write a minimal but structurally valid bundle stamped for a strain.

    Schema 3 requires a ``derived_from`` chain, so this stamps a clean chassis
    source (committed, not dirty) beside the bundle — the wrong-strain check
    under test is about ``build_params``, orthogonal to the chain, and a
    chainless schema-3 cache is independently rejected by guard (a).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "initial_state.json").write_text("{}")
    with open(cache_dir / "sim_data_cache.dill", "wb") as f:
        dill.dump({"configs": {}}, f)
    chassis = cache_dir / "parca_state.pkl"
    chassis.write_bytes(b"fake-chassis")
    (cache_dir / "parca_state.provenance.json").write_text(json.dumps({
        "schema": "chassis-provenance/1", "layer": "chassis",
        "code": {"v2ecoli": {"commit": "abc123", "dirty": False,
                             "source": "git"}},
    }))
    write_cache_version(str(cache_dir), build_params=build_params,
                        sources=[{"layer": "chassis", "path": str(chassis)}])


# ---------------------------------------------------------------------------
# Half 1: the fingerprint itself must diverge between strains.
# ---------------------------------------------------------------------------


def test_new_genes_changes_fingerprint():
    """Two bundles differing only in new_genes must not share inputs_hash."""
    a = compute_cache_version(build_params=STRAIN_A)
    b = compute_cache_version(build_params=STRAIN_B)
    assert a.inputs_hash != b.inputs_hash
    # ...and the strain identity actually lands in the recorded build_params.
    assert a.build_params["new_genes"] == "vioABCDE_MG1655_v2"

    # Same strain -> identical hash (same files, same context).
    a_again = compute_cache_version(build_params=STRAIN_A)
    assert a.inputs_hash == a_again.inputs_hash


def test_bundle_overrides_and_perturbations_change_fingerprint():
    base = compute_cache_version()
    overridden = compute_cache_version(
        build_params={"bundle_overrides": "/overlay/ko-trpR.tsv"})
    perturbed = compute_cache_version(
        build_params={"perturbations": "deadbeef" * 8})

    assert base.inputs_hash != overridden.inputs_hash
    assert base.inputs_hash != perturbed.inputs_hash
    assert overridden.inputs_hash != perturbed.inputs_hash


# ---------------------------------------------------------------------------
# Half 2: verify_cache_version must reject a request for a different strain.
# ---------------------------------------------------------------------------


def test_wrong_strain_cache_is_rejected(tmp_path):
    """Request strain B against a cache built for strain A -> StaleCacheError.

    This is the core P1-6 regression: before the fix this verified clean
    because verify echoed the stored params instead of comparing against the
    request.
    """
    cache_dir = tmp_path / "strain_a_cache"
    _write_strain_cache(cache_dir, STRAIN_A)

    with pytest.raises(StaleCacheError, match="wrong strain/condition"):
        verify_cache_version(str(cache_dir), expected_build_params=STRAIN_B)


def test_same_strain_cache_verifies_clean(tmp_path):
    """Request strain A against strain A's cache -> no raise (positive case)."""
    cache_dir = tmp_path / "strain_a_cache"
    _write_strain_cache(cache_dir, STRAIN_A)

    # Must not raise.
    verify_cache_version(str(cache_dir), expected_build_params=STRAIN_A)


def test_no_expected_params_preserves_self_verification(tmp_path):
    """Omitting expected_build_params keeps the legacy self-verify behavior:

    a real non-basal (strain-A) bundle still verifies against itself, so
    existing callers that don't yet know the request are not broken.
    """
    cache_dir = tmp_path / "strain_a_cache"
    _write_strain_cache(cache_dir, STRAIN_A)

    verify_cache_version(str(cache_dir))  # no raise


def test_wrong_condition_cache_is_rejected(tmp_path):
    """The comparison is general — a wrong nutrient condition is caught too."""
    cache_dir = tmp_path / "acetate_cache"
    _write_strain_cache(
        cache_dir, {"condition": "acetate", "fixed_media": "minimal_acetate"})

    with pytest.raises(StaleCacheError, match="wrong strain/condition"):
        verify_cache_version(
            str(cache_dir),
            expected_build_params={"condition": None, "fixed_media": None})
