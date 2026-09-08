"""PARCA_REVIEW.md A3 + A6: fail loud instead of silently writing a
partially-fit or incomplete ParCa cache/pickle.

A3 (``v2ecoli/processes/parca/steps/step_09_final_adjustments.py``): the
three ``mechanistic_*`` fits were wrapped in ``try/except: print(...);
continue``, so a failure still produced a ``parca_state.pkl`` that is
byte-shaped exactly like a complete one. Fixed: a per-fit ``{label:
"ok"|"error"}`` status is always recorded at the ``mechanistic_fit_status``
output port, and a failure now raises (aborting the step, so the CLI never
reaches its ``pickle.dump``) unless ``allow_partial_fit=True`` (CLI:
``--allow-partial-fit``) is passed.

A6 (``v2ecoli/core.py``'s ``_write_sim_input_bundle``): per-config build
failures were collected and printed, then the bundle was written and
fingerprinted as valid anyway. Fixed: ``REQUIRED_CACHE_CONFIG_NAMES``
(``ecoli-mass-listener``, ``ecoli-metabolism`` — the two configs whose
absence PARCA_REVIEW documents as crashing the online sim) must be present
in ``configs`` or the build raises before writing ``sim_data_cache.dill`` /
``cache_version.json``. The config set actually built is also recorded onto
``CacheVersion.configs`` (not folded into ``inputs_hash`` — see that
field's docstring) so ``verify_cache_version`` can independently reject a
stored bundle that recorded an incomplete required subset, as a second line
of defense for any write path other than ``_write_sim_input_bundle``.

Everything here is hermetic: no ParCa fixture, no real SimulationDataEcoli,
no multi-hour pipeline run — consistent with
``tests/test_parca_ports_and_wiring.py``'s "validate the architecture
without running the full ParCa pipeline" approach. The A3 tests exercise
``FinalAdjustmentsStep.update()`` directly with a synthetic port-value dict
(mirroring ``make_sim_data_facade``'s documented contract) and monkeypatch
the heavy ``create_bulk_container`` call out of the way; the A6 tests
exercise ``v2ecoli.core._write_sim_input_bundle`` with a fake loader and
``v2ecoli.library.cache_version.{write,verify}_cache_version`` directly.

Fix-round 1 (Medium finding): ``scripts/build_cache.py`` and
``scripts/build_condition_cache.py`` each made a second, redundant
``write_cache_version(cache_dir, repo_root=repo_root)`` call *after*
``save_sim_input`` had already written a complete ``cache_version.json``
(configs + build_params + context) — clobbering it back to empty
``configs``/``build_params`` on every real cache build through those two
scripts, which silently no-op'd A6's verify-time second line of defense
for the normal build path. Fixed by dropping the redundant call in
``build_condition_cache.py`` (it wasn't even needed to pick up the
``condition.json`` manifest hash — that write happens *after* it) and by
reading the already-written version back in ``build_cache.py`` instead of
re-deriving+overwriting it. See
``test_build_scripts_do_not_clobber_configs_or_build_params`` below.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# A3 — Step 9 mechanistic fits must not silently land a partial pickle.
# ---------------------------------------------------------------------------

def _make_step9_state():
    """Minimal port-value dict FinalAdjustmentsStep.update() needs.

    Only the keys the step's code path actually reads (see
    ``make_sim_data_facade``'s docstring for which port names populate
    which sim_data-shaped attribute) — everything else in INPUT_PORTS is
    only needed for *composite* wiring, not a direct unit call.
    """
    return {
        'transcription': MagicMock(name='transcription'),
        'metabolism': MagicMock(name='metabolism'),
        'cell_specs': {},
        'constants': MagicMock(name='constants'),
    }


@pytest.fixture
def step9_core():
    from bigraph_schema import allocate_core
    return allocate_core()


def _patch_bulk_container(monkeypatch):
    """create_bulk_container drives real bulk-molecule fitting machinery
    that needs a fully-hydrated SimulationDataEcoli — irrelevant to the
    try/except-loop behavior under test, so replace it with a stub."""
    import v2ecoli.processes.parca.steps.step_09_final_adjustments as step9
    monkeypatch.setattr(step9, 'create_bulk_container',
                        lambda *a, **k: MagicMock(name='bulk_container'))


def test_partial_mechanistic_fit_aborts_by_default(monkeypatch, step9_core):
    """A mechanistic_* fit raising aborts the step (no pickle-worthy output)
    unless allow_partial_fit is set — the direct regression test for A3."""
    from v2ecoli.processes.parca.steps.step_09_final_adjustments import (
        FinalAdjustmentsStep,
    )

    _patch_bulk_container(monkeypatch)
    state = _make_step9_state()
    state['metabolism'].set_mechanistic_supply_constants.side_effect = (
        ValueError("Could not find positive forward and reverse kcat for CYS[c]"))

    step = FinalAdjustmentsStep(config={}, core=step9_core)  # default: allow_partial_fit=False

    with pytest.raises(RuntimeError, match="mechanistic_supply"):
        step.update(state)

    # Execution must not reach the ppGpp kinetics fit after the abort —
    # proof the step stopped rather than "logged and continued".
    state['transcription'].set_ppgpp_kinetics_parameters.assert_not_called()


def test_partial_mechanistic_fit_allowed_with_flag(monkeypatch, step9_core):
    """allow_partial_fit=True opts back into the old behavior: the step
    completes and records {label: "error"} instead of raising."""
    from v2ecoli.processes.parca.steps.step_09_final_adjustments import (
        FinalAdjustmentsStep,
    )

    _patch_bulk_container(monkeypatch)
    state = _make_step9_state()
    state['metabolism'].set_mechanistic_supply_constants.side_effect = (
        ValueError("Could not find positive forward and reverse kcat for CYS[c]"))

    step = FinalAdjustmentsStep(config={'allow_partial_fit': True}, core=step9_core)

    result = step.update(state)

    assert result['mechanistic_fit_status'] == {
        'mechanistic_supply': 'error',
        'mechanistic_export': 'ok',
        'mechanistic_uptake': 'ok',
    }
    # The pipeline DID continue past the failure this time.
    state['transcription'].set_ppgpp_kinetics_parameters.assert_called_once()


def test_all_mechanistic_fits_ok_records_all_ok(monkeypatch, step9_core):
    """Calibration-neutral success path: when nothing fails, status is all
    'ok' and the step behaves exactly as before (no raise either way)."""
    from v2ecoli.processes.parca.steps.step_09_final_adjustments import (
        FinalAdjustmentsStep,
    )

    _patch_bulk_container(monkeypatch)
    state = _make_step9_state()

    step = FinalAdjustmentsStep(config={}, core=step9_core)
    result = step.update(state)

    assert result['mechanistic_fit_status'] == {
        'mechanistic_supply': 'ok',
        'mechanistic_export': 'ok',
        'mechanistic_uptake': 'ok',
    }


def test_cli_allow_partial_fit_flag_threads_to_build_parca_composite():
    """cli/parca.py must expose --allow-partial-fit and forward it into
    build_parca_composite(...).

    A full ``main()`` invocation is deliberately avoided: it unconditionally
    monkeypatches the real Step classes' ``.update`` (module-global, not
    instance-scoped) for per-step checkpointing, which would leak across the
    rest of the test session. Assert the wiring statically instead.
    """
    from v2ecoli.cli import parca as parca_cli

    # The FLAG is asserted on the parser's behaviour rather than on source
    # text: a substring check passes just as happily when a flag is defined
    # and then dropped on the floor. (It also survives the parser moving out
    # of main() into _build_arg_parser(), which is what broke the old form.)
    parser = parca_cli._build_arg_parser()
    assert parser.parse_args([]).allow_partial_fit is False
    assert parser.parse_args(["--allow-partial-fit"]).allow_partial_fit is True

    # The FORWARDING is still a static check: a real main() invocation
    # monkeypatches the Step classes' module-global ``.update`` for per-step
    # checkpointing, which would leak across the rest of the session.
    src = inspect.getsource(parca_cli.main)
    assert 'allow_partial_fit=args.allow_partial_fit' in src, (
        "cli/parca.py:main() no longer forwards --allow-partial-fit into "
        "build_parca_composite(...)")


# ---------------------------------------------------------------------------
# A6 — an incomplete sim-input bundle must not be fingerprinted as valid.
# ---------------------------------------------------------------------------

class _FakeUniqueMolecule:
    unique_molecule_definitions: dict = {}


class _FakeInternalState:
    unique_molecule = _FakeUniqueMolecule()


class _FakeSimData:
    """Module-level (picklable) stand-in for a hydrated SimulationDataEcoli
    — only the two attributes ``_write_sim_input_bundle`` reads past the
    configs loop (unique_molecule_definitions, expectedDryMassIncreaseDict)."""

    internal_state = _FakeInternalState()
    expectedDryMassIncreaseDict: dict = {}


class _FakeLoader:
    """Duck-types the surface ``_write_sim_input_bundle`` needs from a
    ``LoadSimData`` instance, without a real SimulationDataEcoli."""

    def __init__(self, fail_names=()):
        self._fail_names = set(fail_names)
        self.sim_data = _FakeSimData()

    def generate_initial_state(self):
        return {}

    def get_config_by_name(self, name):
        if name in self._fail_names:
            raise ValueError(f"boom building {name}")
        return {'name': name}


def _chassis_sources(cache_dir):
    """A clean chassis source + sidecar beside ``cache_dir`` so a schema-3
    cache has the ``derived_from`` chain verify_cache_version's guard (a)
    requires — orthogonal to the A6 configs check under test here."""
    import json

    chassis = cache_dir / "parca_state.pkl"
    chassis.write_bytes(b"fake-chassis")
    (cache_dir / "parca_state.provenance.json").write_text(json.dumps({
        "schema": "chassis-provenance/1", "layer": "chassis",
        "code": {"v2ecoli": {"commit": "abc123", "dirty": False,
                             "source": "git"}},
    }))
    return [{"layer": "chassis", "path": str(chassis)}]


def test_incomplete_bundle_build_aborts(tmp_path):
    """A required config (PARCA_REVIEW A6: ecoli-mass-listener /
    ecoli-metabolism) failing to build raises before any bundle file that
    verify_cache_version would treat as a valid marker is written."""
    from v2ecoli.core import _write_sim_input_bundle

    bundle_dir = tmp_path / "cache"
    loader = _FakeLoader(fail_names={"ecoli-mass-listener"})

    with pytest.raises(RuntimeError, match="ecoli-mass-listener"):
        _write_sim_input_bundle(loader, str(bundle_dir))

    assert not (bundle_dir / "cache_version.json").exists(), (
        "cache_version.json must not exist after an aborted build — it is "
        "the marker verify_cache_version treats as 'this bundle is valid'")
    assert not (bundle_dir / "sim_data_cache.dill").exists()


def test_complete_bundle_build_succeeds(tmp_path):
    """Sanity/positive-control: no required config missing -> build proceeds
    normally and writes the marker."""
    from v2ecoli.core import _write_sim_input_bundle

    bundle_dir = tmp_path / "cache"
    loader = _FakeLoader(fail_names={"some-non-required-config"})

    _write_sim_input_bundle(loader, str(bundle_dir))

    assert (bundle_dir / "cache_version.json").exists()
    assert (bundle_dir / "sim_data_cache.dill").exists()


def test_incomplete_bundle_fails_verification(tmp_path):
    """verify_cache_version rejects a stored bundle whose recorded configs
    are missing a required entry, independent of the build-time guard —
    the second line of defense named in PARCA_REVIEW A6."""
    from v2ecoli.library.cache_version import (
        REQUIRED_CACHE_CONFIG_NAMES,
        StaleCacheError,
        verify_cache_version,
        write_cache_version,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    incomplete_configs = sorted(
        set(REQUIRED_CACHE_CONFIG_NAMES) - {"ecoli-mass-listener"}
        | {"some-other-config"})
    write_cache_version(str(cache_dir), configs=incomplete_configs,
                        sources=_chassis_sources(cache_dir))

    with pytest.raises(StaleCacheError, match="ecoli-mass-listener"):
        verify_cache_version(str(cache_dir))


def test_complete_bundle_configs_pass_verification(tmp_path):
    """Positive control for the verify-time check: a stored config set that
    covers REQUIRED_CACHE_CONFIG_NAMES verifies clean."""
    from v2ecoli.library.cache_version import (
        REQUIRED_CACHE_CONFIG_NAMES,
        verify_cache_version,
        write_cache_version,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    write_cache_version(
        str(cache_dir),
        configs=list(REQUIRED_CACHE_CONFIG_NAMES) + ["some-other-config"],
        sources=_chassis_sources(cache_dir))

    verify_cache_version(str(cache_dir))  # no raise


def test_empty_recorded_configs_does_not_fail_verification(tmp_path):
    """A stored bundle with no recorded configs (pre-A6, or a caller that
    never passed configs=...) is not asserted against — only the build
    path is a hard gate for those; verify_cache_version can't tell 'nothing
    recorded' from 'nothing required' from the marker alone."""
    from v2ecoli.library.cache_version import verify_cache_version, write_cache_version

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # configs=None -> recorded as (); sources declare the schema-3 chain.
    write_cache_version(str(cache_dir), sources=_chassis_sources(cache_dir))

    verify_cache_version(str(cache_dir))  # no raise


# ---------------------------------------------------------------------------
# Fix-round 1 — scripts/build_cache.py + scripts/build_condition_cache.py
# must not clobber the configs/build_params that save_sim_input already
# wrote correctly.
# ---------------------------------------------------------------------------

def test_build_scripts_do_not_clobber_configs_or_build_params(tmp_path):
    """Regression for the Fix-round-1 Medium finding.

    Both scripts funnel their real bundle-write through
    ``v2ecoli.core._write_sim_input_bundle`` (the shared body of
    ``save_cache``/``save_sim_input``), which already writes a complete
    ``cache_version.json`` — configs (A6) + build_params (T3/A7-A9) — as its
    last step. Before this fix, each script then made a SECOND
    ``write_cache_version(cache_dir, repo_root=repo_root)`` call with no
    ``configs=``/``build_params=``, silently overwriting that file back to
    an empty ``configs`` tuple and all-``None`` ``build_params``.

    This exercises the real write path with a fake loader (hermetic — no
    ParCa fixture) and asserts:
      1. The fixed sequence (write once, read back — what
         ``build_cache.py`` now does; ``build_condition_cache.py`` just
         doesn't write again at all) leaves ``configs``/``build_params``
         intact and non-empty/non-default.
      2. Re-introducing the OLD buggy second call (a bare
         ``write_cache_version(cache_dir)``) is what clobbers them — i.e.
         this test would have caught the exact bug reported, proving it's
         not a vacuous assertion.
    """
    from v2ecoli.core import _write_sim_input_bundle
    from v2ecoli.library.cache_version import (
        REQUIRED_CACHE_CONFIG_NAMES,
        read_cache_version,
        write_cache_version,
    )

    bundle_dir = tmp_path / "cache"
    loader = _FakeLoader()  # no fail_names -> every config builds

    _write_sim_input_bundle(
        loader, str(bundle_dir),
        seed=3, condition="acetate", fixed_media="minimal_acetate")

    # 1. What build_cache.py's FIXED sequence does: read back, don't
    #    rewrite. build_condition_cache.py's fix is simpler still (no
    #    second call at all) but the persisted file is the same either way.
    fixed = read_cache_version(str(bundle_dir))
    assert fixed.configs, "configs must be non-empty after the fixed sequence"
    assert set(REQUIRED_CACHE_CONFIG_NAMES) <= set(fixed.configs)
    assert fixed.build_params == {
        "condition": "acetate",
        "fixed_media": "minimal_acetate",
        "seed": 3,
        "n_seeds": fixed.build_params["n_seeds"],  # resolved independently
        "condition_manifest_hash": None,
        # P1-6 strain-identity keys, None for this unperturbed WT build.
        "new_genes": None,
        "bundle_overrides": None,
        "bundle_manifest": None,
        "perturbations": None,
    }

    # 2. Proof this is a real regression guard, not a vacuous assertion:
    #    the OLD buggy pattern — a bare second write_cache_version(cache_dir)
    #    with no configs=/build_params= — is exactly what clobbers them.
    write_cache_version(str(bundle_dir))
    clobbered = read_cache_version(str(bundle_dir))
    assert clobbered.configs == (), (
        "sanity check: the pre-fix pattern really did drop configs — if "
        "this assertion ever fails, write_cache_version's default changed "
        "and the fixed-sequence assertions above are the ones that matter")
    assert clobbered.build_params["condition"] is None
