"""Shared infrastructure for v2ecoli composites.

Provides the bigraph-schema ``core`` (with ECOLI_TYPES registered) and the
cache-loading/saving helpers used by every architecture's
``@composite_generator`` function in ``v2ecoli/composites/``.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import warnings
from typing import Any

import dill
from bigraph_schema import allocate_core

from v2ecoli.cache import load_initial_state, save_initial_state, save_json
from v2ecoli.library.cache_version import (
    REQUIRED_CACHE_CONFIG_NAMES,
    StaleCacheError,
    verify_cache_version,
    write_cache_version,
)
# Import at module load so the shared pint UnitRegistry has
# nucleotide/amino_acid/count defined before any dill.load hydrates
# a Quantity whose unit string references those names.
from v2ecoli.library.unit_bridge import rebind_cache_quantities  # noqa: F401
from v2ecoli.types import ECOLI_TYPES


__all__ = [
    "build_core",
    "register_ecoli_core",
    "load_cache_bundle",
    "save_cache",
    "save_sim_input",
    "StaleCacheError",
]


def register_ecoli_core(core):
    """Register ecoli types/links onto an EXISTING bigraph-schema core.

    Behavior-preserving split of ``build_core``'s post-``allocate_core()``
    body: this is the piece a ``@composite_generator``'s ``core_extensions``
    hook needs (see ``v2ecoli.composites.ecoli_baseline``) so a generic
    runner can provision a bare core via ``core_extensions`` alone, without
    calling ``build_core()`` itself. ``build_core()`` below is now just
    ``register_ecoli_core(allocate_core())``.
    """
    core.register_types(ECOLI_TYPES)
    # Register emitters as links so they're discoverable (dashboard scans
    # core.link_registry for Emitter subclasses). Parquet stays the default
    # (workspace.yaml runtime.default_emitter = parquet). Be defensive: these
    # carry heavy optional deps — register what imports successfully and never
    # let an emitter import break build_core.
    try:
        from process_bigraph.emitter import SQLiteEmitter, RAMEmitter
        core.register_link("SQLiteEmitter", SQLiteEmitter)
        core.register_link("RAMEmitter", RAMEmitter)
    except Exception:
        pass
    try:
        from viva_emitters import ParquetEmitter
        core.register_link("ParquetEmitter", ParquetEmitter)
    except Exception:
        pass
    try:
        from viva_emitters import XArrayEmitter
        core.register_link("XArrayEmitter", XArrayEmitter)
    except Exception:
        pass
    try:
        # Bounded-RAM per-cell phenotype sink for the colony composite (streams
        # a scalar panel to zarr instead of accumulating the cells map in RAM).
        from v2ecoli.colony_emitter import ColonyPhenotypeRecorder
        core.register_link("ColonyPhenotypeRecorder", ColonyPhenotypeRecorder)
    except Exception:
        pass
    # Placeholder labeled-vector types so a SERIALIZED composite (the dashboard
    # composite-runner builds documents via build_generator, which doesn't run
    # the derivers' initialize() side-effects) can resolve ``overwrite[<vec>]``
    # port schemas at realize time. CountsDeriver / RnapData re-register these
    # with their real shape + element labels in initialize(); the labels are
    # what the output_metadata walker reads. Resolution only needs the type to
    # exist, so a bare array placeholder is enough.
    for _vec in ('monomer_counts_vec', 'rna_init_event_per_cistron_vec'):
        try:
            core.register_type(_vec, {'_inherit': 'array', '_data': 'int64'})
        except Exception:
            pass
    # Pulled-in external composites (pbg-ketchup): register its Process classes
    # so local:KetchupEstimator resolves in dashboard runs. Guarded — a missing
    # pbg_ketchup must never break build_core for the rest of v2ecoli.
    try:
        from pbg_ketchup import KetchupEstimator, KetchupDynamicEstimator
        core.register_link("KetchupEstimator", KetchupEstimator)
        core.register_link("KetchupDynamicEstimator", KetchupDynamicEstimator)
    except Exception:
        pass
    # Pulled-in external reactor physics (pbg-bioreactordesign): register
    # BiRDTransportProcess so local:BiRDTransportProcess resolves in the mbp-03
    # coupled-reactor composite. Guarded — a missing pbg_bioreactordesign must
    # never break build_core for the rest of v2ecoli.
    try:
        from pbg_bioreactordesign import BiRDTransportProcess
        core.register_link("BiRDTransportProcess", BiRDTransportProcess)
        # BiRDTransportHours: the seconds->hours time-base adapter the coupled
        # composite actually wires (v2ecoli steps in seconds; the transport math
        # is in hours). See v2ecoli/steps/bird_transport_hours.py.
        from v2ecoli.steps.bird_transport_hours import BiRDTransportHours
        core.register_link("BiRDTransportHours", BiRDTransportHours)
    except Exception:
        pass
    # Report-card Steps (resolved by name in the comparison harness).
    try:
        from scripts._compare.report_cards import REPORT_CARD_STEPS
        core.register_links(REPORT_CARD_STEPS)
    except Exception:  # noqa: BLE001 — never let card registration break build_core
        pass
    return core


def build_core():
    """Create and configure a bigraph-schema core with ecoli types.

    Also registers ``LineageProcess`` for the ``ray:`` address protocol (item 101/109) --
    unlike other composite-specific registrations (colony's ``_register_colony_core``,
    lineage_ray_batch's ``register_ray_lineage`` via its own ``core_extensions``), this one
    lives here, on the CORE BUILDER ITSELF, because ``run_pbg.py`` only runs
    ``apply_core_extensions`` on its ``--composite-id`` branch -- the ``/compose/v1`` raw-document
    branch has no ``composite_id`` at all, so a raw ``.pbg`` document with a ``ray:LineageProcess``
    address would otherwise fail to resolve under ``PBG_CORE_BUILDER=v2ecoli.core:build_core``.

    Deliberately calls ONLY ``register_ray_lineage`` (a pure registry write: ``register_types`` +
    ``register_process_class``, no Ray connection) -- NOT ``prewarm_lineage_pool``. Confirmed
    directly (``process_bigraph/protocols/ray.py:528-536``): ``RayProtocolRuntime.__init__`` calls
    ``ray.init()`` eagerly whenever Ray isn't already running. Calling ``prewarm_lineage_pool``
    unconditionally here would make EVERY caller of ``build_core()`` -- chain-dispatch's per-
    generation jobs, local scripts, tests, anything -- eagerly try to init Ray, whether or not it
    ever resolves a ``ray:`` address. Known, accepted trade-off from omitting it: a raw document
    dispatched through ``/compose/v1/run-document`` (the one path this unblocks) gets the ray:
    protocol's own DEFAULT pool sizing (``os.cpu_count()``) on first real resolution, not the
    cluster-derived ``RAY_SHARDS_DEFAULT`` -- callers on that path who need correct sizing must
    still call ``prewarm_lineage_pool`` themselves before dispatch, same as ``lineage_ray_batch``'s
    own composite_generator already does via its own ``core_extensions``.
    """
    from v2ecoli.workflow.batch_lineage_ray import register_ray_lineage

    core = register_ecoli_core(allocate_core())
    register_ray_lineage(core)
    return core


# Importing v2ecoli.core also registers the pulled-in pbg-ketchup composite
# *generators* (the @composite_generator decorators fire on import), so the
# dashboard's run subprocess — which does `from v2ecoli.core import build_core`
# then looks up the generator in the registry — can resolve ketchup_baseline /
# ketchup_dynamic. Guarded so it's a no-op when pbg-ketchup isn't installed.
try:
    import pbg_ketchup.composites  # noqa: F401
except Exception:
    pass


_SKIP_CACHE_VERIFY_ENV = "V2ECOLI_SKIP_CACHE_VERIFY"
_skip_cache_verify_warned = False


def _cache_verify_skipped() -> bool:
    """Whether the cache-version staleness check should be bypassed.

    Reads ``V2ECOLI_SKIP_CACHE_VERIFY`` at call time (not import time) so
    tests/scripts can toggle it via ``monkeypatch.setenv`` /
    ``os.environ`` without needing a fresh process. This is an escape
    hatch for deliberate cross-version work (e.g. comparing a cache built
    on a different commit); it warns once per process so a skip never
    goes unnoticed in logs.
    """
    global _skip_cache_verify_warned
    if not os.environ.get(_SKIP_CACHE_VERIFY_ENV):
        return False
    if not _skip_cache_verify_warned:
        warnings.warn(
            f"{_SKIP_CACHE_VERIFY_ENV} is set: skipping ParCa cache "
            "staleness verification. A stale or mis-calibrated cache may "
            "load silently. Unset this env var for normal use.",
            stacklevel=2,
        )
        _skip_cache_verify_warned = True
    return True


@functools.lru_cache(maxsize=4)
def _load_cache_bundle_cached(cache_dir):
    """Raw loader — memoized by cache_dir.

    Staleness verification is deliberately NOT done here: this function's
    result is memoized by ``cache_dir`` alone, so a check performed only on
    the first (cache-miss) call would silently stop running on every
    subsequent call for the same ``cache_dir`` within this process — and
    the escape-hatch env var wouldn't be re-read at call time either. See
    ``load_cache_bundle``, the public entry point, which verifies on every
    call before delegating to this memoized loader.
    """
    initial_state = load_initial_state(
        os.path.join(cache_dir, 'initial_state.json'))
    cache_path = os.path.join(cache_dir, 'sim_data_cache.dill')
    # Side-effectful imports (upstream vEcoli's bigraph_types) can
    # replace pint.application_registry after unit_bridge has
    # registered nucleotide/amino_acid/count on our ureg. Reassert
    # the app registry so dill.load's Quantity unpickle resolves
    # those custom units against the registry that has them.
    import pint
    from v2ecoli.types.quantity import ureg
    pint.set_application_registry(ureg)
    with open(cache_path, 'rb') as f:
        cache = dill.load(f)
    rebind_cache_quantities(cache)
    return initial_state, cache


def load_cache_bundle(cache_dir: str) -> dict[str, Any]:
    """Load the ParCa cache bundle from ``cache_dir``.

    Returns a flat dict containing ``initial_state`` plus whatever keys the
    underlying dill cache provides (typically ``configs``, ``unique_names``,
    ``dry_mass_inc_dict``).

    Verifies ``cache_dir`` against the current cache fingerprint
    (``v2ecoli.library.cache_version.verify_cache_version``) on every call,
    raising ``StaleCacheError`` if the cache was built from a different
    ParCa fixture / sim_data / unit-bridge / composite code. Set
    ``V2ECOLI_SKIP_CACHE_VERIFY=1`` to bypass this for deliberate
    cross-version work; doing so logs a one-time warning.

    The heavy work (reading the dill, rebinding pint Quantities onto the
    shared UnitRegistry) is memoized per ``cache_dir``; ``initial_state`` is
    deep-copied because ``build_document`` mutates it, while the cache dict
    is returned by reference (read-only).
    """
    if not _cache_verify_skipped():
        verify_cache_version(cache_dir)
    initial_state, cache = _load_cache_bundle_cached(cache_dir)
    # The translation_supply / trna_charging selector flags were removed from
    # the elongation process config_schema (model choice is now a wiring
    # choice). Strip them from pre-existing caches so old caches still build.
    # Idempotent; safe to mutate the memoized cache once.
    pe_cfg = cache.get("configs", {}).get("ecoli-polypeptide-elongation")
    if isinstance(pe_cfg, dict):
        pe_cfg.pop("translation_supply", None)
        pe_cfg.pop("trna_charging", None)
    return {"initial_state": copy.deepcopy(initial_state), **cache}


_CACHE_CONFIG_NAMES = [
    'post-division-mass-listener', 'ecoli-mass-listener', 'media_update',
    'exchange_data', 'ecoli-tf-unbinding', 'ecoli-tf-binding',
    'ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation',
    'ecoli-transcript-initiation', 'ecoli-polypeptide-initiation',
    'ecoli-chromosome-replication', 'ecoli-protein-degradation',
    'ecoli-rna-degradation', 'ecoli-complexation',
    'ecoli-transcript-elongation', 'ecoli-polypeptide-elongation',
    'ecoli-chromosome-structure', 'ecoli-metabolism',
    'RNA_counts_listener', 'rna_synth_prob_listener',
    'monomer_counts_listener', 'dna_supercoiling_listener',
    'ribosome_data_listener', 'rnap_data_listener',
    'unique_molecule_counts', 'allocator',
]

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_perturbations(perturbations) -> str | None:
    """Stable, JSON-safe fingerprint of an in-memory sim_data perturbation.

    ``perturbations`` is whatever strain-defining mutation was baked into the
    sim_data before the bundle was written (e.g. new_gene_cache's ``applied``
    dict of ids / indices / per-target values). It becomes a
    ``build_params['perturbations']`` value that folds into ``inputs_hash`` and
    is compared requested-vs-stored, so two perturbed strains no longer share a
    cache fingerprint (P1-6). Returns ``None`` for a falsy/absent perturbation
    (the wild-type build), a short hex digest otherwise. Never raises —
    numpy scalars / non-serializable leaves fall back to ``repr`` rather than
    crashing bundle-saving.
    """
    if not perturbations:
        return None
    if isinstance(perturbations, str):
        payload = perturbations.encode()
    else:
        payload = json.dumps(
            perturbations, sort_keys=True, default=repr).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve_n_seeds() -> int | None:
    """The fit's actual V2PARCA_N_SEEDS, for recording into build_params.

    Lazy + defensive: step_05 pulls in the heavy ParCa stack
    (stochastic_arrow, ...), which callers of the generic composite path
    (not building a cache bundle) shouldn't be forced to import. Never
    raises — a missing/broken import just records ``None`` rather than
    crashing bundle-saving (PARCA_REVIEW A9's "gracefully handle a package
    not being importable" applies here too).
    """
    try:
        from v2ecoli.processes.parca.steps.step_05_fit_condition import (
            resolved_n_seeds,
        )
        return resolved_n_seeds()
    except Exception:
        return None


def _write_sim_input_bundle(loader, bundle_dir, *, seed=None, condition=None,
                            fixed_media=None, condition_manifest_hash=None,
                            new_genes=None, bundle_overrides=None,
                            bundle_manifest=None, perturbations=None,
                            sources=None):
    """Write the simulation-input bundle from an instantiated LoadSimData.

    Shared body of ``save_cache`` (path-based) and ``save_sim_input``
    (live-object). Emits ``initial_state.json``, ``sim_data_cache.dill``,
    ``metadata.json``, and the cache-version marker into ``bundle_dir``.

    ``seed``/``condition``/``fixed_media``/``condition_manifest_hash`` are
    the actual build parameters this specific bundle was built with
    (PARCA_REVIEW A7) — recorded into ``cache_version.json``'s
    ``build_params`` (folded into ``inputs_hash``) so two bundles built with
    different parameters no longer produce byte-identical fingerprints.
    ``n_seeds`` is resolved independently (A8) since it isn't a parameter of
    ``LoadSimData`` — it governs the *fit* upstream of this bundle-writing
    step, not sim-data hydration.

    ``new_genes``/``bundle_overrides``/``bundle_manifest``/``perturbations``
    (P1-6) identify WHICH STRAIN this bundle is, so a wild-type cache and a
    new-gene / knockout / perturbed cache no longer share a fingerprint and a
    wrong-strain ``--cache-dir`` is caught by ``verify_cache_version`` instead
    of silently mis-calibrating the sim. ``perturbations`` is fingerprinted to
    a stable digest (see ``_fingerprint_perturbations``); the other three are
    recorded verbatim.

    ``sources`` (schema 3) declares the upstream artifacts this bundle was
    DERIVED FROM — each a ``{"layer","path"}`` dict, e.g. the ParCa
    ``parca_state.pkl`` chassis. It is threaded into ``write_cache_version`` so
    the ``derived_from`` provenance chain is recorded and folded into
    ``inputs_hash``. Default ``None`` (no declared chain) is backward
    compatible for callers that don't yet know their sources.
    """
    os.makedirs(bundle_dir, exist_ok=True)

    state = loader.generate_initial_state()
    save_initial_state(state, os.path.join(bundle_dir, 'initial_state.json'))

    configs = {}
    failed: list[tuple[str, Exception]] = []
    for name in _CACHE_CONFIG_NAMES:
        try:
            configs[name] = loader.get_config_by_name(name)
        except Exception as e:  # noqa: BLE001 — surface the name+error, don't lose it
            # Don't crash the whole cache build on one bad config: the
            # legacy vEcoli sim_data doesn't always expose every
            # config-getter's inputs (e.g. ``ecoli-metabolism-redux``
            # needs redux-specific attrs).  But DO print each failure
            # loudly — silently dropping ``ecoli-mass-listener`` /
            # ``ecoli-metabolism`` produces a cache that crashes the
            # online sim's Equilibrium step with a divide-by-zero on
            # ``listeners.mass.cell_mass``, which is obscure to debug.
            failed.append((name, e))
    if failed:
        print(f"  sim-input bundle: {len(failed)} config(s) failed to build "
              f"and were omitted:")
        for name, exc in failed:
            print(f"    - {name}: {type(exc).__name__}: {exc}")

    # PARCA_REVIEW A6: a bundle missing a REQUIRED config must never be
    # fingerprinted as valid — it passes verify_cache_version today and the
    # sim then dies on a divide-by-zero in listeners.mass.cell_mass /
    # crashes in Equilibrium. Abort before writing sim_data_cache.dill,
    # simData.cPickle, metadata.json, or cache_version.json: no marker means
    # verify_cache_version already refuses this bundle_dir as stale/partial
    # (same convention as an interrupted build). Unlike A3's mechanistic
    # fits, there is no opt-in bypass here — the review's fix is "hard-fail
    # the build", not "write it anyway with a flag".
    missing_required = sorted(
        name for name in REQUIRED_CACHE_CONFIG_NAMES if name not in configs)
    if missing_required:
        failed_by_name = dict(failed)
        detail = "; ".join(
            f"{name}: {type(failed_by_name[name]).__name__}: "
            f"{failed_by_name[name]}" if name in failed_by_name
            else f"{name}: not attempted"
            for name in missing_required)
        raise RuntimeError(
            f"sim-input bundle at {bundle_dir!r} is missing required "
            f"config(s) {missing_required} ({detail}). Refusing to write "
            f"sim_data_cache.dill / cache_version.json for a bundle that "
            f"would pass verify_cache_version but crash the online sim "
            f"(PARCA_REVIEW A6). Fix the underlying config-getter failure "
            f"— there is no bypass for a required config."
        )

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule
        .unique_molecule_definitions.keys())

    dry_mass_inc = getattr(loader.sim_data, 'expectedDryMassIncreaseDict', {})

    cache = {
        'configs': configs,
        'unique_names': unique_names,
        'dry_mass_inc_dict': dry_mass_inc,
    }
    cache_path = os.path.join(bundle_dir, 'sim_data_cache.dill')
    with open(cache_path, 'wb') as f:
        dill.dump(cache, f)

    # Also dump the RAW SimulationData as vEcoli's classic simData.cPickle so the
    # vEcoli composite (ecoli.library.sim_data.LoadSimData, via sim_data_path)
    # can consume the SAME ParCa output as the v2ecoli composite — identical
    # sim_data isolates the MODEL difference in the v2ecoli↔vEcoli comparison.
    # v2ecoli's own composite uses the `configs` dict above and ignores this file.
    import pickle as _pickle
    simdata_path = os.path.join(bundle_dir, 'simData.cPickle')
    with open(simdata_path, 'wb') as f:
        _pickle.dump(loader.sim_data, f, protocol=_pickle.HIGHEST_PROTOCOL)

    # Record the media this bundle was BUILT for. A per-condition bundle bakes
    # media_id into its configs at generation; recording it here lets a later
    # run cheaply detect a stale/wrong-media bundle (e.g. a with_aa bundle that
    # baked 'minimal' before the condition->media fix) without loading the
    # 164MB dill. Pull it from the same configs the composite actually runs on.
    _bundle_media_id = None
    try:
        _bundle_media_id = configs.get('media_update', {}).get('media_id')
    except AttributeError:
        pass
    save_json({'unique_names': unique_names, 'media_id': _bundle_media_id},
              os.path.join(bundle_dir, 'metadata.json'))

    # A rebuild-in-place (e.g. re-running scripts/build_condition_cache.py at
    # the same --cache dir) leaves a condition.json manifest from the PRIOR
    # build already on disk; fold its hash in when no explicit
    # condition_manifest_hash/patch id was supplied. On a fresh build of a
    # patched cache there is no condition.json yet at this point (it's
    # written by the caller after save_sim_input returns), so this stays
    # None there — pass condition_manifest_hash explicitly if that ordering
    # ever changes.
    resolved_manifest_hash = condition_manifest_hash
    if resolved_manifest_hash is None:
        manifest_path = os.path.join(bundle_dir, 'condition.json')
        if os.path.exists(manifest_path):
            resolved_manifest_hash = _hash_file(manifest_path)

    build_params = {
        'condition': condition,
        'fixed_media': fixed_media,
        'seed': seed,
        'n_seeds': _resolve_n_seeds(),
        'condition_manifest_hash': resolved_manifest_hash,
        # P1-6 strain identity.
        'new_genes': new_genes,
        'bundle_overrides': bundle_overrides,
        'bundle_manifest': bundle_manifest,
        'perturbations': _fingerprint_perturbations(perturbations),
    }
    write_cache_version(bundle_dir, build_params=build_params,
                        configs=sorted(configs.keys()), sources=sources)
    print(f"Sim-input bundle saved to {bundle_dir}")


def save_cache(sim_data_path, cache_dir='out/cache', seed=0, sources=None):
    """Generate the simulation-input bundle from a dilled SimulationDataEcoli.

    Prefer ``save_sim_input(sim_data, ...)`` when the SimulationDataEcoli is
    already in memory — this entry point exists for callers that only have a
    pickle path (legacy vEcoli ``simData.cPickle``).

    ``sources`` (schema 3): the upstream artifacts consumed (e.g. the ParCa
    chassis ``{"layer":"chassis","path": sim_data_path}``), recorded into the
    cache's ``derived_from`` provenance chain. Default ``None`` is backward
    compatible.
    """
    from v2ecoli.library.sim_data import LoadSimData
    loader = LoadSimData(sim_data_path=sim_data_path, seed=seed)
    _write_sim_input_bundle(loader, cache_dir, seed=seed, sources=sources)


def save_sim_input(sim_data, bundle_dir='out/cache', seed=0,
                   condition=None, fixed_media=None,
                   condition_manifest_hash=None,
                   new_genes=None, bundle_overrides=None,
                   bundle_manifest=None, perturbations=None,
                   sources=None):
    """Generate the simulation-input bundle from a live ``SimulationDataEcoli``.

    Skips the ~300 MB dill round-trip that ``save_cache`` performs to load
    sim_data from disk. Use this when you already have a hydrated
    ``SimulationDataEcoli`` in hand (e.g. straight off the parca composite or
    its fixture) — the resulting bundle is byte-for-byte equivalent to what
    ``save_cache`` would produce from the same sim_data dilled to a file.

    ``condition`` (e.g. "acetate") and ``fixed_media`` (e.g. "minimal_acetate")
    select a non-default nutrient condition so the generated initial state
    reflects that condition's growth rate / doubling time — both already live in
    the ParCa state's ``condition_to_doubling_time`` / saved media, so no refit
    is needed. Omitted → default basal / minimal (glucose).

    ``condition_manifest_hash`` (PARCA_REVIEW A7): pass the hash/id of a
    patch manifest (e.g. ``build_condition_cache.py``'s in-memory
    ``patch_record`` before its ``condition.json`` is written) when the
    caller already knows it, so it lands in ``cache_version.json``'s
    ``build_params`` on the very first build rather than only on a rebuild
    (see ``_write_sim_input_bundle``'s condition.json auto-detection).
    """
    from v2ecoli.library.sim_data import LoadSimData
    kwargs = {"sim_data": sim_data, "seed": seed}
    if condition is not None:
        kwargs["condition"] = condition
    if fixed_media is not None:
        kwargs["fixed_media"] = fixed_media
    loader = LoadSimData(**kwargs)
    _write_sim_input_bundle(loader, bundle_dir, seed=seed, condition=condition,
                            fixed_media=fixed_media,
                            condition_manifest_hash=condition_manifest_hash,
                            new_genes=new_genes,
                            bundle_overrides=bundle_overrides,
                            bundle_manifest=bundle_manifest,
                            perturbations=perturbations,
                            sources=sources)
