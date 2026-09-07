"""Millard-PDMP baseline composite — drops v2ecoli's tFBA Metabolism in favor
of the Millard 2017 ODE + multi-state LQR controller + FBA bridge.

Goal (Phase 1 milestone): a runnable 600 s simulation that exercises all
~54 WCM processes (transcription, translation, replication, listeners,
division, etc.) with metabolism replaced by the kinetic-ODE stack.

Wiring strategy
---------------
- All non-metabolism processes from baseline.py stay as-is.
- 'ecoli-metabolism' is REMOVED from the execution layers / step registry.
- Three new Process edges replace it (executed in the same layer slot):
    1. millard-with-lqr (MillardWithLQR): COPASI/basico Millard 2017 ODE
       with per-tick PTS_4.kF modulation from LQR control.
    2. lqr-controller (LQRControllerMultiState): reads Millard state from
       ('shared', 'central_metabolites'), emits ('shared', 'lqr_control').
    3. fba-bridge (FBABridge): translates Millard mM → counts. Writes into
       a SEPARATE plain-dict store ('central_metabolite_counts'), NOT the
       WCM's structured bulk array — the shape doesn't match. Downstream
       WCM processes therefore see unchanged bulk counts (metabolism
       writeback is biologically out-of-scope for this milestone; the
       milestone is "runs end-to-end without crashing").

Topology overview
-----------------
shared/
  central_metabolites          ← Millard mM concentrations (Millard outputs)
  central_fluxes               ← Millard fluxes (unused by anyone yet)
  lqr_control                  ← LQR output (consumed by Millard)
  lqr_diagnostics              ← LQR diagnostics
  control_applied              ← Millard's applied-control log
  bridge_diagnostics           ← FBABridge diagnostics
central_metabolite_counts      ← FBABridge writeback in v2ecoli ID space
                                 (parallel to but NOT merged into bulk)
"""

from __future__ import annotations

import binascii
import copy
from typing import Any

import numpy as np

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle


def _derive_process_seed(master_seed: int, process_name: str) -> int:
    return binascii.crc32(process_name.encode("utf-8"), master_seed) & 0x7FFFFFFF


from v2ecoli.composites._helpers import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    seed_mass_listener,
    _normalize_boundary_units,
    _make_instance,
    _get_special_step,
    CachedConfigLoader,
    _expand_flushes,
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
    ALLOCATOR_LAYERS,
    DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)


# ---------------------------------------------------------------------------
# Execution layers — identical to baseline except 'ecoli-metabolism' replaced
# by the three Millard-PDMP edges, run in the same layer slot.
# ---------------------------------------------------------------------------

MILLARD_EDGES = ['millard-pdmp-metabolism', 'lqr-controller']

# Step that consumes FBABridge's parallel-dict output and writes deltas to
# the structured bulk store — kept as a fallback if the inline-bulk-emit
# path in MillardPDMPMetabolism is ever disabled.
MILLARD_BULK_INDEXER = 'millard-bulk-indexer'

# Optional teleonomic growth driver — closes the W₂ gap exposed by
# scripts/compare_pdmp_vs_phase0.py while a proper biomass-flux Process
# (task #21 full form) is built. Off unless `ref_growth_driver` feature
# is enabled in the composite call.
REF_GROWTH_DRIVER = 'ref-growth-driver'

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: standalone
    ['ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation'], FLUSH,

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: protein deg + initiation/replication/complexation
    ['ecoli-protein-degradation'],
    ['ecoli-complexation', 'ecoli-chromosome-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],
    ['ecoli-rna-degradation_requester'],
    ['allocator_2'],
    ['ecoli-rna-degradation_evolver'], FLUSH,

    # Layer 5: elongation
    ['ecoli-polypeptide-elongation_requester', 'ecoli-transcript-elongation_requester'],
    ['allocator_3'],
    ['ecoli-polypeptide-elongation_evolver', 'ecoli-transcript-elongation_evolver'], FLUSH,

    # Layer 6: chromosome structure + Millard-PDMP metabolism replacement
    ['ecoli-chromosome-structure'], FLUSH,
    # Replaces the original ['ecoli-metabolism'] slot. The four pieces are
    # wired as Steps in explicit order so the composite's priority-based
    # flow scheduler invokes them every tick — a top-level Process edge
    # with interval=1.0 was silently NOT scheduled by the step-flow
    # composite (verified empirically: central_metabolites stayed {} after
    # 3 s of run). Order within the tick:
    #   1. millard-with-lqr   — reads previous lqr_control, advances ODE
    #   2. fba-bridge         — translates central_metabolites mM → counts
    #   3. millard-bulk-indexer — applies counts as deltas to bulk
    #   4. lqr-controller     — reads new central_metabolites, emits next u
    # All four in one layer so they share flow tokens — separate layers
    # broke Millard firing (the layer-token chain interacts badly with
    # Process `interval` scheduling). Order within a layer doesn't matter
    # for process-bigraph; data dependencies resolve via input/output
    # topology. Tick t:
    #   - millard-with-lqr reads stale lqr_control, writes central_metabolites
    #   - bulk-indexer reads current central_metabolites, writes bulk
    #   - lqr-controller reads current central_metabolites, writes lqr_control
    # Single combined Process: runs Millard ODE, applies LQR control,
    # emits bulk deltas. Replaces the staged Millard → FBABridge →
    # bulk-indexer chain that hit a process-bigraph wiring quirk where
    # an input read on `central_metabolites` dropped Millard's writes.
    ['millard-pdmp-metabolism'], FLUSH,
    ['lqr-controller'], FLUSH,

    # Layer 7: listeners
    ['counts_deriver', 'ecoli-mass-listener',
     'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener'], FLUSH,

    # Layer 7b (Phase-3): per-process likelihood collector. Must run
    # AFTER the listener layer so the per-process log_likelihood
    # writes from TranscriptInitiation + PolypeptideInitiation are
    # already in the merged state when the collector reads them.
    ['likelihood_collector'], FLUSH,

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
    ['mark_d_period'], FLUSH,
    ['division'],
]

FEATURE_MODULES = {
    'supercoiling': {
        'insert_after': 'ecoli-chromosome-structure',
        'steps': ['dna-supercoiling-step'],
        'listeners': ['dna_supercoiling_listener'],
    },
    'ppgpp_regulation': {
        'insert_before': 'ecoli-transcript-initiation',
        'steps': ['ppgpp-initiation'],
    },
    'trna_attenuation': {
        'insert_before': 'ecoli-transcript-elongation_requester',
        'steps': ['trna-attenuation-config'],
    },
    # Teleonomic Phase-0 reference growth driver. Inserts after the
    # metabolism Process so each tick's order is: Millard → bulk indexer
    # (delta-mode) → ref-growth-driver (scale precursors at μ_ref).
    # Off by default; opt-in via composite features=['ref_growth_driver'].
    'ref_growth_driver': {
        'insert_after': 'millard-pdmp-metabolism',
        'steps': [REF_GROWTH_DRIVER],
    },
    # Opt-in runtime mass-conservation check (shared with baseline.py). Runs
    # after the mass listener; reads environment.exchange (now populated by the
    # Millard edge's medium-exchange accounting) + listeners.mass. Off by
    # default; enable via features=['mass_conservation'].
    'mass_conservation': {
        'insert_after': 'ecoli-mass-listener',
        'steps': ['ecoli-mass-conservation'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']


def build_execution_layers(features=None):
    layers = copy.deepcopy(BASE_EXECUTION_LAYERS)
    for feat_name in (features or []):
        feat = FEATURE_MODULES.get(feat_name)
        if feat is None:
            continue
        if 'insert_after' in feat:
            ref = feat['insert_after']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in feat.get('steps', []):
                        layers.insert(i + 1, [step_name])
                    break
        if 'insert_before' in feat:
            ref = feat['insert_before']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in reversed(feat.get('steps', [])):
                        layers.insert(i, [step_name])
                    break
        for listener in feat.get('listeners', []):
            for layer in layers:
                if isinstance(layer, list) and 'ecoli-mass-listener' in layer:
                    if listener not in layer:
                        layer.append(listener)
                    break
    return _expand_flushes(layers)


FLOW_ORDER = [step for layer in build_execution_layers(DEFAULT_FEATURES) for step in layer]


# ---------------------------------------------------------------------------
# Millard-PDMP edge builders
# ---------------------------------------------------------------------------

def _build_millard_pdmp_edge(core: Any, *, tick_s: float = 1.0,
                             backend: str = "basico"):
    """Build the combined Millard+LQR+bulk-emit Process edge.

    `backend="basico"` uses COPASI/basico (default; supports LQR control
    parameter modulation). `backend="jax"` uses the JIT-compiled
    JAX/Diffrax port — measured 1.8x faster at loose tolerances, but
    omits LQR control until the SBML->JAX translator is extended to
    expose runtime-settable parameters (task #19).
    """
    if backend == "jax":
        from v2ecoli.steps.millard_pdmp_metabolism_jax import (
            MillardPDMPMetabolismJAX as _Cls,
        )
        cfg = {
            "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
            "tick_s": tick_s,
            # Tight tol matches basico's LSODA accuracy; the loose-tol
            # variant produced slightly different bulk deltas that drove
            # Equilibrium's reconciler to fail. Tight tol costs ~2x basico
            # standalone but stays within ~10⁻¹⁰ of basico's trajectory.
            "rtol": 1e-6,
            "atol": 1e-9,
        }
    elif backend == "basico":
        from v2ecoli.steps.millard_pdmp_metabolism import (
            MillardPDMPMetabolism as _Cls,
        )
        cfg = {
            "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
            "tick_s": tick_s,
            "intervals": 10,
            "control_reaction": "PTS_4",
            "control_parameter": "kF",
            "u_clip": 0.5,
        }
    else:
        raise ValueError(f"Unknown PDMP backend: {backend!r}. "
                         "Expected 'basico' or 'jax'.")
    instance = _Cls(config=cfg, core=core)
    in_topo = {
        "lqr_control": ("shared", "lqr_control"),
        "bulk": ("bulk",),
        # listeners.mass provides cell_mass_fg used to compute live cell
        # volume for the mM->count translation (task #15).
        "listeners_mass": ("listeners", "mass"),
    }
    out_topo = {
        "species_concentrations": ("shared", "central_metabolites"),
        # Per-reaction fluxes (mM/s) published to the agent-root
        # ('central_fluxes',) store that FBAFluxCoupler reads (its topology
        # declares ('central_fluxes',)).
        "central_fluxes": ("central_fluxes",),
        "control_applied": ("shared", "control_applied"),
        "bulk": ("bulk",),
        # Per-tick signed medium-exchange counts (O2/glucose/acetate) -> the
        # cell's ('environment', 'exchange') store (WCM convention).
        "environment": ("environment",),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


def _build_millard_with_lqr_edge(core: Any, *, tick_s: float = 1.0):
    """Build the closed-loop MillardWithLQR Process edge."""
    from v2ecoli.steps.millard_with_lqr import MillardWithLQR
    cfg = {
        "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
        "time": tick_s,
        "intervals": 10,
        "control_reaction": "PTS_4",
        "control_parameter": "kF",
        "u_clip": 0.5,
    }
    instance = MillardWithLQR(config=cfg, core=core)
    in_topo = {"lqr_control": ("shared", "lqr_control")}
    out_topo = {
        "species_concentrations": ("shared", "central_metabolites"),
        "fluxes": ("shared", "central_fluxes"),
        "control_applied": ("shared", "control_applied"),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


def _build_lqr_controller_edge(core: Any, *, tick_s: float = 1.0):
    """Build the multi-state LQR controller Process edge."""
    from v2ecoli.steps.lqr_controller_multistate import LQRControllerMultiState
    cfg = {
        "linearization_npz": "v2ecoli/data/millard_linearization.npz",
        "Q_diag_weight": 1.0,
        "R": 0.1,
        "tick_s": tick_s,
    }
    instance = LQRControllerMultiState(config=cfg, core=core)
    in_topo = {"central_metabolites_millard": ("shared", "central_metabolites")}
    out_topo = {
        "lqr_control": ("shared", "lqr_control"),
        "lqr_diagnostics": ("shared", "lqr_diagnostics"),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


def _build_fba_bridge_edge(core: Any, *, tick_s: float = 1.0):
    """Build the FBABridge Process edge.

    The 'v2ecoli_bulk' port wires to a SEPARATE plain-dict store
    ('central_metabolite_counts'), NOT to the WCM's structured bulk array.
    Wiring it to ('bulk',) is not possible: the bridge writes a
    dict-of-{vid: count} update, but the bulk store is a structured numpy
    array with id+count fields and 16k+ rows. Bridging the two requires an
    indexer step that's out-of-scope for the "first working composite"
    milestone.
    """
    from v2ecoli.steps.fba_bridge import FBABridge
    cfg = {
        "mapping_file": "v2ecoli/data/millard_v2ecoli_species_map.yaml",
        "direction": "millard_to_v2ecoli",
        "cell_volume_L": 1.0e-15,
    }
    instance = FBABridge(config=cfg, core=core)
    # millard_to_v2ecoli direction: bridge does NOT read its v2ecoli_bulk
    # input, only writes to it. Removing the input wiring avoids the
    # InPlaceDict input-also-output pattern where output deltas were
    # silently dropped (verified: shared_pool_count=22 but the target
    # dict stayed {} when wired both ways).
    in_topo = {
        "central_metabolites_millard": ("shared", "central_metabolites"),
    }
    out_topo = {
        "v2ecoli_bulk": ("shared", "central_metabolite_counts"),
        "bridge_diagnostics": ("shared", "bridge_diagnostics"),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


# ---------------------------------------------------------------------------
# Step config dispatch — identical to baseline but with metabolism removed
# from the standalone-step registry.
# ---------------------------------------------------------------------------

def _get_step_config(
    loader,
    step_name,
    core,
    process_cache=None,
    master_seed=0,
    ref_growth_flux_source: str | None = None,
    ref_growth_feedback_tau_s: float | None = None,
    ref_growth_feedback_period_ticks: int | None = None,
    transcript_initiation_mode: str | None = None,
    transcript_init_prob_scale: float | None = None,
    polypeptide_init_prob_scale: float | None = None,
    polypeptide_initiation_mode: str | None = None,
):
    from v2ecoli.processes.equilibrium import Equilibrium
    from v2ecoli.processes.two_component_system import TwoComponentSystem
    from v2ecoli.processes.rna_maturation import RnaMaturation
    from v2ecoli.processes.complexation import Complexation
    from v2ecoli.processes.protein_degradation import ProteinDegradation
    from v2ecoli.processes.transcript_initiation import TranscriptInitiation
    from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
    from v2ecoli.processes.chromosome_replication import ChromosomeReplication
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.steps.partition import Requester, Evolver
    from v2ecoli.steps.derivers.mass_deriver import MassDeriver, PostDivisionMassDeriver
    from v2ecoli.steps.derivers.rna_synth_prob import RnaSynthProb
    from v2ecoli.steps.derivers.dna_supercoiling import DnaSupercoiling
    from v2ecoli.steps.derivers.replication_data import ReplicationData
    from v2ecoli.steps.derivers.rnap_data import RnapData
    from v2ecoli.steps.derivers.translation_deriver import TranslationDeriver
    from v2ecoli.steps.derivers.likelihood_collector import LikelihoodCollector
    from v2ecoli.steps.media_update import MediaUpdate
    from v2ecoli.steps.exchange_data import ExchangeData

    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Allocators
    if step_name.startswith('allocator'):
        from v2ecoli.steps.allocator import Allocator
        try:
            alloc_config = loader.get_config_by_name('allocator')
        except Exception:
            alloc_config = {}
        layer_procs = ALLOCATOR_LAYERS.get(step_name, ALL_PARTITIONED)
        if not alloc_config.get('process_names'):
            alloc_config['process_names'] = ALL_PARTITIONED
        alloc_config['layer_processes'] = layer_procs
        instance = _make_instance(Allocator, alloc_config, core)
        topo = instance.topology
        return instance, topo, 'step'

    # Millard-PDMP edges are handled separately (in baseline body), so the
    # config dispatcher never sees them. Return None to skip.
    if step_name in MILLARD_EDGES or step_name in (
            'millard-with-lqr', 'fba-bridge'):
        return None

    # ref-growth-driver: Phase-0 reference growth driver.
    # flux_source overrides come through master_seed → flux_source extras.
    if step_name == REF_GROWTH_DRIVER:
        from v2ecoli.steps.ref_growth_driver import RefGrowthDriver
        driver_config = {
            "seed": _derive_process_seed(master_seed, step_name),
        }
        flux_source = ref_growth_flux_source
        if flux_source:
            driver_config["flux_source"] = flux_source
        if ref_growth_feedback_tau_s is not None:
            driver_config["feedback_tau_s"] = float(ref_growth_feedback_tau_s)
        if ref_growth_feedback_period_ticks is not None:
            driver_config["feedback_period_ticks"] = int(
                ref_growth_feedback_period_ticks)
        instance = _make_instance(
            RefGrowthDriver,
            driver_config,
            core,
        )
        # Reads bulk + listeners.mass (read-only, for closed-loop WATER[c]
        # density regulation); writes only bulk.
        in_topo = {"bulk": ("bulk",), "listeners_mass": ("listeners", "mass")}
        out_topo = {"bulk": ("bulk",)}
        return instance, instance.topology, 'step', in_topo, out_topo

    # millard-bulk-indexer: instantiate from its own module; no sim-data
    # config required. Pass explicit input/output topologies — the
    # indexer READS central_metabolites and WRITES bulk; the default
    # make_edge behavior of reusing topology for both directions would
    # declare it as a central_metabolites writer too, conflicting with
    # millard-with-lqr and silently dropping the latter's update.
    # Phase-3 likelihood collector: no ParCa config, no special-step
    # registration. Instantiate directly with the minimal tick config so
    # we don't fall through to loader.get_config_by_name (which raises
    # → _get_special_step → silently returns None and the step is
    # dropped from the composite).
    if step_name == 'likelihood_collector':
        from v2ecoli.steps.derivers.likelihood_collector import LikelihoodCollector
        instance = _make_instance(LikelihoodCollector, {}, core)
        topo = getattr(instance, 'topology', {})
        if callable(topo):
            topo = topo()
        return instance, topo, 'step'

    if step_name == MILLARD_BULK_INDEXER:
        from v2ecoli.steps.millard_bulk_indexer import MillardBulkIndexer
        instance = _make_instance(
            MillardBulkIndexer,
            {"seed": _derive_process_seed(master_seed, step_name)},
            core,
        )
        in_topo = {
            "bulk": ("bulk",),
            "cm_view": ("shared", "central_metabolites"),
        }
        out_topo = {"bulk": ("bulk",)}
        return instance, instance.topology, 'step', in_topo, out_topo

    # Consolidated counts deriver: one step computing the RNA / monomer /
    # unique-molecule count readouts (byte-identical to the three former
    # listeners). Assemble its config from the three former config names.
    if step_name == 'counts_deriver':
        from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
        merged_cfg = {}
        for cfg_name in ('RNA_counts_listener', 'monomer_counts_listener',
                         'unique_molecule_counts'):
            try:
                merged_cfg.update(loader.get_config_by_name(cfg_name) or {})
            except (KeyError, AttributeError):
                pass
        instance = _make_instance(CountsDeriver, merged_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'


    try:
        config = loader.get_config_by_name(base_name)
    except (KeyError, AttributeError):
        try:
            config = loader.get_config_by_name(step_name)
        except (KeyError, AttributeError):
            return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    # STANDALONE_STEPS — note: ecoli-metabolism is INTENTIONALLY ABSENT.
    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-protein-degradation': ProteinDegradation,
        'ecoli-equilibrium': Equilibrium,
        'ecoli-two-component-system': TwoComponentSystem,
        'ecoli-complexation': Complexation,
        'ecoli-rna-maturation': RnaMaturation,
        'ecoli-transcript-initiation': TranscriptInitiation,
        'ecoli-polypeptide-initiation': PolypeptideInitiation,
        'ecoli-chromosome-replication': ChromosomeReplication,
    }

    SIMPLE_STEPS = {
        'ecoli-mass-listener': MassDeriver,
        'post-division-mass-listener': PostDivisionMassDeriver,
        'rna_synth_prob_listener': RnaSynthProb,
        'dna_supercoiling_listener': DnaSupercoiling,
        'replication_data_listener': ReplicationData,
        'rnap_data_listener': RnapData,
        'ribosome_data_listener': TranslationDeriver,
        'likelihood_collector': LikelihoodCollector,
        'media_update': MediaUpdate,
        'exchange_data': ExchangeData,
    }

    from v2ecoli.library.config_resolver import resolve_config
    config = resolve_config(config) if config else config

    if isinstance(config, dict) and "seed" in config:
        config["seed"] = _derive_process_seed(master_seed, base_name)

    # Phase-2 jump-process opt-ins. Merge mode overrides into the ParCa-
    # generated config so each Process's initialize() picks them up.
    if (
        transcript_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-transcript-initiation'
    ):
        config["pdmp_initiation_mode"] = transcript_initiation_mode
    if (
        transcript_init_prob_scale is not None
        and isinstance(config, dict)
        and base_name == 'ecoli-transcript-initiation'
    ):
        config["transcript_init_prob_scale"] = float(
            transcript_init_prob_scale)
    if (
        polypeptide_init_prob_scale is not None
        and isinstance(config, dict)
        and base_name == 'ecoli-polypeptide-initiation'
    ):
        config["polypeptide_init_prob_scale"] = float(
            polypeptide_init_prob_scale)
    if (
        polypeptide_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-polypeptide-initiation'
    ):
        config["pdmp_initiation_mode"] = polypeptide_initiation_mode

    if base_name in PARTITIONED_PROCESSES:
        proc_cls = PARTITIONED_PROCESSES[base_name]
        if base_name in process_cache:
            process = process_cache[base_name]
        else:
            from v2ecoli.library.ecoli_step import set_current_core
            set_current_core(core)
            process = proc_cls(config)
            set_current_core(None)
            process_cache[base_name] = process
        topology = dict(config.get('topology', {}) or {})
        if not topology:
            topology = getattr(process, 'topology',
                               getattr(proc_cls, 'topology', {}))
            if callable(topology):
                topology = topology()
            topology = dict(topology)

        if step_name.endswith('_requester'):
            instance = Requester({
                'time_step': config.get('time_step', 1),
                'process': process,
            })
            in_topo = dict(topology)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            in_topo['process'] = ('process', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {
                'next_update_time': ('next_update_time', base_name),
                'process': ('process', base_name),
            }
            if 'request' in out_ports:
                out_topo['request'] = ('request', base_name)
            if 'listeners' in out_ports:
                out_topo['listeners'] = topology.get('listeners', ('listeners',))
            return instance, topology, 'step', in_topo, out_topo

        elif step_name.endswith('_evolver'):
            instance = Evolver({
                'time_step': config.get('time_step', 1),
                'process': process,
            })
            in_topo = dict(topology)
            in_topo['allocate'] = ('allocate', base_name)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            in_topo['process'] = ('process', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {
                'next_update_time': ('next_update_time', base_name),
                'process': ('process', base_name),
            }
            for port in out_ports:
                if port in ('next_update_time', 'process', 'allocate',
                            'global_time', 'timestep'):
                    continue
                if port in topology:
                    out_topo[port] = topology[port]
                elif port == 'listeners':
                    out_topo['listeners'] = ('listeners',)
            return instance, topology, 'step', in_topo, out_topo

    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = _make_instance(step_cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    elif step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = _make_instance(cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    return None


def _register_millard_pdmp_links(core):
    """core_extensions hook — register the three Millard-PDMP Process classes."""
    from v2ecoli.steps.millard_with_lqr import register as register_millard
    from v2ecoli.steps.fba_bridge import register as register_bridge
    from v2ecoli.steps.lqr_controller_multistate import register as register_lqr_ms
    register_millard(core)
    register_bridge(core)
    register_lqr_ms(core)
    return core


@composite_generator(
    name="millard_pdmp_baseline",
    description=(
        "Whole-cell E. coli composite with v2ecoli's tFBA Metabolism replaced "
        "by the Millard 2017 ODE + multi-state LQR controller + FBA-bridge. "
        "Phase 1 PDMP milestone: runnable end-to-end; metabolism writeback to "
        "WCM bulk is NOT yet wired (separate `central_metabolite_counts` store)."
    ),
    parameters={
        "seed": {
            "type": "integer", "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string", "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
        "tick_s": {
            "type": "float", "default": 1.0,
            "description": "Millard / LQR / bridge update interval in seconds",
        },
        "backend": {
            "type": "string", "default": "basico",
            "description": (
                "ODE integrator backend for the Millard substep. 'basico' "
                "uses COPASI (full LQR support); 'jax' uses the "
                "JIT-compiled Diffrax port (faster at loose tols; no LQR yet)."
            ),
        },
        "with_ref_growth": {
            "type": "boolean", "default": False,
            "description": (
                "Enable the reference-growth driver — scaffold that drives "
                "precursor pools to compensate for the Millard ODE's "
                "missing biomass equation. See `ref_growth_flux_source` "
                "for the two flux modes."
            ),
        },
        "ref_growth_flux_source": {
            "type": "string", "default": "proportional",
            "description": (
                "Driver flux mode (only used when with_ref_growth=True). "
                "'proportional' scales pools at μ=2.44e-4/s — teleonomic "
                "but moves cm_final only ~2 fg of the 187 fg gap because "
                "precursor turnover (~1.8M ATP/s) is ~1000× larger. "
                "'measured_kfba' injects at constant per-second rates "
                "measured from a 600 s kFBA-baseline run "
                "(scripts/sample_kfba_precursor_fluxes.py → "
                ".pbg/runs/kfba-precursor-fluxes.json); top rates: "
                "GLT 5413/s, ATP 1640/s, UTP 803/s, TTP 787/s."
            ),
        },
        "transcript_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "Phase-2 jump-process opt-in for transcription initiation. "
                "'discrete' (default, legacy): multinomial event distribution "
                "with exact Σ N_i = n_target — bit-identical to baseline. "
                "'poisson': per-promoter Poisson(n_target · p_i) tau-leap; "
                "each promoter becomes an independent continuous-time jump "
                "process whose per-tick marginal Phase-3 inference can "
                "integrate against. Resource cap is the actual inactive-RNAP "
                "pool, not the discrete-time target (avoids 12% asymmetric-"
                "truncation undercount)."
            ),
        },
        "polypeptide_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "Phase-2 jump-process opt-in for translation initiation. "
                "Same dispatch as transcript_initiation_mode but for "
                "PolypeptideInitiation; ribosome activation per protein "
                "becomes per-protein Poisson(n_target · p_i) tau-leap "
                "with the resource cap pinned to min(30S, 50S) instead "
                "of the discrete-time target."
            ),
        },
        "ref_growth_feedback_tau_s": {
            "type": "number", "default": 1.0,
            "description": (
                "Feedback smoothing time-constant for the "
                "consumption_matched ref-growth driver. ``1.0`` (default) "
                "keeps the legacy tight per-tick controller — the driver "
                "fully compensates last tick's variance, so the PDMP "
                "ensemble's per-tick jump-process variance is invisible "
                "at cell_mass. Larger values smooth the consumption "
                "estimate (EMA) so per-tick stochasticity manifests in "
                "pool counts and downstream in mass; the long-run mean "
                "still tracks the kFBA-measured growth rate. Try 60 s "
                "for a 1-minute-window controller."
            ),
        },
        "ref_growth_feedback_period_ticks": {
            "type": "integer", "default": 1,
            "description": (
                "Sprint-8 sparse-injection knob for the "
                "consumption_matched ref-growth driver. ``1`` (default) "
                "acts every tick. Larger values decimate the controller — "
                "it acts every Nth tick, compensating the cumulative "
                "consumption with a period-scaled target injection. "
                "Pools drift open-loop between corrections, letting per-"
                "tick Poisson variance manifest in the PDMP ensemble. "
                "Bypasses the EMA when > 1 (one knob at a time)."
            ),
        },
        "transcript_init_prob_scale": {
            "type": "number", "default": 1.0,
            "description": (
                "Phase-3 sprint-7 ABC-SMC knob. In poisson mode, "
                "multiplies the per-promoter initiation rate by this "
                "scalar before sampling. Default 1.0 reproduces the "
                "unperturbed sampler; values away from 1.0 produce "
                "ensembles at distinguishable parameter settings for "
                "the ABC-SMC inference stub. Only effective when "
                "transcript_initiation_mode='poisson'."
            ),
        },
        "polypeptide_init_prob_scale": {
            "type": "number", "default": 1.0,
            "description": (
                "Phase-3 sprint-10 ABC-SMC knob, mirror of "
                "transcript_init_prob_scale on the translation side. "
                "Only effective when polypeptide_initiation_mode="
                "'poisson'."
            ),
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    core_extensions=[_register_millard_pdmp_links],
)
def millard_pdmp_baseline(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    tick_s: float = 1.0,
    backend: str = "basico",
    with_ref_growth: bool = False,
    ref_growth_flux_source: str = "proportional",
    transcript_initiation_mode: str = "discrete",
    polypeptide_initiation_mode: str = "discrete",
    ref_growth_feedback_tau_s: float = 1.0,
    ref_growth_feedback_period_ticks: int = 1,
    transcript_init_prob_scale: float = 1.0,
    polypeptide_init_prob_scale: float = 1.0,
) -> dict:
    """Build the process-bigraph state document for the Millard-PDMP baseline."""
    if core is None:
        core = build_core()
    _register_millard_pdmp_links(core)

    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

    features = list(DEFAULT_FEATURES)
    if with_ref_growth and 'ref_growth_driver' not in features:
        features.append('ref_growth_driver')

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    for store in ['listeners', 'process',
                  'allocator_rng', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault('mass', {'dry_mass': 0.0, 'cell_mass': 0.0})
    cell_state.setdefault('allocator_rng', np.random.RandomState(seed=seed))

    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'enabled': False,
    })

    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    cell_state.setdefault('request', {})
    cell_state.setdefault('allocate', {})
    for proc_name in ALL_PARTITIONED:
        cell_state['request'].setdefault(proc_name, {'bulk': {}})
        cell_state['allocate'].setdefault(proc_name, {'bulk': {}})
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Agent-root central_fluxes store: Millard publishes per-reaction fluxes
    # (mM/s) here and FBAFluxCoupler reads them (its topology is the agent-
    # root ('central_fluxes',), not the shared/ one).
    cell_state.setdefault('central_fluxes', {})

    # Millard-PDMP shared stores
    cell_state.setdefault('shared', {})
    cell_state['shared'].setdefault('central_metabolites', {})
    cell_state['shared'].setdefault('central_fluxes', {})
    cell_state['shared'].setdefault('lqr_control', {'u': 0.0, 'u_dict': {}})
    cell_state['shared'].setdefault('lqr_diagnostics', {})
    cell_state['shared'].setdefault('control_applied', {})
    cell_state['shared'].setdefault('bridge_diagnostics', {})
    # FBABridge writeback target — kept under 'shared/' (same parent as
    # 'central_metabolites') because top-level dict stores don't accept
    # dict-merge updates in process-bigraph the way nested map stores do
    # (verified empirically: 22 entries the bridge emitted never landed in
    # a top-level dict, but identical wiring through 'shared/' works).
    cell_state['shared'].setdefault('central_metabolite_counts', {})

    # Medium-exchange store the Millard edge writes (signed per-tick counts);
    # map[float] accumulates. Read by the mass-conservation deriver / reactor
    # coupler. Declared so the topology resolves even with no live reactor.
    cell_state.setdefault('environment', {})
    cell_state['environment'].setdefault('exchange', {})

    # Mock loader: cache configs + minimal sim_data (see _helpers.CachedConfigLoader).
    loader = CachedConfigLoader(configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)

    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    _process_cache = {}
    _millard_builders = {
        'millard-pdmp-metabolism': _build_millard_pdmp_edge,
        'millard-with-lqr':        _build_millard_with_lqr_edge,
        'lqr-controller':          _build_lqr_controller_edge,
        'fba-bridge':              _build_fba_bridge_edge,
    }
    for step_name in flow_order:
        if step_name in _millard_builders:
            if step_name == 'millard-pdmp-metabolism':
                cell_state[step_name] = _millard_builders[step_name](
                    core, tick_s=tick_s, backend=backend)
            else:
                cell_state[step_name] = _millard_builders[step_name](
                    core, tick_s=tick_s)
            continue
        config = _get_step_config(
            loader, step_name, core, _process_cache, master_seed=seed,
            ref_growth_flux_source=ref_growth_flux_source,
            ref_growth_feedback_tau_s=ref_growth_feedback_tau_s,
            ref_growth_feedback_period_ticks=ref_growth_feedback_period_ticks,
            transcript_initiation_mode=transcript_initiation_mode,
            transcript_init_prob_scale=transcript_init_prob_scale,
            polypeptide_initiation_mode=polypeptide_initiation_mode,
            polypeptide_init_prob_scale=polypeptide_init_prob_scale,
        )
        if config is not None:
            if len(config) == 5:
                instance, topology, edge_type, in_topo, out_topo = config
                cell_state[step_name] = make_edge(
                    instance, topology, input_topology=in_topo,
                    output_topology=out_topo, edge_type=edge_type)
            else:
                instance, topology, edge_type = config
                cell_state[step_name] = make_edge(
                    instance, topology, edge_type=edge_type)

    for proc_name, proc_instance in _process_cache.items():
        cell_state['process'][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    seed_mass_listener(cell_state, core)

    inject_flow_dependencies(
        cell_state, flow_order, layers=execution_layers)

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': flow_order,
    }
