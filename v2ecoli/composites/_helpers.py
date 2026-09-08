"""Shared helpers for the v2ecoli composite generators.

Exported names (all are considered semi-private implementation details):
  - make_edge
  - inject_flow_dependencies
  - _seed_state_from_defaults
  - seed_mass_listener
  - _normalize_boundary_units
  - _make_instance
  - _get_special_step
  - _expand_flushes
  - FLUSH
  - PARTITIONED_PROCESSES
  - ALL_PARTITIONED
  - ALLOCATOR_LAYERS
"""

from __future__ import annotations

import copy
import warnings


# Framework-generic per-agent emitter-lifecycle registry (register / get /
# unregister + finalize) now lives in pbg-emitters. v2ecoli re-exports the
# parquet-named wrappers below so existing call sites keep working.
from viva_emitters.lifecycle import (
    register_emitter as register_parquet_emitter,
    get_emitter as get_parquet_emitter,  # noqa: F401 — re-exported for external call sites
    unregister_emitter as _unregister_emitter,
    finalize_emitter_for_agent,  # noqa: F401 — re-exported for external call sites
)

# ---------------------------------------------------------------------------
# Process imports (needed for PARTITIONED_PROCESSES and _instantiate_step)
# ---------------------------------------------------------------------------
from v2ecoli.processes.rna_degradation import RnaDegradation
from v2ecoli.processes.transcript_elongation import TranscriptElongation
from v2ecoli.processes.polypeptide_elongation import SteadyStatePolypeptideElongation


# ---------------------------------------------------------------------------
# FLUSH sentinel and helper
# ---------------------------------------------------------------------------

# FLUSH marks a position where the UniqueNumpyUpdater buffer should be
# drained before the next layer runs.  Expanded to a real step by
# _expand_flushes() at build time.
FLUSH = '__unique_flush__'


# --- Cache-driven config loader -------------------------------------------
# Composite generators build from a cache bundle (pre-resolved configs) rather
# than a live ParCa sim_data object. CachedConfigLoader is the small stand-in
# they pass to step builders: it serves configs by name and exposes the few
# sim_data attributes those builders read (unique-molecule definitions +
# expected dry-mass increase). Previously each generator (baseline,
# baseline_millard) defined its own nested
# _CachedLoader with a 4-level inner-class mock; this is the single flattened,
# module-level version.

class _MockUniqueMolecule:
    def __init__(self, names):
        self.unique_molecule_definitions = {n: {} for n in names}


class _MockInternalState:
    def __init__(self, names):
        self.unique_molecule = _MockUniqueMolecule(names)


class _MockSimData:
    def __init__(self, unique_names, dry_mass_inc_dict):
        self.internal_state = _MockInternalState(unique_names)
        self.expectedDryMassIncreaseDict = dry_mass_inc_dict or {}


class CachedConfigLoader:
    """Serve pre-resolved configs from the cache bundle + a minimal sim_data
    stand-in. Replaces the per-generator nested ``_CachedLoader``."""

    def __init__(self, configs, unique_names, dry_mass_inc_dict,
                 cache_dir='out/cache'):
        self._configs = configs
        self.unique_names = unique_names
        self.cache_dir = cache_dir
        self.sim_data = _MockSimData(unique_names, dry_mass_inc_dict)

    def get_config_by_name(self, name):
        try:
            return self._configs[name]
        except KeyError:
            raise KeyError(f'Unknown: {name}')


def _expand_flushes(layers):
    """Replace each FLUSH sentinel with a real [unique_update_N] sub-layer.

    N is assigned in declaration order so state keys stay stable across
    architecture variants (baseline, colony, baseline_millard).
    """
    out, n = [], 0
    for layer in layers:
        if layer == FLUSH:
            n += 1
            out.append([f'unique_update_{n}'])
        else:
            out.append(layer)
    return out


# ---------------------------------------------------------------------------
# Partitioned process registry
# ---------------------------------------------------------------------------

PARTITIONED_PROCESSES = {
    'ecoli-rna-degradation': RnaDegradation,
    'ecoli-transcript-elongation': TranscriptElongation,
    'ecoli-polypeptide-elongation': SteadyStatePolypeptideElongation,
}

ALL_PARTITIONED = list(PARTITIONED_PROCESSES.keys())


# ---------------------------------------------------------------------------
# Canonical visualization set for single-cell architectures.
# ---------------------------------------------------------------------------
# Shared by baseline / baseline_millard — both resolve to the
# same observables.mass / unique-molecule layout so the same viz tiles apply.
# Opt-in: every study built on one of the v2ecoli single-cell composites
# (baseline / baseline_millard) that wants the legacy
# WorkflowVisualization + NetworkVisualization panels must declare them
# explicitly in its study.yaml `visualizations:` block. Auto-attaching them
# here used to be the default, but the panels rendered confusingly empty for
# planning-not-yet-run studies and for any study whose runs.db wasn't
# populated yet. Re-enable per study by copying the two-entry list below into
# the study's spec, OR opt back into project-wide auto-attach by setting
# ``visualizations=v2ecoli_default_single_cell_visualizations()`` on the
# specific @composite_generator call.
#
# History: the earlier port-wired TimeSeriesPlots came out with empty y-data
# (build_viz_composite only honors `inputs_map` for port→observable wiring); a
# self-contained TimeSeriesFromObservables set that read runs.db directly
# replaced them, but those were generic line plots. The current set renders the
# REAL native analyses via ParquetAnalysisView (see below), which reads the
# run's hive-partitioned parquet sweep + hydrated ParCa sim_data — the same
# analyses that produced the showcase-2 figure gallery.
DEFAULT_SINGLE_CELL_VISUALIZATIONS: list[dict] = [
    # Native-analysis gallery: each panel is a real v2ecoli analysis
    # (v2ecoli/workflow/analyses/) rendered by ParquetAnalysisView from the run's
    # hive-partitioned parquet sweep (installed by the dashboard's
    # inject_analysis_parquet_emitters) + a hydrated ParCa sim_data. The
    # dashboard injects `sweep_dir` + `sim_data_path` per run. These reproduce the
    # showcase-2 figure gallery live in the Composite Explorer instead of the
    # generic observable line plots. An analysis whose scale needs more cells than
    # a one-off run produced (multivariant/multiseed) degrades to the single
    # available cell; ParquetAnalysisView renders an explanatory panel on any
    # missing prerequisite rather than failing the whole tab.
    {
        'name': 'mass_fraction_summary',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Mass fraction summary',
                   'analysis': 'mass_fraction_summary_view'},
    },
    {
        'name': 'cell_mass',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Cell mass over time', 'analysis': 'cell_mass'},
    },
    {
        'name': 'mass_fraction_voronoi',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Mass composition (Voronoi)',
                   'analysis': 'mass_fraction_voronoi'},
    },
    {
        'name': 'replication',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Chromosome replication', 'analysis': 'replication'},
    },
    {
        'name': 'chromosome_state',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Chromosome state (animated)',
                   'analysis': 'chromosome_state_view'},
    },
    {
        'name': 'ribosome_components',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Ribosome components',
                   'analysis': 'ribosome_components'},
    },
    {
        'name': 'central_carbon_metabolism',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Central carbon metabolism (FBA)',
                   'analysis': 'central_carbon_metabolism_scatter'},
    },
    {
        'name': 'protein_counts_validation',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Protein counts vs. proteomics',
                   'analysis': 'protein_counts_validation'},
    },
    {
        'name': 'doubling_time',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Doubling time', 'analysis': 'doubling_time_line'},
    },
]


# A seeds × generations batch produces the cell groupings the single-cell set
# can only degrade to, so it adds the multigeneration + multiseed analyses on
# top of it — the same figures vEcoli's workflow emits from its multiGeneration
# and multiSeed analysis channels. Same rendering path (ParquetAnalysisView over
# the run's hive parquet sweep + hydrated ParCa sim_data), so a panel whose scale
# the batch didn't actually cover (a 1-seed run asking for multiseed) renders an
# explanatory panel rather than failing the tab.
DEFAULT_BATCH_VISUALIZATIONS: list[dict] = DEFAULT_SINGLE_CELL_VISUALIZATIONS + [
    {
        'name': 'mass_growth_across_generations',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Mass growth across generations',
                   'analysis': 'mass_growth_across_generations'},
    },
    {
        'name': 'ribosome_usage',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Ribosome usage (lineage)',
                   'analysis': 'ribosome_usage'},
    },
    {
        'name': 'ribosome_production',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Ribosome production (lineage)',
                   'analysis': 'ribosome_production'},
    },
    {
        'name': 'new_gene_counts',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'New gene counts (lineage)',
                   'analysis': 'new_gene_counts'},
    },
    {
        'name': 'doubling_time_distribution',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Doubling time distribution (across seeds)',
                   'analysis': 'doubling_time_distribution'},
    },
    {
        'name': 'subgenerational_expression',
        'address': 'local:!v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView',
        'config': {'title': 'Subgenerational expression (across seeds)',
                   'analysis': 'subgenerational_expression_table'},
    },
]


def v2ecoli_default_single_cell_visualizations() -> list[dict]:
    """The legacy workflow + topology viz spec, available for explicit opt-in.

    Returns a fresh list each call so the consumer can mutate without
    polluting other studies' specs.
    """
    return [
        {
            'name': 'workflow',
            'address': 'local:!v2ecoli.visualizations.workflow.WorkflowVisualization',
            'config': {'title': 'v2ecoli — single-cell lifecycle'},
        },
        {
            'name': 'topology',
            'address': 'local:!v2ecoli.visualizations.network.NetworkVisualization',
            'config': {'title': 'Process topology'},
        },
    ]

ALLOCATOR_LAYERS = {
    # RNA degradation shares water with polymerizations
    'allocator_2': ['ecoli-rna-degradation'],
    # Elongation processes compete for NTPs / charged tRNAs
    'allocator_3': ['ecoli-polypeptide-elongation',
                    'ecoli-transcript-elongation'],
}


# ---------------------------------------------------------------------------
# Emitter override (module-level)
# ---------------------------------------------------------------------------
# When a Study runner needs persistent time-series history, it can set
# ``_EMITTER_OVERRIDE`` to a dict of SQLiteEmitter-config kwargs (e.g.
# ``{'file_path': '.../workspace/studies/<name>', 'db_file': 'runs.db',
#   'name': 'baseline-steady-state'}``) BEFORE building the composite.
# When set, the special 'emitter' step in baseline /
# baseline_millard swaps RAMEmitter for SQLiteEmitter and expands the emit
# schema to cover the per-listener fields the dnaa investigation reads.
#
# Use the ``sqlite_emitter()`` context manager below for safety —
# ensures the override is cleared even on exceptions.
_EMITTER_OVERRIDE: dict | None = None

# Parallel override for the ParquetEmitter path. When set, the 'emitter' step
# materialises as a v2ecoli ParquetEmitter writing to a hive-partitioned dir,
# independent of the SQLite override above. Both may be set simultaneously
# (parquet wins — sqlite stays available for dashboard's Simulations-DB tab).
_PARQUET_EMITTER_OVERRIDE: dict | None = None

# When True (and no parquet/sqlite override is set), the 'emitter' step
# materialises as a minimal RAMEmitter capturing ONLY global_time — instead of
# the default full-state RAMEmitter (bulk + chromosome + listeners, multi-MB
# per row). Used by callers that emit out-of-band (e.g. the workflow's external
# XArrayEmitter) and don't want the internal emitter wasting memory.
_NULL_EMITTER_OVERRIDE: bool = False

# The generator-declared default emitter (see viva_superpowers
# composite_generator's ``emitters=`` convention + ``emitter_defaults``). A
# ``{address, config, paths?}`` dict that the 'emitter' step materialises when
# NO external override (parquet / sqlite / null) is set — i.e. the composite's
# own default sink travels with the generator instead of being toggled by a
# caller. This is the lowest-priority *declared* source; the RAMEmitter remains
# the final fallback when even this is None. The baseline generator sets it
# from its own ``@composite_generator(emitters=[...])`` entry around the build.
_DEFAULT_EMITTER_DECL: dict | None = None

# Per-agent registry of live ParquetEmitter step instances (framework-generic
# registry imported at module top from ``viva_emitters.lifecycle``). Populated
# when ``_get_special_step('emitter')`` constructs an emitter under a parquet
# override; consulted by ``Division.next_update`` so the parent's trailing
# partial batch can be ``close(success=True)``-d before the agent is
# ``_remove``-d (mirrors vEcoli's pre-divide ``self.emitter.finalize()`` hook
# in ``ecoli/experiments/ecoli_master_sim.py`` / ``engine_process.py``).


def unregister_parquet_emitter(agent_id: str) -> None:
    """Drop the registered emitter for ``agent_id`` (thin pbg-emitters wrapper)."""
    _unregister_emitter(agent_id)


def set_emitter_override(config: dict | None) -> None:
    """Set the module-level SQLite emitter override. Pass None to clear."""
    global _EMITTER_OVERRIDE
    _EMITTER_OVERRIDE = config


def set_parquet_emitter_override(config: dict | None) -> None:
    """Set the module-level Parquet emitter override. Pass None to clear."""
    global _PARQUET_EMITTER_OVERRIDE
    _PARQUET_EMITTER_OVERRIDE = config


def set_null_emitter_override(flag: bool) -> None:
    """Minimise the internal 'emitter' step to global_time only (see
    ``_NULL_EMITTER_OVERRIDE``). Pass False to restore the full-state default."""
    global _NULL_EMITTER_OVERRIDE
    _NULL_EMITTER_OVERRIDE = bool(flag)


# {leaf_name: exchange_key} for the opt-in exchange_flux_listener feature step.
# Set by ecoli_baseline's generator (from its ``exchange_fluxes`` param) around a
# build so _get_special_step can build the config-less feature step with the
# caller's flux map, then restored — same external-override discipline as the
# emitter overrides above. Empty = the listener declares/emits nothing.
_EXCHANGE_FLUXES_OVERRIDE: dict = {}


def set_exchange_fluxes_override(fluxes: dict | None) -> None:
    """Set the flux map the ``exchange_flux_listener`` feature step is built with
    (see ``_EXCHANGE_FLUXES_OVERRIDE``). Pass None/{} to clear."""
    global _EXCHANGE_FLUXES_OVERRIDE
    _EXCHANGE_FLUXES_OVERRIDE = dict(fluxes or {})


# WHICH QUANTITY those leaves carry — "counts" (a lineage-cumulative molecule
# total) or "gdcw" (a per-tick mmol/gDCW/h rate). Threaded the same way as the
# flux map above so a study declares it once. Default preserves the deriver's
# own default, so an undeclared build is unchanged.
_EXCHANGE_FLUX_BASIS_OVERRIDE: str = "counts"


def set_exchange_flux_basis_override(basis: str | None) -> None:
    """Set the ``basis`` the ``exchange_flux_listener`` feature step is built with
    (see ``_EXCHANGE_FLUX_BASIS_OVERRIDE``). Pass None to restore the default."""
    global _EXCHANGE_FLUX_BASIS_OVERRIDE
    _EXCHANGE_FLUX_BASIS_OVERRIDE = str(basis or "counts")


def set_default_emitter_decl(decl: dict | None) -> None:
    """Set the generator-declared default emitter (see ``_DEFAULT_EMITTER_DECL``).

    ``decl`` is a ``{address, config, paths?}`` dict from a generator's
    ``@composite_generator(emitters=[...])`` declaration (read via
    ``viva_superpowers.composite_generator.emitter_defaults``). Pass None to
    clear. The 'emitter' step uses it only when no external parquet / sqlite /
    null override is active. A generator should set this around its build and
    clear it in a ``finally`` so it never leaks into a later composite built in
    the same process.
    """
    global _DEFAULT_EMITTER_DECL
    _DEFAULT_EMITTER_DECL = decl


def _merge_emit_paths(emit_schema: dict, topo: dict, emit_paths) -> None:
    """Add caller-declared EXTRA emit store paths to a parquet emit
    schema/topology, in place — a general, domain-agnostic capability.

    The baseline parquet emitter captures a fixed set (``global_time`` /
    ``bulk`` / ``listeners``). A composite or config that wants to persist
    additional stores declares them via the emitter config's ``emit_paths``: a
    list of store paths, each a sequence of store-node segments (e.g.
    ``["some_store", "sub_key"]`` or ``["compartment", "global", "volume"]``).
    This honors them without any subsystem-specific schema baked into the
    framework — the declaring config owns which stores matter.

    Emit-schema leaf types collapse to ``node`` in the emitter, so only the
    topology (store roots) selects what is captured: each path's first segment
    becomes a top-level emit key wired to its own store root, and the nested
    schema carries ``node`` at the path's leaf. Paths sharing a prefix merge.
    """
    for path in emit_paths or []:
        segments = [str(s) for s in path]
        if not segments:
            continue
        node = emit_schema
        for segment in segments[:-1]:
            child = node.get(segment)
            if not isinstance(child, dict):
                child = {}
                node[segment] = child
            node = child
        leaf = segments[-1]
        if not isinstance(node.get(leaf), dict):
            node[leaf] = "node"
        topo[segments[0]] = (segments[0],)


def _build_declared_emitter(decl: dict, listeners_schema: dict, core,
                             *, allow_ram_fallback: bool = False):
    """Materialise the generator-declared default emitter step.

    Maps ``decl['address']`` (the registered emitter link, with or without a
    ``local:`` prefix) to the right emitter class and an emit-schema/topology
    appropriate for v2ecoli. ``decl['config']`` is merged in as the base
    config. Returns ``(instance, topo)``.

    Only the emitters v2ecoli actually ships are recognised; an unknown
    address raises rather than silently falling back, so a typo in a
    generator's ``emitters=`` declaration surfaces at build time.

    ``allow_ram_fallback`` (default ``False``): when the declared address is
    ``ParquetEmitter`` but ``viva_emitters`` fails to import, the historical
    behavior degraded to an in-memory RAMEmitter with only a warning — a run
    that believed it was persisting parquet to disk would silently keep
    everything in RAM and lose ALL of it on exit, with no error anywhere
    (CD2 pipeline audit finding P1-4, §2.10). ``viva-emitters[parquet,xarray]``
    is a BASE v2ecoli dependency today, so this import failing normally means
    the active environment is stale/incomplete (e.g. ``uv sync`` needed), not
    a genuinely absent optional extra — exactly the case that must not fail
    silently. Set ``allow_ram_fallback=True`` to opt into the old degrade-and-
    warn behavior (tests/debug only).
    """
    from process_bigraph.emitter import RAMEmitter, SQLiteEmitter

    address = (decl.get("address") or "").split(":", 1)[-1]
    cfg_in = dict(decl.get("config") or {})

    if address == "ParquetEmitter":
        try:
            from viva_emitters import ParquetEmitter
        except ImportError as exc:
            if not allow_ram_fallback:
                raise RuntimeError(
                    "generator declared a ParquetEmitter default but "
                    f"`from viva_emitters import ParquetEmitter` failed "
                    f"({exc}). viva-emitters[parquet,xarray] is a BASE "
                    "v2ecoli dependency, so this usually means the active "
                    "venv is stale/incomplete — run `uv sync` — rather than "
                    "a genuinely missing optional extra. Refusing to "
                    "silently degrade to an in-memory RAMEmitter: that would "
                    "let the run complete and exit having persisted NOTHING "
                    "to disk (CD2 pipeline audit finding P1-4). If you "
                    "really want the degraded in-memory behavior (tests/"
                    "debug only), pass allow_ram_fallback=True to "
                    "_build_declared_emitter()."
                ) from exc
            # Explicit opt-in: degrade to the historical full-capture
            # in-memory RAMEmitter — the pre-declaration default — and warn.
            warnings.warn(
                "generator declared a ParquetEmitter default but "
                f"`from viva_emitters import ParquetEmitter` failed ({exc}); "
                "falling back to in-memory RAMEmitter because "
                "allow_ram_fallback=True was passed explicitly. ALL output "
                "will be lost on exit instead of persisted to parquet.")
            emit_schema = {
                "global_time": "float", "bulk": "array",
                "listeners": listeners_schema,
                "full_chromosome": "node", "active_replisome": "node",
                "active_RNAP": "node", "chromosome_domain": "node",
            }
            topo = {
                "global_time": ("global_time",), "bulk": ("bulk",),
                "listeners": ("listeners",),
                "full_chromosome": ("unique", "full_chromosome"),
                "active_replisome": ("unique", "active_replisome"),
                "active_RNAP": ("unique", "active_RNAP"),
                "chromosome_domain": ("unique", "chromosome_domain"),
            }
            return RAMEmitter({"emit": emit_schema}, core), topo
        # Reuse the vEcoli-shaped preset so the hive layout + dtype overrides
        # match upstream, seeded by whatever the declaration provided.
        from v2ecoli.library.emitter_presets import parquet_vecoli
        out_dir = cfg_in.pop("out_dir", None)
        if out_dir is None:
            ws_root = _find_workspace_root()
            out_dir = (str(ws_root / ".pbg" / "parquet-runs")
                       if ws_root is not None else "out/parquet")
        # Thread the run-identity fields the declaration carries into the
        # preset so each cell writes its OWN hive partition. Without this,
        # variant/lineage_seed/agent_id fall back to parquet_vecoli's defaults
        # (variant=0, lineage_seed=0, agent_id="1") and every variant in a
        # multivariant sweep collapses onto partition ``variant=0`` — the
        # multivariant KPI analysis then sees all variants as one. experiment_id
        # likewise defaults to the env var / "default" only when the decl does
        # not supply it (so a declared experiment_id actually reaches the
        # emitter instead of silently no-opping).
        _preset_kwargs = {
            "experiment_id": cfg_in.pop(
                "experiment_id",
                os.environ.get("V2ECOLI_EMITTER_EXPERIMENT_ID", "default")),
        }
        for _idkey in ("variant", "lineage_seed", "agent_id", "generation"):
            if _idkey in cfg_in:
                _preset_kwargs[_idkey] = cfg_in.pop(_idkey)
        preset = parquet_vecoli(out_dir=out_dir, **_preset_kwargs)
        emit_schema = {
            "global_time": "float",
            "bulk": "array[integer]",
            "listeners": listeners_schema,
        }
        topo = {
            "global_time": ("global_time",),
            "bulk": ("bulk",),
            "listeners": ("listeners",),
        }
        # Config-declared EXTRA emit store paths (domain-agnostic; see
        # _merge_emit_paths). Popped so it does not reach the emitter config.
        _merge_emit_paths(emit_schema, topo, cfg_in.pop("emit_paths", None))
        cfg = {"emit": emit_schema, **preset, **cfg_in}
        return ParquetEmitter(cfg, core), topo

    if address == "XArrayEmitter":
        # Single-cell, in-document XArray capture. NOT a raw XArrayEmitter:
        # ``SingleCellXArrayEmitter`` wraps one and defers its construction to the
        # first ``update()`` so the view/output_metadata are discovered from the
        # REALIZED composite state (the pre-realize listener tree only carries a
        # handful of leaves — Task 4 / C2) and the structured ``bulk`` record
        # array is projected to plain counts before emitting (Task 4 / C1). The
        # ``decl['config']`` skeleton (out_uri/metadata/transducer/writer/
        # strategy) flows through ``cfg_in``; the step completes it lazily.
        # Imported here (function-level) to avoid an import cycle with
        # ``ecoli_baseline`` (which imports this module at top level).
        from v2ecoli.composites.ecoli_baseline import SingleCellXArrayEmitter
        topo = {
            "global_time": ("global_time",),
            "bulk": ("bulk",),
            "listeners": ("listeners",),
        }
        return SingleCellXArrayEmitter(cfg_in, core), topo

    if address == "SQLiteEmitter":
        emit_schema = {"global_time": "float", "listeners": listeners_schema}
        topo = {"global_time": ("global_time",), "listeners": ("listeners",)}
        cfg = {"emit": emit_schema, **cfg_in}
        return SQLiteEmitter(cfg, core), topo

    if address == "RAMEmitter":
        emit_schema = {"global_time": "float", "listeners": listeners_schema}
        topo = {"global_time": ("global_time",), "listeners": ("listeners",)}
        cfg = {"emit": emit_schema, **cfg_in}
        return RAMEmitter(cfg, core), topo

    raise ValueError(
        f"declared default emitter address {decl.get('address')!r} is not "
        "recognised (expected one of ParquetEmitter, XArrayEmitter, "
        "SQLiteEmitter, RAMEmitter)"
    )


import contextlib  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402


def _find_workspace_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) looking for ``workspace.yaml``.

    Returns the directory that contains ``workspace.yaml``, or ``None`` if
    none is found before hitting the filesystem root. Cheap, no imports.
    """
    cur = Path(start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / 'workspace.yaml').is_file():
            return d
    return None


def _ensure_study_columns(db_path: str) -> None:
    """Idempotently add study_slug / investigation_slug columns to ``simulations``.

    Safe to call on a missing DB (creates the parent dir + an empty DB with
    just the columns we add — the SQLiteEmitter's _init_history_db will
    create the rest of the schema on its own init).
    """
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        # Mirror the SQLiteEmitter's table layout — without recreating the
        # full schema, we just need the `simulations` table to exist before
        # we ALTER. If SQLiteEmitter has already initialized it, the CREATE
        # is a no-op.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id TEXT PRIMARY KEY,
                name TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                elapsed_seconds REAL,
                composite_config TEXT,
                metadata TEXT,
                emit_schema TEXT
            )
        ''')
        existing = {row[1] for row in conn.execute(
            'PRAGMA table_info(simulations)')}
        if 'study_slug' not in existing:
            conn.execute('ALTER TABLE simulations ADD COLUMN study_slug TEXT')
        if 'investigation_slug' not in existing:
            conn.execute(
                'ALTER TABLE simulations ADD COLUMN investigation_slug TEXT')
    finally:
        conn.close()


def _stamp_study_metadata(
    db_path: str,
    simulation_id: str,
    study_slug: str | None,
    investigation_slug: str | None,
) -> None:
    """Write study_slug + investigation_slug for ``simulation_id``.

    Upserts a row keyed on ``simulation_id`` and leaves any existing
    SQLiteEmitter-managed fields (name, emit_schema, ...) untouched. Use
    after the SQLiteEmitter step has been constructed (which inserts the
    initial row) — or before, in which case ``started_at`` defaults to
    now-UTC and the emitter will overwrite it on its own insert.
    """
    if study_slug is None and investigation_slug is None:
        return
    import datetime
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ')
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            'INSERT INTO simulations '
            '(simulation_id, started_at, study_slug, investigation_slug) '
            'VALUES (?, ?, ?, ?) '
            'ON CONFLICT(simulation_id) DO UPDATE SET '
            '  study_slug = COALESCE(excluded.study_slug, simulations.study_slug), '
            '  investigation_slug = COALESCE(excluded.investigation_slug, simulations.investigation_slug)',
            (simulation_id, now_iso, study_slug, investigation_slug),
        )
    finally:
        conn.close()


@contextlib.contextmanager
def sqlite_emitter(*, file_path: str | None = None,
                   db_file: str | None = None,
                   name: str | None = None,
                   simulation_id: str | None = None,
                   subsample: int = 1,
                   study_slug: str | None = None,
                   investigation_slug: str | None = None):
    """Context manager: build composite with a SQLiteEmitter step.

    By default writes to ``<workspace_root>/.pbg/composite-runs.db`` — the
    same workspace-shared DB the vivarium-workbench's Simulations DB tab
    aggregates from. Pass ``study_slug`` / ``investigation_slug`` to tag the
    run; the slugs are stored as columns on the ``simulations`` row so the
    dashboard can group / filter by them.

    To keep using a per-study DB, pass ``file_path`` (and optionally
    ``db_file``) explicitly — the old behavior is preserved.

    Usage::

        with sqlite_emitter(name='baseline-seed0',
                            study_slug='dnaa-01-expression-dynamics',
                            investigation_slug='dnaa-replication'):
            doc = baseline(cache_dir='out/cache')
            comp = Composite(doc, core=core)
            comp.update({}, 60.0 * 60)
    """
    # Resolve workspace-default DB path when caller didn't pin one.
    if file_path is None:
        ws_root = _find_workspace_root()
        if ws_root is None:
            raise RuntimeError(
                'sqlite_emitter(): no file_path given and no workspace.yaml '
                'found walking up from cwd; either run from inside a '
                'workspace or pass file_path explicitly.')
        file_path = str(ws_root / '.pbg')
        if db_file is None:
            db_file = 'composite-runs.db'
    else:
        if db_file is None:
            db_file = 'runs.db'

    # Pin the simulation_id upfront so we can stamp metadata around the
    # emitter's own init insert.
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())

    cfg = {
        'file_path': file_path,
        'db_file': db_file,
        'simulation_id': simulation_id,
    }
    if name is not None:
        cfg['name'] = name
    if subsample != 1:
        cfg['subsample'] = subsample

    db_path = os.path.join(file_path, db_file)
    # Ensure our extra columns exist on the simulations table BEFORE the
    # SQLiteEmitter step runs its own _init_history_db (which is a no-op for
    # already-existing tables). Then stamp the slugs. Safe to run unconditionally
    # — the helpers are idempotent and tolerate a missing file.
    _ensure_study_columns(db_path)
    _stamp_study_metadata(db_path, simulation_id, study_slug, investigation_slug)

    set_emitter_override(cfg)
    try:
        yield cfg
    finally:
        # Re-stamp in case the emitter's own insert ran after ours and we
        # need to ensure the slugs survived. ON CONFLICT DO UPDATE preserves
        # any non-NULL value, so this is a belt-and-suspenders no-op when
        # things already stuck.
        try:
            _stamp_study_metadata(
                db_path, simulation_id, study_slug, investigation_slug)
        except sqlite3.OperationalError:
            pass
        set_emitter_override(None)


class ParquetEmitterContext:
    """Yielded from ``parquet_emitter()``. Bind the composite via
    :meth:`bind` so its accumulated rows flush on context exit; this
    closes the friction-#3 hole where the context manager couldn't
    enforce its own lifecycle correctness (only ``configuration/`` landed,
    no ``history/`` or ``success/``).

    .cfg holds the emitter config dict (back-compat with the dict that
    pre-2026-05-28 ``parquet_emitter()`` yielded directly).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._composite = None
        self._flushed = False

    def bind(self, composite) -> None:
        """Register the composite for auto-flush on context exit."""
        self._composite = composite

    def flush(self, composite=None, *, success: bool = True) -> int:
        """Flush now. Uses the bound composite if ``composite`` is None.

        Returns the number of ParquetEmitter instances flushed. Safe to
        call multiple times — ``flush_all_in_composite`` skips already-
        closed instances."""
        target = composite if composite is not None else self._composite
        if target is None:
            raise ValueError(
                "ParquetEmitterContext.flush(): no composite given and "
                "none bound; call .bind(composite) first.")
        n = flush_parquet(target, success=success)
        self._flushed = True
        return n


@contextlib.contextmanager
def parquet_emitter(*, out_dir: str | None = None,
                    experiment_id: str | None = None,
                    variant: int = 0,
                    lineage_seed: int = 0,
                    generation: int | None = None,
                    agent_id: str = "1",
                    batch_size: int = 400,
                    threaded: bool = True,
                    study_slug: str | None = None,
                    investigation_slug: str | None = None,
                    extra_metadata: dict | None = None):
    """Context manager: build composite with a ParquetEmitter step.

    Workspace-default ``out_dir`` is ``<workspace_root>/.pbg/parquet-runs/``.
    Hive partitioning matches vEcoli's layout
    (``experiment_id=.../variant=.../lineage_seed=.../generation=.../agent_id=...``),
    so anyone with a vEcoli analysis pipeline can point at the output dir
    directly.

    Usage (auto-flush — recommended, since 2026-05-28)::

        with parquet_emitter(experiment_id='baseline-seed0',
                             study_slug='dnaa-01-expression-dynamics') as emit:
            doc = baseline(cache_dir='out/cache')
            comp = Composite(doc, core=core)
            emit.bind(comp)             # <- enables auto-flush on exit
            comp.update({}, 60.0 * 60)
        # ParquetEmitter.close(success=True) called automatically; if an
        # exception fired in the with-block, close(success=False) instead.

    Legacy usage (still works for callers that already manage the flush)::

        with parquet_emitter(...):
            comp = ...
            comp.update(...)
            flush_parquet(comp, success=True)   # explicit, no .bind()

    If you neither call .bind() nor flush_parquet(), the trailing partial
    batch and the success sentinel are lost: the override clears but the
    final rows never flush. Always .bind() (or call flush_parquet explicitly).

    Storage trade-off vs ``sqlite_emitter()``: Parquet is column-oriented and
    typically 3-5x smaller on disk for v2ecoli-shaped runs (sparse arrays,
    listener-heavy schema). The dashboard's Simulations-DB tab reads the sqlite
    DB, so use ``sqlite_emitter()`` if you need a run to appear there; analyses
    and the per-study parquet views read the parquet output directly.
    """
    if out_dir is None:
        ws_root = _find_workspace_root()
        if ws_root is None:
            raise RuntimeError(
                "parquet_emitter(): no out_dir given and no workspace.yaml "
                "found walking up from cwd; either run from inside a "
                "workspace or pass out_dir explicitly.")
        out_dir = str(ws_root / ".pbg" / "parquet-runs")

    if experiment_id is None:
        experiment_id = (os.environ.get("V2ECOLI_EMITTER_EXPERIMENT_ID")
                         or ("default-" + uuid.uuid4().hex[:8]))

    # Build the emitter config dict via the vEcoli-shaped preset so the
    # hive layout + dtype overrides match upstream conventions exactly.
    from v2ecoli.library.emitter_presets import parquet_vecoli
    cfg = parquet_vecoli(
        out_dir=out_dir,
        experiment_id=experiment_id,
        variant=variant,
        lineage_seed=lineage_seed,
        agent_id=agent_id,
        generation=generation,
        batch_size=batch_size,
        threaded=threaded,
        extra_metadata=extra_metadata,
    )
    # Carry the study/investigation slugs in metadata so downstream readers
    # can group hive-partitioned runs by study without crossing into the
    # SQLite simulations-table convention.
    if study_slug or investigation_slug:
        meta = dict(cfg.get("metadata") or {})
        if study_slug:
            meta["study_slug"] = study_slug
        if investigation_slug:
            meta["investigation_slug"] = investigation_slug
        cfg["metadata"] = meta

    ctx = ParquetEmitterContext(cfg)
    set_parquet_emitter_override(cfg)
    exc_fired = False
    try:
        yield ctx
    except BaseException:
        exc_fired = True
        raise
    finally:
        # Auto-flush if bound but not explicitly flushed. On exception,
        # close with success=False so the sentinel honestly reflects the
        # failed run. Swallow auto-flush errors so they can't mask the
        # caller's original exception.
        if ctx._composite is not None and not ctx._flushed:
            try:
                ctx.flush(success=not exc_fired)
            except Exception:
                pass
        set_parquet_emitter_override(None)


def flush_parquet(composite, *, success: bool = True) -> int:
    """Flush all ParquetEmitter steps inside ``composite`` to disk.

    The ``parquet_emitter()`` context manager only sets / clears the global
    override — it never sees the composite, so it cannot trigger ``close()``
    on the emitter step. Without an explicit flush the trailing partial
    batch (rows after the last batch_size flush) stays in memory; only the
    ``configuration/`` parquet lands, no ``history/`` or ``success/``.

    Call this right before the runner exits its ``with parquet_emitter(...)``
    block, after the simulation loop completes::

        with parquet_emitter(out_dir=..., experiment_id=...):
            composite = build_composite(...)
            ... sim loop ...
            flush_parquet(composite, success=True)

    Returns the number of ParquetEmitter instances flushed (0 when the
    [parquet] extra is not installed or the composite has no parquet
    emitter step).
    """
    try:
        from viva_emitters import ParquetEmitter
    except ImportError:
        return 0
    return ParquetEmitter.flush_all_in_composite(composite, success=success)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _seed_state_from_defaults(cell_state):
    """Walk each edge's port_defaults and inject values into cell_state.

    Each step instance provides port_defaults() which returns a nested dict
    of default values for ports that need pre-population.
    """
    for edge in list(cell_state.values()):
        if not (isinstance(edge, dict) and 'instance' in edge):
            continue
        instance = edge['instance']
        try:
            defaults = instance.port_defaults()
        except (AttributeError, Exception):
            continue
        if not defaults:
            continue
        for port_name, wire_path in edge.get('inputs', {}).items():
            port_default = defaults.get(port_name)
            if port_default is None or not isinstance(wire_path, list):
                continue
            if isinstance(port_default, dict):
                _inject_nested_defaults(cell_state, wire_path, port_default)
            else:
                _inject_port_default(cell_state, wire_path, {'_default': port_default})


def _inject_nested_defaults(state, wire_path, defaults_dict):
    """Recursively inject nested default values into state."""
    target = state
    for segment in wire_path:
        if not isinstance(target, dict):
            return
        target = target.setdefault(segment, {})
    if not isinstance(target, dict):
        return
    for key, val in defaults_dict.items():
        if isinstance(val, dict):
            sub = target.setdefault(key, {})
            if isinstance(sub, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, dict):
                        sub2 = sub.setdefault(k2, {})
                        if isinstance(sub2, dict):
                            for k3, v3 in v2.items():
                                sub2.setdefault(k3, v3)
                    else:
                        sub.setdefault(k2, v2)
        else:
            target.setdefault(key, val)


def _inject_port_default(state, wire_path, port_schema):
    """Inject _default values along wire_path into state."""
    if '_default' in port_schema:
        default = port_schema['_default']
        target = state
        for segment in wire_path[:-1]:
            if not isinstance(target, dict):
                return
            target = target.setdefault(segment, {})
        if isinstance(target, dict) and wire_path:
            key = wire_path[-1]
            current = target.get(key)
            if current is None or (
                    isinstance(current, (list, dict, tuple))
                    and len(current) == 0):
                target[key] = default
        return

    target = state
    for segment in wire_path:
        if not isinstance(target, dict):
            return
        target = target.setdefault(segment, {})
    if not isinstance(target, dict):
        return
    for key, subport in port_schema.items():
        if key.startswith('_') or key == '*' or not isinstance(subport, dict):
            continue
        _inject_port_default(target, [key], subport)


def seed_mass_listener(cell_state, core):
    """Run mass listener once to populate initial mass values."""
    for name in ['post-division-mass-listener', 'ecoli-mass-listener']:
        edge = cell_state.get(name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'next_update'):
            continue

        view = {}
        wires = edge.get('inputs', {})
        for port, wire in wires.items():
            if isinstance(wire, list) and wire:
                target = cell_state
                for seg in wire:
                    if isinstance(target, dict):
                        target = target.get(seg)
                    else:
                        target = None
                        break
                if target is not None:
                    view[port] = target

        for key in ['bulk']:
            arr = view.get(key)
            if arr is not None and hasattr(arr, 'flags'):
                try:
                    arr.flags.writeable = True
                except ValueError:
                    view[key] = arr.copy()
                    view[key].flags.writeable = True
        for uname, uarr in view.get('unique', {}).items():
            if hasattr(uarr, 'flags'):
                try:
                    uarr.flags.writeable = True
                except ValueError:
                    pass

        try:
            delta = instance.next_update(1.0, view)
            if delta and 'listeners' in delta:
                mass = delta['listeners'].get('mass', {})
                cell_state['listeners']['mass'].update(mass)
        except Exception as exc:
            # Don't crash composite construction if the mass listener fails
            # on partial state — but surface the failure. A silent miss here
            # leaves cell_mass=0, which surfaces several layers downstream
            # as a non-finite y_init in Equilibrium's ODE solver.
            warnings.warn(
                f"seed_mass_listener({name}) failed: "
                f"{type(exc).__name__}: {exc}. "
                f"cell_mass / dry_mass left unset.",
                RuntimeWarning,
                stacklevel=2,
            )
        break


def list_paths(path):
    """Convert tuple paths to list paths. Flatten _path dicts."""
    if isinstance(path, tuple):
        return list(path)
    elif isinstance(path, dict):
        if '_path' in path:
            result = {}
            for key, subpath in path.items():
                if key == '_path':
                    continue
                result[key] = list_paths(subpath)
            return result
        return {key: list_paths(subpath) for key, subpath in path.items()}
    return path


def inject_flow_dependencies(cell_state, flow_order, layers=None):
    """Add flow token wiring and priorities to enforce execution order.

    When layers is provided, uses layer-based tokens: all steps in a layer
    depend on the previous layer's token and produce the current layer's
    token.
    """
    if layers is None:
        n = len(flow_order)
        for i, step_name in enumerate(flow_order):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                continue
            edge['priority'] = float(n - i)
            if i == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']
            if i > 0:
                edge.setdefault('inputs', {})[f'_flow_in_{i}'] = [f'_flow_token_{i-1}']
            if i < n - 1:
                edge.setdefault('outputs', {})[f'_flow_out_{i}'] = [f'_flow_token_{i}']
        return

    n_layers = len(layers)
    step_idx = 0
    total_steps = sum(len(layer) for layer in layers)

    for layer_idx, layer in enumerate(layers):
        for j, step_name in enumerate(layer):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                step_idx += 1
                continue

            edge['priority'] = float(total_steps - step_idx)

            if layer_idx == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']

            if layer_idx > 0:
                edge.setdefault('inputs', {})[f'_layer_in_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx - 1}']

            if layer_idx < n_layers - 1:
                edge.setdefault('outputs', {})[f'_layer_out_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx}']

            step_idx += 1


def make_edge(instance, topology, input_topology=None, output_topology=None,
              edge_type='step', config=None):
    """Create an edge dict for a process/step instance."""
    wires = list_paths(topology)
    in_wires = list_paths(input_topology) if input_topology is not None else wires
    out_wires = list_paths(output_topology) if output_topology is not None else wires
    state = {'priority': 1.0} if edge_type == 'step' else {'interval': 1.0}

    inputs_schema = {}
    outputs_schema = {}
    if hasattr(instance, 'inputs'):
        try:
            inputs_schema = instance.inputs()
        except Exception:
            pass
    if hasattr(instance, 'outputs'):
        try:
            outputs_schema = instance.outputs()
        except Exception:
            pass

    cls = type(instance)
    address = f'local:{cls.__module__}.{cls.__qualname__}'
    raw_config = config or getattr(instance, '_raw_config', {})

    state.update({
        '_type': edge_type,
        'address': address,
        'config': raw_config,
        '_inputs': inputs_schema,
        '_outputs': outputs_schema,
        'instance': instance,
        'inputs': copy.deepcopy(in_wires),
        'outputs': copy.deepcopy(out_wires),
    })
    return state


def _normalize_boundary_units(cell_state):
    """Re-create pint Quantities in boundary.external with the current registry."""
    boundary = cell_state.get('boundary', {})
    external = boundary.get('external', {})
    if not isinstance(external, dict):
        return
    from v2ecoli.library.unit_bridge import unum_to_pint
    for key, val in external.items():
        q = unum_to_pint(val)
        if hasattr(q, 'magnitude') and hasattr(q, 'units'):
            external[key] = float(q.magnitude)


# ---------------------------------------------------------------------------
# Step instantiation helpers
# ---------------------------------------------------------------------------

def _make_instance(cls, config, core):
    """Instantiate a Step/Process class, trying multiple signatures."""
    from v2ecoli.library.ecoli_step import set_current_core
    set_current_core(core)
    try:
        return cls(parameters=config)
    except TypeError:
        try:
            return cls(config=config, core=core)
        except TypeError:
            return cls(config)
    finally:
        set_current_core(None)


def _get_special_step(loader, step_name, core):
    """Handle steps that aren't in LoadSimData's config map."""
    from v2ecoli.steps.unique_update import UniqueUpdate

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule
        .unique_molecule_definitions.keys())

    if step_name.startswith('unique_update'):
        UNIQUE_PLURAL = {
            'full_chromosome': 'full_chromosomes',
            'chromosome_domain': 'chromosome_domains',
            'active_replisome': 'active_replisomes',
            'oriC': 'oriCs',
            'promoter': 'promoters',
            'chromosomal_segment': 'chromosomal_segments',
            'DnaA_box': 'DnaA_boxes',
            'active_RNAP': 'active_RNAPs',
            'RNA': 'RNAs',
            'gene': 'genes',
            'active_ribosome': 'active_ribosome',
        }
        unique_topo_v1 = {}
        unique_names_v1 = []
        for name in unique_names:
            plural = UNIQUE_PLURAL.get(name, name)
            unique_topo_v1[plural] = ('unique', name)
            unique_names_v1.append(plural)
        config = {'unique_names': unique_names_v1, 'unique_topo': unique_topo_v1}
        instance = _make_instance(UniqueUpdate, config, core)
        return instance, unique_topo_v1, 'step'

    if step_name == 'ecoli-mass-conservation':
        from v2ecoli.steps.derivers.mass_conservation_deriver import MassConservationDeriver
        from v2ecoli.types.quantity import ureg as units
        # Per-molecule masses (fg/count) of the metabolic exchange molecules,
        # keyed by their environment name (compartment-stripped), so the
        # listener can convert per-tick exchange counts into a mass.
        exchange_masses = {}
        try:
            from v2ecoli.library.unit_bridge import unum_to_pint
            from v2ecoli.library.config_resolver import resolve_config
            # resolve_config realizes the method specs (get_masses) into callables.
            met_cfg = resolve_config(loader.get_config_by_name('ecoli-metabolism'))
            exchange_molecules = list(met_cfg['exchange_molecules'])
            mws = unum_to_pint(met_cfg['get_masses'](exchange_molecules))
            n_avogadro = 6.02214076e23  # molecules / mol
            for mol, mw in zip(exchange_molecules, mws):
                g_per_molecule = mw.to(units.g / units.mol).magnitude / n_avogadro
                exchange_masses[mol[:-3]] = (g_per_molecule * units.g).to(units.fg)
        except Exception:
            # Listener still runs (residual == Δdry_mass) if masses are
            # unavailable; better an observable signal than a build failure.
            exchange_masses = {}
        # tolerance + warmup_ticks fall back to the Step's schema defaults
        # (5% cumulative drift, 10-tick warmup).
        config = {'exchange_masses': exchange_masses}
        instance = _make_instance(MassConservationDeriver, config, core)
        return instance, instance.topology, 'step'

    if step_name == 'exchange_flux_listener':
        from v2ecoli.steps.derivers.exchange_flux_listener import ExchangeFluxListener
        # Flux map comes from the external override the generator sets from its
        # ``exchange_fluxes`` param (same discipline as the emitter overrides).
        config = {'fluxes': dict(_EXCHANGE_FLUXES_OVERRIDE),
                  'basis': str(_EXCHANGE_FLUX_BASIS_OVERRIDE)}
        instance = _make_instance(ExchangeFluxListener, config, core)
        return instance, instance.topology, 'step'

    if step_name == 'global_clock':
        from v2ecoli.steps.global_clock import GlobalClock
        instance = GlobalClock(config={}, core=core)
        topo = {
            'global_time': ('global_time',),
            'next_update_time': ('next_update_time',),
        }
        return instance, topo, 'process'

    if step_name == 'emitter':
        from process_bigraph.emitter import RAMEmitter, SQLiteEmitter
        # ParquetEmitter is optional — the import lives behind an extra so
        # workspaces without the [parquet] extra still build composites.
        # Imported directly from pbg-emitters (the upstream library);
        # ``v2ecoli.library.parquet_emitter`` is just a re-export shim.
        try:
            from viva_emitters import ParquetEmitter
        except ImportError:
            ParquetEmitter = None  # type: ignore[assignment]
        # Mass listener fields — always emitted, used by the workflow report.
        mass_schema = {
            'cell_mass': 'float',
            'water_mass': 'float',
            'dry_mass': 'float',
            'protein_mass': 'float',
            'rna_mass': 'float',
            'rRna_mass': 'float',
            'tRna_mass': 'float',
            'mRna_mass': 'float',
            'dna_mass': 'float',
            'smallMolecule_mass': 'float',
            'instantaneous_growth_rate': 'float',
            'volume': 'float',
        }
        listeners_schema = {'mass': mass_schema}

        # When a caller-set override is active (typical for study runs that
        # need persistent SQLite history), expand the emit schema to capture
        # the dnaa-investigation readouts: monomer_counts / rna_counts /
        # rna_synth_prob / rnap_data. Cheap to add — they're already in the
        # per-step state tree.
        # Parquet override wins over SQLite — see set_parquet_emitter_override.
        parquet_override = _PARQUET_EMITTER_OVERRIDE
        override = _EMITTER_OVERRIDE
        default_decl = _DEFAULT_EMITTER_DECL
        if (parquet_override is not None or override is not None
                or default_decl is not None):
            # Only the leaves the dnaa-investigation readouts actually consume.
            # NB: ``monomer_counts`` is a flat array store (not nested under
            # a ``monomerCounts`` key) — confirmed at runtime via the
            # composite-emitted state shape.
            listeners_schema.update({
                'monomer_counts': 'array[integer]',
                'rna_counts': {
                    'mRNA_counts': 'array[integer]',
                },
                'rna_synth_prob': {
                    'n_bound_TF_per_TU': 'array[float]',
                    'n_actual_bound': 'array[integer]',
                    'total_rna_init': 'integer',
                },
                'rnap_data': {
                    'rna_init_event': 'array[integer]',
                    'did_initialize': 'integer',
                    # Phase-3 sprint-1 per-tick log-likelihood; needed
                    # alongside rna_init_event so post-hoc inference
                    # can compare observation to model.
                    'log_likelihood': 'float',
                },
                'ribosome_data': {
                    # Phase-3 sprint-2 per-tick log-likelihood from
                    # PolypeptideInitiation.
                    'log_likelihood': 'float',
                },
                'replication_data': {
                    'number_of_oric': 'integer',
                    'free_DnaA_boxes': 'integer',
                    'total_DnaA_boxes': 'integer',
                    # Per-tick replication-fork genomic positions (variable length:
                    # 0 before initiation, 2/4/… after). Emitted as a ragged list
                    # so the chromosome-state GIF can place the replisomes; the
                    # listener default is [] so pre-initiation ticks are empty.
                    'fork_coordinates': 'array[integer]',
                },
                # dnaa-3 in-sim DnaA-box occupancy observer (dnaa_box_binding
                # listener). Mirrors DnaaBoxBinding.outputs() leaf-for-leaf so
                # the parquet emitter captures the occupancy columns.
                #
                # CRITICAL: these MUST be overwrite[...] (not plain float/integer).
                # The dnaA_binding store leaves do NOT exist in the resume dill
                # (unlike replication_data, whose leaves the parent emits), so
                # the leaf type is established by this emit-schema declaration.
                # A plain `float` apply ACCUMULATES (state += delta) — the listener
                # would report a running SUM over every tick (free DnaA-ATP grew
                # 27k->57k nM, occupancy >1) instead of the per-tick value. The
                # `overwrite[...]` apply replaces, matching the step's outputs().
                'dnaA_binding': {
                    'free_DnaA_ATP_nM': 'overwrite[float]',
                    'free_DnaA_ADP_nM': 'overwrite[float]',
                    'oric': {
                        'high_affinity_occupied': 'overwrite[float]',
                        'low_affinity_occupied': 'overwrite[float]',
                        'n_bound': 'overwrite[integer]',
                        'n_total': 'overwrite[integer]',
                    },
                    'dnaap': {
                        'occupied': 'overwrite[float]',
                        'n_bound': 'overwrite[integer]',
                        'n_total': 'overwrite[integer]',
                    },
                    'chromosome': {
                        'occupied': 'overwrite[float]',
                        'occupied_count': 'overwrite[integer]',
                        'n_total': 'overwrite[integer]',
                    },
                    'total_DnaA_bound': 'overwrite[integer]',
                },
                # Phase-3 sprint-3: aggregate likelihood listener
                # written by LikelihoodCollector. Single-scalar per-tick
                # observable for downstream inference tools (ABC-SMC,
                # SBC) — read directly from the emitter store without
                # re-aggregating per-process fields.
                'likelihood': {
                    'transcript_init': 'float',
                    'polypeptide_init': 'float',
                    'total': 'float',
                },
            })

        if parquet_override is not None:
            # Parquet path — column-oriented hive-partitioned output. Unlike
            # the SQLite path, this captures the raw ``bulk`` count array
            # (~25k molecules) in addition to global_time + listeners. Parquet
            # is column-oriented and compresses the wide bulk vector far better
            # than SQLite's row store, so the disk-cost concern documented in
            # the SQLite branch below does not apply here — and downstream
            # analyses (and vEcoli parity) need per-molecule counts.
            if ParquetEmitter is None:
                raise ImportError(
                    "ParquetEmitter override set but [parquet] extra not "
                    "installed. Run: pip install 'v2ecoli[parquet]'"
                )
            emit_schema = {
                'global_time': 'float',
                'bulk': 'array[integer]',
                'listeners': listeners_schema,
            }
            topo = {
                'global_time': ('global_time',),
                'bulk': ('bulk',),
                'listeners': ('listeners',),
            }
            # Config-declared EXTRA emit store paths (domain-agnostic; see
            # _merge_emit_paths). Popped so it does not reach the emitter config.
            _merge_emit_paths(
                emit_schema, topo, dict(parquet_override).pop("emit_paths", None))
            parquet_override = {k: v for k, v in parquet_override.items()
                                if k != "emit_paths"}
            cfg = {'emit': emit_schema, **parquet_override}
            instance = ParquetEmitter(cfg, core)
            # Register under the override's metadata.agent_id (the runner's
            # per-generation identity: "0", "00", ...) so Division can flush
            # this emitter's trailing partial batch + write the success
            # sentinel before the agent subtree is torn down at _remove.
            _agent_id = (parquet_override.get('metadata') or {}).get('agent_id')
            if _agent_id is not None:
                register_parquet_emitter(str(_agent_id), instance)
        elif override is not None:
            # Persistent SQLite path (default-baseline + per-study run
            # scripts). Capture ONLY global_time + listeners. The raw
            # ``bulk`` store (~25k molecules) and the unique-node structures
            # (full_chromosome / active_replisome / active_RNAP /
            # chromosome_domain) are MULTI-MB PER ROW — emitting them every
            # tick produced a 5.5 GB default-baseline db (1.75 MB × 2461
            # rows) that timed out the dashboard's post-hoc chart extraction
            # (blank study pages). No viz/test reads them: study yamls
            # declare listeners.* paths, and raw bulk-id paths aren't even
            # json_extract-able. The bulk-derived observable studies DO use
            # (monomer_counts) lives under listeners and is kept.
            emit_schema = {
                'global_time': 'float',
                'listeners': listeners_schema,
            }
            topo = {
                'global_time': ('global_time',),
                'listeners': ('listeners',),
            }
            cfg = {'emit': emit_schema, **override}
            instance = SQLiteEmitter(cfg, core)
        elif _NULL_EMITTER_OVERRIDE:
            # Minimal RAMEmitter (global_time only): the caller emits out of
            # band (e.g. the workflow's external XArrayEmitter), so the internal
            # emitter must not waste memory capturing full state every tick.
            emit_schema = {'global_time': 'float'}
            topo = {'global_time': ('global_time',)}
            instance = RAMEmitter({'emit': emit_schema}, core)
        elif default_decl is not None:
            # Generator-declared default sink (e.g. baseline ships a
            # ParquetEmitter via @composite_generator(emitters=[...])). Lower
            # priority than every external override above; higher than the bare
            # RAMEmitter fallback below. The composite's own default emitter
            # thus travels with the generator for standalone runs.
            #
            # allow_ram_fallback defaults to False here: a run that declares
            # ParquetEmitter as its default sink expects to persist to disk,
            # so a failed build must raise rather than silently keep
            # everything in RAM and lose it all on exit (P1-4, §2.10).
            instance, topo = _build_declared_emitter(
                default_decl, listeners_schema, core)
        else:
            # In-memory RAMEmitter: no disk cost, so keep the full capture
            # (bulk + chromosome/replisome structures) for callers that
            # want the complete state tree in RAM.
            emit_schema = {
                'global_time': 'float',
                'bulk': 'array',
                'listeners': listeners_schema,
                'full_chromosome': 'node',
                'active_replisome': 'node',
                'active_RNAP': 'node',
                'chromosome_domain': 'node',
            }
            topo = {
                'global_time': ('global_time',),
                'bulk': ('bulk',),
                'listeners': ('listeners',),
                'full_chromosome': ('unique', 'full_chromosome'),
                'active_replisome': ('unique', 'active_replisome'),
                'active_RNAP': ('unique', 'active_RNAP'),
                'chromosome_domain': ('unique', 'chromosome_domain'),
            }
            instance = RAMEmitter({'emit': emit_schema}, core)

        return instance, topo, 'step'

    if step_name == 'ppgpp-initiation':
        from v2ecoli.steps.ppgpp_initiation import PpgppInitiation
        try:
            ti_config = loader.get_config_by_name('ecoli-transcript-initiation')
        except (KeyError, AttributeError):
            ti_config = {}
        ppgpp_config = {
            'ppgpp': ti_config.get('ppgpp', ''),
            'synth_prob': ti_config.get('synth_prob'),
            'copy_number': ti_config.get('copy_number', 1),
            'n_avogadro': ti_config.get('n_avogadro', 0),
            'cell_density': ti_config.get('cell_density', 0),
            'get_rnap_active_fraction_from_ppGpp': ti_config.get(
                'get_rnap_active_fraction_from_ppGpp'),
            'trna_attenuation': ti_config.get('trna_attenuation', False),
            'attenuated_rna_indices': ti_config.get('attenuated_rna_indices', []),
            'attenuation_adjustments': ti_config.get('attenuation_adjustments', []),
        }
        from v2ecoli.library.config_resolver import resolve_config
        ppgpp_config = resolve_config(ppgpp_config)
        instance = _make_instance(PpgppInitiation, ppgpp_config, core)
        topo = {
            'bulk': ('bulk',),
            'listeners': ('listeners',),
            'ppgpp_state': ('ppgpp_state',),
        }
        return instance, topo, 'step'

    if step_name == 'trna-attenuation-config':
        from v2ecoli.steps.trna_attenuation import TrnaAttenuationConfig
        try:
            te_config = loader.get_config_by_name('ecoli-transcript-elongation')
        except (KeyError, AttributeError):
            te_config = {}
        att_config = {
            'get_attenuation_stop_probabilities': te_config.get(
                'get_attenuation_stop_probabilities'),
            'attenuated_rna_indices': te_config.get(
                'attenuated_rna_indices', []),
            'location_lookup': te_config.get('location_lookup', {}),
            'cell_density': te_config.get('cell_density', 0),
            'n_avogadro': te_config.get('n_avogadro', 0),
            'charged_trnas': te_config.get('charged_trnas', []),
        }
        instance = _make_instance(TrnaAttenuationConfig, att_config, core)
        topo = {
            'attenuation_config': ('attenuation_config',),
        }
        return instance, topo, 'step'

    if step_name == 'replication_data_listener':
        from v2ecoli.steps.derivers.replication_data import ReplicationData
        config = {'time_step': 1}
        instance = ReplicationData(config=config, core=core)
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    if step_name == 'dnaa_box_binding_listener':
        # dnaa-3 read-only DnaA-box occupancy observer (the PROVIDED model).
        # Config defaults (K_d / pool sizes) live in the step's config_schema;
        # an empty config uses them. Placed in an early layer (see baseline.py)
        # so its listeners.mass read resolves to the prior tick.
        #
        # CRITICAL: this is a READ-ONLY listener. The step reads ``bulk`` (a
        # structured-array store), the ``DnaA_box`` unique store, and
        # ``listeners.mass`` — but writes ONLY ``listeners.dnaA_binding``. If we
        # returned a 3-tuple, make_edge() would mirror the FULL topology onto
        # both the input AND output wires, wiring ``bulk`` as an updatable
        # output. The engine then tries to apply an (absent) delta onto the
        # bulk structured array → ``ufunc 'add' did not contain a loop`` crash.
        # So we split the topology: ALL ports are inputs; the ONLY output wire
        # is ``listeners`` (where dnaA_binding lives). Mirrors the partitioned-
        # step pattern above (in_topo/out_topo split).
        from v2ecoli.steps.derivers.dnaa_box_binding import DnaaBoxBinding
        instance = _make_instance(
            DnaaBoxBinding, {'time_step': 1}, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        # Wire exactly the ports the schema declares: in_topo from inputs()
        # keys, out_topo from outputs() keys. This keeps ``bulk`` /
        # ``mass_listener`` / ``DnaA_boxes`` as pure reads and ``listeners`` as
        # the sole write target.
        in_keys = set(instance.inputs().keys())
        out_keys = set(instance.outputs().keys())
        in_topo = {k: v for k, v in topology.items() if k in in_keys}
        out_topo = {k: v for k, v in topology.items() if k in out_keys}
        return instance, topology, 'step', in_topo, out_topo

    if step_name == 'mark_d_period':
        from v2ecoli.steps.division import MarkDPeriod
        instance = MarkDPeriod(config={}, core=core)
        topo = {
            'full_chromosome': ('unique', 'full_chromosome'),
            'global_time': ('global_time',),
            'divide': ('divide',),
        }
        return instance, topo, 'step'

    if step_name == 'division':
        from v2ecoli.steps.division import Division
        try:
            div_config = loader.get_config_by_name('division')
        except Exception:
            div_config = {}
        div_config.setdefault('agent_id', '0')
        div_config.setdefault('division_threshold', 'mass_distribution')
        dry_mass_inc = getattr(getattr(loader, 'sim_data', None),
                               'expectedDryMassIncreaseDict', {})
        from v2ecoli.library.unit_bridge import unum_to_pint
        dry_mass_inc = {k: unum_to_pint(v) for k, v in dry_mass_inc.items()}
        div_config.setdefault('dry_mass_inc_dict', dry_mass_inc)
        if hasattr(loader, '_configs'):
            div_config['configs'] = loader._configs
        div_config.setdefault('unique_names', getattr(loader, 'unique_names', []))
        div_config.setdefault('cache_dir', getattr(loader, 'cache_dir', 'out/cache'))
        # Thread the injected/swapped-process spec (fork_repo/fork_sim_data/
        # swap_processes/…) so the DivisionStep re-supplies it to each daughter's
        # baseline() rebuild — otherwise daughters revert to plain FBA baseline
        # and a swapped process (e.g. metabolism-redux) is lost across division,
        # crashing the re-realization with KeyError: 'current_timeline'. Only set
        # when non-empty so the normal baseline (None) leaves daughters unchanged.
        _injected = getattr(loader, '_injected_processes', None)
        if _injected:
            div_config['injected_processes'] = _injected
        # Same reasoning, for a config_overrides / knockouts PERTURBATION. Without
        # this, a variant applied as config_overrides is correct in generation 1
        # and silently reverts to the unperturbed cached configs at division
        # (#505). Only set when non-empty so the plain baseline (None) leaves
        # daughters byte-for-byte unchanged.
        _overrides = getattr(loader, '_config_overrides', None)
        if _overrides:
            div_config['config_overrides'] = _overrides
        # Same reasoning, for DECLARED MEASUREMENTS. baseline() threads the flux
        # map through the module-level override above and CLEARS it in its
        # finally, so a daughter rebuilt mid-run is built with an empty map: the
        # feature step declares no leaves and every exchange-flux leaf sits at
        # its 0.0 default for the rest of the lineage. Measured on a two-
        # generation run: generation 1 carries the data, generation 2 reads
        # exactly zero on every sample while dry mass, growth rate and division
        # are all normal — i.e. the cell is fine and only the measurement is
        # gone, which is the shape that gets mistaken for a result.
        # Captured here rather than read at daughter-build time because by then
        # the override has been restored.
        _fluxes = dict(_EXCHANGE_FLUXES_OVERRIDE)
        if _fluxes:
            div_config['exchange_fluxes'] = _fluxes
            # The BASIS travels with the map for the same reason the map does: a
            # daughter rebuilt without it silently reverts to counts and emits a
            # running total under a rate's leaf name — the two are different
            # quantities, and nothing downstream can tell which one it got.
            div_config['exchange_flux_basis'] = str(_EXCHANGE_FLUX_BASIS_OVERRIDE)
        instance = _make_instance(Division, div_config, core)
        topo = {
            'bulk': ('bulk',),
            'unique': ('unique',),
            'listeners': ('listeners',),
            'environment': ('environment',),
            'boundary': ('boundary',),
            'global_time': ('global_time',),
            'division_threshold': ('division_threshold',),
            'media_id': ('environment', 'media_id'),
            # D-period flag raised by MarkDPeriod (same store), so the Division
            # step can divide on the D-period instead of the mass threshold.
            'divide': ('divide',),
            'agents': ('..',),
        }
        return instance, topo, 'step'

    return None
