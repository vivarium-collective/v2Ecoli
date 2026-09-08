"""Baseline whole-cell E. coli composite (55 processes, partitioned).

Upstream-parity architecture: the partitioned model matches the
vivarium-collective/vEcoli composite tick-for-tick. See AGENTS.md.

Shared helpers (``make_edge``, ``inject_flow_dependencies``,
``_seed_state_from_defaults``, ``seed_mass_listener``,
``_normalize_boundary_units``, ``_make_instance``, ``_get_special_step``,
module-level constants) live in ``v2ecoli.composites._helpers``.
Architecture-specific helpers (``build_execution_layers``, ``DEFAULT_FEATURES``,
``_get_step_config``) are defined as private module-level functions here.
"""

from __future__ import annotations

import binascii
import copy
import os
from typing import Any

import numpy as np


def _apply_match_simdata(cell_state: dict, *, match_simdata: str, seed: int,
                         condition: str = "basal") -> dict:
    """Overlay matched-initial-state bulk counts onto ``cell_state`` IN PLACE.

    Declarative counterpart of ``scripts/run_comparison_ensemble.py``'s
    ``--match-vecoli-simdata``/``--match-initial-state`` flags: reuses the
    SAME mechanism (not a reimplementation) —
    ``_vecoli_reference_state`` builds the genuine upstream vEcoli engine
    from ``match_simdata`` (a reference ``simData.cPickle`` path) and reads
    its pre-run bulk molecule counts; ``_apply_bulk_overlay`` overwrites the
    matching v2 bulk counts in place, molecule-id by molecule-id, for
    whatever is present in both engines. Net effect: this composite's t=0
    bulk state becomes the reference's, so a candidate/reference pair driven
    from the same ``simData.cPickle`` starts identical (removes the
    stochastic low-copy sampling divergence — e.g. SpoT — that otherwise
    dominates single-seed comparisons; see run_comparison_ensemble.py's
    "Matched-initial-state seeding" section for the full rationale).

    ``condition`` defaults to "basal" (matching ``build_vivarium_ecoli``'s
    own default) — Task 1 only wires the single-param (``match_simdata``)
    path; per-config condition threading is Task 4's job (materializing the
    comparison investigation's paired studies).

    Imports the ``scripts`` package lazily so the default (``match_simdata``
    unset) path never pays for it.
    """
    from types import SimpleNamespace
    from scripts.run_comparison_ensemble import (
        _vecoli_reference_state, _apply_bulk_overlay)
    ref_bulk, _ref_unique = _vecoli_reference_state(
        os.path.abspath(match_simdata), condition, seed,
        os.environ.get("V2E_VECOLI_DIR"))
    fake_composite = SimpleNamespace(state={"agents": {"0": cell_state}})
    return _apply_bulk_overlay(fake_composite, ref_bulk)


def _independent_founder_state(sim_data_path: str, seed: int,
                               condition: str = "basal") -> dict:
    """Re-draw a fresh founder (t=0 cell state) for this lineage_seed.

    Opt-in alternative to loading the SHARED cached initial_state: draws a new
    founder from ``sim_data_path`` seeded by ``seed`` via the SAME generator the
    cache build uses (``LoadSimData(sim_data_path, seed).generate_initial_state()``
    — see ``v2ecoli/core.py``'s ``_write_sim_input_bundle``), then round-trips
    through ``save_initial_state``/``load_initial_state`` so the result is
    byte-format-identical to a cached founder. Gives each seed of a multiseed
    ensemble an INDEPENDENT founder, so the spread reflects real cell-to-cell
    founder variability rather than only downstream per-process stochasticity.
    Slower than the cached path (loads sim_data + regenerates initial conditions
    per seed) — opt-in via ``independent_founders``.
    """
    import tempfile
    from v2ecoli.library.sim_data import LoadSimData
    from v2ecoli.cache import save_initial_state, load_initial_state
    kwargs = {"sim_data_path": os.path.abspath(sim_data_path), "seed": seed}
    if condition and condition != "basal":
        kwargs["condition"] = condition
    loader = LoadSimData(**kwargs)
    state = loader.generate_initial_state()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "initial_state.json")
        save_initial_state(state, p)
        return load_initial_state(p)


def _derive_process_seed(master_seed: int, process_name: str) -> int:
    """Derive a per-process RNG seed from (master_seed, process_name).

    Without this, all stochastic processes inherit the same cache-derived seed,
    so multi-seed ensembles collapse to bit-identical trajectories. CRC32 with
    master_seed as the initial state, keyed by process name, gives each process
    its own 32-bit positive seed that varies across master_seeds (ensemble
    diversity) but is stable per (master_seed, process_name) pair (reproducibility).
    Mirrors the pattern already used in v2ecoli/steps/division.py.
    """
    return binascii.crc32(process_name.encode("utf-8"), master_seed) & 0x7FFFFFFF


def _is_plain_numeric_leaf(v: Any) -> bool:
    """True for listener leaf values the XArrayEmitter transducer can write
    straight into a fixed-dtype DataArray slot: python/numpy int/float/bool
    scalars, or numeric ``numpy.ndarray``s with more than one element.

    False for ``pint.Quantity`` (no unit-stripping hook in viva_emitters'
    view/transducer), ``str``/``bytes``/``list``/``tuple`` (the transducer
    only walks plain dicts, not these), and — critically — length-0 or
    length-1 numeric arrays.

    The size<=1 exclusion mirrors ``extract_output_metadata_from_state``'s own
    "scalar, no coord needed" threshold (``arr.ndim==0 or arr.size<=1``) and
    guards against a real viva_emitters promotion-trap bug (Task 1 spike,
    gotcha #5b): a leaf with no declared output_metadata coord starts with
    ``spec.coord is None`` and goes through the *dynamic* promote-or-drop
    write path on its first ``update()``. If that first write is itself a
    length-0/1 array (common in v2ecoli — many listener leaves start
    empty/singleton until a biological event first populates them),
    ``_promote_port`` allocates a tiny slot AND mutates ``spec.coord`` to a
    real (non-``None``) array; every SUBSEQUENT write then takes the
    "declared coord" direct-write branch with no promote/drop safety net, so
    the first time the leaf grows to its real, different length the write
    crashes with a broadcast ``ValueError``. Excluding size<=1 leaves up
    front keeps every kept leaf on a stable, provably-safe path.
    """
    import pint
    if isinstance(v, pint.Quantity):
        return False
    if isinstance(v, (str, bytes, list, tuple)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating, bool)):
        return True
    if isinstance(v, np.ndarray):
        return np.issubdtype(v.dtype, np.number) and v.size > 1
    return False


def _listener_leaf_paths(listeners: dict, *, prefix: str = "listeners"):
    """Yield dotted ``listeners.<...>`` leaf paths for plain-numeric values.

    Recurses ``listeners`` (a nested dict of namespace -> leaf -> value);
    yields a path for each leaf that survives ``_is_plain_numeric_leaf``.
    Non-numeric / ragged leaves (pint.Quantity, str, list, size<=1 arrays)
    are silently dropped — see ``_is_plain_numeric_leaf`` for why.
    """
    for k, v in listeners.items():
        p = f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from _listener_leaf_paths(v, prefix=p)
        elif _is_plain_numeric_leaf(v):
            yield p


def filter_listener_paths(listener_paths, emit_paths):
    """Restrict discovered ``listeners.<...>`` leaf paths to an emit-path allowlist.

    ``emit_paths`` entries may be dotted strings (``"listeners.mass"``) or
    tuple/list paths (``["listeners", "mass"]``); an entry matches a leaf path
    exactly or as a dotted prefix (``"listeners.mass"`` keeps
    ``"listeners.mass.cell_mass"``). A falsy/empty ``emit_paths`` returns the
    input unchanged, so the default (emit every numeric listener leaf) is
    preserved and the feature is opt-in. This is the mechanism a study uses to
    emit only the handful of paths it needs instead of the full listener dump.
    """
    if not emit_paths:
        return list(listener_paths)
    prefixes = [".".join(str(x) for x in p) if isinstance(p, (list, tuple)) else str(p)
                for p in emit_paths]
    return [lp for lp in listener_paths
            if any(lp == pre or lp.startswith(pre + ".") for pre in prefixes)]


def _single_cell_xarray_config(*, out_uri: str, metadata: dict | None = None,
                                buffer_size: int = 600) -> dict:
    """Build the STATIC XArrayEmitter ``config`` skeleton for a single-cell,
    agent-relative, in-document capture.

    Pure — no IO, no Composite construction, no ``cell_state`` inspection. The
    two realize-dependent keys — ``view`` and ``output_metadata`` — are NOT set
    here; they are discovered from the REALIZED composite state at run time by
    ``SingleCellXArrayEmitter`` (see its docstring and Task 4 / C2). This is why
    the helper no longer takes ``cell_state``: at document-build time the
    listener tree is only partially materialised (``core.realize()`` fills the
    schema-defaulted listener namespaces only when the ``Composite`` is
    constructed), so a view built here would starve the capture down to the
    handful of pre-seeded leaves (Task 3 concern #1). The lazy step instead
    reads the fully-realized listener tree on its first ``update()``.

    Encodes the static parts of the recipe validated by Task 1's spike (see
    ``.superpowers/sdd/2026-08-10-single-cell-xarray-emitter/task-1-report.md``):

      * ``strategy="flat"``, ``emit_root=[]`` — the in-document emitter Step is
        co-located inside ``agents/0`` and receives bare
        ``{"global_time": ..., "bulk": ..., "listeners": ...}`` payloads, NOT
        wrapped in an ``{"agents": {id: ...}}`` envelope (that's the
        lineage-runner's ``strategy="colony"`` pattern, not this one).
      * ``metadata`` must be non-empty (an empty dict silently skips
        XArrayEmitter's partition setup and crashes the first ``update()`` —
        Task 1 gotcha #1). Callers pass their real experiment_id/variant/seed.
      * bounded, streaming buffer (``subsample(1)`` + small buffer), zarr v3
        writer — copied verbatim from the spike.

    Args:
        out_uri: zarr store path/URI.
        metadata: non-empty run-identity metadata (experiment_id / variant /
            lineage_seed). Falls back to a non-empty placeholder if omitted.
        buffer_size: transducer buffer size (streaming, bounded), in emit steps.
            Default 600 — matches the viva-emitters library default; flushes a
            handful of times per generation rather than every few steps.

    Returns:
        The static XArrayEmitter config skeleton (no ``view`` /
        ``output_metadata`` — those are added lazily at run time).
    """
    return {
        "emit": {"global_time": "float", "bulk": "array[integer]", "listeners": "tree"},
        "out_uri": str(out_uri),
        "strategy": "flat",
        "emit_root": [],
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": buffer_size},
        },
        "writer": {
            "backend": "zarr",
            "store": str(out_uri),
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": dict(metadata) if metadata else {
            "experiment_id": "single_cell", "variant": 0, "lineage_seed": 0},
        "metadata_keys": [],
        "metadata_validators": {},
        "provenance": {},
        "debug": False,
    }


def _resolve_xarray_out_uri(experiment_id: str, out_dir: str = "") -> str:
    """Resolve the zarr store path for the in-document single-cell
    XArrayEmitter (``baseline()``'s ``emitter=="xarray"`` branch).

    Mirrors the sqlite branch's workspace-shared-root convention
    (``_find_workspace_root`` -> ``<ws>/.pbg/...``): prefers the workspace's
    ``.pbg/xarray-runs/`` dir so a workspace-hosted run's zarr store lives
    alongside the sqlite history db (``.pbg/composite-runs.db``) and parquet
    hive dir (``.pbg/parquet-runs/``); falls back to ``out/xarray`` when no
    ``workspace.yaml`` is found (e.g. a bare-checkout build).

    ``out_dir`` is threaded through explicitly (rather than resolved
    unconditionally) so a caller — e.g. Task 4's real-run integration test —
    can target a tmp dir without needing a real workspace on disk.

    Args:
        experiment_id: names the zarr store (``<experiment_id>.zarr``).
        out_dir: explicit output directory override. Empty (default) falls
            back to the workspace-root / ``out/xarray`` resolution above.

    Returns:
        The zarr store path as a string. Not created on disk here — the
        XArrayEmitter's writer creates it lazily on first write.
    """
    from pathlib import Path
    if out_dir:
        base = Path(out_dir)
    else:
        ws_root = _find_workspace_root()
        base = (ws_root / ".pbg" / "xarray-runs") if ws_root is not None else Path("out/xarray")
    return str(base / f"{experiment_id}.zarr")


from viva_superpowers.composite_generator import composite_generator, emitter_defaults

from v2ecoli.core import build_core, load_cache_bundle, register_ecoli_core

# ---------------------------------------------------------------------------
# Shared helpers and constants
# ---------------------------------------------------------------------------
from v2ecoli.composites._helpers import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    seed_mass_listener,
    _normalize_boundary_units,
    _make_instance,
    _get_special_step,
    _expand_flushes,
    set_default_emitter_decl,
    set_emitter_override,
    set_null_emitter_override,
    set_exchange_fluxes_override,
    set_exchange_flux_basis_override,
    _find_workspace_root,
    CachedConfigLoader,
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
    ALLOCATOR_LAYERS,
    DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)
# Re-export: baseline's biological store grouping (compartment -> molecular
# class), distilled from the retired `biological` composite. Pure annotation —
# baseline's EMITTED paths stay flat (global_time/bulk/listeners) so analyses are
# unaffected; display tools (Composite Explorer / loom, follow-up) read this to
# render the flat stores grouped as biology. See store_groups.py.
from v2ecoli.composites.store_groups import STORE_GROUPS as STORE_GROUPS  # noqa: PLC0414

from process_bigraph.emitter import Emitter


class SingleCellXArrayEmitter(Emitter):
    """In-document, agent-relative XArrayEmitter — built LAZILY from realized state.

    Subclasses ``process_bigraph.emitter.Emitter`` (not a bare ``Step``) so the
    composite_generator convention recognises it as the document's observation
    sink (``process_bigraph.emitter._node_is_emitter`` short-circuits on
    ``isinstance(instance, Emitter)``). Without that, ``CompositeSpec._with_emitters``
    treats the document as observing nothing and injects the generator's declared
    ParquetEmitter default at the top level with an empty ``out_dir`` — which then
    fails ``Composite`` realize.

    Swapped into the single ``agents/0/emitter`` key in place (never as an extra
    document Step: adding sibling Steps perturbs process_bigraph's scheduling and
    trips a pre-existing metabolism fragility — Task 1 gotcha #4). It wraps a real
    ``viva_emitters.XArrayEmitter`` but defers its construction to the FIRST
    ``update()`` so the view/output_metadata are discovered from the fully
    REALIZED composite state, resolving the two Task-4 blockers:

      * **C2 (realized-state view).** At document-build time the listener tree is
        only partially materialised — ``core.realize()`` fills the schema-defaulted
        listener namespaces only when the ``Composite`` is constructed, and per-tick
        listener vectors are populated only once processes run. Building the view
        then captures ~4 leaves. This step instead reaches the driving ``Composite``
        via ``get_current_composite()`` (the ``run()`` contextvar) on its first
        ``update()`` and reads the full realized ``agents/0`` subtree — yielding the
        complete listener set (~50-115 leaves) plus process-instance-derived named
        coord labels via ``output_metadata(full_state)``. Mirrors the multigen
        runner's "warm 1 tick, then discover coords" pattern.

      * **C1 (structured ``bulk``).** ``cell_state["bulk"]`` is a numpy *record*
        array (fields ``id``/``count``/``*_submass``); the transducer cannot cast
        it to ``<i8``. This step projects ``bulk["count"]`` to a plain int64 vector
        inside ``update()`` before emitting (no field-projection hook exists in the
        emitter's view machinery).

    The trailing partial buffer is flushed by ``close_emitter()`` (called by
    ``v2ecoli.build_composite``'s run-wrap): the emitter only auto-flushes a FULL
    buffer mid-run, and ``flush(final=False)`` asserts a full buffer, so a partial
    tail can only be written via ``close()``/``flush(final=True)``.
    """

    def __init__(self, config, core):
        super().__init__(config, core)
        self._em = None
        self._leaf_key_paths: list | None = None
        # Declared bulk molecule ids to surface as scalar observables under
        # listeners.observable_bulk.<id> (the two-arm comparison's bulk KPI hook —
        # e.g. VIOLACEIN[c] titer, mecillinam[p]-EG10606-MONOMER[i] drug-target
        # complex). Emitting under the `listeners` root reuses the existing view
        # machinery and gives BOTH engines an identical path to compare on.
        self._obs_bulk_ids: list = list(config.get("observable_bulk_ids") or [])

    def inputs(self):
        return {"global_time": "float", "bulk": "array[integer]", "listeners": "tree"}

    def outputs(self):
        return {}

    def _lazy_init(self):
        from process_bigraph.composite import get_current_composite
        from viva_emitters import XArrayEmitter
        from v2ecoli.library.xarray_run import (
            view_from_emit_paths, extract_output_metadata_from_state)
        from v2ecoli.library.output_metadata import output_metadata as _named_output_metadata

        comp = get_current_composite()
        if comp is None:
            raise RuntimeError(
                "SingleCellXArrayEmitter.update() ran outside a Composite.run() "
                "context — get_current_composite() returned None, so the "
                "realized listener tree cannot be discovered.")
        full_state = comp.state
        cell = (full_state.get("agents") or {}).get("0") or full_state
        listener_paths = list(_listener_leaf_paths(cell.get("listeners") or {}))
        # Optional emit-path allowlist: when the config declares ``emit_paths``,
        # restrict the emitted listener leaves to those under the declared paths
        # (see ``filter_listener_paths``). Absent/empty keeps the default (every
        # numeric listener leaf), so this is backward-compatible. Declared bulk
        # observables (below) are always kept — they are requested explicitly via
        # ``observable_bulk_ids``, not the listener sweep. This is how a study emits
        # only the handful of paths it needs instead of the full ~400-column dump.
        listener_paths = filter_listener_paths(
            listener_paths, self.config.get("emit_paths"))
        # Declared bulk observables ride under a synthetic listeners.observable_bulk
        # group so the SAME listener view captures them (both engines share the path).
        listener_paths += [f"listeners.observable_bulk.{i}" for i in self._obs_bulk_ids]
        # Listener view (unmodified helper) + manual bulk entry. root=() so the
        # read path resolves to () + ("bulk",) == ("bulk",); LeafView.path (the
        # OUTPUT var name) must be non-empty, hence "bulk".
        view = view_from_emit_paths(listener_paths)
        view.append({"root": (), "variables": {"bulk": [{"path": "bulk", "dtype": "<i8"}]}})
        named_metadata = _named_output_metadata(full_state)
        output_metadata_ = extract_output_metadata_from_state(
            full_state, view, named_metadata=named_metadata)

        cfg = {**dict(self.config), "view": view, "output_metadata": output_metadata_}
        self._em = XArrayEmitter(cfg, self.core)
        # listener_paths are "listeners.<ns>.<leaf>"; store the key path relative
        # to the listeners root (drop the leading "listeners").
        self._leaf_key_paths = [tuple(p.split(".")[1:]) for p in listener_paths]

    def update(self, state, interval=None):
        if self._em is not None and getattr(self._em, "_closed", False):
            # F3: build_composite's run-end flush hook calls close_emitter()
            # after each run(), which finalizes the zarr writer. A SECOND run()
            # on the same xarray composite would drive updates into a closed
            # writer and silently no-op/corrupt. Fail loudly and actionably
            # instead — this is a single-run sink by design.
            raise RuntimeError(
                "SingleCellXArrayEmitter was already closed by the run-end flush "
                "hook: the in-document single-cell XArray sink supports ONE run() "
                "per build_composite(...). Rebuild the composite (a fresh "
                "build_composite(..., emitter='xarray')) for another run, or use "
                "emitter='parquet' if you need to resume/extend a run.")
        if self._em is None:
            self._lazy_init()
        # C1: project the structured bulk record array to a plain int64 vector.
        bulk_counts = np.asarray(state["bulk"]["count"], dtype=np.int64)
        # Filter listeners to exactly the declared leaves (the transducer raises
        # on any undeclared emit path once sim_tix > 0).
        src = state.get("listeners") or {}
        filtered: dict = {}
        for path in self._leaf_key_paths or []:
            cur = src
            ok = True
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if not ok:
                continue
            cursor = filtered
            for k in path[:-1]:
                cursor = cursor.setdefault(k, {})
            cursor[path[-1]] = cur
        # Declared bulk observables → listeners.observable_bulk.<id> scalars,
        # selected by molecule id from the bulk record (state["bulk"] carries both
        # "id" and "count"). A missing id emits 0.0 so the trace stays continuous.
        if self._obs_bulk_ids:
            ids = state["bulk"]["id"]
            grp = filtered.setdefault("observable_bulk", {})
            for mol in self._obs_bulk_ids:
                hit = np.where(ids == mol)[0]
                grp[mol] = float(bulk_counts[hit[0]]) if len(hit) else 0.0
        self._em.update({
            "global_time": state["global_time"],
            "bulk": bulk_counts,
            "listeners": filtered,
        })
        return {}

    def close_emitter(self):
        """Flush the trailing partial buffer and finalize the zarr store.

        Idempotent. Swallows the known viva_emitters ``flush(final=True)`` assert
        (buffer exactly full at close) ONLY when at least one buffer already
        reached disk mid-run — those rows are safe and only the just-flushed
        trailing buffer tripped the boundary assert. If NOTHING was ever written
        (``num_writes <= 0``), the store would be left empty/without a success
        marker, so the assert is re-raised with context instead of masked (F4).
        """
        if self._em is None or getattr(self._em, "_closed", False):
            return
        try:
            self._em.close(success=True)
        except AssertionError:
            writer = getattr(self._em, "writer", None)
            num_writes = getattr(writer, "num_writes", 0)
            if num_writes and num_writes > 0:
                # benign trailing-buffer boundary — earlier buffers are on disk
                return
            raise RuntimeError(
                "SingleCellXArrayEmitter.close_emitter(): the XArray final flush "
                "asserted before any buffer reached disk — the zarr store at "
                f"{(self.config or {}).get('out_uri')!r} is likely empty/unfinalized. "
                "This is NOT the benign buffer-full-at-close boundary.")


# ---------------------------------------------------------------------------
# Execution layers (partitioned / baseline architecture)
# ---------------------------------------------------------------------------

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment (sequential sub-steps)
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: standalone (no partitioning needed)
    ['ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation'], FLUSH,

    # NOTE: the dnaA-investigation mechanism steps — dnaa-3 (dnaa-box-binding +
    # dnaa_box_binding_listener), dnaa-4 (autoregulation in transcript_initiation),
    # and dnaa-5 (rida / ddah / dars + library/locus_copy_number) — remain in the
    # tree as DORMANT infrastructure but are NOT wired into the default baseline.
    # Main's default model is the pre-investigation WCM; these are activated only
    # by the dnaa-replication investigation (draft PR), which re-adds the layers.

    # Layer 3: TF binding

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: protein degradation (standalone — no resource competition)
    ['ecoli-protein-degradation'],

    # Layer 4b: standalone initiation/replication/complexation
    ['ecoli-complexation', 'ecoli-chromosome-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],
    # RNA degradation still partitioned (shares water with other processes)
    ['ecoli-rna-degradation_requester'],
    ['allocator_2'],
    ['ecoli-rna-degradation_evolver'], FLUSH,

    # Layer 5: partition layer 3 -- elongation requesters (parallel)
    ['ecoli-polypeptide-elongation_requester', 'ecoli-transcript-elongation_requester'],
    ['allocator_3'],
    # Layer 5: partition layer 3 -- elongation evolvers (parallel)
    ['ecoli-polypeptide-elongation_evolver', 'ecoli-transcript-elongation_evolver'], FLUSH,

    # Layer 6: chromosome structure + metabolism (sequential)
    ['ecoli-chromosome-structure'], FLUSH,
    ['ecoli-metabolism'], FLUSH,

    # Layer 7: listeners (parallel)
    ['counts_deriver', 'ecoli-mass-listener',
     'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener'], FLUSH,

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
    # Opt-in runtime mass-conservation check. Runs after the mass listener
    # (dry_mass) and metabolism (environment.exchange). OFF by default: the
    # residual is not yet calibrated (net boundary exchange measures ~55x the
    # cell-mass change — see mass_conservation.py STATUS), so on a healthy run
    # it warns every tick. Enable per investigation via
    # enable_features('mass_conservation') before build_composite.
    'mass_conservation': {
        'insert_after': 'ecoli-mass-listener',
        'steps': ['ecoli-mass-conservation'],
    },
    # Opt-in: re-home named environment.exchange fluxes onto
    # listeners.exchange_flux.<name> so the compact XArray view (listeners-only)
    # carries them. Enabled automatically when the generator's exchange_fluxes
    # param is non-empty; the flux map is threaded via set_exchange_fluxes_override.
    'exchange_flux': {
        'insert_after': 'ecoli-mass-listener',
        'steps': ['exchange_flux_listener'],
    },
    # Opt-in: native cell-shape geometry (periplasm/cytoplasm volume split +
    # outer surface area), ported from vEcoli ecoli/processes/shape.py.
    # Populates `periplasm.global.volume`/`cytoplasm.global.volume`/
    # `boundary.outer_surface_area` (the vEcoli ecoli-shape store paths) for any
    # injected subsystem that needs the geometry split (e.g. an antibiotic
    # transport chain), which otherwise finds nothing writing those stores. Runs
    # right after the mass listener so it reads this tick's `listeners.mass.volume`.
    # A general feature: enable explicitly (features=['cell_geometry']) or via an
    # injected subsystem's `requires_features: ['cell_geometry']`; a no-op otherwise.
    'cell_geometry': {
        'insert_after': 'ecoli-mass-listener',
        'steps': ['cell_geometry_step'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']  # trna_attenuation + mass_conservation off by default

# Opt-in feature modules enabled by a caller for the NEXT build (the generator
# itself fixes the base set to DEFAULT_FEATURES). Mirrors the emitter-override
# pattern. Use enable_features('mass_conservation', ...) before build_composite.
_EXTRA_FEATURES: list = []


def enable_features(*names: str) -> None:
    """Enable opt-in feature modules (e.g. 'mass_conservation') for the next
    baseline build. Call before ``build_composite("ecoli_baseline", ...)``; pass no
    args to clear."""
    global _EXTRA_FEATURES
    _EXTRA_FEATURES = list(names)


def build_execution_layers(features=None):
    """Build EXECUTION_LAYERS from base + enabled feature modules."""
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
        # Replace an existing step name with another in-place (same layer/order).
        for old_name, new_name in (feat.get('replace') or {}).items():
            for layer in layers:
                if isinstance(layer, list):
                    layer[:] = [new_name if s == old_name else s for s in layer]
    return _expand_flushes(layers)


# Convenience re-export: ordered list of all step names in execution order.
FLOW_ORDER = [step for layer in build_execution_layers(DEFAULT_FEATURES) for step in layer]


# ---------------------------------------------------------------------------
# Step instantiation (partitioned / baseline architecture)
# ---------------------------------------------------------------------------

def _get_step_config(
    loader,
    step_name,
    core,
    process_cache=None,
    master_seed=0,
    transcript_initiation_mode: str | None = None,
    polypeptide_initiation_mode: str | None = None,
):
    """Get (instance, topology, edge_type[, in_topo, out_topo]) for a step.

    master_seed: investigation-level seed; per-process seeds are derived via
    _derive_process_seed(master_seed, base_name) so each stochastic process
    gets a distinct, reproducible seed across an ensemble.

    transcript_initiation_mode / polypeptide_initiation_mode: Phase-2 opt-in
    for the respective Process's jump-process kinetics. ``"discrete"``
    (default) → legacy multinomial sampling; ``"poisson"`` → per-target
    Poisson tau-leap. See the respective ``pdmp_initiation_mode`` config
    on each Process.
    """
    from v2ecoli.processes.equilibrium import Equilibrium
    from v2ecoli.processes.two_component_system import TwoComponentSystem
    from v2ecoli.processes.rna_maturation import RnaMaturation
    from v2ecoli.processes.complexation import Complexation
    from v2ecoli.steps.dnaa_box_binding import DnaABoxBinding
    from v2ecoli.steps.rida import Rida
    from v2ecoli.steps.ddah import Ddah
    from v2ecoli.steps.dars import Dars
    from v2ecoli.processes.protein_degradation import ProteinDegradation
    from v2ecoli.processes.transcript_initiation import TranscriptInitiation
    from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
    from v2ecoli.processes.chromosome_replication import ChromosomeReplication
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.processes.metabolism import Metabolism
    from v2ecoli.steps.partition import Requester, Evolver
    from v2ecoli.steps.derivers.mass_deriver import MassDeriver, PostDivisionMassDeriver
    from v2ecoli.steps.derivers.rna_synth_prob import RnaSynthProb
    from v2ecoli.steps.derivers.dna_supercoiling import DnaSupercoiling
    from v2ecoli.steps.derivers.replication_data import ReplicationData
    from v2ecoli.steps.derivers.rnap_data import RnapData
    from v2ecoli.steps.derivers.translation_deriver import TranslationDeriver
    from v2ecoli.steps.media_update import MediaUpdate
    from v2ecoli.steps.exchange_data import ExchangeData

    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Handle allocators
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

    # Consolidated counts deriver: one step computing the RNA / monomer /
    # unique-molecule count readouts (byte-identical to the three former
    # listeners). Assemble its config from the three former config names.
    if step_name == 'counts_deriver':
        from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
        # Flat merge of the three former configs. Order matters: unique's
        # unique_ids (the one actually used) overwrites monomer's unused copy.
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

    # dnaa-3 Phase 2: DnaA-box binding step. No ParCa-generated config — built
    # from class defaults + cell_density / n_avogadro from the equilibrium
    # config + bulk_mass_data / submass_indices from tf_binding (used to
    # update DnaA_box.massDiff_* when DnaA moves bulk → bound).
    if step_name == 'dnaa-box-binding':
        try:
            eq_cfg = loader.get_config_by_name('ecoli-equilibrium') or {}
        except (KeyError, AttributeError):
            eq_cfg = {}
        try:
            tf_cfg = loader.get_config_by_name('ecoli-tf-binding') or {}
        except (KeyError, AttributeError):
            tf_cfg = {}
        dnaa_cfg = {
            'cell_density': eq_cfg.get('cell_density', 1100.0),
            'n_avogadro': eq_cfg.get('n_avogadro', 6.02214076e23),
            'seed': _derive_process_seed(master_seed, 'dnaa-box-binding'),
            'time_step': 1,
            'bulk_mass_data': tf_cfg.get('bulk_mass_data'),
            'bulk_molecule_ids': tf_cfg.get('bulk_molecule_ids'),
            'submass_indices': tf_cfg.get('submass_indices'),
        }
        instance = _make_instance(DnaABoxBinding, dnaa_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # dnaa-5: RIDA — replisome-coupled DnaA-ATP inactivation. No ParCa config;
    # built from class defaults. rate_multiplier=0.0 gives the rida-knockout
    # variant (set via env RIDA_RATE_MULTIPLIER for the knockout sweep).
    if step_name == 'rida':
        rida_cfg = {
            'rate_multiplier': float(os.environ.get('RIDA_RATE_MULTIPLIER', '1.0')),
            'seed': _derive_process_seed(master_seed, 'rida'),
            'time_step': 1,
        }
        instance = _make_instance(Rida, rida_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # dnaa-5: DDAH — datA-locus-coupled DnaA-ATP hydrolysis.
    if step_name == 'ddah':
        ddah_cfg = {
            'rate_multiplier': float(os.environ.get('DDAH_RATE_MULTIPLIER', '1.0')),
            'seed': _derive_process_seed(master_seed, 'ddah'),
            'time_step': 1,
        }
        instance = _make_instance(Ddah, ddah_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # dnaa-5: DARS1/DARS2 — locus-copy-number-coupled DnaA reactivation.
    if step_name == 'dars':
        dars_cfg = {
            'dars1_multiplier': float(os.environ.get('DARS1_RATE_MULTIPLIER', '1.0')),
            'dars2_multiplier': float(os.environ.get('DARS2_RATE_MULTIPLIER', '1.0')),
            'seed': _derive_process_seed(master_seed, 'dars'),
            'time_step': 1,
        }
        instance = _make_instance(Dars, dars_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # cell_geometry feature: native cell-shape geometry deriver. No ParCa
    # config; built from class defaults (width_um=1.0, matching vEcoli
    # shape.py's default width).
    if step_name == 'cell_geometry_step':
        from v2ecoli.steps.derivers.cell_geometry import CellGeometry
        instance = _make_instance(CellGeometry, {}, core)
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

    # _instantiate_step inlined here (baseline/partitioned version)
    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-metabolism': Metabolism,
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
        'media_update': MediaUpdate,
        'exchange_data': ExchangeData,
    }

    from v2ecoli.library.config_resolver import resolve_config
    config = resolve_config(config) if config else config

    if isinstance(config, dict) and "seed" in config:
        config["seed"] = _derive_process_seed(master_seed, base_name)

    # Phase-2: thread {transcript,polypeptide}_initiation_mode overrides
    # into the ParCa-generated config so each Process's initialize()
    # picks up the jump-process mode.
    if (
        transcript_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-transcript-initiation'
    ):
        config["pdmp_initiation_mode"] = transcript_initiation_mode
    if (
        polypeptide_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-polypeptide-initiation'
    ):
        config["pdmp_initiation_mode"] = polypeptide_initiation_mode

    # Partitioned processes: wrap with generic Requester/Evolver
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
            }, core=core)
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
            }, core=core)
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

    # Standalone steps
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


def _build_batch_document(
    core: Any,
    *,
    seed: int,
    n_seeds: int,
    n_generations: int,
    single_daughters: bool,
    time_step: float,
    max_duration: float,
    cache_dir: str,
    out_dir: str,
    experiment_id: str,
    emitter: str,
    analyses: Any,
    study: str,
    parallel: str,
    variants: dict | None,
    knockouts: list[str] | None,
    config_overrides: dict | None,
    media: str,
    independent_founders: bool = False,
    founder_sim_data: str = "",
    variant: int = 0,
    injected_processes: dict | None = None,
    features: list | None = None,
    ppgpp_regulation: bool = True,
    trna_attenuation: bool = False,
    supercoiling: bool = False,
    mass_conservation: bool = False,
    exchange_fluxes: dict | None = None,
    exchange_flux_basis: str | None = None,
    transcript_initiation_mode: str = "discrete",
    polypeptide_initiation_mode: str = "discrete",
    initial_carry_state_path: str = "",
    initial_generation_index: int = 0,
    daughter_state_out_path: str = "",
) -> dict:
    """Build the batch-orchestrator document (seeds × generations lineage).

    ``baseline`` dispatches here when ``n_seeds>1`` or ``n_generations>1``.
    Returns a cheap one-step document whose ``BatchBaselineRunner`` fans out one
    baseline lineage per seed at RUN time (Ray, sequential fallback), emitting to
    a shared hive-partitioned parquet sweep (+ per-lineage zarr) and then running
    the post-simulation analysis flush. Absorbs the former ``batch_baseline``
    composite; every per-seed cell is built by ``baseline`` itself.

    Parameter mapping: baseline's ``seed`` is the lineage base seed (seeds are
    ``seed .. seed+n_seeds-1``); ``knockouts`` + ``config_overrides`` fold into
    the runner's panel-wide ``base_config_overrides`` (applied to every seed);
    ``media`` threads through to each per-seed ``baseline`` build.

    ``initial_carry_state_path``/``initial_generation_index``/
    ``daughter_state_out_path`` (backlog item 34): a wave orchestrator's own
    per-seed-per-generation checkpoint/resume keys, passed straight through to
    ``BatchBaselineRunner`` -> ``run_workflow`` -> ``meta_composite.py``'s
    per-branch ``LineageProcess`` config. Empty/0 (default) = today's
    single-invocation-runs-every-generation behavior, unchanged.
    """
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.perturbations import translation_efficiency_override
    from v2ecoli.steps.batch_baseline_runner import (
        BatchBaselineRunner, DEFAULT_EMITTER)
    from v2ecoli.composites._helpers import _make_instance, make_edge

    # Emitter reconciliation: the single-cell sinks 'sqlite'/'null' are
    # meaningless for a multi-lineage sweep. A batch accepts parquet | xarray |
    # both. (baseline's default 'parquet' is valid here; pass 'both' for the
    # per-lineage zarr the dashboard per-run charts read.)
    batch_emitter = emitter or "parquet"
    if batch_emitter in ("sqlite", "null"):
        raise ValueError(
            f"emitter={emitter!r} is single-cell only; a batch run "
            "(n_seeds>1 or n_generations>1) accepts 'parquet', 'xarray', or "
            "'both'.")

    # Knockouts + config_overrides -> panel-wide base_config_overrides, applied to
    # EVERY seed's lineage (not one variant arm). Resolve knockouts once against
    # the cache; a caller config_override key wins on a clash.
    base_config_overrides: dict = {}
    if knockouts:
        bundle = load_cache_bundle(cache_dir)
        base_config_overrides = translation_efficiency_override(
            bundle, list(knockouts))
    if config_overrides:
        base_config_overrides = {**base_config_overrides, **config_overrides}

    runner_config = {
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "base_seed": int(seed),
        "single_daughters": bool(single_daughters),
        "time_step": float(time_step),
        "max_duration": float(max_duration),
        "variants": dict(variants or {}),
        # Base offset for every branch's variant_index (mirrors `seed` above
        # offsetting each seed) — threaded through runner_config ->
        # build_workflow_config -> expand_branches -> _lineage_node so a batch
        # dispatch partitions its emitter output starting at the caller's
        # requested variant index instead of always colliding on variant=0
        # (P0-10 batch-mode coverage fix; same threading pattern as
        # injected_processes below).
        "variant": int(variant),
        "cache_dir": cache_dir,
        "out_dir": out_dir,
        "experiment_id": experiment_id,
        "emitter": batch_emitter or DEFAULT_EMITTER,
        "analyses": analyses,
        "study": study,
        "parallel": parallel or "",
        "base_config_overrides": base_config_overrides,
        "media": media,
        "independent_founders": independent_founders,
        "founder_sim_data": founder_sim_data,
        # Per-cell biological build kwargs (metabolism-redux/violacein swap,
        # feature toggles, exchange-flux readouts, PDMP initiation modes).
        # WITHOUT these in the runner config they never reach build_workflow_config
        # -> _lineage_node -> each generation's baseline() build, so an injected
        # batch run silently degrades to basal FBA (audit: batch mode dropped
        # injected_processes). Thread them so every generation cell is built with
        # the SAME biological configuration the single-cell path uses.
        "injected_processes": dict(injected_processes or {}),
        "features": list(features or []),
        "ppgpp_regulation": bool(ppgpp_regulation),
        "trna_attenuation": bool(trna_attenuation),
        "supercoiling": bool(supercoiling),
        "mass_conservation": bool(mass_conservation),
        "exchange_fluxes": dict(exchange_fluxes or {}),
        "exchange_flux_basis": exchange_flux_basis or "",
        "transcript_initiation_mode": transcript_initiation_mode or "discrete",
        "polypeptide_initiation_mode": polypeptide_initiation_mode or "discrete",
        "initial_carry_state_path": initial_carry_state_path,
        "initial_generation_index": int(initial_generation_index),
        "daughter_state_out_path": daughter_state_out_path,
    }
    runner = _make_instance(BatchBaselineRunner, runner_config, core)
    state = {
        "batch": {},  # empty; the runner writes per-seed results here at run time
        "global_time": 0.0,
        "batch_runner": make_edge(
            runner, BatchBaselineRunner.topology, edge_type="step",
            config=runner_config),
    }
    # Install a top-level observation sink so the batch orchestrator document is
    # not left "observing nothing". Without an emitter node here,
    # CompositeSpec._with_emitters installs the generator's declared ParquetEmitter
    # default with an EMPTY config, which raises "ParquetEmitter requires either
    # config['out_dir'] or config['out_uri']" at Composite() realize -- before any
    # step runs (issue #496). Resolve out_dir exactly as the single-cell path does.
    from process_bigraph.emitter import install_emitters  # noqa: PLC0415
    _emit_out_dir = out_dir
    if not _emit_out_dir:
        _ws_root = _find_workspace_root()
        _emit_out_dir = (str(_ws_root / ".pbg" / "parquet-runs")
                         if _ws_root is not None else "out/parquet")
    state = install_emitters(
        state,
        [{"address": "local:ParquetEmitter",
          "config": {"out_dir": _emit_out_dir},
          "paths": ["global_time"]}],
        core=core)
    return {"state": state}


def validate_seed_exchange_species(injected_processes: dict | None) -> list:
    """Validate ``injected_processes["seed_exchange_species"]`` and return it.

    Returns ``[]`` when the key is absent. Raises ``ValueError`` on a shape that
    would seed the WRONG keys or none at all -- which in this store means a run
    that completes clean and reads zero, so every case here fails loud.

    ⚠ Called BEFORE the batch/lineage branch as well as on the single-cell path.
    The seeding itself can only happen where the initial state is assembled, but
    the *validation* must not: ``n_seeds>1`` / ``n_generations>1`` /
    ``stop_at_division`` return a batch document early, and a malformed
    declaration would otherwise be stored verbatim and only raise inside each
    worker after dispatch. Multi-seed is the production shape, so the guard has
    to hold there first.

    ⚠ Presence, not truthiness: an explicitly empty/None/false value is a
    distinct thing from a mistyped one and must not skip the type check.
    """
    if injected_processes is None:
        return []
    if "seed_exchange_species" not in injected_processes:
        return []
    declared = injected_processes["seed_exchange_species"]
    if declared is None:
        return []
    # A bare string is iterable, so `seed_exchange_species: "MY-PRODUCT"` -- the
    # natural single-item form in a config -- would otherwise seed one key per
    # CHARACTER and never the declared species. A dict would iterate keys and
    # silently discard its values.
    if not isinstance(declared, (list, tuple, set, frozenset)):
        raise ValueError(
            "injected_processes['seed_exchange_species'] takes a LIST of "
            f"exchange species ids; got {type(declared).__name__}. "
            "A single species must still be a list: ['MY-PRODUCT'].")
    for species in declared:
        if not isinstance(species, str) or not species:
            raise ValueError(
                "injected_processes['seed_exchange_species'] takes exchange "
                f"species ids (non-empty strings); got {species!r}.")
        if species.endswith("]"):
            raise ValueError(
                "injected_processes['seed_exchange_species'] takes BARE "
                f"species ids without a compartment suffix; got {species!r}. "
                "Writers of environment.exchange strip the compartment, so a "
                "tagged id would seed a key nothing ever writes to.")
    return list(declared)


def assert_injection_sourcing(injected_processes: dict | None) -> None:
    """Enforce v2ecoli's native-only injection policy.

    v2ecoli builds injected processes fork-free off its OWN bundle simData.
    Fork-sourcing has been removed, so a non-empty ``injected_processes.fork_repo``
    is now a hard error — inject native pbg processes (with ``fork_repo`` empty)
    instead.
    """
    if not injected_processes:
        return
    fork_repo = injected_processes.get("fork_repo") or ""
    if fork_repo:
        raise ValueError(
            f"injected_processes.fork_repo={fork_repo!r} is set, but fork-sourcing "
            f"has been removed — v2ecoli is native-only. Use native pbg processes "
            f"(fork_repo empty) for injected add/swap.")


WCM_PARAMETERS = {
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
        "match_simdata": {
            "type": "string",
            "default": None,
            "description": "Optional path to a REFERENCE vEcoli simData.cPickle "
                           "(e.g. from a paired comparison run). When set, this "
                           "composite's initial bulk molecule counts are "
                           "overlaid from that reference's genuine-vEcoli "
                           "pre-run state, so a candidate/reference pair driven "
                           "from the same simData start from an identical t=0 — "
                           "the same matched-initial-state mechanism "
                           "scripts/run_comparison_ensemble.py applies via "
                           "--match-vecoli-simdata/--match-initial-state, "
                           "expressed declaratively. Default None/empty leaves "
                           "the baseline's own cached initial state untouched.",
        },
        "match_condition": {
            "type": "string",
            "default": "basal",
            "description": "vEcoli growth condition to build the REFERENCE "
                           "engine under when resolving match_simdata's "
                           "overlay (matches build_vivarium_ecoli's own "
                           "condition param). Only consulted when "
                           "match_simdata is set. A paired comparison must "
                           "pass the SAME condition its config drives on "
                           "both engines -- e.g. a with_aa config's candidate "
                           "must set match_condition='with_aa', not the "
                           "default 'basal', or the overlay is drawn from "
                           "the wrong reference condition. Default 'basal' "
                           "for back-compat with match_simdata callers that "
                           "predate per-config condition threading (Task 4).",
        },
        "transcript_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "Phase-2 jump-process opt-in for transcription initiation. "
                "'discrete' (default, legacy): multinomial event distribution "
                "with exact Σ N_i = n_target. 'poisson': per-promoter "
                "Poisson(n_target · p_i) tau-leap."
            ),
        },
        "polypeptide_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "Phase-2 jump-process opt-in for translation initiation. "
                "Same dispatch as transcript_initiation_mode but for "
                "PolypeptideInitiation; ribosome activation per protein "
                "becomes per-protein Poisson(n_target · p_i) tau-leap "
                "instead of one global multinomial draw."
            ),
        },
        "config_overrides": {
            "type": "map",
            "default": {},
            "description": "Declarative '<process>.<key>': value config overrides (variants)",
        },
        "knockouts": {
            "type": "list",
            "default": [],
            "description": "Genes to knock out at the translation level — EcoCyc "
                           "gene ids (EG10526) or monomer ids (LACY-MONOMER[c]). "
                           "Each named gene's translation efficiency is zeroed on "
                           "the cached polypeptide-initiation config, so no protein "
                           "is made — a functional knockout with no ParCa re-fit. "
                           "Empty = plain baseline. See v2ecoli.perturbations.",
        },
        "media": {
            "type": "string",
            "default": "minimal",
            "description": "Initial growth medium — any condition in the cache's "
                           "saved_media (e.g. 'minimal_plus_amino_acids', "
                           "'minimal_succinate', 'minimal_minus_oxygen'). Sets the "
                           "environment's initial media_id so media_update shifts "
                           "the cell onto that condition on the first tick and "
                           "metabolism responds (e.g. amino-acid-rich media grows "
                           "faster) — a lightweight media perturbation from the "
                           "existing cache, no ParCa re-fit. Default 'minimal' = "
                           "unchanged. For a rigorously-calibrated condition, run a "
                           "per-condition ParCa cache instead (see showcase-4).",
        },
        "independent_founders": {
            "type": "boolean",
            "default": False,
            "description": "Opt-in: re-draw the founder (t=0 cell state) per "
                           "lineage_seed from founder_sim_data instead of loading "
                           "the shared cached initial_state, so a multiseed "
                           "ensemble's spread reflects true cell-to-cell founder "
                           "variability, not just downstream per-process "
                           "stochasticity. Requires founder_sim_data. Slower "
                           "(loads sim_data + regenerates initial conditions per "
                           "seed); default False keeps the fast shared-founder path.",
        },
        "founder_sim_data": {
            "type": "string",
            "default": "",
            "description": "Path to the v2 simData.cPickle to re-draw per-seed "
                           "founders from when independent_founders is set — the "
                           "same sim_data the cache was built from. Each seed gets "
                           "LoadSimData(sim_data_path, seed).generate_initial_state(). "
                           "Ignored unless independent_founders=True.",
        },
        "features": {
            "type": "list",
            "default": [],
            "description": "Opt-in feature-module names to insert in addition to "
                           "the boolean toggles (e.g. ['mass_conservation']). "
                           "Each must be a key in FEATURE_MODULES.",
        },
        # --- Biological feature toggles (insert/remove feature-module steps) ---
        "ppgpp_regulation": {
            "type": "bool",
            "default": True,
            "description": "Enable ppGpp-mediated regulation of transcription "
                           "initiation (ppgpp-initiation step). On by default.",
        },
        "trna_attenuation": {
            "type": "bool",
            "default": False,
            "description": "Enable tRNA transcriptional attenuation "
                           "(trna-attenuation-config step). Off by default.",
        },
        "supercoiling": {
            "type": "bool",
            "default": False,
            "description": "Enable DNA supercoiling dynamics (dna-supercoiling-step "
                           "+ dna_supercoiling_listener). Off by default.",
        },
        "mass_conservation": {
            "type": "bool",
            "default": False,
            "description": "Enable the opt-in runtime mass-conservation check "
                           "(ecoli-mass-conservation step). Off by default — the "
                           "residual is not yet calibrated, so it warns each tick.",
        },
        "exchange_fluxes": {
            "type": "map",
            "default": {},
            "description": "{leaf_name: exchange_key} — re-home named "
                           "environment.exchange fluxes onto "
                           "listeners.exchange_flux.<leaf> so the listeners-only "
                           "XArray view carries them (e.g. "
                           "{'glucose_exchange': 'GLC[p]'}). Empty = off.",
        },
        "exchange_flux_basis": {
            "type": "string",
            "choices": ["counts", "gdcw"],
            "default": "counts",
            "description": "WHICH QUANTITY the exchange_flux leaves carry. "
                           "'counts' re-homes environment.exchange verbatim — a "
                           "LINEAGE-CUMULATIVE molecule total that does not reset "
                           "at division, so its time-average is not a rate. "
                           "'gdcw' differences it and normalises to mmol/gDCW/h, "
                           "which is the quantity a genuine vEcoli reports for "
                           "its own exchanges and therefore the one that is "
                           "comparable across engines. These are different "
                           "measurements, not different units.",
        },
        # --- Observation sink selection ---
        "emitter": {
            "type": "string",
            "choices": ["parquet", "sqlite", "xarray", "null", "both"],
            "default": "parquet",
            "description": "Observation sink. Single-cell: parquet (hive "
                           "column store, default), sqlite (time-series db), "
                           "xarray (in-memory arrays), or null (global_time only). "
                           "Batch (n_seeds>1 / n_generations>1): parquet, xarray "
                           "(per-lineage zarr), or 'both' (parquet + zarr; what "
                           "the dashboard per-run charts read). 'both' is batch "
                           "only; 'sqlite'/'null' are single-cell only.",
        },
        "emitter_out_dir": {
            "type": "string",
            "default": "",
            "description": "Single-cell observation-sink output directory "
                           "override. Empty (default) = unchanged behavior: "
                           "each emitter resolves its own default location "
                           "(parquet -> workspace .pbg/parquet-runs, sqlite -> "
                           "workspace .pbg or out/, xarray -> workspace "
                           ".pbg/xarray-runs or out/xarray). Set this to pin "
                           "the sink to an explicit directory instead — e.g. "
                           "a standalone/provisioned run with no workspace on "
                           "disk (a generic runner building this composite via "
                           "core_extensions alone, with no dashboard around "
                           "it). Ignored for emitter='null'. Batch runs use "
                           "the separate out_dir param instead.",
        },
        "injected_processes": {
            "type": "map",
            "default": {},
            "description": "Native process-injection spec "
                           "{add_processes, swap_processes, process_configs, "
                           "topology, time_step}; empty = none. fork_repo must be "
                           "empty — fork-sourcing is removed, v2ecoli is native-only. "
                           "Also honoured: seed_bulk_species (a list of "
                           "{id, molar_mass_g_per_mol} SPEC DICTS, ids "
                           "compartment-TAGGED) and seed_exchange_species (a LIST "
                           "of BARE environment.exchange id strings, seeded at 0.0 "
                           "so an injected process's secretion has a key to land "
                           "in). Note the two differ in BOTH shape and id "
                           "convention.",
        },
        # --- Batch / lineage knobs (absorbed from the former batch_baseline) ----
        # n_seeds>1 OR n_generations>1 switches baseline from a single 55-process
        # cell document to a one-step batch-orchestrator document that fans out
        # one baseline lineage per seed at RUN time (Ray, sequential fallback),
        # emits to a shared sweep, then runs the post-sim analysis flush. The
        # single-cell default (n_seeds=1, n_generations=1) is unchanged.
        "n_seeds": {
            "type": "integer",
            "default": 1,
            "description": "Number of independent seeds to run (vEcoli's "
                           "n_init_sims); seeds are seed .. seed+n_seeds-1. 1 = a "
                           "single cell (default). >1 launches a batch run.",
        },
        "n_generations": {
            "type": "integer",
            "default": 1,
            "description": "Cell-division generations to follow per seed lineage — "
                           "engages the division-aware LineageProcess stop under "
                           "batch mode (n_seeds>1 or n_generations>1) OR under "
                           "stop_at_division=True. It has NO effect on the PLAIN "
                           "single-cell default (n_seeds=1, n_generations=1, "
                           "stop_at_division=False): that run has no division-stop "
                           "and simulates the full requested step count, continuing "
                           "past the cell's own division (see issue #495). To bound "
                           "a single-cell run to one cell cycle, pass "
                           "stop_at_division=True (routes through the lineage path "
                           "at n_seeds=1, generations=n_generations).",
        },
        "stop_at_division": {
            "type": "bool",
            "default": False,
            "description": "Single-cell division-stop opt-in (issue #495, Option "
                           "A). True bounds a single-cell run to ONE cell cycle by "
                           "routing this build through the lineage machinery at "
                           "n_seeds=1, generations=n_generations, where "
                           "LineageProcess stops at the first division. Two "
                           "consequences: (i) observations take the lineage "
                           "'generation=N/agent_id' layout, not the flat "
                           "single-cell agents/0 layout; (ii) INCOMPATIBLE with "
                           "match_simdata (raises ValueError). False (default) = "
                           "unchanged full-budget single-cell run.",
        },
        "single_daughters": {
            "type": "bool",
            "default": True,
            "description": "Batch runs only: follow ONE daughter per division "
                           "(vEcoli's default). Off = binary-tree lineage.",
        },
        "time_step": {
            "type": "float",
            "default": 1.0,
            "description": "Batch runs only: simulation time step in seconds.",
        },
        "max_duration": {
            "type": "float",
            "default": 3600.0,
            "description": "Batch runs only: per-generation sim-time cap (seconds).",
        },
        "variants": {
            "type": "map",
            "default": {},
            "description": "Batch runs only: vEcoli-style variant grid "
                           "({name: {target, value}}) crossed with the seed range.",
        },
        "variant": {
            "type": "integer",
            "default": 0,
            "description": "This cell's variant index in a multivariant sweep. "
                           "Stamped into the parquet hive partition column "
                           "(variant=<idx>) so each variant's rows are stored "
                           "and analysed separately; defaults to 0 for a single "
                           "(baseline) arm. Set per branch at fan-out time.",
        },
        "out_dir": {
            "type": "string",
            "default": "",
            "description": "Batch runs only: output root for the parquet sweep + "
                           "zarr stores. Empty = this run's own dir under the "
                           "workbench, else out/batch_baseline.",
        },
        "experiment_id": {
            "type": "string",
            "default": "baseline",
            "description": "Batch runs only: id stamped into the parquet "
                           "partitions and zarr store names.",
        },
        "analyses": {
            "type": "string",
            "choices": ["applicable", "none"],
            "default": "applicable",
            "description": "Batch runs only: 'applicable' runs every ported "
                           "analysis at the scales this batch covers; 'none' skips.",
        },
        "study": {
            "type": "string",
            "default": "",
            "description": "Batch runs only: owning study slug for the flush's "
                           "outputs. Empty = infer from out_dir.",
        },
        "parallel": {
            "type": "string",
            "default": "ray",
            "description": "Batch runs only: 'ray' to fan out across worker "
                           "processes; '' for sequential.",
        },
        "initial_carry_state_path": {
            "type": "string",
            "default": "",
            "description": "Batch runs only, per-generation checkpoint/resume "
                           "(backlog item 34): path to a prior generation's "
                           "saved daughter state, threaded to every branch's "
                           "LineageProcess. Empty = fresh lineage at generation "
                           "0 (default, unchanged single-invocation behavior). "
                           "Set together with initial_generation_index by a "
                           "wave orchestrator resuming a checkpointed lineage.",
        },
        "initial_generation_index": {
            "type": "integer",
            "default": 0,
            "description": "Batch runs only, per-generation checkpoint/resume: "
                           "the generation index this invocation resumes at. "
                           "Must be 0 when initial_carry_state_path is empty "
                           "(enforced by LineageProcess at run time).",
        },
        "daughter_state_out_path": {
            "type": "string",
            "default": "",
            "description": "Batch runs only, per-generation checkpoint/resume: "
                           "path to persist this invocation's daughter state "
                           "to, for the next generation's job to resume from. "
                           "Empty = no checkpoint hand-off.",
        },
}


# --- Batch-mode parameter coverage (fail-loud guard) ------------------------
# Every WCM_PARAMETERS key must appear in EXACTLY ONE of these two sets. The
# batch dispatch (baseline() -> _build_batch_document) then either forwards the
# key into the batch-orchestrator document (so it reaches every generation's
# per-cell baseline() build via BatchBaselineRunner -> build_workflow_config ->
# _lineage_node -> LineageProcess) or, for a single-cell-only key, refuses to run
# a batch when it is set to a non-default value. This makes it structurally
# impossible for a new baseline() kwarg to be silently dropped in batch mode —
# the exact defect that dropped `injected_processes` and degraded an injected
# metabolism-redux/violacein batch to a basal FBA lineage (pipeline audit).
_BATCH_FORWARDED_PARAMETERS = frozenset({
    # Dispatch switches (consumed by the batch/lineage routing itself).
    "n_seeds", "n_generations", "stop_at_division",
    # Threaded into the batch document / runner config -> workflow config.
    "seed", "cache_dir", "config_overrides", "knockouts", "media",
    "independent_founders", "founder_sim_data",
    "single_daughters", "time_step", "max_duration", "variants", "variant",
    "out_dir", "experiment_id", "analyses", "study", "parallel", "emitter",
    "initial_carry_state_path", "initial_generation_index",
    "daughter_state_out_path",
    # Per-cell biological build kwargs threaded so every generation cell is built
    # with the same biology as the single-cell path (audit fix).
    "injected_processes", "features", "ppgpp_regulation", "trna_attenuation",
    "supercoiling", "mass_conservation", "exchange_fluxes", "exchange_flux_basis",
    "transcript_initiation_mode", "polypeptide_initiation_mode",
})
# Single-cell-only knobs: build-time overlays / sinks the run-time lineage
# fan-out has no wiring for. A non-default value in batch mode raises (mirrors
# the match_simdata guard) rather than being silently ignored.
_BATCH_INCOMPATIBLE_PARAMETERS = frozenset({
    "match_simdata",     # build-time single-cell initial-state overlay
    "match_condition",   # only consulted alongside match_simdata
    "emitter_out_dir",   # single-cell sink dir; batch uses out_dir instead
})


def _assert_batch_parameter_coverage(values: dict) -> None:
    """Enforce that every WCM parameter is classified for batch mode.

    ``values`` is the caller's local namespace (name -> value). Raises a
    developer-facing ``RuntimeError`` if a WCM_PARAMETERS key is unclassified
    (a new kwarg was added without deciding its batch behavior), and a
    user-facing ``ValueError`` if a batch-incompatible key is set to a
    non-default value in batch mode.
    """
    classified = _BATCH_FORWARDED_PARAMETERS | _BATCH_INCOMPATIBLE_PARAMETERS
    overlap = _BATCH_FORWARDED_PARAMETERS & _BATCH_INCOMPATIBLE_PARAMETERS
    if overlap:
        raise RuntimeError(
            f"ecoli_baseline batch guard misconfigured: parameter(s) "
            f"{sorted(overlap)} are listed as BOTH forwarded and "
            "batch-incompatible; a WCM parameter must be in exactly one set.")
    unclassified = set(WCM_PARAMETERS) - classified
    if unclassified:
        raise RuntimeError(
            f"ecoli_baseline batch guard: WCM parameter(s) {sorted(unclassified)} "
            "are neither forwarded into the batch document nor listed "
            "batch-incompatible. A new baseline() kwarg was added without "
            "deciding its batch behavior — classify it in "
            "_BATCH_FORWARDED_PARAMETERS (and actually thread it through "
            "_build_batch_document -> runner_config -> build_workflow_config -> "
            "_lineage_node) or _BATCH_INCOMPATIBLE_PARAMETERS. This guard exists "
            "so the next dropped kwarg fails loudly instead of silently "
            "no-op'ing in batch mode (see the injected_processes audit).")
    for key in sorted(_BATCH_INCOMPATIBLE_PARAMETERS):
        if key not in values:
            continue
        default = WCM_PARAMETERS[key].get("default")
        if values[key] != default:
            raise ValueError(
                f"ecoli_baseline: {key}={values[key]!r} is a single-cell-only "
                "parameter and is not supported in batch mode (n_seeds>1 or "
                "n_generations>1); running the batch would silently drop it. "
                f"Pass n_seeds=1, n_generations=1, or leave {key} at its "
                f"default ({default!r}).")


@composite_generator(
    name="ecoli_baseline",
    description="55-process partitioned whole-cell E. coli model — upstream-parity architecture",
    parameters=WCM_PARAMETERS,
    default_n_steps=2700,
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    # Lets a generic runner (e.g. process_bigraph.workflow.provision) provision
    # a BARE core with exactly this composite's required types/links, without
    # needing to import v2ecoli.core.build_core directly. See
    # v2ecoli.core.register_ecoli_core + v2ecoli/__init__.py's register_types
    # convention hook (same function, two discovery paths).
    core_extensions=[register_ecoli_core],
    emitters=[
        {
            # Default observation sink for standalone builds: a vEcoli-shaped
            # ParquetEmitter (hive-partitioned, column-oriented — captures the
            # raw bulk count array + listeners that downstream/vEcoli-parity
            # analyses need). out_dir is omitted on purpose: the emitter step
            # resolves it to <workspace>/.pbg/parquet-runs. External overrides
            # (set_parquet_emitter_override / set_emitter_override) still win.
            "address": "local:ParquetEmitter",
            "config": {},
            "paths": ["global_time", "bulk", "listeners"],
        },
    ],
)
def baseline(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    match_simdata: str | None = None,
    match_condition: str = "basal",
    transcript_initiation_mode: str = "discrete",
    polypeptide_initiation_mode: str = "discrete",
    config_overrides: dict | None = None,
    knockouts: list[str] | None = None,
    media: str = "minimal",
    independent_founders: bool = False,
    founder_sim_data: str = "",
    features: list | None = None,
    ppgpp_regulation: bool = True,
    trna_attenuation: bool = False,
    supercoiling: bool = False,
    mass_conservation: bool = False,
    exchange_fluxes: dict | None = None,
    exchange_flux_basis: str | None = None,
    emitter: str = "parquet",
    emitter_out_dir: str = "",
    bundle: dict | None = None,
    injected_processes: dict | None = None,
    n_seeds: int = 1,
    n_generations: int = 1,
    stop_at_division: bool = False,
    single_daughters: bool = True,
    time_step: float = 1.0,
    max_duration: float = 3600.0,
    variants: dict | None = None,
    variant: int = 0,
    out_dir: str = "",
    experiment_id: str = "baseline",
    analyses: Any = "applicable",
    study: str = "",
    parallel: str = "ray",
    initial_carry_state_path: str = "",
    initial_generation_index: int = 0,
    daughter_state_out_path: str = "",
) -> dict:
    """Build the process-bigraph state document for the baseline architecture.

    Migrated from ``v2ecoli/generate.py:build_document`` +
    ``v2ecoli/composite.py:_build_from_cache``.  Returns a plain dict
    suitable for ``Composite(doc, core=core)``; does NOT wrap in Composite.

    The biological feature set is assembled from the boolean toggles
    (``ppgpp_regulation`` on by default; the rest off) plus any opt-in
    modules a caller registered via :func:`enable_features` (back-compat).

    Args:
        core: bigraph-schema core.  If None, one is created via build_core().
        seed: Random seed for stochastic initialisation.
        cache_dir: Path to the ParCa cache directory (must contain
            ``initial_state.json`` and ``sim_data_cache.dill``).
        match_simdata: optional path to a reference vEcoli simData.cPickle.
            When set, overlays that reference's genuine-vEcoli pre-run bulk
            molecule counts onto this composite's initial state (the same
            matched-initial-state mechanism run_comparison_ensemble.py
            applies via --match-vecoli-simdata/--match-initial-state).
            None (default) leaves the baseline's own cached initial state
            unchanged.
        match_condition: vEcoli growth condition to build the reference
            engine under when resolving match_simdata's overlay. Only
            consulted when match_simdata is set. Threads a paired
            comparison's per-config condition into the matched-init overlay
            (Task 4's materializer sets this to the same condition it drives
            on the reference `vecoli` composite); defaults to "basal" for
            back-compat.
        transcript_initiation_mode: Phase-2 opt-in for the PDMP transcript
            initiation dispatch — ``discrete`` (default) or the piecewise-
            deterministic mode.
        polypeptide_initiation_mode: same dispatch as
            ``transcript_initiation_mode`` but for polypeptide initiation.
        config_overrides: declarative '<process>.<key>': value config patches
            (variants), applied on top of any knockout patch.
        knockouts: genes to knock out at the translation level (EcoCyc gene ids
            or monomer ids). Each gene's translation efficiency is zeroed on the
            cached polypeptide-initiation config — a functional knockout with no
            ParCa re-fit. Empty/None = plain baseline.
        media: initial growth medium — any condition in the cache's saved_media.
            Sets the environment's initial media_id so media_update shifts the
            cell onto that condition on the first tick and metabolism responds
            (lightweight media perturbation from the existing cache, no ParCa
            re-fit). 'minimal' (default) leaves it unchanged.
        n_seeds, n_generations: >1 on either switches baseline to a BATCH run —
            a one-step orchestrator document that fans out one baseline lineage
            per seed (seeds seed..seed+n_seeds-1) at run time and flushes the
            ported analyses (absorbs the former batch_baseline composite). The
            other batch knobs (single_daughters, time_step, max_duration,
            variants, out_dir, experiment_id, analyses, study, parallel,
            initial_carry_state_path, initial_generation_index,
            daughter_state_out_path) apply only in batch mode;
            knockouts/media/config_overrides carry through to every seed.
            n_seeds==1, n_generations==1 (default) = single cell.

            NOTE (issue #495): the division-aware stop lives ONLY in the
            lineage/batch machinery (BatchBaselineRunner ->
            LineageProcess._run_until_division). The plain single-cell default
            path built below has NO division-stop: the in-cell Division step
            still fires and structurally splits state into daughters, but nothing
            tells the *run* to stop, so it simulates the full requested step
            budget and keeps going past division (now simulating the daughter).
            n_generations therefore does NOT bound a plain single-cell run to one
            cell cycle — it is inert unless n_seeds>1 or n_generations>1 flips
            this into batch mode. To bound a single-cell run to one cell cycle,
            pass stop_at_division=True (see below); that routes this single-cell
            build through the same lineage machinery at n_seeds=1, generations=1
            so LineageProcess stops at the first division.
        stop_at_division: opt-in that bounds a single-cell run to ONE cell cycle
            (issue #495, Option A). When True, this build is routed through the
            lineage/batch machinery at n_seeds=1 and generations=n_generations
            (default 1), where LineageProcess._run_until_division halts the run at
            the first division instead of continuing into the daughter. Two
            consequences of routing through the lineage path:
              (i) the emitted observations take the lineage
                  ``generation=N/agent_id`` layout, NOT the flat single-cell
                  agents/0 layout the plain single-cell build emits; and
              (ii) it is INCOMPATIBLE with match_simdata (a build-time,
                  single-cell initial-state overlay that the run-time lineage
                  fan-out has no wiring for) — combining them raises ValueError.
            False (default) = unchanged full-budget single-cell behavior.
        initial_carry_state_path, initial_generation_index,
            daughter_state_out_path: batch-mode-only per-generation
            checkpoint/resume (backlog item 34) — a wave orchestrator's own
            resume hand-off, passed straight through to each branch's
            LineageProcess. Empty/0 (default) = unchanged single-invocation
            behavior; see BatchBaselineRunner/LineageProcess for the contract.
        ppgpp_regulation: insert the ppGpp-regulation feature module (default on).
        trna_attenuation: insert the tRNA-attenuation feature module (default off).
        supercoiling: insert the DNA-supercoiling feature module (default off).
        mass_conservation: insert the mass-conservation check (default off).
        emitter: observation sink for the internal 'emitter' step — one of
            ``parquet`` (default), ``sqlite``, ``xarray``, ``null``.
        emitter_out_dir: explicit output-directory override for the chosen
            single-cell emitter. Empty (default) = unchanged behavior (each
            sink resolves its own workspace-relative default). Ignored for
            ``emitter="null"``.
        bundle: optional pre-loaded cache bundle (as returned by
            ``load_cache_bundle``). When given, the cache is not re-read from
            ``cache_dir`` — lets callers building many composites from the same
            cache (ensembles, sweeps) load it once and reuse it.

    Returns:
        Process-bigraph document dict with keys ``state``,
        ``skip_initial_steps``, ``sequential_steps``, ``flow_order``.
    """
    if core is None:
        core = build_core()

    # Option A single-cell division-stop guard (issue #495): stop_at_division
    # routes the single-cell build through the lineage machinery, which fans out
    # per-seed lineages at RUN time — match_simdata is a build-time, single-cell
    # initial-state overlay with no wiring in that run-time path, so the two are
    # mutually exclusive (same reason the n_seeds>1/n_generations>1 batch path
    # rejects match_simdata below). Fail loud rather than silently dropping the
    # overlay. A future single-cell orchestrator (Option B) that keeps the flat
    # single-cell build AND stops at division could support both.
    if stop_at_division and match_simdata:
        raise ValueError(
            "stop_at_division=True is incompatible with match_simdata: Option A "
            "(issue #495) bounds the run to one cell cycle by routing through the "
            "lineage machinery, whose run-time per-seed fan-out has no wiring for "
            "the build-time single-cell match_simdata overlay. Pass one or the "
            "other (a future single-cell orchestrator / Option B could support "
            "both).")

    # Batch / lineage dispatch: n_seeds>1, n_generations>1, OR stop_at_division
    # turns baseline from a single 55-process cell into a one-step
    # batch-orchestrator document (absorbs the former batch_baseline composite).
    # stop_at_division routes the single-cell defaults (n_seeds=1,
    # n_generations=1) through this same lineage path so LineageProcess stops at
    # the first division (Option A). The plain single-cell path below is untouched
    # for n_seeds==1, n_generations==1, stop_at_division=False (bit-identical to
    # plain baseline).
    if int(n_seeds) > 1 or int(n_generations) > 1 or stop_at_division:
        # Validate the injected seeding declaration BEFORE dispatching: the
        # batch document is built here but each cell's baseline() runs inside a
        # worker, so a malformed value would otherwise ride through and raise
        # once per lineage after dispatch, in the shape production actually uses.
        validate_seed_exchange_species(injected_processes)
        if match_simdata:
            # Batch mode builds per-seed lineages via BatchBaselineRunner at
            # RUN time, outside this document-building call, so match_simdata
            # (a single-cell, build-time overlay) has no wiring there yet.
            # Fail loud rather than silently ignoring it.
            raise ValueError(
                "match_simdata is not yet supported with n_seeds>1 or "
                "n_generations>1 (batch mode); pass n_seeds=1, "
                "n_generations=1 or omit match_simdata.")
        # Fail-loud coverage guard: every WCM_PARAMETERS key must be either
        # forwarded into the batch document below or explicitly listed as
        # batch-incompatible, and a set-but-unforwarded batch-incompatible key
        # raises here rather than silently no-op'ing in batch mode (the class of
        # bug that dropped injected_processes -> basal FBA). See
        # _assert_batch_parameter_coverage.
        _assert_batch_parameter_coverage(locals())
        return _build_batch_document(
            core, seed=seed, n_seeds=n_seeds, n_generations=n_generations,
            single_daughters=single_daughters, time_step=time_step,
            max_duration=max_duration, cache_dir=cache_dir, out_dir=out_dir,
            experiment_id=experiment_id, emitter=emitter, analyses=analyses,
            study=study, parallel=parallel, variants=variants, variant=variant,
            knockouts=knockouts, config_overrides=config_overrides, media=media,
            independent_founders=independent_founders, founder_sim_data=founder_sim_data,
            injected_processes=injected_processes, features=features,
            ppgpp_regulation=ppgpp_regulation, trna_attenuation=trna_attenuation,
            supercoiling=supercoiling, mass_conservation=mass_conservation,
            exchange_fluxes=exchange_fluxes,
            exchange_flux_basis=exchange_flux_basis,
            transcript_initiation_mode=transcript_initiation_mode,
            polypeptide_initiation_mode=polypeptide_initiation_mode,
            initial_carry_state_path=initial_carry_state_path,
            initial_generation_index=initial_generation_index,
            daughter_state_out_path=daughter_state_out_path)

    if bundle is None:
        bundle = load_cache_bundle(cache_dir)

    # Translation-level gene knockouts (design PR #341, folded in from the former
    # KO_baseline composite). Resolve the knockouts against this same cache bundle
    # into a `translation_efficiencies` config override and merge it UNDER any
    # caller config_overrides — an explicit override key wins on a clash (they are
    # being deliberate). An unknown/non-coding gene raises here, at build time,
    # not silently mid-run. Empty knockouts = plain baseline (no cache touch).
    if knockouts:
        from v2ecoli.perturbations import translation_efficiency_override
        ko = translation_efficiency_override(bundle, list(knockouts))
        config_overrides = {**ko, **(config_overrides or {})}

    # Deep-copy initial_state: a reused bundle (e.g. one load_cache_bundle()
    # shared across many baseline() calls, as in parameter sweeps / UQ ensembles)
    # otherwise hands every composite the SAME initial_state arrays. v2ecoli's
    # in-place bulk arrays then mutate that shared state during run(), so each
    # subsequent build resumes from the previous run's advanced state (mass
    # accumulates across samples, eventually triggering a spurious mid-run
    # division). configs is already deep-copied below for the same reason —
    # initial_state needs the same isolation.
    if independent_founders and founder_sim_data:
        # Opt-in: draw a fresh founder for THIS lineage's seed instead of the
        # shared cached one, so a multiseed ensemble varies at t=0 too (real
        # cell-to-cell founder variability), not only in downstream stochastics.
        initial_state = _independent_founder_state(founder_sim_data, seed, match_condition)
    else:
        initial_state = copy.deepcopy(bundle["initial_state"])

    # Injected bulk-species seeding (opt-in, drug-agnostic). The ParCa cache
    # bundle is built without any injected subsystem's extra species, so an
    # injected process that reads them from the bulk store would raise "Names not
    # found in bulk_names". The candidate loads a pre-built bundle and never
    # re-runs LoadSimData, so we seed the declared species directly onto the
    # bundle-loaded columnar bulk store — count 0, correct submass — with no
    # ParCa/cache rebuild. No-op (unchanged baseline) when both flags are False.
    # Drug-agnostic bulk-species seeding: an injected subsystem declares the bulk
    # species + molar masses it needs seeded via
    # injected_processes["seed_bulk_species"], so the engine holds no drug
    # knowledge (e.g. an antibiotic arm seeds its drug/complex species — the
    # ParCa cache bundle is built without them). count 0 + correct submass,
    # idempotent, no ParCa rebuild.
    _seed_specs = (injected_processes or {}).get("seed_bulk_species")
    if _seed_specs:
        from v2ecoli.library.sim_data import seed_bulk_species
        initial_state["bulk"] = seed_bulk_species(initial_state["bulk"], _seed_specs)

    # Product-agnostic EXCHANGE-store seeding, the environment.exchange sibling of
    # the bulk seeding above: an injected subsystem declares the exchange keys it
    # needs present via injected_processes["seed_exchange_species"], so the engine
    # holds no pathway knowledge (the caller supplies the ids).
    #
    # Why it is needed at all: environment.exchange is a map[float] store
    # initialised from the cache bundle with only the media's external molecules.
    # A bare-float map leaf ACCUMULATES (state + update) -- it updates keys that
    # already exist and never ADDS one. So an injected metabolism that secretes a
    # species the ParCa bundle never registered writes into a key nobody created,
    # and every downstream reader (exchange-flux listeners, a coupled environment)
    # sees nothing. The process runs, the run completes clean, and the product
    # reads bit-exact zero with nothing raising.
    #
    # setdefault, not assignment: a species the bundle already carries keeps its
    # real initial value. Opt-in -- absent or empty leaves the built document
    # byte-identical to today.
    #
    # ⚠ BARE names, no compartment suffix. Every writer of this store strips the
    # compartment (metabolism.py emits `str(molecule[:-3])`), so a compartment-
    # tagged id would seed a key no writer ever touches -- a clean build with a
    # zero product, i.e. the very failure this seam closes. NOTE this differs
    # from the `seed_bulk_species` sibling above, whose ids ARE compartment-
    # tagged ("X[c]"); the two stores use different id conventions.
    _exchange_seed = validate_seed_exchange_species(injected_processes)
    if _exchange_seed:
        _exchange = initial_state.setdefault("environment", {}).setdefault(
            "exchange", {})
        for _species in _exchange_seed:
            _exchange.setdefault(_species, 0.0)

    configs = bundle["configs"]
    if config_overrides:
        # Deep-copy before patching: load_cache_bundle returns the cache dict
        # by reference (lru_cache-shared); mutating it would corrupt other runs.
        configs = copy.deepcopy(configs)
        for path, value in config_overrides.items():
            proc, _, key = path.partition(".")
            if not key:
                raise ValueError(f"override path {path!r} must be '<process>.<key>'.")
            if "." in key:
                raise ValueError(
                    f"override path {path!r}: nested keys not supported; "
                    "use '<process>.<top-level-key>'.")
            if proc not in configs:
                raise KeyError(f"override target process {proc!r} not in cache configs.")
            configs[proc][key] = value
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

    # Assemble the feature set from the explicit boolean toggles. Order follows
    # FEATURE_MODULES so insertions are deterministic. The legacy
    # _EXTRA_FEATURES / enable_features() global is still honoured (back-compat
    # for callers like scripts/pr_session_report.py and the mass-conservation
    # behavior test) and unions in any modules it requested.
    _toggle_features = {
        'ppgpp_regulation': ppgpp_regulation,
        'trna_attenuation': trna_attenuation,
        'supercoiling': supercoiling,
        'mass_conservation': mass_conservation,
    }
    _requested_features = list(features or [])
    features = [name for name, on in _toggle_features.items() if on]
    # exchange_fluxes (non-empty) auto-enables the exchange_flux feature; its map
    # is threaded to the feature step via the external override set below.
    _exchange_fluxes = dict(exchange_fluxes or {})
    # WHICH QUANTITY those leaves carry. Threaded beside the map because the two
    # are meaningless apart: "counts" is a lineage-cumulative molecule total and
    # "gdcw" a per-tick mmol/gDCW/h rate, and a leaf carrying one under the
    # other's name is not a unit error but a different measurement. None keeps
    # the deriver's own default, so an undeclared build is unchanged.
    _exchange_flux_basis = str(exchange_flux_basis or "counts")
    if _exchange_fluxes and 'exchange_flux' not in _requested_features:
        _requested_features.append('exchange_flux')
    # Neutral feature-dependency seam: ANY injected subsystem can declare the
    # general features it needs by name via
    # injected_processes["requires_features"], so the engine never hardcodes a
    # drug->feature link. e.g. the antibiotic transport process declares
    # cell_geometry (it divides counts by periplasm/cytoplasm volume and reads
    # boundary.outer_surface_area, which nothing else in the candidate populates).
    for _rf in (injected_processes or {}).get("requires_features", []) or []:
        if _rf not in _requested_features:
            _requested_features.append(_rf)
    for f in _EXTRA_FEATURES:
        if f not in features:
            features.append(f)
    # Explicit per-call opt-in feature modules (e.g. mass_conservation).
    for f in _requested_features:
        if f not in features:
            features.append(f)

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    # Matched-initial-state (opt-in, declarative): overlay a reference vEcoli's
    # pre-run bulk counts onto this composite's initial state so a paired
    # candidate/reference comparison starts from an identical t=0. See
    # _apply_match_simdata for the reused mechanism. No-op when unset (the
    # default), leaving the cache's own initial state untouched. match_condition
    # threads the config's own condition (Task 4's materializer) into the
    # reference-engine build; defaults to "basal" for back-compat.
    if match_simdata:
        _apply_match_simdata(cell_state, match_simdata=match_simdata, seed=seed,
                              condition=match_condition)

    # Media perturbation (from the existing cache — no ParCa re-fit). The cache's
    # initial environment is 'minimal'; the media_update step swaps in a different
    # condition's external concentrations on the first tick when the environment's
    # media_id differs from its own (config-seeded) current id. So a media change
    # is just: set the initial environment.media_id to a saved_media condition.
    # Validate here so a typo'd condition fails at build time, not silently.
    if media and media != cell_state.get('environment', {}).get('media_id'):
        _saved = (configs.get('media_update') or {}).get('saved_media') or {}
        if media not in _saved:
            raise ValueError(
                f"media={media!r} is not a condition in the cache's saved_media. "
                f"Available: {sorted(_saved)}")
        cell_state.setdefault('environment', {})['media_id'] = media

    # Pre-create virtual stores
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

    # Pre-create feature module stores
    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'enabled': False,
    })
    # cell_geometry feature: the `periplasm` /
    # `cytoplasm` compartment stores are built entirely by the injected vEcoli
    # subsystem's own port materialization (antibiotic-transport-odeint declares
    # periplasm.global.volume / .potential + cytoplasm.global.volume;
    # concentrations_deriver declares periplasm.concentrations.<id>), which lets
    # pbg infer flexible schemas for their dynamic leaves. CellGeometry's own
    # `quantity[...]` output declaration then supplies the applyable schema for
    # the two volume leaves it writes each tick. Deliberately NO pre-seed here:
    # pre-typing part of `periplasm` rigidifies the store so the concentrations
    # leaf can no longer be applied ("apply(None, ...)").

    # Initialize next_update_time for all partitioned processes
    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    # Pre-create shared request/allocate/process stores
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

    # Mock loader: serves cache configs + a minimal sim_data stand-in (see
    # CachedConfigLoader in _helpers — replaces the old nested _CachedLoader).
    loader = CachedConfigLoader(
        configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)
    # Carry the injected/swapped-process spec onto the loader so the DivisionStep
    # config path (_helpers._get_special_step, 'division' branch) can thread it
    # into each daughter's own baseline() rebuild. Without this, daughters revert
    # to the plain FBA baseline at division and a swapped process (e.g. metabolism
    # -> metabolism-redux) is lost across generations. Normal FBA baseline has
    # injected_processes=None -> daughters rebuild plain baseline unchanged.
    loader._injected_processes = injected_processes

    # Same discipline for config_overrides (a variant / sensitivity perturbation)
    # and knockouts — knockouts are already folded INTO config_overrides above, so
    # this single stash carries both. Without it, a perturbation applied as
    # config_overrides is correct in generation 1 and silently reverts to the
    # unperturbed cached configs at division (#505). Empty/None -> daughters
    # rebuild the plain baseline unchanged.
    loader._config_overrides = config_overrides

    # Build execution layers for the requested feature set
    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    # Resolve the 'emitter' param to the right override / declared-default so
    # the internal 'emitter' step materialises the chosen sink. The selection
    # only adjusts this generator's OWN scoped knobs (the declared default decl
    # + the sqlite/null overrides it sets here); it is restored in the finally
    # so nothing leaks into a later composite built in the same process. Any
    # EXTERNAL override a caller set before this build (set_parquet_emitter_override
    # / set_emitter_override / set_null_emitter_override) still wins, because
    # _get_special_step checks those external overrides before the declared
    # default and we only set our own override when none is already active.
    import v2ecoli.composites._helpers as _h  # noqa: PLC0415

    if emitter not in ("parquet", "sqlite", "xarray", "null"):
        raise ValueError(
            f"emitter={emitter!r} not recognised; expected one of "
            "parquet, sqlite, xarray, null.")

    _emitter_decls = emitter_defaults(baseline)
    _default_decl = _emitter_decls[0] if _emitter_decls else None
    if _default_decl is not None:
        # Thread the run-identity fields into the declared default (parquet)
        # emitter's config so its hive partition columns are correct per cell.
        # experiment_id and variant are declared baseline() params that
        # otherwise never reach the emitter (_build_declared_emitter would fall
        # back to "default" / variant=0), so every variant in a multivariant
        # sweep would collapse onto partition ``variant=0`` and the
        # multivariant KPI analysis would see them as one. emitter_out_dir, when
        # set, still pins out_dir instead of the workspace-relative default.
        _decl_cfg = {
            **_default_decl.get("config", {}),
            "experiment_id": experiment_id,
            "variant": int(variant),
        }
        if emitter_out_dir:
            _decl_cfg["out_dir"] = emitter_out_dir
        _default_decl = {**_default_decl, "config": _decl_cfg}

    # Snapshot external overrides so we can detect 'caller already pinned one'
    # and restore them exactly on exit.
    _ext_parquet = _h._PARQUET_EMITTER_OVERRIDE
    _ext_sqlite = _h._EMITTER_OVERRIDE
    _ext_null = _h._NULL_EMITTER_OVERRIDE
    _any_external = (_ext_parquet is not None or _ext_sqlite is not None
                     or _ext_null)

    set_default_emitter_decl(_default_decl)

    if emitter == "xarray" and not _any_external:
        # In-document, agent-relative XArrayEmitter (single-cell / plain
        # build_composite path — no lineage workflow runner). Reuses the same
        # declared-emitter mechanism the parquet default travels through
        # (set_default_emitter_decl -> _get_special_step ->
        # _build_declared_emitter's XArrayEmitter branch): that branch builds a
        # SingleCellXArrayEmitter wired to the agent-relative
        # global_time/bulk/listeners ports and seeded with the static config
        # skeleton below. The step defers building the real XArrayEmitter to its
        # first update(), where it reads the REALIZED composite state (via the
        # run() contextvar) to discover the FULL listener view + named coords
        # (Task 4 / C2) and projects the structured bulk record array to counts
        # (Task 4 / C1). See SingleCellXArrayEmitter + task-{1,4}-report.md in
        # .superpowers/sdd/2026-08-10-single-cell-xarray-emitter/.
        #
        # This REPLACES the single existing 'emitter' key in place (no new
        # document Steps) — see Task 1 gotcha #4: adding extra document
        # Steps, even read-only ones, perturbs process_bigraph's step
        # scheduling enough to trip a pre-existing metabolism numerical
        # fragility. The lineage workflow runner (v2ecoli.workflow.lineage)
        # still owns its own OUT-OF-BAND XArrayEmitter for multi-generation
        # sweeps (n_seeds>1 / n_generations>1 dispatches to
        # _build_batch_document before this branch is ever reached), so this
        # only changes the plain single-cell build_composite path.
        import warnings  # noqa: PLC0415
        warnings.warn(
            "emitter='xarray' uses the in-document single-cell XArray sink: it "
            "streams bulk + listeners to zarr with bounded memory and is "
            "validated for short/moderate runs. Its per-leaf view is discovered "
            "from the first tick's realized shapes, so a VERY long run may hit an "
            "upstream viva_emitters ragged-vector limitation (a listener leaf that "
            "later changes length or disappears). For long single-cell runs "
            "prefer emitter='parquet' (the robust default).",
            stacklevel=2,
        )
        _xr_out = _resolve_xarray_out_uri(experiment_id, emitter_out_dir or out_dir)
        # Static config skeleton only — view/output_metadata are discovered
        # lazily from the REALIZED state by SingleCellXArrayEmitter at run time
        # (Task 4 / C2). The real run-identity metadata is baked in here.
        _xr_cfg = _single_cell_xarray_config(
            out_uri=_xr_out,
            metadata={
                "experiment_id": experiment_id,
                "variant": int(variant),
                "lineage_seed": int(seed),
            },
        )
        set_default_emitter_decl({
            "address": "local:XArrayEmitter",
            "config": _xr_cfg,
            "paths": ["global_time", "bulk", "listeners"],
        })
    elif emitter == "sqlite" and not _any_external:
        # Minimal persistent SQLite sink. Resolve the workspace-shared DB (the
        # dashboard's Simulations-DB tab aggregates from it); fall back to out/.
        # emitter_out_dir, when set, pins this instead of the workspace lookup.
        if emitter_out_dir:
            _sqlite_dir = emitter_out_dir
        else:
            _ws_root = _find_workspace_root()
            _sqlite_dir = (str(_ws_root / ".pbg") if _ws_root is not None
                           else "out")
        set_emitter_override({
            "file_path": _sqlite_dir,
            "db_file": "composite-runs.db",
        })
    elif emitter == "null" and not _any_external:
        set_null_emitter_override(True)
    # emitter == "parquet": the declared parquet default (set above) is used.

    _process_cache = {}
    # Thread the flux map to the exchange_flux_listener feature step (built via
    # _get_special_step) for the duration of this build; restored in finally.
    set_exchange_fluxes_override(_exchange_fluxes)
    set_exchange_flux_basis_override(_exchange_flux_basis)
    try:
        for step_name in flow_order:
            config = _get_step_config(
                loader, step_name, core, _process_cache, master_seed=seed,
                transcript_initiation_mode=transcript_initiation_mode,
                polypeptide_initiation_mode=polypeptide_initiation_mode,
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
    finally:
        set_default_emitter_decl(None)
        # Restore external overrides to exactly their pre-build values (we only
        # ever changed them when none was active, so this clears ours).
        set_emitter_override(_ext_sqlite)
        set_null_emitter_override(_ext_null)
        set_exchange_fluxes_override({})
        set_exchange_flux_basis_override(None)

    # Place shared PartitionedProcess instances in the process store
    for proc_name, proc_instance in _process_cache.items():
        cell_state['process'][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    seed_mass_listener(cell_state, core)

    # Shape step (Skalnik et al. 2023): derive the capsule cell geometry from
    # mass — length from volume = mass/density, fixed width. Reads the whole
    # listeners.mass sub-store, writes the top-level 'shape' store. Added as a
    # FINAL execution layer (after the mass listener) so inject_flow_dependencies
    # wires it into the per-tick flow: it recomputes length/volume from the
    # current cell_mass every step, so the envelope tracks growth over the sim.
    # (The 'shape' store is seeded with all keys: a map[float] store only merges
    # onto existing keys.)
    if core is not None:
        from v2ecoli.cell_shape import ShapeStep, zero_shape
        core.register_link("ShapeStep", ShapeStep)
        cell_state['shape'] = zero_shape()
        cell_state['shape_step'] = {
            '_type': 'step',
            # Self-resolving module-path address (the `!` form → importlib), so the
            # node realizes in ANY core — including the run subprocess's workspace
            # build_core, which does not call allocate_core and so has no
            # register_link("ShapeStep"). A bare `local:ShapeStep` only resolves
            # where that side-effect ran (e.g. the server's resolve core), which is
            # why interactive Apply worked but a detached Run failed with
            # "no link found at address: {'protocol': 'local', 'data': 'ShapeStep'}".
            'address': 'local:!v2ecoli.cell_shape.ShapeStep',
            'config': {'width_um': 1.0, 'density_g_per_ml': 1.1,
                       'periplasm_fraction': 0.2},
            'inputs': {'mass': ['listeners', 'mass']},
            'outputs': {'shape': ['shape']},
        }
        execution_layers = execution_layers + [['shape_step']]
        flow_order = [step for layer in execution_layers for step in layer]

    inject_flow_dependencies(
        cell_state, flow_order, layers=execution_layers)

    if injected_processes and (
            injected_processes.get("add_processes")
            or injected_processes.get("swap_processes")
            or injected_processes.get("exclude_processes")):
        assert_injection_sourcing(injected_processes)
        # Thread baseline()'s own cache_dir onto the injection spec. The native
        # resolver builds an injected process's config from the bundle sim_data
        # (build_native_redux_config), gated on the spec carrying `cache_dir` —
        # but resolve_injections sees only the spec, never baseline()'s own args.
        # Without this seed the redux SWAP gets an empty config (0 metabolites,
        # 0 homeostatic targets) and collapses to one tick while reporting success
        # (sms-ecoli#210 Gate 0). A caller-supplied cache_dir on the spec wins
        # (deliberate override, e.g. a per-seed cache); else seed baseline's.
        # Copy, don't mutate — the same spec is reused across per-seed builds.
        # (v2ecoli#667: right for the deployment; that PR closed for the wrong
        # reason — its no-op was only against the in-repo resolver, while the
        # image's resolver consumes this key.)
        injected_processes = {
            **injected_processes,
            "cache_dir": injected_processes.get("cache_dir") or cache_dir,
        }
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "..", "scripts"))
        from scripts._compare.inject import (
            resolve_injections, apply_injected_processes, remove_processes)
        # Add half: convert + inject the new processes (add_processes plus the
        # TARGETS of swap_processes). resolve_injections needs the fork repo only
        # when there is something to add.
        if (injected_processes.get("add_processes")
                or injected_processes.get("swap_processes")):
            # fork_repo is guaranteed empty here (assert_injection_sourcing raises
            # on a non-empty one — fork-sourcing is removed). resolve_injections'
            # native path builds specs off the candidate's own bundle simData.
            specs = resolve_injections(injected_processes.get("fork_repo") or "",
                                       injected_processes)
            apply_injected_processes(cell_state, flow_order, core, specs)
        # Remove half: drop the swapped-out SOURCES and any exclude_processes, so
        # a swap is a true replace (not a co-existing add).
        remove_processes(cell_state, flow_order,
                         list((injected_processes.get("swap_processes") or {}).keys()))
        remove_processes(cell_state, flow_order,
                         list(injected_processes.get("exclude_processes") or []))

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    # Issue #495: this plain single-cell (n_seeds==1, n_generations==1,
    # stop_at_division=False) document has NO division-stop. The in-cell Division
    # step fires and structurally splits state at division, but nothing halts the
    # *run* — it simulates the full requested step budget and keeps going past
    # division (now the daughter). Surface that here, at document-build time, so
    # it is not a silent surprise. Suppressed in two cases:
    #  - stop_at_division=True: this build is routed through the lineage path
    #    (early return above) and DOES stop at division, so the note is moot. The
    #    early return already prevents reaching here; the explicit check is a
    #    belt-and-braces guard so the warning can never fire for that opt-in.
    #  - a lineage/daughter build is in flight (an emitter override is active):
    #    the batch path — BatchBaselineRunner -> LineageProcess — builds its
    #    per-generation cell through this same single-cell branch but DOES stop at
    #    division out of band, so the note would be misleading there.
    # warnings' default once-per-location filter keeps this to a single line per
    # process.
    from v2ecoli.composites._helpers import (  # noqa: PLC0415
        _EMITTER_OVERRIDE, _NULL_EMITTER_OVERRIDE, _PARQUET_EMITTER_OVERRIDE)
    _lineage_context = (
        _PARQUET_EMITTER_OVERRIDE is not None
        or _EMITTER_OVERRIDE is not None
        or bool(_NULL_EMITTER_OVERRIDE))
    if not _lineage_context and not stop_at_division:
        import warnings  # noqa: PLC0415
        warnings.warn(
            "ecoli_baseline single-cell mode (n_seeds=1, n_generations=1) has "
            "no division-stop: if this cell divides within the requested step "
            "budget, the run continues PAST division (simulating the daughter) "
            "rather than stopping — n_steps controls how far past division it "
            "runs, and n_generations is inert here (issue #495). To bound the "
            "run to one cell cycle, pass stop_at_division=True; it routes this "
            "single-cell build through the lineage machinery (n_seeds=1, "
            "generations=1) so LineageProcess stops at the first division.",
            stacklevel=2,
        )

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': flow_order,
    }
