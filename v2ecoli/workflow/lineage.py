"""LineageProcess — one (variant, seed) lineage as an embeddable Process.

Wraps a baseline cell composite (the EcoliWCM embedding pattern) and runs it
generation-by-generation, carrying a single daughter forward (vEcoli's
``single_daughters=true`` default). Variant overrides are applied at build
time. Each generation emits partitioned parquet (default), a hive-partitioned
zarr store via an external XArrayEmitter (``emitter == "xarray"`` — the
validated v2ecoli/library/xarray_run.py pattern) with its own metadata, or
BOTH (``emitter == "both"``), which is what a batch run wants: the parquet
sweep is what the DuckDB analyses read, the zarr store is what the workspace's
xarray emitter and the dashboard's per-run charts read. The meta-composite
ticks this process via update(); it reports ``complete`` when ``generations``
cells have been run.
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from v2ecoli.library.quantity_helpers import fg_magnitude

from process_bigraph import Process


def select_carry_daughter(agents_before, agents_now, mother_snapshot):
    """State to seed the next generation (single-daughter lineage), or None.

    The inner baseline composite's Division step already splits the mother
    into ``…0`` / ``…1`` daughters and adds them to its agents map. Carry the
    ``…0`` daughter's biological state DIRECTLY — re-dividing it would halve an
    already-halved cell, producing quarter-mass, slow-growing daughters (the
    multigeneration bug this guards against). Only when no structural daughter
    surfaced (a divide-flag / exception signal with no agents-map change) fall
    back to dividing the pre-run mother snapshot exactly once.
    """
    keys = ("bulk", "unique", "environment", "boundary")
    new_ids = set(agents_now) - set(agents_before)
    d0_id = next((i for i in sorted(new_ids) if i.endswith("0")), None)
    if d0_id is not None:
        dcell = agents_now.get(d0_id, {}) or {}
        return {k: dcell.get(k) for k in keys}
    if mother_snapshot and mother_snapshot.get("bulk") is not None:
        from v2ecoli.library.division import divide_cell

        d1, _d2 = divide_cell(mother_snapshot)
        return d1
    return None


# Substores under ``environment`` that are RE-DERIVED every tick by their owning
# Step and must therefore be taken from the freshly-built daughter, not inherited
# from the mother. ``exchange_data`` (the FBA import constraints, written each tick
# by the ExchangeData step) is realized as an overwrite store (ListenerStore) in a
# fresh build, but carrying the mother's *raw* dict drops that updater so the
# rebuilt daughter falls back to the ``map[float]`` default — which ACCUMULATES on
# apply. The per-tick bound write (e.g. glucose uptake = cap) then adds up instead
# of overwriting, ballooning the bound across the generation and silently voiding
# every exchange constraint in generations >= 1. Keeping the fresh substore is both
# correct (ExchangeData re-derives it from boundary.external on the first tick) and
# the minimal fix.
_FRESH_ENVIRONMENT_SUBSTORES = ("exchange_data",)


def apply_carry_state(agent, carry_state):
    """Overlay an inherited daughter's biological state onto a fresh agent doc.

    Carries ``bulk``/``unique``/``environment``/``boundary`` from ``carry_state``,
    but PRESERVES the fresh agent's derived ``environment`` substores listed in
    :data:`_FRESH_ENVIRONMENT_SUBSTORES` so their overwrite updaters survive the
    daughter rebuild (see the note there).
    """
    for key in ("bulk", "unique", "environment", "boundary"):
        if key not in carry_state:
            continue
        if key == "environment":
            fresh_env = agent.get("environment") or {}
            carried_env = dict(carry_state["environment"] or {})
            for sub in _FRESH_ENVIRONMENT_SUBSTORES:
                if sub in fresh_env:
                    carried_env[sub] = fresh_env[sub]
            agent["environment"] = carried_env
        else:
            agent[key] = carry_state[key]


def _estimate_state_mb(state) -> float:
    """Rough MB of a carry/checkpoint state: the summed ``nbytes`` of its numpy
    arrays (``bulk`` + the ``unique`` structured arrays, which dominate). Cheap
    enough for the per-generation hot path; used only for the checkpoint log line
    so an oversized or stalled write is visible in the run log instead of silent.
    """
    if not isinstance(state, dict):
        return 0.0
    total = 0
    for key in ("bulk", "unique", "environment", "boundary"):
        val = state.get(key)
        if hasattr(val, "nbytes"):
            total += int(val.nbytes)
        elif isinstance(val, dict):
            for v in val.values():
                if hasattr(v, "nbytes"):
                    total += int(v.nbytes)
    return total / 1e6


def _apply_lineage_offset(injected_processes, offset):
    """Expose the cumulative lineage-time ``offset`` (summed duration of the
    generations completed before the current one) to every injected process.

    The inner composite's ``global_time`` RESTARTS at 0 each generation (each
    generation is a freshly-built composite — see :class:`LineageProcess`), so
    an injected process that needs to reason about ABSOLUTE / cumulative lineage
    time cannot get it from ``global_time`` alone. This sets
    ``lineage_time_offset`` on EVERY injected process config so any such process
    can read ``global_time + lineage_time_offset``; a process that does not
    declare/read the key simply ignores it. Domain- and process-agnostic: no
    process is named here.

    Returns a COPY (the caller's dict is never mutated). A no-op for an
    empty/None injected block; for generation 0 / single-generation runs the
    offset is 0.0, which every reader treats as "use local time" — unchanged
    behavior.
    """
    if not injected_processes:
        return injected_processes
    out = copy.deepcopy(injected_processes)
    pcfg = out.get("process_configs")
    if isinstance(pcfg, dict):
        for process_config in pcfg.values():
            if isinstance(process_config, dict):
                process_config["lineage_time_offset"] = float(offset)
    return out


# Default xarray view: scalar mass gauges (no vector coord arrays needed).
# Override via emitter_arg["view"] (JSON list roots are accepted). Leaves the
# composite doesn't emit are filtered out at open time (xarray is strict).
DEFAULT_XARRAY_VIEW = [
    {
        "root": ("listeners", "mass"),
        "variables": {
            name: [{"path": name, "dtype": "<f4"}]
            for name in (
                "dry_mass",
                "cell_mass",
                "protein_mass",
                "rna_mass",
                "dna_mass",
            )
        },
    }
]


class LineageProcess(Process):
    config_schema = {
        "cache_dir": {"_type": "string", "_default": "out/cache"},
        "seed": {"_type": "integer", "_default": 0},
        "lineage_seed": {"_type": "integer", "_default": 0},
        "variant_index": {"_type": "integer", "_default": 0},
        "variant_name": {"_type": "string", "_default": "baseline"},
        "config_overrides": {"_default": {}},
        "generations": {"_type": "integer", "_default": 1},
        "single_daughters": {"_type": "boolean", "_default": True},
        # Per-generation checkpoint/resume (backlog item 34): externalizes the
        # daughter-state hand-off vEcoli-private's Nextflow driver does via task
        # I/O (sim.nf), so a wave orchestrator can retry at generation
        # granularity instead of whole-lineage granularity. All three default
        # to empty/0, which is today's single-invocation-runs-every-generation
        # behavior, unchanged.
        "initial_carry_state_path": {"_type": "string", "_default": ""},
        "initial_generation_index": {"_type": "integer", "_default": 0},
        "daughter_state_out_path": {"_type": "string", "_default": ""},
        # Item 115: a SINGLE-lineage-run caller (chain-dispatch, one process
        # invocation per generation) already gets per-generation resume via the
        # three fields above -- an external scheduler computes each generation's
        # own literal daughter_state_out_path before submitting that generation's
        # job. A pbg-native lineage has no such external scheduler: ALL of a
        # lineage's generations run inside ONE continuous process invocation, so
        # no caller can pre-compute N literal per-generation paths ahead of time
        # -- the process itself must derive each one as it goes. When set, this
        # is a DIRECTORY PREFIX (not a literal file path): each generation's own
        # checkpoint is written to "{checkpoint_dir}/gen_{generation:04d}.pkl",
        # a distinct key per generation so a write failure at generation N can
        # never corrupt generation N-1's already-durable checkpoint (an
        # overwrite-in-place scheme would risk exactly that -- the one thing a
        # checkpoint meant to survive a crash cannot afford). Takes priority
        # over daughter_state_out_path when both are set (real precedence, not
        # silently ignored) since a literal path can only ever describe ONE
        # generation's own destination. Empty by default: unused, so an existing
        # single-generation caller (chain-dispatch) is entirely unaffected.
        "checkpoint_dir": {"_type": "string", "_default": ""},
        "experiment_id": {"_type": "string", "_default": "default"},
        "out_dir": {"_type": "string", "_default": "out/workflow"},
        "max_duration_per_gen": {"_type": "float", "_default": 3600.0},
        "time_step": {"_type": "float", "_default": 1.0},
        "media": {"_type": "string", "_default": "minimal"},
        # Gate 1b / v2ecoli#693: seeds sharing a cache_dir share a FOUNDER cell,
        # because _load_cache_bundle_cached is memoised on cache_dir alone and
        # returns the initial state by reference. So an M-seed sweep off one
        # cache varies the trajectory but not the starting cell -- measured at
        # 251/16321 species differing between two seeds, against 754 changing in
        # a single timestep of one seed. Opt in to re-drawing a founder per
        # lineage_seed (v2ecoli#712).
        "independent_founders": {"_type": "boolean", "_default": False},
        "founder_sim_data": {"_type": "string", "_default": ""},
        # "parquet" (default), "xarray", or "both". xarray drives an external
        # XArrayEmitter per lineage (validated multigen pattern); the internal
        # baseline emitter step then falls back to RAM (not read). "both" keeps
        # the internal parquet emitter AND drives the external XArrayEmitter.
        "emitter": {"_type": "string", "_default": "parquet"},
        "emitter_arg": {"_default": {}},
        # Config-declared EXTRA emit store paths (domain-agnostic): a list of
        # store paths (each a list of store-node segments, e.g.
        # ["some_store", "sub_key"]) to persist beyond the baseline parquet set.
        # `quote` keeps the nested list verbatim (same reason as
        # injected_processes below). Threaded to the per-generation parquet
        # emitter override, which honors it via _merge_emit_paths.
        "emit_paths": {"_type": "quote", "_default": []},
        # A generation that ran and emitted NOTHING is a failed generation, not
        # a successful one. Checked at the end of every generation, inside the
        # process every lineage shape passes through (LineageStep/Nextflow,
        # lineage_ray_batch, chain-dispatch's BatchBaselineRunner, the local
        # meta-composite) -- so an empty emit fails loud at its source, with
        # the generation/agent/out_dir in the message, instead of surfacing as
        # a generic "no emitted output" from a downstream gate (or not at all).
        # See _assert_generation_emitted. Opt out ONLY for a stubbed test.
        "require_output": {"_type": "boolean", "_default": True},
        # `quote` (NOT a bare {"_default": {}}): the injected-processes block is a
        # heterogeneous, config-shaped dict — it carries each injected process's
        # own `process_configs`, some of which are list- or nested-shaped (e.g. a
        # per-process schedule like `[[time, {key: value}]]`). Without an explicit
        # `_type`, bigraph-schema infers a schema for this key from its `{}`
        # default and coerces the value against it, mangling a nested list
        # (`[[100, {"key": 1.0}]]` -> `[[100, 100]]`) — which then crashes the
        # generation-1 composite rebuild that re-realizes this config. `quote`
        # stores the block verbatim, so an injected process's structured config
        # survives realize.
        "injected_processes": {"_type": "quote", "_default": {}},
        # Per-cell biological build kwargs, forwarded to each generation's
        # baseline() build so a batch/lineage run engages the SAME biology as the
        # single-cell path (audit: batch mode dropped these -> basal FBA). `features`
        # and the toggles also ride inside injected_processes for the injected-arm
        # convention; _build_generation resolves both via _feature_flag.
        "features": {"_default": []},
        "ppgpp_regulation": {"_type": "boolean", "_default": True},
        "trna_attenuation": {"_type": "boolean", "_default": False},
        "supercoiling": {"_type": "boolean", "_default": False},
        "mass_conservation": {"_type": "boolean", "_default": False},
        "exchange_fluxes": {"_default": {}},
        "exchange_flux_basis": {"_type": "string", "_default": ""},
        "transcript_initiation_mode": {"_type": "string", "_default": "discrete"},
        "polypeptide_initiation_mode": {"_type": "string", "_default": "discrete"},
    }

    def initialize(self, config):
        self._composite = None
        carry_path = str(config.get("initial_carry_state_path") or "")
        gen_index = int(config.get("initial_generation_index") or 0)
        if not carry_path and gen_index != 0:
            # A nonzero start with no state to seed it silently mislabels a
            # fresh cell as a later generation (wrong parquet/zarr partition,
            # wrong summary["generation"]) instead of failing loudly.
            raise ValueError(
                "LineageProcess: initial_generation_index must be 0 when "
                "initial_carry_state_path is empty."
            )
        self._generation = (
            gen_index  # 0-based current generation; >0 resumes a checkpointed wave
        )
        # Under single_daughters=True (the only supported mode, enforced below
        # in update()), the phylogeny walk is deterministic: select_carry_daughter
        # always keeps the "...0" daughter, so a continuous single-process run
        # reaches agent_id "0"*(generation+1) by the time it starts generation
        # `generation`. A per-generation chain job (backlog item 34) restores
        # `_generation` from `initial_generation_index` above but must restore
        # `_agent_id` to match, or every chain job resolves to the SAME agent_id
        # ("0") regardless of which generation it's actually resuming — which the
        # xarray/zarr emitter reads as "generation 1" (`len(agent_id)`) every
        # time, mistaking gen1+'s real pre-existing S3 content (from the shared
        # per-seed prefix) for a collision on a supposedly-fresh store.
        self._agent_id = "0" * (gen_index + 1)
        self._gen_elapsed = 0.0
        # Cumulative duration (s) of the generations completed BEFORE the
        # current one — the lineage-time offset exposed to every injected process
        # (as `lineage_time_offset`) so one that reasons about cumulative lineage
        # time can add it to the inner composite's per-generation `global_time`
        # (which restarts at 0 each generation). 0.0 for generation 0; grows by
        # each generation's duration as it completes (see update()).
        self._lineage_offset = 0.0
        self._carry_state: dict | None = None
        if carry_path:
            from v2ecoli.cache import load_initial_state

            self._carry_state = load_initial_state(carry_path)
        self._complete = False
        self._summaries: list[dict] = []
        # Per-generation checkpoint hand-off (backlog item 34/35): each
        # generation is a SEPARATE process invocation, so self._summaries would
        # otherwise only ever contain THIS generation's own entry, and a
        # per-seed summary.json written from it would silently lose every prior
        # generation's history the moment the next generation's job overwrites
        # it. Restoring the accumulated list here (saved alongside the daughter
        # state, see update() below) makes each write authoritative for the
        # seed's FULL history so far, matching what the analysis step already
        # expects from a single-invocation run's summary.json.
        if self._carry_state and "_prior_summaries" in self._carry_state:
            self._summaries = list(self._carry_state.pop("_prior_summaries"))
        self._needs_build = True  # True → call _build_generation on next tick
        # xarray emitter state (only used when config["emitter"] == "xarray")
        self._xarray_em = None  # live XArrayEmitter for the current gen
        self._xarray_pending = False  # True → open on first populated emit tick
        self._xarray_view = None  # filtered view in use for this lineage
        self._xarray_store = None  # zarr store path (stable across gens)
        # Emitted-output bookkeeping for _assert_generation_emitted: the parquet
        # emitter the current generation's inner composite was built with
        # (captured at build time, since Division may pop it from the registry
        # before the generation ends), and the number of populated xarray emits.
        self._parquet_em = None
        self._xarray_emits = 0

    def _is_xarray(self) -> bool:
        """True when this lineage drives the external XArrayEmitter."""
        return self.config.get("emitter", "parquet") in ("xarray", "both")

    def _is_parquet(self) -> bool:
        """True when the inner composite's own emitter writes the hive parquet
        sweep. Mutually exclusive with the null override, NOT with xarray —
        ``emitter == "both"`` runs the two side by side."""
        return self.config.get("emitter", "parquet") in ("parquet", "both")

    def inputs(self):
        return {}

    def outputs(self):
        return {"summary": "map", "complete": "boolean"}

    # --- build / run helpers (stubbed in unit tests) ---------------------

    def _build_generation(self):
        from process_bigraph import Composite
        from v2ecoli.core import build_core
        from v2ecoli.composites.ecoli_baseline import baseline, seed_mass_listener

        core = build_core()
        gen_seed = (int(self.config["seed"]) + self._generation) % (2**31)
        overrides = dict(self.config.get("config_overrides") or {})
        # Fresh emitted-output bookkeeping for this generation.
        self._parquet_em = None
        self._xarray_emits = 0

        # Forward baseline()'s feature-selection kwargs from the config so an
        # injected candidate arm actually engages the features it declares. The
        # injected subsystem's bulk-species seeds + feature needs (e.g.
        # cell_geometry, which supplies periplasm/cytoplasm.global.volume +
        # boundary.outer_surface_area so a downstream mol/(volume*N_A) conversion
        # does not divide by zero) ride generically inside `injected_processes`
        # (`seed_bulk_species` / `requires_features`) — the engine reads them, so
        # nothing subsystem-specific is threaded here. The harness forwards `features`
        # under `injected_processes`; fall back to a top-level config key.
        _injected = self.config.get("injected_processes") or {}

        def _feature_flag(key, default):
            return _injected.get(key, self.config.get(key, default))

        _features = _feature_flag("features", None)

        # Expose this generation's cumulative lineage-time offset to every
        # injected process (as `lineage_time_offset`), so one that reasons about
        # cumulative lineage time can add it to the per-generation `global_time`
        # (which restarts at 0 here). Process-agnostic; offset 0.0 (generation 0
        # / single-generation runs) is a no-op for every reader — see
        # _apply_lineage_offset.
        _injected_for_build = _apply_lineage_offset(
            self.config.get("injected_processes"), self._lineage_offset
        )

        # Per-cell biological build kwargs, shared by both emitter branches below
        # so an injected batch/lineage run builds every generation cell with the
        # SAME biology as the single-cell path (audit: batch mode used to drop
        # these -> basal FBA). Each rides inside injected_processes OR a top-level
        # config key (see _feature_flag); the toggles keep baseline()'s own
        # defaults (ppgpp on, the rest off) when unset.
        _bio_kwargs = dict(
            cache_dir=self.config["cache_dir"],
            config_overrides=overrides,
            media=self.config.get("media", "minimal"),
            independent_founders=bool(self.config.get("independent_founders", False)),
            founder_sim_data=str(self.config.get("founder_sim_data", "") or ""),
            features=_features,
            injected_processes=_injected_for_build,
            ppgpp_regulation=bool(_feature_flag("ppgpp_regulation", True)),
            trna_attenuation=bool(_feature_flag("trna_attenuation", False)),
            supercoiling=bool(_feature_flag("supercoiling", False)),
            mass_conservation=bool(_feature_flag("mass_conservation", False)),
            exchange_fluxes=_feature_flag("exchange_fluxes", None) or None,
            exchange_flux_basis=_feature_flag("exchange_flux_basis", None) or None,
            transcript_initiation_mode=(
                _feature_flag("transcript_initiation_mode", "discrete") or "discrete"
            ),
            polypeptide_initiation_mode=(
                _feature_flag("polypeptide_initiation_mode", "discrete") or "discrete"
            ),
        )

        # The inner composite's own emitter step writes the hive parquet sweep;
        # under a pure-xarray lineage it is minimised to global_time only
        # (set_null_emitter_override) because we emit out of band instead. The
        # XArrayEmitter is opened lazily on the first populated emit tick (see
        # _emit_xarray), so the view can be filtered against real state — xarray
        # is strict about missing emit paths. "both" takes the parquet override
        # AND arms the xarray path.
        if self._is_parquet():
            from v2ecoli.composites._helpers import set_parquet_emitter_override
            from v2ecoli.library.emitter_presets import parquet_vecoli

            emitter_cfg = parquet_vecoli(
                out_dir=self.config["out_dir"],
                experiment_id=self.config["experiment_id"],
                variant=int(self.config["variant_index"]),
                lineage_seed=int(self.config["lineage_seed"]),
                agent_id=self._agent_id,
                generation=self._generation,
            )
            # Config-declared EXTRA emit store paths, passed through generically
            # so a run can persist stores beyond the baseline set (the emitter
            # honors them via _merge_emit_paths — domain-agnostic).
            emit_paths = self.config.get("emit_paths")
            if emit_paths:
                emitter_cfg["emit_paths"] = list(emit_paths)
            set_parquet_emitter_override(emitter_cfg)
            try:
                doc = baseline(core=core, seed=gen_seed, **_bio_kwargs)
            finally:
                set_parquet_emitter_override(None)
            # Capture THIS generation's emitter now: the build registered it
            # under self._agent_id, and Division may finalize + pop it before
            # the generation ends (see _finalize_parquet), so a lookup at the
            # end would miss generation 0. None here means the inner composite
            # was built WITHOUT a parquet sink -- which the end-of-generation
            # check refuses (require_output).
            from v2ecoli.composites._helpers import get_parquet_emitter

            self._parquet_em = get_parquet_emitter(self._agent_id)
        else:
            from v2ecoli.composites._helpers import set_null_emitter_override

            set_null_emitter_override(True)
            try:
                doc = baseline(core=core, seed=gen_seed, **_bio_kwargs)
            finally:
                set_null_emitter_override(False)
        if self._is_xarray():
            self._xarray_pending = True

        if self._carry_state is not None:
            agent = doc["state"]["agents"]["0"]
            apply_carry_state(agent, self._carry_state)
            agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
            seed_mass_listener(agent, core)

        self._composite = Composite(doc, core=core)
        self._core = core
        self._gen_elapsed = 0.0

    def _open_xarray_emitter(self, emit_cell):
        """Open an XArrayEmitter for the current generation, filtering the view
        against ``emit_cell`` (populated state) and discovering vector coords.
        Mirrors the validated v2ecoli/library/xarray_run.py pattern."""
        import os
        import shutil
        from v2ecoli.library.xarray_run import (
            _build_emitter,
            filter_view_to_existing_leaves,
            extract_output_metadata_from_state,
        )

        from v2ecoli.cache import is_s3_uri

        arg = dict(self.config.get("emitter_arg") or {})
        raw_view = arg.get("view") or DEFAULT_XARRAY_VIEW
        raw_view = [dict(e, root=tuple(e["root"])) for e in raw_view]
        transducer = arg.get("transducer") or {}
        buf = (transducer.get("buffer") or {}).get("size")
        # Default 600 (viva-emitters library default: a handful of flushes per
        # generation, not one every few steps); floor 3 since the transducer
        # requires buffer.size > 2.
        buf = max(3, int(buf or 600))
        predicate = transducer.get("predicate")
        # buffers_per_chunk defaults to 1 here (not the shared build_emitter_config
        # default of 10) -- Boyan Beronov's own documented guidance (his pending
        # vEcoli doc commit, CovertLab/vEcoli@febe3817): for immutable object
        # storage (S3 Standard -- our own backend for this dispatch path), a value
        # >1 means each chunk flush re-copies previously-written objects rather
        # than appending cleanly. ecoli_baseline.py's own single-cell path already
        # makes this same override explicitly; this path silently inherited the
        # shared default instead. setdefault, not assignment, so an explicit
        # caller-supplied value still wins.
        writer = dict(arg.get("writer") or {})
        writer.setdefault("buffers_per_chunk", 1)
        out_dir = arg.get("out_dir") or self.config["out_dir"]
        out_is_s3 = is_s3_uri(out_dir)

        wrapped = {"agents": {"0": emit_cell}}
        view = filter_view_to_existing_leaves(wrapped, raw_view)
        # required_leaves: declared KPI columns that MUST be present. A required leaf
        # filtered out of the view means it was absent from composite state -- e.g. the
        # redux swap / injected process did not apply, so the column is ABSENT (not zero).
        # Fail loudly rather than silently emitting a wild-type run. Checked BEFORE the
        # empty-view skip below, so an all-missing view still raises instead of skipping.
        required = arg.get("required_leaves") or []
        if required:
            present_leaves: set[str] = set()
            for _e in view:
                present_leaves.update((_e.get("variables") or {}).keys())
            missing = [
                leaf
                for leaf in required
                if str(leaf).split(".")[-1] not in present_leaves
            ]
            if missing:
                raise ValueError(
                    f"LineageProcess: required emitter leaf(s) {missing} absent from "
                    f"composite state at generation {self._generation} (present: "
                    f"{sorted(present_leaves)}). A missing KPI column usually means an "
                    f"injected process/swap did not apply -- refusing to emit a "
                    f"silently-wild-type run. Drop 'required_leaves' from emitter_arg to "
                    f"downgrade to warn-and-skip."
                )
        if not view:
            warnings.warn(
                "LineageProcess: xarray view has no leaves present in "
                "composite state; skipping xarray emission."
            )
            self._xarray_pending = False
            return
        output_metadata = extract_output_metadata_from_state(wrapped, view)

        if self._xarray_store is None:
            self._xarray_store = os.path.join(
                out_dir,
                f"{self.config['experiment_id']}_v{int(self.config['variant_index'])}"
                f"_s{int(self.config['lineage_seed'])}.zarr",
            )
        if not out_is_s3:
            # Local-filesystem-only bookkeeping: zarr's own S3 store (opened via
            # zarr.open_group(store=...) inside pbg-emitters) handles "fresh
            # store" / "create the prefix" semantics itself for s3:// URIs — an
            # os.path.exists/os.makedirs call on an s3:// string is meaningless
            # (checks/creates a bogus local path, never the real object prefix).
            if self._generation == 0 and os.path.exists(self._xarray_store):
                shutil.rmtree(self._xarray_store)  # fresh store for a new lineage
            os.makedirs(out_dir, exist_ok=True)

        metadata_base = {
            "experiment_id": self.config["experiment_id"],
            "variant": int(self.config["variant_index"]),
            "lineage_seed": int(self.config["lineage_seed"]),
            "time_step": float(self.config.get("time_step", 1.0)),
            "max_duration": float(self.config["max_duration_per_gen"]),
        }
        self._xarray_view = view
        self._xarray_em = _build_emitter(
            core=self._core,
            store_path=self._xarray_store,
            view=view,
            metadata_base=metadata_base,
            generation=self._generation,
            agent_id=self._agent_id,
            buffer_size=buf,
            output_metadata=output_metadata,
            writer=writer,
            predicate=predicate,
        )
        self._xarray_pending = False

    def _emit_xarray(self, agents_now):
        """Emit the inner cell's filtered state to the xarray emitter (opening
        it lazily on the first populated tick)."""
        emit_cell = agents_now.get("0")  # inner composite always names the cell "0"
        if not isinstance(emit_cell, dict):
            return
        if self._xarray_pending and self._xarray_em is None:
            self._open_xarray_emitter(emit_cell)
        if self._xarray_em is None:
            return
        from v2ecoli.library.xarray_run import _filter_agent_state

        payload = _filter_agent_state(emit_cell, self._xarray_view)
        try:
            self._xarray_em.update(
                {
                    "time": float(self._gen_elapsed),
                    "global_time": float(self._gen_elapsed),
                    "agents": {self._agent_id: payload},
                }
            )
            self._xarray_emits += 1
        except Exception as e:
            warnings.warn(
                f"LineageProcess: xarray emit failed at generation "
                f"{self._generation} t={self._gen_elapsed}: {e}"
            )

    def _finalize_parquet(self) -> None:
        """Close this generation's parquet emitter, however the generation ended.

        Two DISJOINT cases, so both are attempted (``close()`` is idempotent and
        ``finalize_emitter_for_agent`` pops, so the redundant one is a no-op):

        * **Timed out, no division** — the agent subtree is still in
          ``self._composite``, so ``flush_parquet`` finds the live emitter.
        * **Divided** — ``Division`` has already returned
          ``{'agents': {'_remove': [...]}}`` and torn the subtree out, so
          ``flush_parquet`` finds nothing to close. The emitter is still in the
          process-global registry under the metadata ``agent_id`` it was built
          with (``self._agent_id``: "0", "00", ...).

        ``Division`` cannot derive that key. Inside the composite the cell is
        always the ``agents/0`` key, and the parquet override it reads to
        recover the runner's identity is already cleared by
        ``_build_generation`` before the composite is constructed -- so its
        lookup falls back to "0" and MISSES for every generation after the
        first. The finalize therefore happens here, in the object that owns the
        key. Without it a generation >= 1 silently loses its trailing batch AND
        its ``success/`` sentinel while the summary still records
        ``divided: true`` with a full duration -- and a missing sentinel drops
        the whole generation from any analysis that filters on ``success_sql``,
        not just its last few hundred ticks (v2ecoli#687).
        """
        from v2ecoli.composites._helpers import (
            finalize_emitter_for_agent,
            flush_parquet,
        )

        try:
            flush_parquet(self._composite, success=True)
        except Exception as e:
            warnings.warn(
                f"LineageProcess: parquet flush failed for "
                f"generation {self._generation} ({self._agent_id}): {e}"
            )
        try:
            finalize_emitter_for_agent(self._agent_id, success=True)
        except Exception as e:
            warnings.warn(
                f"LineageProcess: parquet finalize failed for "
                f"generation {self._generation} ({self._agent_id}): {e}"
            )

    def _assert_generation_emitted(self) -> None:
        """Refuse to close a generation that ran and emitted nothing.

        This is the emit-path guard at its SOURCE. Every lineage shape passes
        through this process -- LineageStep (Nextflow), ``lineage_ray_batch``
        (pbg-native), chain-dispatch's ``BatchBaselineRunner`` and the local
        meta-composite -- so a generation whose sink received no rows, or whose
        rows never reached storage, fails here with the generation, agent id and
        destination named, rather than (at best) as a generic "no emitted output"
        from a downstream gate that cannot say WHICH generation or WHY.

        What is checked, per sink:

        * **parquet** (``emitter`` = ``parquet``/``both``): the inner composite
          must have been built WITH the lineage's parquet sink (``_parquet_em``
          captured at build time), that sink must have received at least one
          row (``num_emits``), and its history partition must hold at least one
          non-empty ``*.pq`` -- listed through the emitter's own fsspec
          filesystem, so an ``s3://`` out_dir is checked for real rather than
          assumed. "Could not list" is reported as a warning, not a failure:
          "no output" and "could not look" are different answers.
        * **xarray-only** (``emitter`` = ``xarray``): at least one populated emit
          reached the XArrayEmitter. An empty view (every leaf filtered out) is
          exactly the metadata-only zarr store CD2 Run 4 produced.

        ``emitter="null"`` lineages emit nothing BY DESIGN (model browsing,
        division-behaviour tests) and are not checked. ``require_output=False``
        disables the check -- for stubbed unit tests only.
        """
        if not self.config.get("require_output", True):
            return
        where = (
            f"generation {self._generation} (agent_id={self._agent_id!r}, "
            f"experiment_id={self.config.get('experiment_id')!r}, "
            f"out_dir={self.config.get('out_dir')!r})"
        )
        if self._is_parquet():
            em = self._parquet_em
            if em is None:
                raise RuntimeError(
                    f"LineageProcess: {where} completed but its inner composite was "
                    f"built WITHOUT the lineage's parquet emitter (no ParquetEmitter "
                    f"was registered for this generation's agent_id). Nothing this "
                    f"generation computed was persisted -- refusing to report it as "
                    f"a completed generation. Set require_output=False only for a "
                    f"stubbed test."
                )
            num_emits = int(getattr(em, "num_emits", 0) or 0)
            if num_emits <= 0:
                raise RuntimeError(
                    f"LineageProcess: {where} completed but its parquet emitter "
                    f"received 0 rows over {self._gen_elapsed:.0f}s of simulated "
                    f"time. The 'emitter' step never fired, so the generation ran "
                    f"unobserved -- refusing to report it as a completed generation."
                )
            self._assert_history_landed(em, where, num_emits)
        elif self._is_xarray():
            if self._xarray_emits <= 0:
                raise RuntimeError(
                    f"LineageProcess: {where} completed but 0 populated emits "
                    f"reached the XArrayEmitter (store={self._xarray_store!r}). "
                    f"An empty view (every declared leaf filtered out of the "
                    f"composite state) leaves a metadata-only zarr store with no "
                    f"data chunk -- refusing to report it as a completed generation."
                )

    @staticmethod
    def _assert_history_landed(em, where: str, num_emits: int) -> None:
        """Verify at least one non-empty history parquet exists for ``em``'s
        partition, through the emitter's own filesystem (local or s3)."""
        out_uri = getattr(em, "out_uri", None)
        fs = getattr(em, "filesystem", None)
        if not out_uri or fs is None:
            return  # a duck-typed stand-in without a filesystem: nothing to list
        history_dir = os.path.join(
            str(out_uri),
            str(getattr(em, "experiment_id", "") or "default"),
            "history",
            str(getattr(em, "partitioning_path", "") or ""),
        )
        try:
            entries = fs.ls(history_dir, detail=True)
        except FileNotFoundError:
            entries = []
        except Exception as e:  # noqa: BLE001 -- "could not look" is not "no output"
            warnings.warn(
                f"LineageProcess: could not list {history_dir!r} to verify that "
                f"{where} persisted its {num_emits} emitted row(s): {e!r}. The "
                f"generation is recorded, but its output is UNVERIFIED."
            )
            return
        for entry in entries or []:
            name = str(entry.get("name", "") if isinstance(entry, dict) else entry)
            size = int(entry.get("size", 0) or 0) if isinstance(entry, dict) else 1
            if name.endswith((".pq", ".parquet")) and size > 0:
                return
        raise RuntimeError(
            f"LineageProcess: {where} emitted {num_emits} row(s) but no non-empty "
            f"history parquet exists under {history_dir!r} after the flush. The "
            f"rows never reached storage (a failed background write, or a sink "
            f"pointed somewhere else) -- refusing to report it as a completed "
            f"generation."
        )

    def _run_until_division(self, interval):
        """Run the internal composite for ``interval`` seconds. Returns
        ``(divided, daughter_cell_data_or_None, final_dry_mass)``."""
        agents = self._composite.state.get("agents") or {}
        agents_before = set(agents.keys())
        # Snapshot the mother's divisible state BEFORE running: the inner
        # Division step removes the mother mid-run (and adds daughters), so
        # reading after the run samples an already-divided daughter. Only the
        # snapshot is used for the exception/divide-flag fallback path.
        mother = agents.get(self._agent_id) or next(iter(agents.values()), {})
        mother_snapshot = (
            {k: mother.get(k) for k in ("bulk", "unique", "environment", "boundary")}
            if isinstance(mother, dict)
            else None
        )

        divided = False
        try:
            self._composite.run(interval)
        except Exception as e:
            # A genuine division surfaces as a structural agents-map update that
            # process-bigraph raises through; its message mentions divide/division.
            # But a plain runtime error whose message merely CONTAINS that
            # substring (e.g. ZeroDivisionError: "float division by zero") must
            # NOT be mistaken for a division — doing so silently masks real
            # failures as phantom divisions. Only a genuine division signal is
            # honored, and never silently.
            from v2ecoli.library.division import is_division_exception

            if not is_division_exception(e):
                raise
            warnings.warn(
                f"LineageProcess: treating a raised exception as a division "
                f"signal at t={self._gen_elapsed}: {e!r}"
            )
            divided = True
        self._gen_elapsed += interval

        agents_now = self._composite.state.get("agents") or {}
        agents_after = set(agents_now.keys())
        if agents_before and agents_after != agents_before:
            divided = True
        # MarkDPeriod sets a divide flag without changing the agents map; honor it
        # too (mirrors the three-signal detection in v2ecoli/bridge.py).
        # The inner composite always names its single cell "0" (see baseline()
        # + _emit_xarray), whereas self._agent_id accumulates phylogeny suffixes
        # ("0" -> "00" -> ...) across generations. Look the survivor up by the
        # inner key, falling back to the sole agent — otherwise generations >= 1
        # never see the divide flag and run to max_duration_per_gen without
        # dividing (matches the resilient lookups above/below).
        survivor = agents_now.get(self._agent_id) or next(iter(agents_now.values()), {})
        divide_flag = isinstance(survivor, dict) and bool(survivor.get("divide"))
        if divide_flag:
            divided = True

        cell = agents_now.get(self._agent_id) or next(iter(agents_now.values()), {})
        dry_mass = fg_magnitude(
            cell.get("listeners", {}).get("mass", {}).get("dry_mass", 0.0)
        )

        # sms-ecoli#210, item106: temporary, opt-in diagnostic for Run 3's real,
        # not-yet-root-caused one-tick collapse (global_time reaches ~1.0 with no
        # exception at all) — reports exactly which of the 3 division signals
        # fired plus the real state values behind them, since static reading of
        # this function alone couldn't distinguish the cases. Silent unless
        # LINEAGE_DEBUG_DIVISION=1 is set; never touches production behavior.
        if os.environ.get("LINEAGE_DEBUG_DIVISION") == "1":
            print(
                f"[lineage-debug] t={self._gen_elapsed} divided={divided} "
                f"structural_agents_change={agents_before and agents_after != agents_before} "
                f"divide_flag={divide_flag} dry_mass={dry_mass} "
                f"agents_before={sorted(agents_before)} agents_after={sorted(agents_after)}",
                flush=True,
            )

        if self._is_xarray():
            self._emit_xarray(agents_now)

        daughter = None
        if divided:
            daughter = select_carry_daughter(agents_before, agents_now, mother_snapshot)
        return divided, daughter, dry_mass

    # --- main tick -------------------------------------------------------

    def update(self, state, interval):
        if not self.config.get("single_daughters", True):
            raise NotImplementedError(
                "single_daughters=False (binary-tree lineage) is deferred; "
                "MVP supports the single-lineage walk only."
            )
        if self._complete:
            return {"complete": True}
        if self._needs_build:
            self._build_generation()
            self._needs_build = False

        divided, daughter, dry_mass = self._run_until_division(interval)
        timed_out = self._gen_elapsed >= float(self.config["max_duration_per_gen"])
        if not (divided or timed_out):
            return {}

        # End of this generation: flush/close whichever emitters are live, then
        # record the summary. Independent checks, NOT if/else — under
        # ``emitter == "both"`` the parquet buffer must still be flushed, or the
        # generation's trailing rows (every row, for a generation shorter than
        # the emitter's 400-row batch) never land and the sweep has no history
        # parquet for the analyses to read.
        # Observability for the division→checkpoint window. This is exactly where
        # dispatch 313 stalled — idle, no error, right before the checkpoint write
        # (sms-ecoli#210). The emitter flush and the checkpoint write are both S3
        # I/O; without these markers a stall in either is indistinguishable and
        # invisible. Printed (flushed) so it lands in the run log, and timed so a
        # slow/blocked step is obvious rather than silent.
        _t_flush = time.monotonic()
        print(
            f"[LineageProcess] gen {self._generation}: end (divided={divided} "
            f"timed_out={timed_out}); flushing emitters...",
            flush=True,
        )
        if self._is_xarray() and self._xarray_em is not None:
            try:
                self._xarray_em.close(success=True)
            except Exception as e:
                warnings.warn(
                    f"LineageProcess: xarray close failed for "
                    f"generation {self._generation}: {e}"
                )
            self._xarray_em = None
        if self._is_xarray():
            self._xarray_pending = False
        if self._is_parquet():
            self._finalize_parquet()
        print(
            f"[LineageProcess] gen {self._generation}: emitters flushed in "
            f"{time.monotonic() - _t_flush:.1f}s",
            flush=True,
        )
        # AFTER the flush (the trailing batch is what lands a short
        # generation's history at all), BEFORE the summary/checkpoint: a
        # generation that emitted nothing must not be recorded as completed.
        self._assert_generation_emitted()
        self._summaries.append(
            {
                "generation": self._generation,
                "agent_id": self._agent_id,
                "duration": self._gen_elapsed,
                "dry_mass": dry_mass,
                "divided": bool(divided),
            }
        )
        # This generation is done: fold its duration into the cumulative
        # lineage-time offset so the NEXT generation's injected processes see the
        # correct cumulative lineage time (see _apply_lineage_offset /
        # _build_generation). Mirrors the analyses' own per-generation cumulative
        # reconstruction (sum of prior-generation durations).
        self._lineage_offset += self._gen_elapsed

        # Per-generation checkpoint hand-off (backlog item 34): persist whatever
        # would otherwise only ever live in self._carry_state, so a wave
        # orchestrator can feed it to the NEXT generation's own process
        # invocation. Fires regardless of which branch follows — a
        # one-wave-per-invocation caller (generations=1) always takes the
        # "complete" branch below, but still needs THIS generation's daughter
        # written out. No daughter (timed out without dividing) means nothing
        # to hand off, mirroring self._carry_state staying None in that case.
        checkpoint_dir = str(self.config.get("checkpoint_dir") or "")
        if checkpoint_dir:
            out_path = f"{checkpoint_dir.rstrip('/')}/gen_{self._generation:04d}.pkl"
        else:
            out_path = str(self.config.get("daughter_state_out_path") or "")
        if out_path and daughter is not None:
            from v2ecoli.cache import save_initial_state

            payload = dict(daughter)
            payload["_prior_summaries"] = list(self._summaries)
            # Log size + growth BEFORE the write: this is the stall point in 313,
            # and the size (and its per-generation growth) is the signal that a
            # lineage is running away — carried in the log so an over-growing run
            # is visible immediately, not inferred from a hang after the fact.
            mb = _estimate_state_mb(daughter)
            prev = getattr(self, "_last_checkpoint_mb", 0.0)
            if prev and mb > 1.5 * prev:
                warnings.warn(
                    f"LineageProcess: gen {self._generation} carry state is "
                    f"{mb:.1f}MB, up {mb / prev:.1f}x from the previous "
                    f"generation ({prev:.1f}MB). A lineage whose per-generation "
                    f"state keeps growing is not reaching steady-state division "
                    f"size (over-growth); the checkpoint reflects it and the "
                    f"write gets progressively heavier."
                )
            self._last_checkpoint_mb = mb
            _t_ckpt = time.monotonic()
            print(
                f"[LineageProcess] gen {self._generation}: writing checkpoint "
                f"(~{mb:.1f}MB) -> {out_path}",
                flush=True,
            )
            save_initial_state(payload, out_path)
            print(
                f"[LineageProcess] gen {self._generation}: checkpoint written "
                f"in {time.monotonic() - _t_ckpt:.1f}s",
                flush=True,
            )

        self._generation += 1
        if self._generation >= int(self.config["generations"]):
            self._complete = True
            self._composite = None
            return {"complete": True, "summary": {"generations": self._summaries}}

        # Carry daughter 0 forward; rebuild a fresh composite next tick.
        from v2ecoli.steps.division import daughter_phylogeny_id

        self._carry_state = daughter
        self._agent_id = daughter_phylogeny_id(self._agent_id)[0]
        self._composite = None
        self._needs_build = True
        return {"summary": {"generations": self._summaries}}
