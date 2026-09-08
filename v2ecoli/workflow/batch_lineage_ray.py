"""Process-bigraph-native multiseed batch dispatch — item 101.

``batch_baseline``/``BatchBaselineRunner`` already wires N seed-lineages together, but does it
by calling Ray's raw Python API directly from inside one opaque Step (``run_workflow``) — the
seed-level fan-out is invisible to process-bigraph's own object model, structurally identical to
chain-dispatch's own external-orchestration defect, one layer down (see backlog item 98's own
memory record for the full trace).

This module builds the same seeds-x-generations shape a different way: N real ``LineageProcess``
nodes, each a genuine node in the composite's own state tree, each addressed via process-bigraph's
real ``ray:`` protocol (the same mechanism the colony composite already proves works across real
physical AWS nodes — see ``v2ecoli/colony.py``). ``LineageProcess`` itself needs no changes: it
already runs a lineage's own generation-to-generation progression using real structural deltas
(``_add``/``_remove`` at each division) — that part was already pbg-native. What's new here is
composing many of them, natively, instead of hiding the fan-out inside a Step's own Python loop.

Pool sizing is NOT left to the ``ray:`` protocol's own bare default. Empirically confirmed
(2026-09-01, local test, ``process_bigraph.protocols.ray``): the actor pool defaults to
``os.cpu_count()`` evaluated on whichever node the driver runs on -- on this ecosystem's AWS Batch
MNP topology that is the HEAD node specifically (which itself runs with ``--num-cpus=0`` in Ray's
own resource accounting, so it never RUNS an actor, but the pool-size CALCULATION still uses its
real hardware core count -- a number with zero relationship to how many workers/cores exist
elsewhere in the cluster). A real local test (8 actors, 3s each) measured pool_size=2 -> 80.75s
wall vs. pool_size=8 -> 9.09s wall, a ~9x difference from that one setting. This is the exact
mechanism that produces a "whole seeds x generations matrix running one square at a time"
bottleneck if left unsized.

``n_workers`` therefore defaults to ``None``, NOT a concrete small int (a real item-101 incident,
2026-09-01: an earlier concrete default of ``2`` silently shadowed the cluster-derived value below,
since ``RayProtocolRuntime`` only reads its env var when given ``None`` explicitly). With ``None``,
``prewarm_lineage_pool`` passes it straight through to ``get_or_create_runtime``, which falls
through to the ``RAY_SHARDS_DEFAULT`` env var -- viva-api's own dispatch code already computes this
correctly from real per-node vCPUs x real node count for every multi-node dispatch. Override
``n_workers`` explicitly only when deliberately capping concurrency below the cluster's real
capacity (e.g. fewer lineages than available shards).
"""

from __future__ import annotations

from typing import Any


def register_ray_lineage(core: Any) -> Any:
    """Register ``LineageProcess`` for the ``ray:`` address protocol and register the
    protocol's own types on ``core``. Mirrors ``v2ecoli/colony.py::make_colony``'s own
    ``transport == 'ray'`` branch -- same registration calls, applied to ``LineageProcess``
    instead of ``EcoliWCM``.

    Returns ``core`` (mutated in place; returned for chaining, matching the rest of this
    module's own functions).
    """
    from process_bigraph.protocols import ray as ray_protocol
    from v2ecoli.workflow.lineage import LineageProcess

    ray_protocol.register_types(core)
    ray_protocol.register_process_class("LineageProcess", LineageProcess)
    return core


def prewarm_lineage_pool(core: Any, n_workers: int | None) -> Any:
    """Size the ``ray:`` actor pool BEFORE any ``ray:LineageProcess`` address is resolved.

    ``RayProtocolRuntime`` sizes its pool for a (class_name, config) key on FIRST creation only
    -- "``n_shards_default`` ... honored only on creation" (``process_bigraph/protocols/ray.py``
    docstring). ``Composite.__init__`` resolves every ``ray:`` address as it builds the state
    tree, which would otherwise create the runtime with the protocol's own default pool size
    (``os.cpu_count()``) before this module gets a chance to size it correctly. Calling this
    FIRST, with the real target concurrency, wins that race deliberately.

    ``n_workers=None`` (the recommended default -- see this module's own docstring) passes
    straight through to ``get_or_create_runtime``, which falls through to the cluster-derived
    ``RAY_SHARDS_DEFAULT`` env var. Pass a concrete int only to deliberately override that.
    """
    from process_bigraph.protocols.ray import get_or_create_runtime

    if n_workers is not None and n_workers < 1:
        raise ValueError(f"prewarm_lineage_pool: n_workers must be >= 1, got {n_workers}")
    get_or_create_runtime(core, n_shards_default=n_workers)
    return core


def build_lineage_ray_batch_document(
    *,
    n_seeds: int,
    n_generations: int,
    base_seed: int = 0,
    cache_dir: str = "out/cache",
    out_dir: str = "",
    experiment_id: str = "batch_lineage_ray",
    emitter: str = "both",
    max_duration_per_gen: float = 3600.0,
    time_step: float = 1.0,
    media: str = "minimal",
    variants: dict | None = None,
    injected_processes: dict | None = None,
    config_overrides: dict | None = None,
    emitter_arg: dict | None = None,
    seed_overrides: dict[Any, dict[str, Any]] | None = None,
    exchange_fluxes: dict | None = None,
    exchange_flux_basis: str | None = None,
    variant_grid: list[dict] | None = None,
) -> dict:
    """Build a document with N real ``ray:LineageProcess`` nodes, one per seed.

    ``seed_overrides`` (item 115): per-seed overrides, keyed by seed number --
    accepts either int or str keys (a caller building the request in Python has
    ints; one round-tripping it through JSON, e.g. ``--params`` over HTTP, has
    strings; both are looked up so neither caller shape silently misses). Two
    real gaps this closes, found comparing this design against Jim's own
    Nextflow-dispatch plan (viva-api PR#405):

    - **Resume**: a seed's own entry may set ``initial_carry_state_path`` +
      ``initial_generation_index`` to resume that ONE lineage from a specific
      prior checkpoint instead of generation 0 -- the same fields chain-dispatch
      already uses for its own per-generation resume (``LineageProcess`` needed
      no changes for THIS half; the fields were already there, just never
      threaded from this document builder).
    - **Variant-specific caching**: a seed's own entry may set ``cache_dir`` to
      point that lineage at a strain-specific ParCa cache instead of the
      batch-wide default -- real, needed for Run1/Run2 (K4/J3), whose real
      strain identity lives in a variant-specific cache, not a shared baseline
      one (``v2ecoli/workflow/batch_lineage_ray.py``'s own prior comment here:
      "A real variant sweep across ray:-distributed lineages is real, separate,
      not-yet-scoped work" -- this is that work, scoped to the per-seed case).

    ``exchange_fluxes``/``exchange_flux_basis`` (item 106): ``ecoli_baseline.baseline()`` and
    ``LineageProcess`` both already accept these (a caller-supplied exchange-species-to-flux-column
    map, plus the units basis those columns are reported in -- e.g. ``{"violacein_exchange":
    "VIOLACEIN"}``/``"gdcw"``), but this document builder never threaded them onto a lineage's own
    config -- the same class of gap ``variants``/``injected_processes`` had before item109/#663.
    Needed for real CD2 Run 2 KPI reporting (a violacein-exchange flux column), not just raw state.

    Omitted entirely (the default): every lineage starts fresh at generation 0
    against the one shared ``cache_dir`` -- today's exact behavior, unchanged.

    Unlike ``_build_batch_document`` (``v2ecoli/composites/ecoli_baseline.py``), which returns a
    single-Step document whose ``BatchBaselineRunner`` fans seeds out internally at run time, this
    returns N real state-tree nodes -- the seeds x generations shape is represented in the
    document itself, not hidden inside one Step's own Python loop.

    Each lineage's ``interval`` is set to ``max_duration_per_gen``. ``Composite.run(interval)``
    takes TOTAL SIMULATED TIME to advance, not a tick/step count (confirmed directly from its own
    source, 2026-09-01 -- an easy, real mistake to make from the CLI's own ``-n/--steps`` naming
    in ``run_pbg.py``, which is a different, unrelated entrypoint). The caller is expected to call
    ``composite.run(n_generations * max_duration_per_gen)`` -- enough total simulated time for
    each lineage to reach its own division point (or its own per-generation timeout) once per
    generation, matching ``LineageProcess.update()``'s own one-call-per-generation contract.

    Per-seed lineage_seed follows ``base_seed + i`` (mirrors ``BatchBaselineRunner``'s own
    ``seeds = list(range(base_seed, base_seed + n_seeds))`` convention, so results stay directly
    comparable against the existing mechanism).
    """
    if n_seeds < 1:
        raise ValueError(f"build_lineage_ray_batch_document: n_seeds must be >= 1, got {n_seeds}")
    if n_generations < 1:
        raise ValueError(
            f"build_lineage_ray_batch_document: n_generations must be >= 1, got {n_generations}"
        )

    from v2ecoli.steps.batch_baseline_runner import resolve_out_dir

    resolved_out_dir = resolve_out_dir(out_dir)

    # ``variant_grid``: one entry per variant to sweep, each a dict of LineageProcess
    # config keys -- any of ``variant_index``, ``variant_name``, ``config_overrides``.
    # LineageProcess is itself "one (variant, seed) lineage" (its own module docstring)
    # and applies ``variant_index`` + ``config_overrides`` via ``baseline()`` at each
    # generation build, so the sweep is a genuine (variant, seed) cross-product of real
    # ``ray:``-distributed nodes -- not the older single-shared-override shape. ``None``/
    # ``[]`` means one implicit variant, preserving the seeds-only node naming and the
    # legacy ``variants`` single-shared-override behavior.
    grid = variant_grid if variant_grid else [None]
    swept = bool(variant_grid)

    state: dict[str, Any] = {"lineages": {}}
    for v_pos, variant in enumerate(grid):
        variant = dict(variant or {})
        variant_index = int(variant.get("variant_index", v_pos))
        variant_name = variant.get("variant_name")
        # Merge overrides: shared ``config_overrides``, then this variant's own, then the
        # legacy single-shared ``variants`` dict (applied to every node, back-compat).
        merged_overrides: dict[str, Any] = {}
        if config_overrides:
            merged_overrides.update(config_overrides)
        if variant.get("config_overrides"):
            merged_overrides.update(variant["config_overrides"])
        if variants:
            merged_overrides.update(variants)

        for i in range(n_seeds):
            seed = base_seed + i
            node_name = (
                f"lineage_v{variant_index:03d}_s{seed:04d}" if swept
                else f"lineage_{seed:04d}"
            )
            # Per-node per-generation checkpoint destination (item 115 / #680):
            # LineageProcess checkpoints after every generation, but only when
            # given a real dir. Disambiguate by variant when sweeping so two
            # variants of one seed do not collide; keep the seeds-only path
            # unchanged when not swept.
            checkpoint_dir = (
                f"{resolved_out_dir.rstrip('/')}/checkpoints/{experiment_id}/"
                + (f"v{variant_index:03d}_s{seed:04d}" if swept
                   else f"seed_{seed:04d}")
            )
            config: dict[str, Any] = {
                "cache_dir": cache_dir,
                "seed": seed,
                "lineage_seed": seed,
                "variant_index": variant_index,
                "generations": int(n_generations),
                "single_daughters": True,
                "experiment_id": experiment_id,
                "out_dir": resolved_out_dir,
                "max_duration_per_gen": float(max_duration_per_gen),
                "time_step": float(time_step),
                "media": media,
                "emitter": emitter,
                "checkpoint_dir": checkpoint_dir,
            }
            if variant_name:
                config["variant_name"] = variant_name
            if merged_overrides:
                config["config_overrides"] = dict(merged_overrides)
            if injected_processes:
                config["injected_processes"] = dict(injected_processes)
            if emitter_arg:
                config["emitter_arg"] = dict(emitter_arg)
            # Per-seed cache/resume override (#680), keyed by seed (int or str).
            override = None
            if seed_overrides:
                override = seed_overrides.get(seed)
                if override is None:
                    override = seed_overrides.get(str(seed))
            if override:
                if "cache_dir" in override:
                    config["cache_dir"] = override["cache_dir"]
                if "initial_carry_state_path" in override:
                    config["initial_carry_state_path"] = override["initial_carry_state_path"]
                if "initial_generation_index" in override:
                    config["initial_generation_index"] = int(override["initial_generation_index"])
            # Declared exchange-flux measurements (#691), forwarded per node.
            if exchange_fluxes:
                config["exchange_fluxes"] = dict(exchange_fluxes)
            if exchange_flux_basis:
                config["exchange_flux_basis"] = exchange_flux_basis

            state[node_name] = {
                "_type": "process",
                "address": "ray:LineageProcess",
                "config": config,
                "interval": float(max_duration_per_gen),
                "inputs": {},
                "outputs": {
                    "summary": ["lineages", node_name, "summary"],
                    "complete": ["lineages", node_name, "complete"],
                },
            }

    return {"state": state}


def build_lineage_ray_composite(
    *,
    n_seeds: int,
    n_generations: int,
    n_workers: int,
    **document_kwargs: Any,
) -> Any:
    """Build a ready-to-run ``Composite`` wiring ``n_seeds`` real ``ray:LineageProcess`` nodes.

    ``n_workers`` is the real target concurrency (see this module's own docstring on why this is
    required, never defaulted) -- typically the real number of Ray worker slots available across
    the whole MNP cluster's non-head nodes, NOT ``os.cpu_count()`` on whichever single node the
    driver happens to run on.

    Caller is responsible for ``composite.run(n_generations)`` and for tearing down the pool
    afterward (``process_bigraph.protocols.ray.shutdown_all_runtimes()``) -- this function only
    builds, it does not run or clean up, mirroring ``v2ecoli/colony.py::make_colony``'s own
    build-only contract.
    """
    from process_bigraph import Composite
    from v2ecoli.core import build_core

    core = build_core()
    register_ray_lineage(core)
    # MUST happen before Composite() resolves any ray: address below -- see
    # prewarm_lineage_pool's own docstring for why order matters here.
    prewarm_lineage_pool(core, n_workers)

    doc = build_lineage_ray_batch_document(
        n_seeds=n_seeds, n_generations=n_generations, **document_kwargs
    )
    return Composite({**doc, "parallel_processes": True}, core=core)
