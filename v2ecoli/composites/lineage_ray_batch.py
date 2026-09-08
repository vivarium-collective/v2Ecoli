"""Registers ``v2ecoli.workflow.batch_lineage_ray`` as a process-bigraph ``@composite_generator``
(item 101) so it resolves through compose/v1's EXISTING ``--composite-id`` mechanism the same way
``baseline``/``ecoli_colony`` already do — zero new viva-api/dispatch code needed, matching this
project's own "reuse existing patterns first" rule.

Thin registration shim, mirroring ``v2ecoli/composites/ecoli_colony.py``'s own shape exactly.
"""
from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.workflow.batch_lineage_ray import (
    build_lineage_ray_batch_document,
    prewarm_lineage_pool,
    register_ray_lineage,
)


@composite_generator(
    name="lineage_ray_batch",
    description=(
        "Process-bigraph-native multiseed batch: N real LineageProcess nodes, one per seed, "
        "wired directly into the composite's own state tree and addressed via the ray: protocol "
        "-- unlike batch_baseline's BatchBaselineRunner, which fans seeds out via raw Ray calls "
        "hidden inside one opaque Step."
    ),
    parameters={
        "n_seeds": {
            "type": "integer",
            "default": 2,
            "description": "Number of independent seed-lineages.",
        },
        "n_generations": {
            "type": "integer",
            "default": 1,
            "description": "Generations to run per lineage.",
        },
        "n_workers": {
            "type": "integer",
            "default": None,
            "description": (
                "Real target concurrency for the ray: actor pool. Defaults to None, which "
                "correctly falls through to the cluster-derived RAY_SHARDS_DEFAULT env var "
                "viva-api's own dispatch code already computes from real per-node vCPUs x real "
                "node count -- no per-request tuning needed for the common case. See "
                "batch_lineage_ray's own module docstring: an earlier concrete default of 2 "
                "silently shadowed that computation, producing a ~9x wall-time penalty (the "
                "'one seed at a time' bottleneck this composite exists to avoid). Override "
                "explicitly only to deliberately cap concurrency below the cluster's real capacity."
            ),
        },
        "base_seed": {"type": "integer", "default": 0, "description": "First seed; seeds are contiguous."},
        "cache_dir": {"type": "string", "default": "out/cache", "description": "Path to the ParCa cache directory."},
        "out_dir": {"type": "string", "default": "", "description": "Output dir; resolved at run time if empty."},
        "experiment_id": {"type": "string", "default": "lineage_ray_batch", "description": "Experiment id."},
        "emitter": {"type": "string", "default": "both", "description": "'parquet' | 'xarray' | 'both'."},
        "max_duration_per_gen": {
            "type": "number",
            "default": 3600.0,
            "description": (
                "Per-generation sim-time cap (seconds). Caller must run the resulting composite "
                "for n_generations * max_duration_per_gen of TOTAL SIMULATED TIME -- "
                "Composite.run(interval) takes total time to advance, not a tick count."
            ),
        },
        "time_step": {"type": "number", "default": 1.0, "description": "Integration timestep (seconds)."},
        "media": {"type": "string", "default": "minimal", "description": "Media condition."},
        "variants": {
            "type": "object",
            "default": None,
            "description": (
                "Strain/variant overrides, quote-typed and threaded verbatim into "
                "build_lineage_ray_batch_document -> each LineageProcess's own config -- "
                "already-correct plumbing (item109/#653), only unexposed at this thin wrapper "
                "until now."
            ),
        },
        "injected_processes": {
            "type": "object",
            "default": None,
            "description": (
                "Process swap/add/exclude block, same shape "
                "viva_api.simulation.simulation_service_ray.injected_processes_from_config "
                "already builds for chain-dispatch's own ecoli_baseline path -- "
                "{'swap_processes':..., 'add_processes':..., 'exclude_processes':..., "
                "'fork_repo':''}. Threaded verbatim; LineageProcess._build_generation already "
                "consumes it per-generation."
            ),
        },
        "config_overrides": {
            "type": "object",
            "default": None,
            "description": "Raw per-generation config overrides, threaded verbatim.",
        },
        "emitter_arg": {
            "type": "object",
            "default": None,
            "description": (
                "XArrayEmitter view override (e.g. {'view': [...dotted paths...]}). Without it "
                "every lineage falls back to LineageProcess.DEFAULT_XARRAY_VIEW (mass only) -- "
                "a document that needs a specific KPI column (e.g. a product exchange flux) "
                "cannot get it without this. Add 'required_leaves': [<dotted leaf paths>] to "
                "make a declared-but-absent KPI leaf raise instead of silently emitting."
            ),
        },
        "variant_grid": {
            "type": "array",
            "default": None,
            "description": (
                "One entry per variant to sweep, each a dict of LineageProcess config keys "
                "(any of variant_index, variant_name, config_overrides). Crossed with every seed "
                "-> real (variant, seed) ray:LineageProcess nodes. Empty/None = one implicit "
                "variant (the seeds-only shape). The genuine multi-variant sweep the older "
                "single-shared 'variants' override could not express."
            ),
        },
        "seed_overrides": {
            "type": "object",
            "default": None,
            "description": (
                "Per-seed overrides (item 115), keyed by seed number (int or str). A seed's own "
                "entry may set 'initial_carry_state_path' + 'initial_generation_index' to resume "
                "that ONE lineage from a specific prior checkpoint instead of generation 0, and/or "
                "'cache_dir' to point it at a strain-specific ParCa cache instead of the batch-wide "
                "default -- real, needed for a genuine variant sweep (e.g. Run1/Run2's own K4/J3 "
                "strains, each with its own cache). Every lineage already gets a real, working "
                "per-generation checkpoint directory automatically (no config needed for that half) "
                "-- this is only for a caller that wants to explicitly resume a specific seed or "
                "vary its cache. Omitted entirely: every lineage starts fresh against the shared "
                "cache_dir, today's exact behavior, unchanged."
            ),
        },
        "exchange_fluxes": {
            "type": "object",
            "default": None,
            "description": (
                "Item 106: exchange-species-to-flux-column map (e.g. {'violacein_exchange': "
                "'VIOLACEIN', 'glucose_exchange': 'GLC'}), threaded verbatim into each lineage's "
                "own LineageProcess config -- already-supported by ecoli_baseline.baseline()/"
                "LineageProcess, only unexposed at this thin wrapper until now. Needed to get a "
                "real product-exchange-flux column out of a batch run rather than only raw state."
            ),
        },
        "exchange_flux_basis": {
            "type": "string",
            "default": None,
            "description": (
                "Item 106: units basis for exchange_fluxes columns (e.g. 'gdcw'). Required "
                "alongside exchange_fluxes -- omitting it used to inherit 'counts', which is "
                "lineage-cumulative and wrong for a per-generation flux reading."
            ),
        },
    },
    visualizations=[],
    core_extensions=[register_ray_lineage],
)
def lineage_ray_batch(
    core: Any = None,
    *,
    n_seeds: int = 2,
    n_generations: int = 1,
    n_workers: int | None = None,
    base_seed: int = 0,
    cache_dir: str = "out/cache",
    out_dir: str = "",
    experiment_id: str = "lineage_ray_batch",
    emitter: str = "both",
    max_duration_per_gen: float = 3600.0,
    time_step: float = 1.0,
    media: str = "minimal",
    variants: dict | None = None,
    injected_processes: dict | None = None,
    config_overrides: dict | None = None,
    emitter_arg: dict | None = None,
    seed_overrides: dict | None = None,
    exchange_fluxes: dict | None = None,
    exchange_flux_basis: str | None = None,
    variant_grid: list[dict] | None = None,
) -> dict:
    """Build the lineage_ray_batch composite document.

    ``core_extensions`` (``register_ray_lineage``) already ran by the time this function's body
    executes -- ``core`` arrives with the ray: protocol's own types and ``LineageProcess``
    registered. This function's own job is the ONE piece ``core_extensions`` cannot do (it takes
    no parameters): size the actor pool to the real, caller-supplied ``n_workers`` BEFORE
    returning a document any ``ray:`` address in it could be resolved from -- see
    ``prewarm_lineage_pool``'s own docstring for why this ordering is load-bearing, not
    incidental.

    Falls back to building its own core (same registration ``core_extensions`` would have done)
    when called directly with no core, matching ``ecoli_colony.colony``'s own fallback shape.
    """
    if core is None:
        from v2ecoli.core import build_core

        core = build_core()
        register_ray_lineage(core)

    prewarm_lineage_pool(core, n_workers)

    doc = build_lineage_ray_batch_document(
        n_seeds=n_seeds,
        n_generations=n_generations,
        base_seed=base_seed,
        cache_dir=cache_dir,
        out_dir=out_dir,
        experiment_id=experiment_id,
        emitter=emitter,
        max_duration_per_gen=max_duration_per_gen,
        time_step=time_step,
        media=media,
        variants=variants,
        injected_processes=injected_processes,
        config_overrides=config_overrides,
        emitter_arg=emitter_arg,
        seed_overrides=seed_overrides,
        exchange_fluxes=exchange_fluxes,
        exchange_flux_basis=exchange_flux_basis,
        variant_grid=variant_grid,
    )
    return doc
