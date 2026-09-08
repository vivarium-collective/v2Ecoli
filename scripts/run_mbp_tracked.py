"""Run mbp-* simulation variants past first division, tracked in the
workspace-shared sqlite registry.

This is the v2 runner. v1 used ``sqlite_emitter()`` as a context manager
which injects a SQLiteEmitter Step into the composite — that path hits
``UNIQUE constraint failed: history.simulation_id, history.step`` at the
first division (both daughters try to emit at the same step). v2 uses
:func:`v2ecoli.library.sqlite_run.run_multigen_sqlite` with external
emitter management + lineage-following (``single_daughters=True``) so
runs continue past division.

Each run is also tagged with ``study_slug`` + ``investigation_slug`` so
the dashboard's Simulations DB tab groups them under the
``multiscale-bioprocess`` investigation.

Currently runnable variants. mbp-03's reactor coupling runs fork-free via the
local ``BiRDTransportHours`` adapter registered in ``v2ecoli/core.py`` (no
upstream bioreactor-transport fork needed).

  mbp-01-time-varying-environment:
    static-env-baseline   — baseline_time_varying_env (env_driver_mode=static)
                            single-cell + division, regression-guard

  mbp-02-population-aggregation:
    aggregator-cpa1       — baseline_population at cells_per_agent=1.0
    aggregator-cpa1e6     — at 1e6 (scaling-factor sweep)
    aggregator-cpa1e9     — at 1e9 (high-density representative sampling)

  mbp-04-multigeneration-runs:
    reactor-bird-coupled-batch-multigen
                          — v2ecoli cells <-> BiRD 0D reactor, batch.
                            Carries the import-gate booleans (v2ecoli#572).

  Cross-investigation reference:
    baseline-reference    — unmodified v2ecoli.composites.ecoli_baseline

Default duration is 120 sim-min (~2 doublings). A variant may override that
via VARIANT_DEFAULTS when its window is a study-enforced param rather than a
runner preference (mbp-04: 240 sim-min); an explicit --duration-sec /
--max-generations always wins.

Usage:
    python scripts/run_mbp_tracked.py [--variant <name>] [--duration-sec N]
                                      [--max-generations N]
"""

from __future__ import annotations

import argparse
import inspect
import os
import sqlite3
import sys
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli.composites._helpers import (
    _ensure_study_columns,
    _stamp_study_metadata,
)
from v2ecoli.core import build_core
from v2ecoli.library.sqlite_run import run_multigen_sqlite
from v2ecoli.library.parquet_run import run_multigen_parquet
from v2ecoli.library.run_provenance import write_run_identity


INVESTIGATION_SLUG = "multiscale-bioprocess"
DEFAULT_DURATION_SEC = 120 * 60   # 120 sim-min (~2 doublings)
DEFAULT_MAX_GENERATIONS = 3
DEFAULT_CHUNK = 1   # per-tick emission (~80 ms/tick on this machine)
DEFAULT_EMITTER = "parquet"   # workspace default (see workspace.yaml.runtime.default_emitter)
DB_PATH = REPO_ROOT / ".pbg" / "composite-runs.db"
# Per-study parquet roots: studies/<study_slug>/parquet-runs/<experiment_id>/...
# This is the convention vivarium-workbench's _latest_parquet_for_study reads
# from (vivarium_workbench/lib/study_charts.py:_latest_parquet_for_study).
# The cross-investigation reference variant (study_slug not a real study)
# also gets a per-slug directory so the dashboard can still discover it via
# the same code path, even though no study.yaml lives there.
#
# Overridable via V2E_STUDIES_ROOT. A remote multi-node entrypoint syncs exactly
# ONE directory to S3 (sms-api's Ray entrypoint: {V2ECOLI_DIR}/.pbg/runs/
# phase0-xarray, and nothing else), so a run whose parquet lands under
# REPO_ROOT/studies/ is computed but never uploaded — dispatch 322 ran
# reactor_bird_coupled cleanly for ~3h and nothing reached S3 for exactly this
# reason. Setting V2E_STUDIES_ROOT to a path under the synced directory makes the
# output land where the entrypoint looks. Unset (the local/dashboard default)
# keeps REPO_ROOT/studies so vivarium-workbench's _latest_parquet_for_study still
# discovers runs unchanged.
STUDIES_ROOT = Path(os.environ.get("V2E_STUDIES_ROOT") or (REPO_ROOT / "studies"))

def _parquet_root_for(study_slug: str) -> Path:
    return STUDIES_ROOT / study_slug / "parquet-runs"

# Agent-rooted emit paths (under agents/<id>/) shared by all variants.
COMMON_AGENT_PATHS = [
    "listeners/mass/cell_mass",
    "listeners/mass/dry_mass",
    "listeners/mass/instantaneous_growth_rate",
    "listeners/monomer_counts/monomerCounts",
    # boundary.external.GLC is per-agent (rooted under agents/<id>/),
    # so it belongs in emit_paths (agent-level), NOT extra_root_paths.
    # Captures the driver-supplied glucose trajectory at the cell
    # boundary for mbp-01's driven-env runs; trivial overhead for
    # other sims (one float per emit).
    "boundary/external/GLC",
]


def _build_baseline(core, cache_dir, *, seed=0):
    from v2ecoli.composites.ecoli_baseline import baseline
    return baseline(core=core, seed=seed, cache_dir=cache_dir)

def _build_baseline_population(core, cache_dir, *, cells_per_agent, seed=0):
    from v2ecoli.composites.ecoli_population import baseline_population
    return baseline_population(
        core=core, seed=seed, cache_dir=cache_dir,
        cells_per_agent=cells_per_agent,
    )

def _build_baseline_time_varying_env(
    core, cache_dir, *,
    seed=0,
    env_driver_mode="static",
    synthetic_trajectory_spec=None,
):
    from v2ecoli.composites.ecoli_time_varying_env import baseline_time_varying_env
    return baseline_time_varying_env(
        core=core, seed=seed, cache_dir=cache_dir,
        env_driver_mode=env_driver_mode,
        synthetic_trajectory_spec=synthetic_trajectory_spec or {},
    )


def _build_reactor_bird_coupled(
    core, cache_dir, *,
    seed=0,
    single_daughters=True,
    carbon_exhaustion_arrest=False,
    cells_per_agent=1.0e9,
    population_growth_mode="representative_doubling",
    initial_glucose_mM=None,
    bird_reactor_config=None,
):
    """mbp-04's coupled composite: v2ecoli cells <-> BiRD 0D reactor.

    Defaults mirror mbp-04's ``enforced_params`` so the variant is
    reproducible without remembering flags (see VARIANT_DEFAULTS for the
    duration / generation half of that contract).
    """
    from v2ecoli.composites.reactor_bird_coupled import reactor_bird_coupled
    # v2ecoli#591 adds the in-composite LineageBookkeeper behind the composite's
    # own ``single_daughters`` flag. Forward it only when the composite accepts
    # it: on a tree predating #591 there is no Step to install and the runners
    # own the pruning, which is the correct behaviour there. This keeps the
    # runner correct on both sides of that merge instead of imposing a
    # landing-order constraint on #591.
    _accepts = inspect.signature(reactor_bird_coupled).parameters
    extra = {}
    if "single_daughters" in _accepts:
        extra["single_daughters"] = single_daughters
    # v2ecoli#592: opt-in substrate-exhaustion arrest, default off. Same
    # signature guard, but NOT the same fallback -- unlike single_daughters,
    # nothing else applies the arrest, so silently dropping it would emit a run
    # that claims a code path it never took. This guard is the one that bites:
    # THIS function always accepts the kwarg, so the caller's own signature check
    # passes and only the composite's does not. Fail loudly on an explicit True.
    if "carbon_exhaustion_arrest" in _accepts:
        extra["carbon_exhaustion_arrest"] = carbon_exhaustion_arrest
    elif carbon_exhaustion_arrest:
        raise ValueError(
            "carbon_exhaustion_arrest=True was requested but this tree's "
            "reactor_bird_coupled does not accept it (predates v2ecoli#592), "
            "and no runner-side fallback applies it. Refusing to build."
        )
    return reactor_bird_coupled(
        core=core, seed=seed, cache_dir=cache_dir,
        **extra,
        cells_per_agent=cells_per_agent,
        population_growth_mode=population_growth_mode,
        initial_glucose_mM=(
            MBP_04_GLUCOSE_MM if initial_glucose_mM is None else initial_glucose_mM
        ),
        bird_reactor_config=bird_reactor_config or dict(MBP_04_REACTOR_CONFIG),
    )


def _composite_of(builder_fn):
    """The composite a builder wraps, or None if it wraps none we track.

    Kept as an explicit mapping rather than an `is` check so adding an
    arrest-capable builder (a fed-batch coupled variant, say) is a one-line
    registration here -- an identity test would silently refuse to run it while
    the dispatcher forwarded the flag correctly.
    """
    if builder_fn is _build_reactor_bird_coupled:
        from v2ecoli.composites.reactor_bird_coupled import (  # noqa: PLC0415
            reactor_bird_coupled,
        )
        return reactor_bird_coupled
    return None


def _composite_accepts_arrest(builder_fn) -> bool:
    """Would this builder's COMPOSITE accept `carbon_exhaustion_arrest`?

    The builder accepting the kwarg is not sufficient -- `_build_reactor_bird_coupled`
    always does, and only the composite it wraps decides whether the arrest can
    actually be applied (v2ecoli#592). Used for the pre-flight in main() so an
    unhonourable request fails before any compute rather than after.
    """
    composite = _composite_of(builder_fn)
    if composite is None:
        return False
    return "carbon_exhaustion_arrest" in inspect.signature(composite).parameters


# (sim_name, study_slug, builder_fn, builder_kwargs, extra_root_paths)
VARIANTS = [
    (
        "baseline-reference-multigen",
        # Cross-investigation reference (not a study); kept under a
        # named pseudo-slug so the dashboard groups it visibly with the
        # other mbp comparison entries without polluting any real study.
        "multiscale-bioprocess-reference",
        _build_baseline,
        {},
        [],
    ),
    (
        "aggregator-cpa1-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "aggregator-cpa1e6-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e6},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "aggregator-cpa1e9-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e9},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "static-env-baseline-multigen",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {},
        [],   # static mode: env-driver doesn't write external_concentrations
    ),
]


# Multi-seed sweep over the cpa scaling axis. Chris's 2026-05-28 Pass 2
# §1.a confirmation validated cpa scaling AT seed=0; this sweep extends
# the evidence to seeds {0, 1, 2} so the report can show variance bands
# across stochastic replicates and answer "is single-seed scaling a
# coincidence or a robust property of representative-sampling under the
# 0D well-mixed assumption?"
#
# 3 cpa × 3 seeds = 9 runs. Each run targets ~120 sim-min (2 doublings)
# and writes into studies/mbp-02-population-aggregation/parquet-runs/
# alongside the existing 3 single-seed runs. Sim names follow
# `aggregator-cpa<N>-seed<S>-multigen` for unambiguous artifact grouping.
_MBP_02_MULTISEED_CPA_VALUES = (1.0, 1.0e6, 1.0e9)
_MBP_02_MULTISEED_SEEDS = (0, 1, 2)

for _cpa in _MBP_02_MULTISEED_CPA_VALUES:
    for _seed in _MBP_02_MULTISEED_SEEDS:
        # Skip the (cpa, seed=0) triple since those already live as the
        # single-seed `aggregator-cpa<N>-multigen` runs above — re-running
        # would duplicate without adding signal.
        if _seed == 0:
            continue
        _cpa_label = "1" if _cpa == 1.0 else ("1e6" if _cpa == 1.0e6 else "1e9")
        VARIANTS.append((
            f"aggregator-cpa{_cpa_label}-seed{_seed}-multigen",
            "mbp-02-population-aggregation",
            _build_baseline_population,
            {"cells_per_agent": _cpa, "seed": _seed},
            [
                "population/total_biomass_gDW",
                "population/cell_count",
                "population/biomass_concentration_gL",
                "population/OD600",
            ],
        ))
del _cpa, _seed, _cpa_label  # housekeeping


# mbp-01 driven-env multigen variants — first time the EnvironmentDriver /
# EnvironmentMirror path runs at multigen scale (unblocked by 60cdbd3).
# Chris's 2026-05-28 review didn't see any env-driven trajectory; these
# produce two: linear-decline (glucose drops over the run) and zero-clamp
# (instant deprivation). Each goes into mbp-01's parquet-runs/ and feeds
# the per-study Charts panel.
_MBP_01_DRIVEN_EXTRA_PATHS: list[str] = []  # boundary.external.GLC moved to COMMON_AGENT_PATHS
VARIANTS.extend([
    # Long-duration batch-phase prefix sim (2026-05-29). Runs the cpa=1e9
    # composite at sim_seconds=21600 (6 h, the Beulig batch-phase
    # endpoint) with single_daughters and max_generations bumped to 12
    # via the runner's --max-generations flag (default cap is 3).
    # Produces the data that mbp-05's expectation-setting preliminary
    # chart needs to know whether v2ecoli's per-cell mass accumulates
    # enough over 6 h to materially close the Beulig density gap, or
    # whether even at this duration the residual gap persists.
    (
        "aggregator-cpa1e9-batch-prefix-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e9},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "linear-decline-glc-multigen",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {
            "env_driver_mode": "synthetic_trajectory",
            "synthetic_trajectory_spec": {
                # 5 mM -> 0 over 60 sim-min (~1 doubling). Bare-name
                # convention per chris_feedback_2026_05_28 §11 resolution.
                "GLC": {
                    "kind": "linear_decline",
                    "start_gL": 5.0, "end_gL": 0.0, "duration_min": 60.0,
                },
            },
        },
        _MBP_01_DRIVEN_EXTRA_PATHS,
    ),
    (
        "zero-clamp-glc-multigen",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {
            "env_driver_mode": "synthetic_trajectory",
            "synthetic_trajectory_spec": {
                "GLC": {"kind": "clamp_to_value", "value_mmolL": 0.0},
            },
        },
        _MBP_01_DRIVEN_EXTRA_PATHS,
    ),
])


# --- mbp-04 multigeneration batch ------------------------------------------
# Promoted out of a scratchpad script (2026-08-24). Both prior mbp-04 runs
# (2026-08-21) were driven by an untracked script, so the study's headline
# trajectory was reproducible only by the session that ran it -- no
# run_identity sidecar, no registry row, no dashboard grouping.
#
# Params are mbp-04's own `enforced_params`: 4.0 g/L glucose (MW 180.156),
# cells_per_agent 1e9 ("calibrated so 4 g/L depletes within the 240-min
# window"), bubble column 1.0 L / 2.0 Lpm / 310.15 K, 240 sim-min.
#
# NOT a graded study. Two of mbp-04's five criteria measure
# reactor.diagnostics.{carbon,nitrogen}_residual, which nothing in this repo
# computes; the headline criterion is undispatchable on both `kind` and `op`.
# This variant produces the TRAJECTORY.
MBP_04_GLUCOSE_MM = 4.0 / 180.156 * 1000.0      # 22.20 mM
MBP_04_REACTOR_CONFIG = {
    "reactor_type": "bubble_column",
    "volume_L": 1.0,
    "gas_flow_rate_Lpm": 2.0,
    "temperature_K": 310.15,
}

VARIANTS.append((
    "reactor-bird-coupled-batch-multigen",
    "mbp-04-multigeneration-runs",
    _build_reactor_bird_coupled,
    {},
    [
        "population/total_biomass_gDW",
        "population/cell_count",
        "population/biomass_concentration_gL",
        "population/OD600",
        "reactor/glucose_medium_mM",
        "reactor/dissolved_o2",
        "reactor/dissolved_co2",
        "reactor/biomass",
    ],
))


# Per-variant agent-level emit paths, added to COMMON_AGENT_PATHS.
#
# The gate booleans are the discriminating observable for v2ecoli#572:
# `constrained_molecules` / `unconstrained_molecules` are boolean masks over
# the 87-entry sim_data.external_state.all_external_exchange_molecules
# (GLC[p] = index 0) -- they answer "did the import gate FIRE?", which is
# sharper than a rate. `external_exchange_fluxes` is emitted DELIBERATELY
# despite being measured all-zeros on 2026-08-21: it is the probe for whether
# #576's batch emitter-sink install changes emission on this path. Do not
# grade anything on it without a known-secreting positive control (see the
# cross-lane zero-exchange-flux hazard).
EXTRA_AGENT_PATHS: dict[str, list[str]] = {
    "reactor-bird-coupled-batch-multigen": [
        "boundary/external/OXYGEN-MOLECULE",
        "listeners/fba_results/external_exchange_fluxes",
        "listeners/fba_results/constrained_molecules",
        "listeners/fba_results/unconstrained_molecules",
    ],
}

# Per-variant duration / generation defaults, applied only when the caller did
# NOT pass the corresponding flag. mbp-04's window is a study-enforced param
# (240 sim-min), not a runner preference -- the global 120-min default would
# silently truncate it, and the 2026-08-21 run 1 was in fact cut short at 135
# sim-min by the generation cap rather than the clock.
VARIANT_DEFAULTS: dict[str, dict] = {
    "reactor-bird-coupled-batch-multigen": {
        "duration_sec": 240 * 60,
        "max_generations": 6,
    },
}


def _register_simulation_row(
    db_path: Path, *, simulation_id: str, name: str,
    study_slug: str, investigation_slug: str, started_at: float,
) -> None:
    """Insert a row into the simulations table that the dashboard reads.

    SQLiteEmitter writes history rows but does NOT itself add a
    simulations table row when constructed externally (the in-composite
    Step path does that automatically). We insert it here so the
    dashboard's /api/simulations endpoint surfaces the run.
    """
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        # Schema is created by SQLiteEmitter on first connect; the table
        # always exists after the first em.update(). Use INSERT OR IGNORE
        # so re-runs of the registration step don't conflict on the
        # PRIMARY KEY (simulation_id).
        conn.execute(
            """
            INSERT OR IGNORE INTO simulations
              (simulation_id, name, started_at, study_slug, investigation_slug)
            VALUES (?, ?, ?, ?, ?)
            """,
            (simulation_id, name, iso, study_slug, investigation_slug),
        )
        conn.commit()
    finally:
        conn.close()


def _count_parquet_rows(out_dir: Path, experiment_id: str) -> int:
    """Count rows emitted under one experiment_id by scanning the
    hive-partitioned parquet tree. Cheap polars scan; column not needed."""
    try:
        import polars as pl
        hive_root = out_dir / experiment_id / "history"
        if not hive_root.exists():
            return 0
        # Lazy scan + count; doesn't materialise data.
        return int(pl.scan_parquet(str(hive_root / "**" / "*.pq")).select(
            pl.len()
        ).collect().item())
    except Exception:
        return 0


def _run_one_variant(
    *, sim_name, study_slug, builder_fn, builder_kwargs, extra_root_paths,
    duration_sec, max_generations, chunk, cache_dir, core, emitter,
    single_daughters=True, carbon_exhaustion_arrest=False,
) -> dict:
    # COMMON_AGENT_PATHS is shared by every variant; EXTRA_AGENT_PATHS adds
    # the observables only one variant needs (see its docstring for why
    # mbp-04 carries the gate booleans).
    agent_paths = COMMON_AGENT_PATHS + EXTRA_AGENT_PATHS.get(sim_name, [])
    # v2ecoli#591: the in-composite LineageBookkeeper is opt-in on the COMPOSITE's
    # own single_daughters flag. A runner run with single_daughters=True against a
    # composite built with the default False silently gets the OLD chunk-dependent
    # behaviour -- while run_identity still records single_daughters: true. Thread
    # it to any builder that accepts it so the two cannot disagree.
    _params = inspect.signature(builder_fn).parameters
    if "single_daughters" in _params:
        builder_kwargs = {**builder_kwargs, "single_daughters": single_daughters}
    # v2ecoli#592: same seam, but NOT the same fallback. `single_daughters` is
    # also honoured runner-side by run_multigen_{parquet,sqlite}, so a builder
    # that cannot take it still gets the requested pruning and run_identity stays
    # truthful. `carbon_exhaustion_arrest` has no such fallback: if the composite
    # cannot take it, nothing else applies it, and recording the request would be
    # the very lie this function exists to prevent. Its default is False, so an
    # explicit True that cannot be honoured is unambiguous -- fail loudly instead
    # of emitting a sidecar that claims a code path the run never took.
    # Only the coupled variant models substrate exhaustion at all; the other 14
    # builders have no arrest to enable and never will, on any tree. So an
    # inapplicable variant is NOT an error -- raising here would abort the whole
    # default sweep on variant 1 (`_build_baseline`) and blame a merge that has
    # nothing to do with it. Skip it, say so, and let run_identity record the
    # effective False, which is truthful. The genuine "you asked for something
    # this tree cannot do" case is caught inside the coupled builder itself.
    _arrest_forwarded = "carbon_exhaustion_arrest" in _params
    if _arrest_forwarded:
        builder_kwargs = {**builder_kwargs,
                          "carbon_exhaustion_arrest": carbon_exhaustion_arrest}
    print(f"\n=== {sim_name} ({study_slug}) ===")
    print(f"  emitter: {emitter}")
    print(f"  duration: {duration_sec}s ({duration_sec/60:.0f} sim-min)")
    print(f"  max_generations: {max_generations}")
    print(f"  kwargs: {builder_kwargs}")
    if carbon_exhaustion_arrest and not _arrest_forwarded:
        print(
            "  NOTE: --carbon-exhaustion-arrest does not apply to this variant "
            f"({builder_fn.__name__} models no substrate exhaustion); "
            "recording carbon_exhaustion_arrest=false."
        )

    simulation_id = str(uuid.uuid4())

    t_build = time.time()
    # Silence the composite's internal generator-declared default emitter
    # (workspace runtime.default_emitter). It is not lineage-aware: at the
    # first division it writes the raw inner agent_id under
    # experiment_id=default and crashes the run with a missing-partition
    # FileNotFoundError. Recording here is the external lineage-following
    # emitter's job (run_multigen_parquet / run_multigen_sqlite).
    from v2ecoli.composites import _helpers as _emit_h
    _prev_null_override = _emit_h._NULL_EMITTER_OVERRIDE
    _emit_h.set_null_emitter_override(True)
    try:
        doc = builder_fn(core, cache_dir, **builder_kwargs)
        from process_bigraph import Composite
        composite = Composite(doc, core=core)
    finally:
        _emit_h.set_null_emitter_override(_prev_null_override)
    build_time = time.time() - t_build

    t_start = time.time()

    if emitter == "sqlite":
        # Workspace-shared sqlite registry; tag for dashboard grouping.
        _ensure_study_columns(str(DB_PATH))
        _register_simulation_row(
            DB_PATH, simulation_id=simulation_id, name=sim_name,
            study_slug=study_slug, investigation_slug=INVESTIGATION_SLUG,
            started_at=t_start,
        )
        _stamp_study_metadata(str(DB_PATH), simulation_id, study_slug, INVESTIGATION_SLUG)

        t_run = time.time()
        result = run_multigen_sqlite(
            composite,
            run_id=simulation_id,
            db_file=str(DB_PATH),
            emit_paths=agent_paths,
            extra_root_paths=extra_root_paths,
            max_steps=duration_sec,
            max_generations=max_generations,
            chunk=chunk,
            single_daughters=single_daughters,
            core=core,
        )
        wall_time = time.time() - t_run

        conn = sqlite3.connect(str(DB_PATH))
        n_rows, max_step = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(step), 0) FROM history WHERE simulation_id = ?",
            (simulation_id,),
        ).fetchone()
        conn.close()

        artifact = str(DB_PATH.relative_to(REPO_ROOT))

    elif emitter == "parquet":
        # Workspace-default emitter (workspace.yaml.runtime.default_emitter).
        # Hive-partitioned per experiment_id/variant/lineage_seed/generation/agent_id.
        # Written under studies/<study_slug>/parquet-runs/ so vivarium-workbench's
        # _latest_parquet_for_study can discover them per-study.
        parquet_root = _parquet_root_for(study_slug)
        parquet_root.mkdir(parents=True, exist_ok=True)

        t_run = time.time()
        result = run_multigen_parquet(
            composite,
            experiment_id=simulation_id,
            out_dir=str(parquet_root),
            emit_paths=agent_paths,
            extra_root_paths=extra_root_paths,
            max_steps=duration_sec,
            max_generations=max_generations,
            chunk=chunk,
            single_daughters=single_daughters,
            core=core,
            study_slug=study_slug,
            investigation_slug=INVESTIGATION_SLUG,
        )
        wall_time = time.time() - t_run

        n_rows = _count_parquet_rows(parquet_root, simulation_id)
        max_step = n_rows - 1 if n_rows > 0 else 0
        artifact = str(
            (parquet_root / simulation_id).relative_to(REPO_ROOT)
        )
        # v2ecoli#472/#473: canonical run_identity.json sidecar, at the actual
        # sweep_dir sim_vector_cache._run_commit reads (out_dir/experiment_id,
        # not out_dir itself — this runner nests one experiment_id per run
        # under a shared per-study parquet_root).
        write_run_identity(
            str(parquet_root / simulation_id), cache_dir=cache_dir,
            design={
                "experiment_id": simulation_id,
                "sim_name": sim_name,
                "study_slug": study_slug,
                "investigation_slug": INVESTIGATION_SLUG,
                "duration_sec": duration_sec,
                "max_generations": max_generations,
                "chunk": chunk,
                "single_daughters": single_daughters,
                # The EFFECTIVE value, not the requested one. Reaching here
                # with _arrest_forwarded False means the arrest was inapplicable
                # to this variant (the NOTE above) -- it may well have been
                # asked for, and recording the request is exactly the lie this
                # threading exists to prevent. The unhonourable-request case
                # never reaches here: the coupled builder raises.
                "carbon_exhaustion_arrest": (
                    carbon_exhaustion_arrest and _arrest_forwarded
                ),
            },
        )

    else:
        raise ValueError(f"unknown emitter {emitter!r}; expected sqlite|parquet")

    print(f"  build: {build_time:.1f}s  run: {wall_time:.1f}s")
    print(f"  result: {result}")
    print(f"  artifact: {artifact}")
    print(f"  rows: {n_rows}  max_step: {max_step}")

    return {
        "simulation_id":   simulation_id,
        "sim_name":        sim_name,
        "study_slug":      study_slug,
        "emitter":         emitter,
        "artifact":        artifact,
        "duration_sec":    duration_sec,
        "build_wall_s":    build_time,
        "run_wall_s":      wall_time,
        "result_steps":    result.get("steps"),
        "result_gens":     result.get("generations"),
        "n_history_rows":  n_rows,
        "max_step":        max_step,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None,
                   help="Run only the named variant (default: all).")
    p.add_argument("--emitter", choices=["parquet", "sqlite"], default=DEFAULT_EMITTER,
                   help=(f"Emitter to capture history with (default: {DEFAULT_EMITTER}, "
                         "the workspace default per workspace.yaml.runtime.default_emitter)."))
    # default=None so a variant's own VARIANT_DEFAULTS entry can supply the
    # value; an explicit flag always wins. Falls back to DEFAULT_* otherwise.
    p.add_argument("--duration-sec", type=int, default=None,
                   help=f"Sim seconds (default: per-variant, else {DEFAULT_DURATION_SEC}).")
    p.add_argument("--max-generations", type=int, default=None,
                   help=f"Generation cap (default: per-variant, else {DEFAULT_MAX_GENERATIONS}).")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                   help=(f"Composite-tick chunk (default {DEFAULT_CHUNK} → "
                         "per-tick emit; larger = sparser emit, faster runtime)."))
    p.add_argument("--cache-dir", default="out/cache")
    # Lineage-following toggle. Default True (preserves existing variant
    # behavior); --no-single-daughters lets BOTH daughters continue at
    # each division, producing exponentially-growing populations (2^N
    # cells after N generations). Memory grows roughly linearly with
    # active agents; cap max_generations to control RSS.
    p.add_argument("--single-daughters", action="store_true", default=True,
                   help="Follow one daughter per division (default).")
    p.add_argument("--no-single-daughters", action="store_false",
                   dest="single_daughters",
                   help=("Continue BOTH daughters at each division. Memory "
                         "scales with active agents; cap --max-generations "
                         "to bound RSS."))
    p.add_argument("--carbon-exhaustion-arrest", action="store_true",
                   default=False,
                   help=("v2ecoli#592: arrest biomass growth once the carbon "
                         "source is exhausted (opt-in; default off)."))
    args = p.parse_args()

    variants = VARIANTS
    if args.variant:
        variants = [v for v in VARIANTS if v[0] == args.variant]
        if not variants:
            available = ", ".join(v[0] for v in VARIANTS)
            sys.exit(f"unknown variant {args.variant!r}; available: {available}")

    print(f"Running {len(variants)} variant(s) under investigation={INVESTIGATION_SLUG}")
    if args.emitter == "sqlite":
        print(f"Workspace DB: {DB_PATH.relative_to(REPO_ROOT)}")
    else:
        print("Parquet roots: studies/<study_slug>/parquet-runs/<simulation_id>/history/...")
    print(f"Per-variant: emitter={args.emitter}  chunk={args.chunk}  "
          f"(duration / max_generations resolved per variant)")

    # Pre-flight: the arrest is only honoured by the coupled composite, and only
    # once v2ecoli#592 has landed. The builder raises rather than emit a sidecar
    # claiming an arrest it never applied -- but the coupled variant is LAST in
    # sweep order and carries the longest window, so without this check a bare
    # `--carbon-exhaustion-arrest` on a pre-#592 tree would burn 14 variants of
    # compute before failing. Fail before building anything instead.
    if args.carbon_exhaustion_arrest:
        _honourable = [
            name for name, _slug, fn, _kw, _extra in variants
            if _composite_accepts_arrest(fn)
        ]
        if not _honourable:
            sys.exit(
                "--carbon-exhaustion-arrest was requested but no selected "
                "variant can honour it: none of "
                f"{[v[0] for v in variants]} builds a composite that accepts "
                "`carbon_exhaustion_arrest`. Either this tree predates "
                "v2ecoli#592, or the selection models no substrate exhaustion "
                "(only the coupled variant does). Refusing rather than running "
                "something other than what was asked for -- the sidecars would "
                "record `false`, truthfully, but no run would carry the arrest."
            )

    core = build_core()
    results = []
    t_all = time.time()
    for sim_name, study_slug, builder_fn, builder_kwargs, extra_root in variants:
        # Explicit flag > per-variant default > global default.
        _vd = VARIANT_DEFAULTS.get(sim_name, {})
        _duration = (args.duration_sec if args.duration_sec is not None
                     else _vd.get("duration_sec", DEFAULT_DURATION_SEC))
        _max_gens = (args.max_generations if args.max_generations is not None
                     else _vd.get("max_generations", DEFAULT_MAX_GENERATIONS))
        result = _run_one_variant(
            sim_name=sim_name, study_slug=study_slug,
            builder_fn=builder_fn, builder_kwargs=builder_kwargs,
            extra_root_paths=extra_root,
            duration_sec=_duration,
            max_generations=_max_gens,
            chunk=args.chunk,
            emitter=args.emitter,
            cache_dir=args.cache_dir, core=core,
            single_daughters=args.single_daughters,
            carbon_exhaustion_arrest=args.carbon_exhaustion_arrest,
        )
        results.append(result)
    total_wall = time.time() - t_all

    print(f"\n{'='*60}")
    print(f"Done — {len(results)} variant(s) in {total_wall:.1f}s wall "
          f"({total_wall/60:.1f} min).")
    for r in results:
        # Sim-minutes come from the runner's own tick counter, NOT from the
        # emitted row count: with chunk>1 the composite advances `chunk` ticks
        # per emitted row, so rows/60 under-reports sim time by a factor of
        # `chunk` (an 8400-tick run at chunk=100 printed "sim=1.4min").
        sim_min = (r.get("result_steps") or 0) / 60.0
        gens = r["result_gens"]
        print(f"  {r['sim_name']:35s} sim={sim_min:5.1f}min  gens={gens}  rows={r['n_history_rows']}")


if __name__ == "__main__":
    main()
