"""reactor_bird_coupled — baseline_population + BiRD reactor + cell<->reactor coupler.

Build-phase wire-up for mbp-03-reactor-coupling (req-2). This is the
multiscale-bioprocess composite: it composes v2ecoli's single-cell whole-cell
model (with the population aggregator) against the BiRD bioreactor transport
physics, mediated by the :class:`ReactorCellCoupler` Step.

Topology (Option B of the v2ecoli <-> pbg-bioreactordesign coupling — BiRD owns
transport, v2ecoli owns biomass + metabolic demand)::

    baseline_population (cell side)
      agents.0.*            single-cell WCM
      population.*          PopulationAggregator output (biomass_concentration_gL, ...)
      environment.*         top-level env store (external_concentrations, media_id)
        environment_driver  EnvironmentDriver in external_store mode (no-op; the
                            coupler is the env source of truth)
        environment_mirror  propagates environment.external_concentrations ->
                            agents.*.boundary.external each tick

    reactor.*               shared reactor stores (mg/L gases, g/L biomass)
      reactor_transport     BiRDTransportProcess (local:BiRDTransportProcess)
      reactor_cell_coupler  ReactorCellCoupler Step

Shared-store / additive-delta convention (mirrors
``pbg_bioreactordesign.composites._factory.make_coupled_document``):

  * ``reactor.dissolved_o2`` / ``reactor.dissolved_co2`` are bare ``float``
    leaves. BOTH BiRDTransportProcess (transport, +) and ReactorCellCoupler
    (consumption, -) write ADDITIVE deltas to them; process-bigraph's ``float``
    ``_apply`` sums the two contributions, so the net change each tick is
    ``transport - consumption``.
  * ``reactor.biomass`` is ``overwrite[float]``: the coupler passes the
    population biomass concentration through (absolute value), and
    BiRDTransportProcess reads it as a read-only input.

Port-schema subtlety (load-bearing — verified empirically against the PB
realize/project path): ReactorCellCoupler declares a single ``reactor`` output
port typed ``InPlaceDict`` (an empty Node). When the ``reactor`` store becomes a
structured tree (because BiRDTransportProcess wires ``dissolved_o2`` /
``dissolved_co2`` as concrete ``float`` leaf outputs), PB projects the coupler's
update per-leaf and DROPS any leaf the port schema doesn't enumerate — so
``reactor.biomass`` would silently never apply. We therefore OVERRIDE the
coupler edge's ``_outputs['reactor']`` with an explicit per-leaf schema
(``biomass: overwrite[float]``, ``dissolved_o2/co2: float``) so biomass routes
(overwrite) while the gases stay additive. The gases alone propagate even
without the override (they have transport leaf-output ports), but biomass does
not; the override fixes both consistently.
"""

from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import _make_instance, make_edge
from v2ecoli.composites.ecoli_population import baseline_population
from v2ecoli.composites.ecoli_time_varying_env import (
    ENVIRONMENT_DRIVER_STEP_NAME,
    ENVIRONMENT_MIRROR_STEP_NAME,
    _empty_environment_store,
)
from v2ecoli.steps.environment_driver import (
    ENV_DRIVER_MODE_EXTERNAL_STORE,
    EnvironmentDriver,
)
from v2ecoli.steps.environment_mirror import EnvironmentMirror
from v2ecoli.steps.reactor_cell_coupler import (
    AMMONIUM_MEDIUM_LEAF,
    GLUCOSE_MEDIUM_LEAF,
    ReactorCellCoupler,
)


# Top-level state keys / step names.
REACTOR_STORE_NAME = "reactor"
REACTOR_TRANSPORT_NAME = "reactor_transport"
REACTOR_CELL_COUPLER_STEP_NAME = "reactor_cell_coupler"


# Default BiRDTransportProcess config (mbp-03 req-2). Time base: hours.
DEFAULT_BIRD_REACTOR_CONFIG: dict[str, Any] = {
    "reactor_type":      "bubble_column",
    "volume_L":          1.0,
    "gas_flow_rate_Lpm": 2.0,
    "temperature_K":     310.15,
}

# Fallback initial dissolved-gas concentrations (mg/L), used only if the BiRD
# equilibrium can't be computed at build time. ~O2 saturation at 37 C.
FALLBACK_DISSOLVED_O2_MGL = 8.0
FALLBACK_DISSOLVED_CO2_MGL = 0.5

# Medium glucose recipe seed (mmol/L) for the ReactorCellCoupler medium
# accumulator. Default ~ M9 batch glucose (4 g/L / 180.16 g/mol =
# 22.2 mM); the coupler draws this pool down by the cell's GLC uptake so
# reactor.glucose_medium_mM is a gradeable remaining-glucose CONCENTRATION
# (vs Beulig reactor_glucose_data.csv). Byproduct leaves seed at 0 (cumulative
# secreted concentration).
DEFAULT_INITIAL_GLUCOSE_MM = 22.2
# Batch ammonium, ~2x the M9 recipe value (which seeds boundary.external at
# 30.272 mM).
#
# ⛔ WHY IT IS NOT THE RECIPE VALUE. At the recipe pool, nitrogen -- not carbon --
# is the binding constraint at the OD10 working point, with almost no margin:
#
#     OD  8.0 -> 2.72 gDW -> 23.3 mM N needed   (77% of a 30.27 mM pool)
#     OD 10.0 -> 3.40 gDW -> 29.1 mM N needed   (96% of pool)
#     OD 12.0 -> 4.08 gDW -> 35.0 mM N needed   (EXCEEDS pool)
#
# `[m@31Aug]` A 40 mM-glucose batch run exhausted ammonium outright by OD ~12.7,
# and growth then continued past exhaustion. Sizing N off the recipe leaves an
# OD10 run on a knife edge where a slightly faster seed hits the wall.
#
# ⊕ Raising it also makes the draw-down behaviourally inert in the working range:
# a finite pool that is never exhausted behaves exactly like the previous
# effectively-infinite one, so the physical change only bites well beyond the
# target -- which is where it SHOULD bite.
DEFAULT_INITIAL_AMMONIUM_MM = 60.0
# Byproduct medium-concentration leaves seeded on the reactor store (mmol/L,
# ADDITIVE float — only the coupler writes them). Mirrors
# reactor_cell_coupler.BYPRODUCT_LEAVES.
MEDIUM_BYPRODUCT_LEAVES = (
    "acetate_mM", "lactate_mM", "formate_mM",
    "ethanol_mM", "pyruvate_mM", "succinate_mM",
)

# BiRDTransportProcess update interval (SECONDS — v2ecoli's global time base).
#
# TIME BASE (the proper fix; mbp-03 T5): the WCM steps in SECONDS while the BiRD
# transport math is in HOURS (``dC = kLa[1/h]*(C*-C)*interval`` with interval
# treated as hours). We wire the v2ecoli-side ``BiRDTransportHours`` adapter
# (local:BiRDTransportHours) which converts the per-second interval to hours
# (/3600) before the upstream transport update — mirroring ReactorCellCoupler's
# own timestep/3600 conversion so both transport and consumption deltas land on
# the same per-second basis. The interval below is therefore a NORMAL per-second
# WCM step (no damping hack).
#
# Stability: the adapter makes the effective per-step factor kLa[1/h]*(dt_s/3600)
# = ~20.8 * (1/3600) ~= 0.006 << 1 for the default bubble_column config, so the
# explicit-Euler transport relaxes monotonically to the gas-liquid equilibrium.
DEFAULT_TRANSPORT_INTERVAL = 1.0


def _transport_equilibrium(bird_config: dict[str, Any]) -> tuple[float, float]:
    """Gas-liquid equilibrium dissolved O2/CO2 (mg/L) for the BiRD config.

    Seeding the shared stores at the transport fixed point (``C* ``) means the
    reactor starts at gas-liquid equilibrium, so the only initial dynamics come
    from the cell's metabolic demand rather than an artificial saturation
    transient. Falls back to fixed values if the BiRD transport module isn't
    importable (the composite can't actually run in that case, but the document
    still builds).
    """
    try:
        from pbg_bioreactordesign import BiRDTransportProcess
        from pbg_bioreactordesign.transport import compute_transport_state

        cfg = {k: spec.get("_default")
               for k, spec in BiRDTransportProcess.config_schema.items()}
        cfg.update(bird_config)
        t = compute_transport_state(cfg, cfg["gas_flow_rate_Lpm"])
        return float(t["cstar_o2"]), float(t["cstar_co2"])
    except Exception:
        return FALLBACK_DISSOLVED_O2_MGL, FALLBACK_DISSOLVED_CO2_MGL


def _reactor_store(
    bird_config: dict[str, Any],
    initial_glucose_mM: float = DEFAULT_INITIAL_GLUCOSE_MM,
    initial_ammonium_mM: float = DEFAULT_INITIAL_AMMONIUM_MM,
) -> dict[str, Any]:
    """Seed the shared reactor store.

    ``dissolved_o2`` / ``dissolved_co2`` are bare floats (additive multi-writer
    shared stores). ``biomass`` is overwrite[float] (single writer = coupler,
    passthrough). Read-only inputs (``glucose``, ``gas_flow_rate_Lpm``,
    ``volume_L``) and the transport diagnostics are seeded so wires resolve.

    ``glucose_medium_mM`` + the ``*_mM`` byproduct leaves are ADDITIVE float
    medium-concentration accumulators the coupler integrates exchange counts into:
    glucose seeds at the medium recipe (drawn down by uptake); byproducts seed at
    0 (accumulate secretion).
    """
    cstar_o2, cstar_co2 = _transport_equilibrium(bird_config)
    medium = {GLUCOSE_MEDIUM_LEAF: float(initial_glucose_mM),
              AMMONIUM_MEDIUM_LEAF: float(initial_ammonium_mM)}
    medium.update({leaf: 0.0 for leaf in MEDIUM_BYPRODUCT_LEAVES})
    return {
        # Additive shared dissolved-gas stores (mg/L) — both transport and the
        # coupler write deltas; the float _apply sums them. Seeded at the
        # gas-liquid equilibrium (C*) so the transport delta starts near zero.
        "dissolved_o2":      cstar_o2,
        "dissolved_co2":     cstar_co2,
        # Overwrite passthrough: coupler writes, transport reads (g/L).
        "biomass":           {"_type": "overwrite[float]", "_default": 0.0},
        # Read-only transport inputs.
        "glucose":           0.0,
        "gas_flow_rate_Lpm": float(bird_config.get("gas_flow_rate_Lpm", 2.0)),
        # BiRD owns volume; the coupler reads it (BiRD config seeds it).
        "volume_L":          float(bird_config.get("volume_L", 1.0)),
        # Transport diagnostics (overwrite readouts).
        "o2_transport_delta":  {"_type": "overwrite[float]", "_default": 0.0},
        "co2_transport_delta": {"_type": "overwrite[float]", "_default": 0.0},
        "kla_o2":              {"_type": "overwrite[float]", "_default": 0.0},
        # kla_co2 is emitted by BiRDTransportProcess.outputs() exactly as kla_o2
        # is, but was never declared or wired here — so the CO2 side of the gas
        # transfer was UNOBSERVABLE while the O2 side was fully instrumented.
        # (Transport itself was always correct; only the readout was missing.)
        "kla_co2":             {"_type": "overwrite[float]", "_default": 0.0},
        "o2_saturation":       {"_type": "overwrite[float]", "_default": 0.0},
        "co2_saturation":      {"_type": "overwrite[float]", "_default": 0.0},
        "gas_holdup":          {"_type": "overwrite[float]", "_default": 0.0},
        # Medium-concentration accumulators (mmol/L; additive — coupler-only).
        **medium,
    }


def add_reactor_coupling(
    document: dict,
    core: Any = None,
    *,
    bird_config: dict | None = None,
    cells_per_agent: float = 1.0,
    initial_glucose_mM: float = DEFAULT_INITIAL_GLUCOSE_MM,
    initial_ammonium_mM: float = DEFAULT_INITIAL_AMMONIUM_MM,
    track_medium: bool = True,
) -> dict:
    """Layer the BiRD reactor + cell<->reactor coupling onto a cell document.

    Shared helper (mbp-09 req-2): applies the SAME reactor/env additions to any
    cell-base document that already has a top-level ``population`` store (from
    :func:`add_population_aggregator`) and a ``flow_order`` with ``media_update``
    — either the WCM ``baseline_population`` or the Millard ``baseline_millard``
    (+ aggregator). Mutates and returns ``document`` in place. Adds:

      * the EnvironmentDriver (external_store mode, no-op) + EnvironmentMirror;
      * the shared ``reactor`` store (additive dissolved gases, overwrite
        biomass/diagnostics);
      * the ``BiRDTransportHours`` transport edge (``local:BiRDTransportHours``);
      * the ``ReactorCellCoupler`` Step (with the per-leaf reactor output-schema
        override so biomass routes overwrite while gases stay additive).

    This is cell-engine-agnostic: the coupler reads
    ``population.biomass_concentration_gL`` + ``agents.*.environment.exchange``
    (O2/CO2 counts), neither of which is WCM-specific.
    """
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    bird_cfg = dict(DEFAULT_BIRD_REACTOR_CONFIG)
    if bird_config:
        bird_cfg.update(bird_config)

    state = document["state"]
    flow_order = document.setdefault("flow_order", [])

    # --- environment hook: driver (external_store) + mirror ---------------
    # The coupler is the env source of truth, so the driver is a no-op
    # (external_store mode); the mirror must still run so agents see the
    # coupler-written boundary concentrations.
    state.setdefault("environment", _empty_environment_store())

    driver_config = {
        "env_driver_mode":           ENV_DRIVER_MODE_EXTERNAL_STORE,
        "synthetic_trajectory_spec": {},
    }
    driver = _make_instance(EnvironmentDriver, driver_config, core)
    state[ENVIRONMENT_DRIVER_STEP_NAME] = make_edge(
        driver, EnvironmentDriver.topology, edge_type="step",
        config=driver_config,
    )

    mirror = _make_instance(EnvironmentMirror, {}, core)
    state[ENVIRONMENT_MIRROR_STEP_NAME] = make_edge(
        mirror, EnvironmentMirror.topology, edge_type="step", config={},
    )

    # Insert driver + mirror BEFORE the unique_update FLUSH that precedes
    # media_update, so the FLUSH commits their writes before exchange_data
    # re-derives metabolism's exchange constraints (identical placement logic
    # to baseline_time_varying_env).
    if "media_update" in flow_order:
        media_idx = flow_order.index("media_update")
        flush_idx = media_idx - 1
        while flush_idx > 0 and not flow_order[flush_idx].startswith("unique_update_"):
            flush_idx -= 1
        insert_at = (flush_idx
                     if flow_order[flush_idx].startswith("unique_update_")
                     else media_idx)
        flow_order.insert(insert_at, ENVIRONMENT_MIRROR_STEP_NAME)
        flow_order.insert(insert_at, ENVIRONMENT_DRIVER_STEP_NAME)
    else:
        flow_order.extend([ENVIRONMENT_DRIVER_STEP_NAME, ENVIRONMENT_MIRROR_STEP_NAME])

    # --- reactor side: shared stores + transport + coupler ----------------
    state[REACTOR_STORE_NAME] = _reactor_store(
        bird_cfg, initial_glucose_mM, initial_ammonium_mM)

    # BiRDTransportHours (local: link registered in build_core) — the
    # seconds->hours time-base adapter over BiRDTransportProcess. Reads the
    # shared dissolved stores + biomass/glucose/gas_flow; emits ADDITIVE deltas
    # to dissolved_o2/co2 (bare float ports) + overwrite diagnostics.
    state[REACTOR_TRANSPORT_NAME] = {
        "_type":   "process",
        "address": "local:BiRDTransportHours",
        "config":  dict(bird_cfg),
        "interval": DEFAULT_TRANSPORT_INTERVAL,
        "inputs": {
            "dissolved_o2":      [REACTOR_STORE_NAME, "dissolved_o2"],
            "dissolved_co2":     [REACTOR_STORE_NAME, "dissolved_co2"],
            "biomass":           [REACTOR_STORE_NAME, "biomass"],
            "glucose":           [REACTOR_STORE_NAME, "glucose"],
            "gas_flow_rate_Lpm": [REACTOR_STORE_NAME, "gas_flow_rate_Lpm"],
        },
        "outputs": {
            # Additive transport deltas -> shared stores.
            "dissolved_o2":  [REACTOR_STORE_NAME, "dissolved_o2"],
            "dissolved_co2": [REACTOR_STORE_NAME, "dissolved_co2"],
            # Diagnostics (overwrite readouts).
            "o2_transport_delta":  [REACTOR_STORE_NAME, "o2_transport_delta"],
            "co2_transport_delta": [REACTOR_STORE_NAME, "co2_transport_delta"],
            "kla_o2":              [REACTOR_STORE_NAME, "kla_o2"],
            "kla_co2":             [REACTOR_STORE_NAME, "kla_co2"],
            "o2_saturation":       [REACTOR_STORE_NAME, "o2_saturation"],
            "co2_saturation":      [REACTOR_STORE_NAME, "co2_saturation"],
            "gas_holdup":          [REACTOR_STORE_NAME, "gas_holdup"],
        },
    }

    # ReactorCellCoupler Step. Reads population.biomass_concentration_gL +
    # reactor dissolved gases + agent fluxes; writes reactor deltas (additive
    # gases, overwrite biomass) + environment.external_concentrations.
    coupler_config = {
        "cells_per_agent":  float(cells_per_agent),
        "reactor_volume_L": float(bird_cfg.get("volume_L", 1.0)),
        "track_medium":     bool(track_medium),
    }
    coupler = _make_instance(ReactorCellCoupler, coupler_config, core)
    coupler_edge = make_edge(
        coupler, ReactorCellCoupler.topology, edge_type="step",
        config=coupler_config,
    )
    # Override the coupler's reactor output schema so biomass routes (overwrite)
    # while dissolved gases + the medium-concentration accumulators stay additive
    # (see module docstring — the bare InPlaceDict port silently drops any leaf
    # the per-leaf schema doesn't enumerate once reactor is a structured tree, so
    # the new *_mM leaves MUST be listed here or the coupler's medium deltas never
    # apply).
    coupler_edge["_outputs"]["reactor"] = {
        "biomass":       "overwrite[float]",
        "dissolved_o2":  "float",
        "dissolved_co2": "float",
        GLUCOSE_MEDIUM_LEAF: "float",
        AMMONIUM_MEDIUM_LEAF: "float",
        "acetate_mM":   "float",
        "lactate_mM":   "float",
        "formate_mM":   "float",
        "ethanol_mM":   "float",
        "pyruvate_mM":  "float",
        "succinate_mM": "float",
    }
    state[REACTOR_CELL_COUPLER_STEP_NAME] = coupler_edge

    # Run the coupler at the END of the flow (after the PopulationAggregator
    # has produced population.biomass_concentration_gL and metabolism has
    # emitted the per-agent exchange fluxes this tick).
    flow_order.append(REACTOR_CELL_COUPLER_STEP_NAME)

    return document


@composite_generator(
    name="reactor_bird_coupled",
    description=(
        "v2ecoli baseline_population coupled to the BiRD bioreactor "
        "(BiRDTransportProcess) via ReactorCellCoupler. The cell population's "
        "metabolic O2/CO2 exchange and the reactor's gas-liquid transport net "
        "additively at shared reactor.dissolved_o2/dissolved_co2 stores; the "
        "EnvironmentDriver runs in external_store mode (the coupler is the env "
        "source) and EnvironmentMirror propagates the reactor-derived boundary "
        "concentrations to every agent."
    ),
    parameters={
        "seed":            {"type": "int",    "default": 0},
        "cache_dir":       {"type": "string", "default": "out/cache"},
        # BiRDTransportProcess config (reactor_type / volume_L /
        # gas_flow_rate_Lpm / temperature_K). Time base hours.
        "bird_reactor_config": {
            "type": "object", "default": dict(DEFAULT_BIRD_REACTOR_CONFIG)},
        # Population-aggregator knobs forwarded to baseline_population.
        "cells_per_agent": {"type": "number", "default": 1.0},
        # "fixed" (default) | "representative_doubling": grow the
        # represented population 2x per generation so the coupled reactor sees an
        # ACCUMULATING biomass / O2 demand instead of the single-lineage plateau.
        "population_growth_mode": {"type": "string", "default": "fixed"},
        # Opt into the #572 substrate-exhaustion growth arrest. Set True for a
        # batch-to-exhaustion run so the cell arrests once glucose is depleted
        # instead of building biomass from phantom internal carbon. Default off
        # -> metabolism unchanged. carbon_source_ids defaults to ["GLC[p]"].
        "carbon_exhaustion_arrest": {"type": "boolean", "default": False},
        # Follow a single lineage past divisions (#588). Set True for the
        # single-lineage coupled batch runs (must match the multigen runner's
        # single_daughters); adds the in-composite LineageBookkeeper so the
        # reactor trajectory is chunk-independent. Default False = no-op.
        "single_daughters": {"type": "boolean", "default": False},
        # Medium glucose recipe seed (mmol/L) for the coupler's drawdown
        # accumulator (substrate/glucose-conc axis).
        "initial_glucose_mM": {"type": "number",
                               "default": DEFAULT_INITIAL_GLUCOSE_MM},
        # Medium ammonium recipe seed (mmol/L) -- a finite, drawn-down pool
        # rather than a static concentration.
        "initial_ammonium_mM": {"type": "number",
                                "default": DEFAULT_INITIAL_AMMONIUM_MM},
        # Per-cell biological build kwarg, forwarded through baseline_population
        # to baseline(). Before this parameter existed there was NO WAY to put a
        # different process in the metabolism slot on the coupled path at all --
        # a missing capability, not a silently-wrong build: naming it in a config
        # raised ValueError from build_generator's override validation.
        # Empty = none, so the default is byte-identical to before.
        "injected_processes": {
            "type": "map",
            "default": {},
            "description": "Process-injection spec {fork_repo, add_processes, "
                           "swap_processes, process_configs, topology, "
                           "time_step}. ⚠ fork_repo is REQUIRED even when empty (\"\" = native); baseline() indexes it directly. Omit the whole parameter for no injection. Forwarded verbatim to "
                           "baseline_population -> baseline().",
        },
    },
)
def reactor_bird_coupled(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    bird_reactor_config: dict | None = None,
    cells_per_agent: float = 1.0,
    population_growth_mode: str = "fixed",
    carbon_exhaustion_arrest: bool = False,
    carbon_source_ids: list | None = None,
    single_daughters: bool = False,
    initial_glucose_mM: float = DEFAULT_INITIAL_GLUCOSE_MM,
    initial_ammonium_mM: float = DEFAULT_INITIAL_AMMONIUM_MM,
    injected_processes: dict | None = None,
) -> dict:
    """Build the reactor_bird_coupled document.

    Extends ``baseline_population`` (cell side) with the BiRD reactor transport
    process + the reactor<->cell coupler, and switches the environment driver to
    ``external_store`` mode with the mirror active.
    """
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    bird_config = dict(DEFAULT_BIRD_REACTOR_CONFIG)
    if bird_reactor_config:
        bird_config.update(bird_reactor_config)

    # --- cell side: baseline + PopulationAggregator -----------------------
    document = baseline_population(
        core, seed=seed, cache_dir=cache_dir, cells_per_agent=cells_per_agent,
        reactor_volume_L=float(bird_config.get("volume_L", 1.0)),
        population_growth_mode=population_growth_mode,
        carbon_exhaustion_arrest=carbon_exhaustion_arrest,
        carbon_source_ids=carbon_source_ids,
        single_daughters=single_daughters,
        injected_processes=injected_processes,
    )

    # --- env hook + reactor + coupler (shared with reactor_bird_coupled_millard)
    return add_reactor_coupling(
        document, core,
        bird_config=bird_config, cells_per_agent=cells_per_agent,
        initial_glucose_mM=initial_glucose_mM,
        initial_ammonium_mM=initial_ammonium_mM,
    )
