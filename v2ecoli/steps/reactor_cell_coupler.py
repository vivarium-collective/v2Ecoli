"""ReactorCellCoupler — bridge v2ecoli's cell population to the BiRD reactor.

Build-phase scaffold for mbp-03 (req-2). PBG Step that runs each emit cycle,
implementing **Option B** of the v2ecoli <-> pbg-bioreactordesign coupling:

  * BiRD owns transport physics + reads biomass as an input.
  * v2ecoli owns biomass + the metabolic exchange demand.

This Step is the thin translator between the two halves. Each cycle it:

1. **Biomass passthrough.** Read ``population.biomass_concentration_gL`` (g/L,
   produced by :class:`PopulationAggregator`) and write it to
   ``reactor.biomass`` (g/L) for ``BiRDTransportProcess`` to consume. This is a
   passthrough/overwrite of the absolute value (the ``reactor.biomass`` port
   must be wired as an overwrite type in the composite).

2. **Dissolved-gas -> environment concentration.** Read ``reactor.dissolved_o2``
   and ``reactor.dissolved_co2`` (mg/L) and write the cell environment's
   ``external_concentrations`` (mM) for the two gases.

   Conversion (the load-bearing one):

       mM = (mg/L) / MW[g/mol]

   because  (mg/L) / (g/mol) = (1e-3 g/L) / (g/mol) = 1e-3 mol/L = 1 mM.
   e.g. 8 mg/L O2 / 31.999 ~= 0.25 mM. These are absolute concentrations
   (overwrite the environment port, not additive).

3. **Metabolic demand -> dissolved-gas delta.** Aggregate each agent's
   metabolic exchange flux into an *additive* dissolved-gas delta (mg/L) the
   reactor accumulates this interval:

       delta[mg/L] = counts * cells_per_agent / N_A[1/mol]
                     * MW[g/mol] * 1000[mg/g] / volume_L

   MW in g/mol numerically equals mg/mmol, so the same constant serves both
   the mM conversion and this mg/L delta.

   NOTE this is COUNT-based and touches no mass listener. An earlier revision
   of this docstring described a flux x biomass form with the biomass taken as
   ``cell_mass_fg * cells_per_agent * 1e-15`` — stale twice over: the code has
   not worked that way for some time, and ``cell_mass`` is TOTAL (wet) mass,
   ~3.33x dry, so using it as a gDW basis is the same error since corrected in
   PopulationAggregator. Do not reintroduce it.

Sign convention (verified against ``v2ecoli/processes/metabolism.py``
:func:`_fba_output_to_deltas`, lines 469-492): ``external_exchange_fluxes`` is
reported on a gDCW basis where the FBA exchange flux feeds
``delta_nutrients = (1/counts_to_molar) * exchange_fluxes`` — described there as
"the cumulative count added to the environment store". So a POSITIVE flux means
the molecule is *secreted* (added to the environment) and a NEGATIVE flux means
*uptake* (removed). Therefore O2 (consumed) carries a negative flux and yields a
NEGATIVE ``reactor.dissolved_o2`` delta (uptake depletes dissolved O2), while
CO2 (produced) carries a positive flux and yields a POSITIVE
``reactor.dissolved_co2`` delta. The Step passes the signed flux straight
through the conversion chain, so the sign is preserved without special-casing.

Time base: ``BiRDTransportProcess`` works in HOURS, while v2ecoli's cell sim
steps in SECONDS. The metabolic-exchange path needs NO interval scaling: it
differences ``environment.exchange``, a cumulative COUNT, and one difference
already covers exactly one WCM step. ``SECONDS_PER_HOUR`` and ``FG_PER_GRAM``
are retained as published constants but are not used by this Step; an earlier
version of this docstring described a per-hour flux conversion that the code
has never performed.

⚠ STANDING HAZARD for anything cumulative added to this Step
------------------------------------------------------------
The Step flow runs ONCE AT CYCLE START, before the processes and before the
PopulationAggregator have written anything (measured: ``calls_per_tick =
[2,1,1,...]``), so the first invocation sees an unpopulated state -- notably
``population.biomass_concentration_gL == 0.0``.

Any quantity that latches "the initial value" on its first invocation latches a
pre-population zero. Two distinct defects have come from this:

* mbp-03's O2 mass balance anchored its numerator after tick 0 while its
  denominator was anchored before it -- a 5.3% phantom deficit read as a leak;
* the per-agent first-observation rule exists precisely because tick 0's
  exchange store is not yet meaningful.

=> Defer the baseline until the store you depend on is live, and latch every
related baseline together so they refer to the same instant.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


# --- Constants --------------------------------------------------------------

# Molecular weights (g/mol == mg/mmol).
MW_O2: float = 31.999
MW_CO2: float = 44.010

# Metabolism exchange-flux map keys (periplasmic compartment). Used for the
# environment.external_concentrations write (the cell-side env port IS keyed with
# the [p] compartment suffix).
O2_ID: str = "OXYGEN-MOLECULE[p]"
CO2_ID: str = "CARBON-DIOXIDE[p]"

# v2ecoli reports environmental exchange as CUMULATIVE molecule COUNTS at
# agents.*.environment.exchange, keyed by the BARE molecule name (no compartment
# suffix). Negative == uptake (removed from the environment), positive ==
# secretion. This is the real consumption source the coupler reads (the molar
# external_exchange_fluxes path the original draft assumed does not exist in
# v2ecoli's live state — see metabolism._fba_output_to_deltas: delta_nutrients is
# "the cumulative count added to the environment store"). ⚠ CUMULATIVE is literal:
# the store is a lineage running total that does not reset at division, so this
# Step DIFFERENCES it per agent to recover this tick's exchange.
O2_EXCHANGE_KEY: str = "OXYGEN-MOLECULE"
CO2_EXCHANGE_KEY: str = "CARBON-DIOXIDE"

SECONDS_PER_HOUR: float = 3600.0
FG_PER_GRAM: float = 1.0e-15
AVOGADRO: float = 6.02214076e23  # 1/mol

DEFAULT_CELLS_PER_AGENT: float = 1.0
DEFAULT_REACTOR_VOLUME_L: float = 1.0

# Medium concentration accumulator (#225 req-3 — close the ungraded
# substrate/glucose-conc + byproducts comparison axes). Maps a reactor store
# leaf (mmol/L, ADDITIVE float) to the bare environment.exchange molecule key the
# cell reports per-step COUNTS for (negative == uptake, positive == secretion).
# Each tick the coupler integrates those counts into a mmol/L delta on the leaf,
# so the reactor accumulates a medium CONCENTRATION the Beulig batch CSVs (also
# mmol/L) can be graded against — the same count->concentration conversion the
# dissolved-gas delta uses, minus the MW factor (mmol/L instead of mg/L).
#
#   delta[mmol/L] = counts * cells_per_agent / N_A[1/mol] * 1000 / volume_L
#
# Glucose (GLC) is UPTAKE (negative counts): seeded at the medium recipe glucose
# (reactor.glucose_medium_mM initial) and DRAWN DOWN. The byproducts are SECRETED
# (positive counts) and seeded at 0 so the leaf == cumulative secreted
# concentration. acetate/pyruvate are tracked too (sim observable) even though
# the WCM does not secrete them in this aerobic state (leaf stays ~0) — that is a
# genuine sim prediction, not a gap.
GLUCOSE_MEDIUM_LEAF: str = "glucose_medium_mM"
# Environment-store id for medium glucose, in the compartment-tagged form the
# rest of this Step's env_concs writes use (EnvironmentMirror resolves it onto
# the bare `GLC` boundary key).
GLUCOSE_ID: str = "GLC[p]"
GLUCOSE_EXCHANGE_KEY: str = "GLC"
# Medium AMMONIUM — the nitrogen source, tracked on exactly the same footing as
# glucose: seeded from the medium recipe and DRAWN DOWN by uptake, rather than
# held at a static concentration the cell can never exhaust.
# ⚠ The compartment tag is [c], NOT [p] as for glucose: measured, the importable
# id in exchange_data is `AMMONIUM[c]`. EnvironmentMirror strips the tag either
# way onto the bare `AMMONIUM` boundary key, but the id is not guessable from
# the glucose case.
AMMONIUM_MEDIUM_LEAF: str = "ammonium_medium_mM"
AMMONIUM_ID: str = "AMMONIUM[c]"
AMMONIUM_EXCHANGE_KEY: str = "AMMONIUM"
# reactor leaf (mmol/L) -> bare environment.exchange byproduct key.
BYPRODUCT_LEAVES: dict[str, str] = {
    "acetate_mM":   "ACET",
    "lactate_mM":   "D-LACTATE",
    "formate_mM":   "FORMATE",
    "ethanol_mM":   "ETOH",
    "pyruvate_mM":  "PYRUVATE",
    "succinate_mM": "SUC",
}


class ReactorCellCoupler(Step):
    """Translate between v2ecoli's cell population and the BiRD reactor stores.

    See module docstring; spec in
    ``studies/mbp-03-*/study.yaml`` req-2 (reactor<->cell coupler).
    """

    name = "reactor_cell_coupler"
    # Bare type names (not {"_default": ...}) so config overrides propagate —
    # see PopulationAggregator's schema comment for the bigraph-schema rationale.
    config_schema = {
        "cells_per_agent":  "float",
        "reactor_volume_L": "float",
        "time_step":        "float",
        "track_medium":     "boolean",
    }
    topology = {
        "population":  ("population",),
        "reactor":     ("reactor",),
        "environment": ("environment",),
        "agents":      ("agents",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.cells_per_agent = float(
            cfg.get("cells_per_agent") or DEFAULT_CELLS_PER_AGENT)
        self.reactor_volume_L = float(
            cfg.get("reactor_volume_L") or DEFAULT_REACTOR_VOLUME_L)
        # Accumulate medium glucose/byproduct concentrations into the reactor
        # store so the substrate/byproducts axes grade (#225 req-3), and publish
        # glucose back to the cell's environment (see next_update).
        #
        # ⚠ The `is None` branch below reads as "default ON", and is not: the
        # config_schema declares `track_medium: boolean`, so the framework fills
        # it as False whenever a caller omits it, and the branch never fires.
        # Direct construction (`ReactorCellCoupler(config={})`) therefore gets
        # medium tracking OFF. Composites switch it on explicitly
        # (`reactor_bird_coupled` passes track_medium=True), which is why this
        # has never shown up in a run. Left as-is rather than changed here:
        # flipping it would alter behaviour for any direct caller relying on
        # today's effective default.
        track = cfg.get("track_medium")
        self.track_medium = True if track is None else bool(track)
        # How many invocations fell back to the configured `cells_per_agent`
        # because no population cell_count was readable. Expected to be exactly
        # 1 in a composite that wires the aggregator (the first pass, before
        # its write commits); a growing count means the population scale is not
        # reaching this Step at all.
        self.scale_fallbacks: int = 0
        # agents.*.environment.exchange is a LINEAGE-CUMULATIVE running total
        # (ecoli_baseline's `exchange_flux_basis` doc: "a LINEAGE-CUMULATIVE
        # molecule total that does not reset at division, so its time-average is
        # not a rate"). This Step needs THIS TICK's exchange, so it differences
        # the store per agent. Keyed by agent_id: at division each daughter
        # inherits the parent's total under a NEW id, so differencing the summed
        # total across agents would DOUBLE-COUNT it: with the store's
        # negative-uptake convention, summing two daughters that each carry the
        # parent's total doubles it in the NEGATIVE direction, i.e. a spurious
        # extra UPTAKE of one full lineage total on the division tick -- not a
        # sign flip, and not secretion. (An earlier version of this comment said
        # "reads as secretion"; that was backwards, and the division test's
        # assertion direction was derived from it.)
        self._prev_exchange: dict[str, dict[str, float]] = {}
        # A leaf's first observation for an agent yields 0.0 rather than a first
        # difference -- the same choice the gdcw deriver makes and for the same
        # reason: the store carries across division, so differencing against an
        # assumed zero would dump a whole generation's accumulation into one
        # tick. Costs one tick of exchange per new agent; a spike would be worse.
        self.first_observation_ticks: int = 0

    def inputs(self) -> dict[str, Any]:
        return {
            "population": InPlaceDict(),
            "reactor":    InPlaceDict(),
            "agents":     InPlaceDict(),
        }

    def outputs(self) -> dict[str, Any]:
        return {
            "reactor":     InPlaceDict(),
            "environment": InPlaceDict(),
        }

    # --- main update -------------------------------------------------------

    def next_update(self, timestep, states):
        update: dict[str, Any] = {}

        reactor_out: dict[str, float] = {}
        env_concs: dict[str, float] = {}

        reactor = states.get("reactor") or {}
        # Reactor volume: prefer the live store, fall back to config default.
        volume_L = _as_float(reactor.get("volume_L"))
        if not volume_L or volume_L <= 0.0:
            volume_L = self.reactor_volume_L

        # 1. Biomass passthrough (g/L) — overwrite the reactor biomass input.
        population = states.get("population") or {}
        biomass_gL = population.get("biomass_concentration_gL")
        if biomass_gL is not None:
            reactor_out["biomass"] = _as_float(biomass_gL)

        # 2. Dissolved gas (mg/L) -> environment concentration (mM).
        do2 = reactor.get("dissolved_o2")
        dco2 = reactor.get("dissolved_co2")
        if do2 is not None:
            env_concs[O2_ID] = _as_float(do2) / MW_O2
        if dco2 is not None:
            env_concs[CO2_ID] = _as_float(dco2) / MW_CO2
        # Medium glucose (mM) -> the cell's environment, on the same footing as
        # the dissolved gases. Before this, `glucose_medium_mM` was written by
        # this Step and read by NOTHING: draining the pool changed nothing the
        # cells experienced, so glucose-limited growth was unreachable however
        # the reactor was configured. The leaf is already seeded at the medium
        # recipe and drawn down below, and is already in mM, so no MW
        # conversion applies. Clamp at zero: an exhausted pool is zero
        # available, never negative.
        if self.track_medium:
            glc_medium = reactor.get(GLUCOSE_MEDIUM_LEAF)
            if glc_medium is not None:
                env_concs[GLUCOSE_ID] = max(_as_float(glc_medium), 0.0)
            nh4_medium = reactor.get(AMMONIUM_MEDIUM_LEAF)
            if nh4_medium is not None:
                env_concs[AMMONIUM_ID] = max(_as_float(nh4_medium), 0.0)

        # 3. Metabolic exchange demand -> additive dissolved-gas delta (mg/L).
        #
        # v2ecoli emits environmental exchange as CUMULATIVE molecule COUNTS at
        # agents.*.environment.exchange (negative == uptake), DIFFERENCED per
        # agent above to recover this tick's exchange. Convert counts/step
        # for one agent, scaled by cells_per_agent (representative-sampling
        # population scale), to a mg/L delta on the shared dissolved store:
        #
        #   delta[mg/L] = counts * cells_per_agent / N_A[1/mol]
        #                 * MW[g/mol] * 1000[mg/g] / volume_L
        #
        # The DIFFERENCED counts cover one WCM step, so NO interval
        # (timestep/3600) scaling is applied here. (Before 2026-08-29 this Step
        # consumed the store's value directly, on the reading that it was already
        # per-step. It is not -- it is a running total -- so the reactor was
        # drained by ~N/2 for an N-tick run: measured 22.2 mM of a 22.2 mM pool
        # gone while the population formed 2.3 mg of biomass, an apparent yield
        # of 0.0006 g/g against a ~0.54 ceiling.) Sign flows straight through: negative O2
        # counts (uptake) -> negative dissolved_o2 delta. This is the per-step
        # mass exchanged between the cell population and the reactor liquid.
        agents = states.get("agents") or {}
        o2_counts = 0.0   # molecules/step, signed (negative == uptake)
        co2_counts = 0.0  # molecules/step, signed (positive == secretion)
        # Medium glucose + byproduct counts (signed; negative == uptake).
        glc_counts = 0.0
        nh4_counts = 0.0
        byproduct_counts = {leaf: 0.0 for leaf in BYPRODUCT_LEAVES}
        seen_agents: set[str] = set()
        for _agent_id, agent_state in agents.items():
            exch = _extract_environment_exchange(agent_state)
            seen_agents.add(_agent_id)
            # F4: decide NEW-ness from whether this agent has been SEEN, before
            # setdefault creates its entry. `not prev` was equivalent only while
            # every tick wrote a baseline for every key; with the absent-key
            # guard below, an agent whose snapshot carries none of our keys
            # leaves `prev` empty forever and would count as new EVERY tick,
            # conflating "many divisions" with "one agent with no exchange data".
            new_agent = _agent_id not in self._prev_exchange
            prev = self._prev_exchange.setdefault(_agent_id, {})

            def _tick_delta(key: str) -> float:
                """This tick's exchange for one molecule: the store is a running
                total, so difference it against this agent's previous reading.

                Two cases yield 0.0 rather than a difference:

                * FIRST OBSERVATION of a (agent, molecule) pair. The store
                  carries across division, so differencing against an assumed
                  zero would dump a whole generation's accumulation into one
                  tick. The gdcw deriver makes the same choice for the same
                  reason (steps/derivers/exchange_flux_listener.py).
                * KEY MISSING OR ``None`` in this tick's snapshot. The emit
                  cadence can snapshot mid-init, so ``exch`` may be missing a
                  key it had last tick, or carry it as ``None`` (see
                  _extract_environment_exchange). Treating either as 0.0 would
                  overwrite the baseline with 0.0 and then emit the whole
                  running total as a spike on the next tick -- the very spike
                  the first-observation rule exists to prevent. Leave the
                  baseline untouched and skip the tick.
                  ⚠ ``None`` must be tested SEPARATELY from absence: a bare
                  ``key not in exch`` passes a present ``None`` through to
                  ``_as_float``, which maps it to 0.0 -- measured at +0.332 mM
                  of spurious secretion followed by -0.498 mM of spurious
                  uptake, against a -0.166 mM normal tick.
                """
                value = exch.get(key)
                if value is None:
                    return 0.0
                total = _as_float(value)
                previous = prev.get(key)
                prev[key] = total
                return 0.0 if previous is None else total - previous

            o2_counts += _tick_delta(O2_EXCHANGE_KEY)
            co2_counts += _tick_delta(CO2_EXCHANGE_KEY)
            if self.track_medium:
                glc_counts += _tick_delta(GLUCOSE_EXCHANGE_KEY)
                nh4_counts += _tick_delta(AMMONIUM_EXCHANGE_KEY)
                for leaf, key in BYPRODUCT_LEAVES.items():
                    byproduct_counts[leaf] += _tick_delta(key)
            if new_agent:
                self.first_observation_ticks += 1
        # Drop agents that are gone (division retires the parent id) so the
        # bookkeeping cannot grow without bound over a long lineage.
        for stale in set(self._prev_exchange) - seen_agents:
            del self._prev_exchange[stale]

        if agents:
            # Population scale. `PopulationAggregator` multiplies biomass and
            # cell_count by 2**doublings under `representative_doubling`, so the
            # followed lineage stands in for a growing population; this Step
            # scaled exchange by `cells_per_agent` ALONE. The reactor therefore
            # saw an accumulating biomass consuming a constant amount of
            # substrate -- by generation 4, 8x the biomass eating 1x the
            # glucose and oxygen. That is a mass-conservation violation, not an
            # accuracy problem, and it makes any carbon/oxygen closure
            # criterion fail by construction.
            #
            # `population.cell_count` already carries cells_per_agent *
            # growth_factor, so deriving the per-agent scale from it keeps ONE
            # authoritative number rather than a second copy that can drift.
            # The aggregator precedes this Step in flow_order, so from the
            # second invocation onward the value is current for this tick.
            #
            # ⚠ NOT on the first. Measured under `Composite.run()`: the
            # aggregator's write is not committed to the store within the first
            # pass, so the coupler's FIRST invocation reads 0.0 and takes the
            # fallback below -- exactly once per composite. (An earlier revision
            # of this comment claimed tick-0 propagation was verified. It is
            # path-dependent, and on the `run()` path used by the harness and
            # the figures script it is false. Corrected after measurement.)
            #
            # Consequence: in `fixed` mode the fallback is harmless, because
            # cell_count = n_agents * cells_per_agent * 1.0 makes the two
            # expressions identical. Under `representative_doubling` at
            # generation g, that one tick of exchange is under-scaled by
            # 2**(g-1). Small, but it is the very error this Step is being
            # fixed for, so the fallback now RECORDS that it fired rather than
            # switching scales silently.
            n_agents = len(agents)
            cell_count = _as_float(population.get("cell_count"))
            if cell_count > 0.0 and n_agents:
                cells_per_agent_effective = cell_count / n_agents
            else:
                cells_per_agent_effective = self.cells_per_agent
                self.scale_fallbacks += 1
            counts_to_mgL = cells_per_agent_effective / AVOGADRO * 1000.0 / volume_L
            o2_delta = o2_counts * counts_to_mgL * MW_O2
            co2_delta = co2_counts * counts_to_mgL * MW_CO2
            # Safety clamp: a single step's uptake must not drive the dissolved
            # store negative (transport replenishes next step). The coupler runs
            # after transport in the flow, so reactor.dissolved_o2 already holds
            # this step's transport delta; cap the consumption at what's present.
            do2_now = _as_float(reactor.get("dissolved_o2"))
            if o2_delta < 0.0 and (do2_now + o2_delta) < 0.0:
                o2_delta = -do2_now
            dco2_now = _as_float(reactor.get("dissolved_co2"))
            if co2_delta < 0.0 and (dco2_now + co2_delta) < 0.0:
                co2_delta = -dco2_now
            reactor_out["dissolved_o2"] = o2_delta
            reactor_out["dissolved_co2"] = co2_delta

            # Republish the dissolved gases POST-consumption.
            #
            # Section 2 above published the cell's view from the PRE-consumption
            # store, i.e. after transport but before this tick's uptake. For an
            # unreplenished pool (glucose, ammonium) that is harmless: the pool
            # reaches zero and stays there, so the cell does see exhaustion.
            # For a REPLENISHED species it is not, because the two effects lock
            # into a sawtooth:
            #
            #   transport adds one tick of OTR   ->  dissolved_o2 = OTR*dt
            #   section 2 publishes that peak    ->  cell reads OTR*dt / MW
            #   the clamp below caps uptake      ->  dissolved_o2 = 0
            #   repeat
            #
            # so the cell reads a CONSTANT equal to one tick of oxygen transfer
            # and never observes the zero. Measured on a 6-generation coupled
            # run: the cell's O2 sat at exactly 0.0011881823 mM for 7,107 of
            # 18,566 ticks while the reactor read 0.0000, and
            #   kLa * C* / 3600 = 20.7868 * 6.5847 / 3600 = 0.03802079 mg/L
            #   observed * MW_O2                          = 0.03802183 mg/L
            # agree to 27 ppm. The apparent concentration therefore scales with
            # the timestep, which made oxygen availability a property of the
            # integration step rather than of the process.
            #
            # Publishing post-consumption reports the concentration the cell
            # leaves behind rather than the one transport handed it. Neither
            # endpoint is the tick average -- the physically exact choice would
            # integrate over the tick -- but the pre-consumption value cannot
            # express scarcity at all under a replenished-then-clamped species,
            # and this one can.
            #
            # ⛔ The clamp above is CORRECT and must stay: it prevents a negative
            # store. The defect was never the clamp, only what was published
            # before it ran.
            # ⛔ O2 ONLY. CO2 is deliberately NOT corrected here, and the
            # symmetry is a trap: co2_counts is "positive == secretion"
            # (:317), so co2_delta > 0 and the same expression publishes a
            # HIGHER CO2 -- the cell reading carbon dioxide it produced within
            # the same tick, an instantaneous self-feedback. That is the
            # opposite direction to the O2 correction, which REMOVES a
            # fictitious supply.
            #
            # It is also not small. Measured by the same direct A/B:
            #   dissolved_co2 0.5 mg/L, 1e7 counts secreted: 0.01136 -> 0.02797 mM (2.5x)
            #                           5e7 counts:          0.01136 -> 0.09439 mM (8.3x)
            #   dissolved_co2 0.05 mg/L, 1e7 counts:         0.00114 -> 0.01774 mM (15.6x)
            #
            # And no CO2 defect exists to fix. The sawtooth needs a species that
            # is REPLENISHED and then CLAMPED to zero; CO2 is secreted, and its
            # clamp (:432) tests `co2_delta < 0.0`, so it fires only on uptake
            # and effectively never. Same reasoning excludes glucose and
            # ammonium: an unreplenished pool reaches zero and stays there, so
            # the cell does see exhaustion (confirmed on a 6-generation run --
            # cell GLC tracked the reactor to 0, max deviation 0.177 mM).
            #
            # ⚠ AND THAT EXEMPTION IS CONDITIONAL, NOT CATEGORICAL: it holds
            # only while CO2 is NET-SECRETED, which is what the clamp's
            # `< 0.0` gate encodes. If a condition ever nets CO2 UPTAKE the
            # clamp fires, the sawtooth becomes reachable, and this correction
            # is absent for it. That is not hypothetical -- an anoxic regime
            # becomes reachable for the first time BECAUSE of this fix (O2 can
            # now leave allowed_exchange_uptake), and an anoxic carbon balance
            # is exactly where the aerobic assumption is least safe.
            #
            # ⚠ THAT EXEMPTION IS CONDITIONAL, NOT CATEGORICAL. It holds only
            # while CO2 is NET-SECRETED, which is exactly what the clamp's
            # `< 0.0` gate encodes. Should a condition ever net CO2 UPTAKE the
            # clamp fires, the sawtooth becomes reachable, and this correction
            # is absent for it. Not hypothetical: an anoxic regime becomes
            # reachable for the first time BECAUSE of this fix -- O2 can now
            # leave allowed_exchange_uptake -- and an anoxic carbon balance is
            # where the aerobic assumption is least safe.
            #
            # ⇒ Correct the one species where the defect is demonstrated. This
            # Step is on the path of every coupled study; a large, unmeasured,
            # opposite-direction change to a second species is regression
            # surface bought for nothing.
            if do2 is not None:
                env_concs[O2_ID] = max(do2_now + o2_delta, 0.0) / MW_O2

            # Medium concentration accumulator (mmol/L) — same count->conc
            # conversion as the dissolved-gas delta minus the MW factor
            # (counts_to_mgL is already counts->mmol/L; *MW gives mg/L).
            if self.track_medium:
                counts_to_mM = counts_to_mgL  # = cells/N_A * 1000 / volume_L
                # Glucose: uptake (negative) -> draw down the seeded medium pool.
                # Clamp the draw at what remains; unlike the dissolved gases
                # (which are replenished by transport) an exhausted medium pool
                # has nothing to restore it, so an unclamped delta drives the
                # concentration negative and the cell would be told it has less
                # than nothing.
                glc_delta = glc_counts * counts_to_mM
                glc_now = _as_float(reactor.get(GLUCOSE_MEDIUM_LEAF))
                if glc_delta < 0.0 and (glc_now + glc_delta) < 0.0:
                    glc_delta = -glc_now
                reactor_out[GLUCOSE_MEDIUM_LEAF] = glc_delta
                # Ammonium: same shape and the same clamp reasoning as glucose —
                # an exhausted medium pool has nothing to restore it, so an
                # unclamped draw would drive the concentration negative and the
                # cell would be told it has less than nothing.
                nh4_delta = nh4_counts * counts_to_mM
                nh4_now = _as_float(reactor.get(AMMONIUM_MEDIUM_LEAF))
                if nh4_delta < 0.0 and (nh4_now + nh4_delta) < 0.0:
                    nh4_delta = -nh4_now
                reactor_out[AMMONIUM_MEDIUM_LEAF] = nh4_delta
                # Byproducts: secretion (positive) -> accumulate from 0.
                for leaf in BYPRODUCT_LEAVES:
                    reactor_out[leaf] = byproduct_counts[leaf] * counts_to_mM

        if reactor_out:
            update["reactor"] = reactor_out
        if env_concs:
            update["environment"] = {"external_concentrations": env_concs}
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)


# --- helpers ----------------------------------------------------------------

def _as_float(value: Any) -> float:
    """Coerce a possibly pint-Quantity scalar to a bare float."""
    if value is None:
        return 0.0
    if hasattr(value, "magnitude"):
        return float(value.magnitude)
    return float(value)


def _extract_environment_exchange(agent_state: dict | Any) -> dict[str, Any]:
    """Return ``environment.exchange`` (CUMULATIVE molecule COUNTS), or {} if absent.

    A dict keyed by the bare molecule name (e.g. ``OXYGEN-MOLECULE``), value =
    the signed molecule count added to the environment **since the start of the
    lineage** (negative == uptake).

    ⚠ This is a RUNNING TOTAL, not a per-step delta, and it does NOT reset at
    division — a daughter inherits the parent's total. Callers must difference
    it per agent to recover one tick's exchange; see ``ReactorCellCoupler`` and
    ``ecoli_baseline``'s ``exchange_flux_basis`` documentation. Reading it as a
    per-step value was the #632 defect: it drained the reactor by elapsed time
    rather than by cell demand, ~N/2 too much over N ticks.

    Defensive against missing intermediate keys (emit cadence can snapshot
    mid-init) — returns {} rather than raising.
    """
    try:
        env = agent_state.get("environment", {}) if hasattr(agent_state, "get") else {}
        exch = env.get("exchange")
        if isinstance(exch, dict):
            return exch
        return {}
    except (AttributeError, KeyError, TypeError):
        return {}
