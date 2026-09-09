"""
Cell division steps for v2ecoli.

MarkDPeriod: Sets division flag after D period has elapsed.
Division: Detects division condition, splits state, produces daughter
    cells via process-bigraph's _add/_remove structural updates.
"""

import numpy as np
import binascii

from bigraph_schema.contract import ProcessContract
from bigraph_schema.schema import Overwrite, Float

from v2ecoli.steps.base import V2Step
from v2ecoli.types.stores import InPlaceDict
from v2ecoli.library.schema import attrs
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.types.quantity import ureg as units


def daughter_phylogeny_id(mother_id):
    return [str(mother_id) + '0', str(mother_id) + '1']


class MarkDPeriod(V2Step):
    """Set division flag after D period has elapsed."""

    name = "mark_d_period"
    config_schema = {}

    contract = ProcessContract(
        summary=(
            "Raise the division flag once the D period has elapsed after "
            "chromosome replication completes: when global_time reaches the "
            "earliest untriggered chromosome division_time, mark that "
            "chromosome triggered and set divide=True."
        ),
        math=[
            "divide := True  and  has_triggered_division[k] := True   when  t_global ≥ min(division_time[¬has_triggered])",
        ],
        symbols={
            "t_global": "global simulation time (s)",
            "division_time": "per-chromosome scheduled division time = replication completion + D period (s)",
        },
        inputs={
            "full_chromosome": "Reads division_time and has_triggered_division; selects the earliest untriggered division_time and marks it triggered.",
            "global_time": "Compared against the earliest untriggered division_time to decide whether to fire.",
            "divide": "Downstream division flag, set True when the D-period deadline is reached.",
        },
        outputs={
            "full_chromosome": "Writes the updated has_triggered_division array (the fired chromosome marked True).",
            "divide": "Writes True when the earliest untriggered chromosome's division_time has been reached.",
            "global_time": "Passed through unchanged (mirrored port).",
        },
        assumptions=[
            "division_time is scheduled at replication completion + D_period.",
            "Only the single earliest untriggered chromosome is marked per step.",
            "Requires at least two chromosome rows before it can fire.",
        ],
    )

    def initialize(self, config):
        self.parameters = config or {}

    def inputs(self):
        return {
            "full_chromosome": InPlaceDict(),
            "global_time": Float(_default=0.0),
            "divide": Overwrite(),
        }

    def outputs(self):
        return self.inputs()

    def next_update(self, timestep, states):
        full_chrom = states.get("full_chromosome")
        if full_chrom is None or not hasattr(full_chrom, 'dtype'):
            return {}

        division_time, has_triggered_division = attrs(
            full_chrom, ["division_time", "has_triggered_division"])

        if len(division_time) < 2:
            return {}

        untriggered = division_time[~has_triggered_division]
        if len(untriggered) == 0:
            return {}
        divide_at_time = untriggered.min()
        if states.get("global_time", 0) >= divide_at_time:
            divide_at_time_index = np.where(division_time == divide_at_time)[0][0]
            has_triggered_division = has_triggered_division.copy()
            has_triggered_division[divide_at_time_index] = True
            return {
                "full_chromosome": {
                    "set": {"has_triggered_division": has_triggered_division}
                },
                "divide": True,
            }
        return {}

    def update(self, state, interval=None):
        return self.next_update(1.0, state)


class Division(V2Step):
    """Detect division and produce daughter cells via _add/_remove.

    When the division condition is met (dry mass >= threshold with 2+
    chromosomes), this step:
    1. Divides the mother cell's state (bulk binomial, unique by domain)
    2. Builds complete daughter cell states with fresh process instances
    3. Returns structural update to remove mother and add daughters
    """

    name = "division"
    config_schema = {}

    contract = ProcessContract(
        summary=(
            "Detect the division condition and split the mother cell into two "
            "daughters via process-bigraph _add/_remove structural updates: bulk "
            "molecules split binomially, unique molecules partitioned by "
            "chromosome domain, daughter composites rebuilt from baseline()."
        ),
        math=[
            "d_period (default): divide  iff  divide_flag ∧ n_chrom ≥ 2   (dry-mass threshold ignored)",
            "legacy: divide  iff  m_dry ≥ threshold ∧ n_chrom ≥ 2",
            "threshold = m_dry,0 + Δm_dry · μ_div,   μ_div ~ Normal(1.0, 0.1)",
            "on trigger: bulk ~ Binomial(n, ½) split;  unique partitioned by chromosome domain → daughters (d0,d1) via _add/_remove",
        ],
        symbols={
            "n_chrom": "number of full chromosomes (Σ _entryState); division requires ≥ 2",
            "m_dry": "cell dry mass, from listeners.mass.dry_mass (fg)",
            "threshold": "dry-mass division threshold (fg); consulted only on the legacy path",
            "Δm_dry": "media-specific dry-mass increment added per generation (fg)",
            "μ_div": "division mass multiplier ~ Normal(1.0, 0.1), CRC32-seeded",
            "divide_flag": "D-period flag raised by MarkDPeriod at the chromosome's division_time",
            "n": "molecule copy count of a bulk species, split ~ Binomial(n, ½) between daughters",
        },
        inputs={
            "listeners": "Reads mass.dry_mass to evaluate the division trigger (legacy mass threshold) and to seed daughter mass listeners.",
            "unique": "Reads full_chromosome _entryState for the n_chrom ≥ 2 gate; the whole unique tree is partitioned by chromosome domain into the two daughters.",
            "divide": "Reads the D-period flag; under d_period (default) division fires when this is set and n_chrom ≥ 2.",
            "division_threshold": "Reads the dry-mass threshold on the legacy path; initialized on first step from media-specific increment × μ_div.",
            "bulk": "Reads the mother's bulk state, split binomially into the two daughters.",
            "environment": "Reads the mother's environment state, carried into both daughter cell states.",
            "boundary": "Reads the mother's boundary state, carried into both daughter cell states.",
            "media_id": "Selects the media-specific dry-mass increment for the legacy threshold initialization.",
            "global_time": "Read as the division timestamp stamped onto both daughters.",
        },
        outputs={
            "agents": "Writes a structural update: _remove the mother agent and _add two rebuilt daughter cell composites (fresh process instances + divided biological state).",
            "division_threshold": "Writes the computed dry-mass division threshold on the legacy (mass_distribution) path.",
        },
        assumptions=[
            "Under d_period=True (vEcoli default) division fires D-period after replication completes (via the divide flag), and the dry-mass threshold is ignored; otherwise the legacy mass-distribution threshold is used.",
            "Division requires at least two full chromosomes.",
            "Bulk molecules split binomially (p=0.5); unique molecules are partitioned by chromosome domain.",
            "The division mass multiplier μ_div is drawn once from Normal(1.0, 0.1) (CRC32-seeded) for the mass_distribution threshold.",
            "Daughters are rebuilt via baseline() with the divided biological state overlaid; the parquet emitter is re-pointed per daughter to avoid clobbering the parent's history.",
        ],
        references=[
            "vEcoli ecoli/processes/cell_division.py (d_period division semantics)",
        ],
    )

    def initialize(self, config):
        self.parameters = config or {}
        self.agent_id = self.parameters.get('agent_id', '0')
        self.dry_mass_inc_dict = self.parameters.get('dry_mass_inc_dict', {})
        self.division_detected = False

        # Configs for rebuilding daughter cell states
        self._configs = self.parameters.get('configs', {})
        self._unique_names = self.parameters.get('unique_names', [])
        self._seed = self.parameters.get('seed', 0)
        self._cache_dir = self.parameters.get('cache_dir', 'out/cache')
        # Injected/swapped-process spec (fork_repo/fork_sim_data/swap_processes/…)
        # threaded from baseline() via _helpers' division config. Passed straight
        # back into each daughter's baseline() rebuild so an injected process
        # (e.g. metabolism-redux) survives division. None for the normal baseline
        # -> daughters rebuild the plain FBA baseline exactly as before.
        self._injected_processes = self.parameters.get('injected_processes')
        # Opt-in FEATURE list (flagella-cascade investigation, 2026-08-06),
        # threaded the same way as injected_processes above -- see the note
        # in ecoli_baseline.py's baseline() (loader._features) and
        # _helpers.py's 'division' branch. None/empty -> daughters rebuild
        # with baseline()'s own defaults exactly as before this fix.
        self._features = self.parameters.get('features')
        # A config_overrides / knockouts perturbation, threaded for the same
        # reason: passed back into each daughter's baseline() rebuild so a
        # variant survives division instead of reverting to the cached configs
        # from generation 2 on (#505). None for the plain baseline.
        self._config_overrides = self.parameters.get('config_overrides')
        # Declared exchange-flux measurements, threaded for the same reason as
        # the injected processes above: baseline() takes them via a module-level
        # override it clears in its own finally, so a daughter rebuilt without
        # them declares no leaves and reports 0.0 for the rest of the lineage.
        self._exchange_fluxes = self.parameters.get('exchange_fluxes')
        self._exchange_flux_basis = self.parameters.get('exchange_flux_basis')
        # vEcoli's default (`d_period=True`): division fires D_period after
        # chromosome replication completes (via the flag MarkDPeriod raises at
        # the chromosome's division_time), and the dry-mass threshold is
        # IGNORED — see vEcoli ecoli/processes/cell_division.py. When False, the
        # legacy mass-distribution threshold is used instead.
        self.d_period = bool(self.parameters.get('d_period', True))

        # Division mass multiplier
        seed = self._seed
        self.division_mass_multiplier = 1
        if self.parameters.get('division_threshold') == 'mass_distribution':
            div_seed = binascii.crc32(b'CellDivision', seed) & 0xFFFFFFFF
            div_rng = np.random.RandomState(seed=div_seed)
            self.division_mass_multiplier = div_rng.normal(loc=1.0, scale=0.1)

    def inputs(self):
        return {
            "bulk": InPlaceDict(),
            "unique": InPlaceDict(),
            "listeners": InPlaceDict(),
            "environment": InPlaceDict(),
            "boundary": InPlaceDict(),
            "global_time": Float(_default=0.0),
            "division_threshold": Overwrite(),
            "media_id": InPlaceDict(),
            # D-period flag MarkDPeriod raises at the chromosome's division_time
            # (consulted only when self.d_period is True).
            "divide": Overwrite(),
            # flagella_nfsim_complexation.py's own state ports (2026-08-19).
            # Absent from most composites -- defaults to {} harmlessly when
            # the flagella_nfsim_complexation feature isn't wired in. Declared
            # here so divide_cell() can actually see and divide them instead
            # of daughters silently getting baseline()'s empty-dict default
            # (see divide_internal_observables/divide_scaffold_species in
            # v2ecoli/library/division.py for the full rationale).
            "nfsim_scaffold_species": InPlaceDict(),
            "nfsim_internal_observables": InPlaceDict(),
        }

    def outputs(self):
        ports = self.inputs()
        ports['agents'] = {'_type': 'map', '_value': 'node'}
        return ports

    def next_update(self, timestep, states):
        # --- Threshold initialization ---
        # Under d_period (vEcoli default) the dry-mass threshold is never
        # consulted, so skip its one-shot initialization entirely.
        if not self.d_period and states.get("division_threshold") == "mass_distribution":
            media_id = states.get("media_id", "minimal")
            dry_mass_inc = self.dry_mass_inc_dict.get(media_id)
            if dry_mass_inc is not None:
                dry_mass = 0.0
                listeners = states.get('listeners')
                if isinstance(listeners, dict):
                    mass = listeners.get('mass', {})
                    if isinstance(mass, dict):
                        dry_mass = mass.get('dry_mass', 0.0)
                # Under units-on-ports, dry_mass is a pint Quantity[fg]; coerce
                # to a plain fg float so the arithmetic below doesn't raise a
                # DimensionalityError (which process-bigraph silently swallows,
                # leaving division_threshold stuck on "mass_distribution").
                dry_mass = fg_magnitude(dry_mass)
                return {
                    "division_threshold": (
                        dry_mass
                        + dry_mass_inc.to(units.fg).magnitude
                        * self.division_mass_multiplier
                    )
                }
            return {}

        # --- Division check ---
        dry_mass = 0.0
        listeners = states.get('listeners')
        if isinstance(listeners, dict):
            mass = listeners.get('mass', {})
            if isinstance(mass, dict):
                dry_mass = mass.get('dry_mass', 0.0)
        # Coerce both sides to plain fg floats — dry_mass may be a Quantity[fg]
        # (units-on-ports); a Quantity vs float comparison raises (and is
        # swallowed), which would silently prevent division.
        dry_mass = fg_magnitude(dry_mass)

        full_chrom = states.get('unique', {}).get('full_chromosome')
        n_chromosomes = 0
        if full_chrom is not None and hasattr(full_chrom, 'dtype'):
            if '_entryState' in full_chrom.dtype.names:
                n_chromosomes = full_chrom['_entryState'].sum()

        # --- Division trigger ---
        if self.d_period:
            # vEcoli default: divide once the D-period has elapsed after
            # replication finished — MarkDPeriod raises `divide` at the
            # chromosome's division_time (= completion + D_period). The dry-mass
            # threshold is IGNORED. Without this, v2's mass threshold fired
            # ~D_period too early, dividing slow-growth cells (acetate,
            # succinate, anaerobic; cycle > the per-gen window) well below their
            # proper size, which also cut them off before the 2nd replication
            # round and suppressed the growth ramp.
            threshold = float('nan')
            if not bool(states.get("divide", False)) or n_chromosomes < 2:
                return {}
        else:
            threshold = fg_magnitude(states.get("division_threshold", float('inf')))
            if dry_mass < threshold or n_chromosomes < 2:
                return {}

        if self.division_detected:
            return {}

        # --- DIVISION ---
        self.division_detected = True
        division_time = states.get("global_time", 0)
        print(f'DIVISION at t={division_time:.0f}s '
              f'(dry_mass={dry_mass:.1f} fg, '
              f'threshold={threshold:.1f} fg, '
              f'chromosomes={n_chromosomes})')

        try:
            cell_data = {
                'bulk': states['bulk'],
                'unique': states['unique'],
                'environment': states.get('environment', {}),
                'boundary': states.get('boundary', {}),
            }
            # flagella_nfsim_complexation.py's ports, 2026-08-19 -- only
            # include when actually present (most composites won't have
            # them) so divide_cell()'s presence check works correctly.
            if states.get('nfsim_scaffold_species') is not None:
                cell_data['nfsim_scaffold_species'] = states['nfsim_scaffold_species']
            if states.get('nfsim_internal_observables') is not None:
                cell_data['nfsim_internal_observables'] = states['nfsim_internal_observables']

            from v2ecoli.library.division import divide_cell
            d1_data, d2_data = divide_cell(cell_data)

            d1_data['global_time'] = division_time
            d2_data['global_time'] = division_time

            d1_id, d2_id = daughter_phylogeny_id(self.agent_id)

            # Build daughter docs via the new composite generator API.
            # baseline() loads wiring from the cache; we then overlay the
            # daughter's divided biological state on top.
            from v2ecoli.composites.ecoli_baseline import baseline, seed_mass_listener
            from v2ecoli.composites._helpers import (
                _PARQUET_EMITTER_OVERRIDE, set_parquet_emitter_override)
            d1_seed = (self._seed + 1) % (2**31)
            d2_seed = (self._seed + 2) % (2**31)

            def _build_daughter_doc(d_data, seed, daughter_id):
                # When a parquet-emitter override is active, the daughter's
                # emitter would otherwise inherit the PARENT's static partition
                # metadata (generation, agent_id) from the global override, and
                # its _write_configuration (run in __init__) would DELETE the
                # parent's fully-written history partition at division — the
                # parent emits its whole life to generation=N/agent_id=<id>, the
                # daughter spawns on the SAME path and wipes it, leaving only
                # the daughter's birth rows (the early-cycle-slice symptom).
                # Re-point the override to the daughter's OWN slot
                # (generation=N+1, agent_id=<parent><suffix>) for the duration
                # of the build so the daughter partitions to
                # generation=N+1/agent_id=P0|P1 and never touches the parent's
                # data. The next runner-driven generation re-wipes and rewrites
                # that daughter slot cleanly.
                import copy as _copy
                _saved = _PARQUET_EMITTER_OVERRIDE
                if _saved is not None:
                    _ovr = _copy.deepcopy(_saved)
                    _meta = _ovr.setdefault('metadata', {})
                    # Derive the daughter slot from the PARENT's override
                    # metadata (the runner's per-generation identity), NOT from
                    # self.agent_id — inside a composite the cell is always the
                    # 'agents/0' key, so self.agent_id is "0" for every runner
                    # generation. Using it would give generation=2/agent_id=00
                    # for ALL gens, colliding with (and wiping) runner gen 2.
                    _parent_aid = str(_meta.get('agent_id', self.agent_id))
                    _parent_gen = _meta.get('generation')
                    if _parent_gen is None:
                        _parent_gen = len(_parent_aid)
                    _suffix = daughter_id[len(str(self.agent_id)):] or '0'
                    _meta['agent_id'] = _parent_aid + _suffix
                    _meta['generation'] = int(_parent_gen) + 1
                    set_parquet_emitter_override(_ovr)
                # Emitter sink for the daughter's INTERNAL 'emitter' step.
                # baseline() defaults to 'parquet', whose declared-default sink
                # has a FIXED partition (generation=1/agent_id=1) with no
                # per-daughter identity. When NO parquet override is active
                # (e.g. runs driven by run_multigen_xarray, which captures the
                # real output via an EXTERNAL XArrayEmitter), both daughters
                # would otherwise spawn ParquetEmitters pinned to that one
                # shared partition and stomp each other's temp files — the next
                # batch flush (~step 400 ≈ 780s of gen 2) raises FileNotFoundError
                # on the temp→final rename, which the runner swallows as a
                # "graceful stop" and truncates the lineage mid-generation.
                # The internal sink is redundant in that path, so build the
                # daughter with the no-op 'null' emitter (global_time-only
                # RAMEmitter) — no files, no collision, gen 2+ runs a full
                # cell cycle. When an override IS active the repointed,
                # per-daughter-partitioned parquet sink above is used instead.
                _daughter_emitter = "parquet" if _saved is not None else "null"
                try:
                    doc = baseline(
                        core=self.core, seed=seed, cache_dir=self._cache_dir,
                        emitter=_daughter_emitter,
                        injected_processes=self._injected_processes,
                        features=self._features,
                        config_overrides=self._config_overrides,
                        exchange_fluxes=self._exchange_fluxes,
                        exchange_flux_basis=self._exchange_flux_basis)
                finally:
                    if _saved is not None:
                        set_parquet_emitter_override(_saved)
                agent = doc['state']['agents']['0']
                for key in ('bulk', 'unique', 'environment', 'boundary',
                            'nfsim_scaffold_species', 'nfsim_internal_observables'):
                    if key in d_data:
                        agent[key] = d_data[key]
                agent['listeners']['mass'] = {'dry_mass': 0.0, 'cell_mass': 0.0}
                seed_mass_listener(agent, self.core)
                # Advance the phylogeny. baseline() always wires the daughter's
                # OWN internal division Step with the default agent_id='0' (see
                # _helpers._get_step_config). Re-point it at this daughter's
                # actual lineage id so that when *this* daughter later divides it
                # removes the correct agent and emits descendant ids
                # (e.g. parent '00' -> daughters '000'/'001'), instead of forever
                # re-emitting '00'/'01'. Without this, _remove:['0'] removes
                # nothing on the 2nd+ division and run_multigen_sqlite's
                # followed-agent-disappeared branch never trips, so generations
                # are under-counted even though gen-3 cells biologically run.
                div_edge = agent.get('division')
                if isinstance(div_edge, dict):
                    if isinstance(div_edge.get('config'), dict):
                        div_edge['config']['agent_id'] = daughter_id
                    inst = div_edge.get('instance')
                    if inst is not None:
                        inst.agent_id = daughter_id
                return doc

            d1_doc = _build_daughter_doc(d1_data, d1_seed, d1_id)
            d2_doc = _build_daughter_doc(d2_data, d2_seed, d2_id)

            d1_cell = d1_doc['state']['agents']['0']
            d2_cell = d2_doc['state']['agents']['0']

            print(f'  DAUGHTERS: {d1_id} (bulk={d1_data["bulk"]["count"].sum()}) '
                  f'+ {d2_id} (bulk={d2_data["bulk"]["count"].sum()})')

            # Mirror vEcoli's pre-divide ``emitter.finalize()`` hook
            # (ecoli_master_sim.py DivisionDetected branch: success=True then
            # finalize()). The parent ParquetEmitter has rows buffered since its
            # last batch flush; without an explicit close() they vanish — and
            # no success sentinel is written — when this _remove update tears
            # the agent subtree down. The close-flush-and-unregister itself is
            # the FRAMEWORK-generic ``finalize_emitter_for_agent`` from
            # pbg-emitters; the v2ecoli-specific glue is only *which key* to
            # finalize: the parent is registered under the parquet override's
            # metadata.agent_id (the runner's per-generation identity: "0",
            # "00", ...). self.agent_id inside the composite is always "0" (the
            # agents/0 key), so for gens >= 2 a self.agent_id lookup misses;
            # use the override metadata as the canonical key.
            from v2ecoli.composites._helpers import finalize_emitter_for_agent
            _ovr_meta = (_PARQUET_EMITTER_OVERRIDE or {}).get('metadata') or {}
            _emitter_key = str(_ovr_meta.get('agent_id', self.agent_id))
            finalize_emitter_for_agent(_emitter_key, success=True)

            return {
                'agents': {
                    '_remove': [self.agent_id],
                    '_add': [
                        (d1_id, d1_cell),
                        (d2_id, d2_cell),
                    ],
                },
            }
        except Exception as e:
            # Surface failures loudly — process-bigraph otherwise swallows
            # exceptions raised from a step's next_update.
            import traceback
            print(f'  DIVISION FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            raise

    def update(self, state, interval=None):
        return self.next_update(1.0, state)
