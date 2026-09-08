"""
======================
Chromosome Replication
======================

Performs initiation, elongation, and termination of active partial chromosomes
that replicate the chromosome.

Mathematical Model
------------------
Chromosome replication proceeds in three modules per timestep:

**Module 1 -- Initiation**

Replication initiates when the cell mass per origin of replication (oriC)
exceeds a critical threshold:

    M_cell / n_oriC  >=  M_critical(tau)

where M_critical (fg) depends on the growth rate via the doubling time tau.
Upon initiation, two replisomes are assembled per oriC (one per replichore),
each consuming 3 trimer + 1 monomer subunits (if mechanistic_replisome=True).
Two new chromosome domains are created as children of the parent domain.

**Module 2 -- Elongation**

Each replication fork extends along its template strand. The elongation
rate v (nt/s) is drawn stochastically around the basal rate (default 967 nt/s).
dNTP consumption is computed via the ``polymerize`` algorithm:

    sequences = buildSequences(template, fork_position, v * dt)
    result = polymerize(sequences, dNTP_counts, rate_limit)

If one dNTP species is limiting, all four are scaled by the same ratio
to maintain stoichiometric balance. Mass increase per fork:

    delta_mass_DNA = sum(elongated_nt_i * weight_i)   [fg]

Each dNTP polymerized releases one pyrophosphate (PPi).

**Module 3 -- Termination**

A fork terminates when its coordinate reaches the replichore length:

    |coordinate| == L_replichore

When both forks of a domain terminate, the domain splits: the parent
domain's two child domains become independent chromosomes. Replisome
subunits are released back to the bulk pool. A D-period timer is set
on the new full chromosome for cell division scheduling.
"""

import os

import numpy as np

from bigraph_schema.contract import ProcessContract

from v2ecoli.library.schema import (
    counts,
    attrs,
    bulk_name_to_idx,
)

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease

# topology_registry removed
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import (
    ACTIVE_REPLISOME_ARRAY,
    ORIC_ARRAY,
    CHROMOSOME_DOMAIN_ARRAY,
    FULL_CHROMOSOME_ARRAY,
)


# Register default topology for this process, associating it with process name
NAME = "ecoli-chromosome-replication"
TOPOLOGY = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "oriCs": ("unique", "oriC"),
    "chromosome_domains": ("unique", "chromosome_domain"),
    "full_chromosomes": ("unique", "full_chromosome"),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "timestep": ("timestep",),
    "global_time": ("global_time",),
}


class ChromosomeReplication(Step):
    """Chromosome Replication Step

    dNTPs are consumed only by chromosome replication; no other process
    competes for them. Runs as a plain Step.
    """

    description = (
        "Chromosome Replication — initiate, elongate, and terminate replication forks.\n\n"
        "1. Initiation: fire when  M_cell / n_oriC ≥ M_critical(τ);  +2 replisomes·oriC, +2 domains.\n"
        "2. Elongation: seqs = buildSequences(template, pos, ν·dt); polymerize(seqs, dNTPs, limit);\n"
        "   Δm_DNA = ∑ᵢ (elongated_ntᵢ · weightᵢ);  PPi released = dNTP polymerized.\n"
        "3. Termination: fork ends when |coordinate| = L_replichore; domain splits, subunits recycled.\n"
        "  M_cell: cell mass (fg); n_oriC: oriC count; M_critical(τ): mass/oriC threshold at doubling time τ;\n"
        "  ν: stochastic elongation rate (nt/s); dt: timestep; L_replichore: replichore length (nt)."
    )

    name = NAME
    topology = TOPOLOGY

    contract = ProcessContract(
        summary=(
            "Initiate, elongate, and terminate replication forks: fire a round of "
            "replication when mass-per-oriC crosses the growth-rate-dependent critical "
            "threshold, extend forks by dNTP polymerization, and split domains into "
            "daughter chromosomes at the terminus."
        ),
        symbols={
            "M_cell": "cell (dry) mass, femtograms (fg)",
            "n_oriC": "number of origins of replication (oriC) currently present (count)",
            "M_critical": "critical mass-per-oriC threshold for initiation (fg), a function of doubling time",
            "τ": "doubling time set by the current media/nutrient condition (min)",
            "ν": "stochastic per-fork elongation rate, nucleotides per second (nt/s)",
            "dt": "simulation timestep (s)",
            "L_replichore": "replichore length; fork terminates when |coordinate| reaches it (nt)",
            "Δm_DNA": "DNA mass added to a fork from elongated nucleotides this step (fg)",
            "PPi": "pyrophosphate released, one per nucleotide polymerized (count)",
        },
        inputs={
            "bulk": (
                "Reads counts of replisome trimer/monomer subunits and of the four dNTPs "
                "(and their PPi id) to gate initiation and cap elongation by the limiting dNTP."
            ),
            "active_replisomes": (
                "Reads each fork's domain_index, right_replichore flag, coordinates and "
                "massDiff_DNA to build the template sequences elongated this step."
            ),
            "oriCs": (
                "Reads the oriC count n_oriC (mass-per-origin trigger) and each oriC's "
                "domain_index to assign daughter domains on initiation."
            ),
            "chromosome_domains": (
                "Reads domain_index and child_domains to find the parent domains that carry "
                "an origin and to detect finished domains at termination."
            ),
            "full_chromosomes": (
                "Reads full-chromosome domain_index so a completed domain's chromosome can be "
                "re-pointed to a child domain and a second daughter chromosome added at termination."
            ),
            "listeners": (
                "Reads listeners.mass.cell_mass (M_cell) to compute mass per origin for the "
                "initiation test."
            ),
            "environment": (
                "Reads environment.media_id to look up the doubling time τ and thus the critical "
                "initiation mass M_critical for the current condition."
            ),
            "timestep": "Reads dt to scale the elongation rate over the step.",
            "global_time": (
                "Reads current simulation time to stamp a new full chromosome's division_time "
                "(global_time + D_period)."
            ),
        },
        outputs={
            "bulk": (
                "On initiation decrements 6 trimer + 2 monomer replisome subunits per oriC; during "
                "elongation subtracts dNTPs used and adds the equal count of PPi; on termination "
                "returns 3 trimer + 1 monomer subunits per deleted replisome."
            ),
            "active_replisomes": (
                "Adds two replisomes per oriC (one per replichore) at initiation, advances fork "
                "coordinates and adds Δm_DNA to massDiff_DNA during elongation, deletes both forks "
                "of a domain at termination."
            ),
            "oriCs": (
                "Re-assigns existing oriCs to new daughter domain indices and adds one new oriC per "
                "existing origin so each daughter keeps an origin."
            ),
            "chromosome_domains": (
                "Adds two child domains per origin (no children yet) and links them as children of "
                "the originating parent domain."
            ),
            "full_chromosomes": (
                "At termination re-points a finished chromosome to its first child domain and adds a "
                "second daughter full_chromosome, each with division_time = global_time + D_period."
            ),
            "listeners": (
                "Writes replication_data.critical_initiation_mass (M_critical, fg) and "
                "critical_mass_per_oriC (M_cell/n_oriC ÷ M_critical, dimensionless)."
            ),
        },
        config={
            "basal_elongation_rate": "Basal replication-fork speed ν (nt/s) that make_elongation_rates draws around.",
            "criticalInitiationMass": "Fallback critical initiation mass (fg) when no media-specific value applies.",
            "get_dna_critical_mass": "Callable mapping the current doubling time τ to the critical initiation mass.",
            "nutrientToDoublingTime": "Map from media_id to doubling time τ, selecting M_critical for the condition.",
            "make_elongation_rates": "Callable producing per-fork stochastic elongation rates from the basal rate.",
            "mechanistic_replisome": "If True, initiation requires (and consumes) explicit replisome subunits; else forks assemble freely.",
            "replichore_lengths": "Per-replichore length L_replichore; a fork terminates when |coordinate| reaches it.",
            "replisome_protein_mass": "Protein mass (fg) added per replisome when mechanistic_replisome is on.",
            "replisome_trimers_subunits": "Bulk ids of the trimeric replisome subunits (6 per oriC consumed/assembled).",
            "replisome_monomers_subunits": "Bulk ids of the monomeric replisome subunits (2 per oriC consumed/assembled).",
            "dntps": "Bulk ids of the four dNTPs consumed during elongation.",
            "ppi": "Bulk id of pyrophosphate released per polymerized nucleotide.",
            "polymerized_dntp_weights": "Per-dNTP polymerized masses used to compute Δm_DNA per fork.",
            "sequences": "Template DNA sequences the forks polymerize against.",
            "replication_coordinate": "Genomic coordinates anchoring the replication templates (stored at init; unused in this build's update path).",
            "D_period": "C+D-period timer added to global_time to schedule division on a new full chromosome.",
            "no_child_place_holder": "Sentinel domain index marking a chromosome domain with no children yet.",
        },
        assumptions=[
            "Initiation is synchronous across all origins in a step (no asynchronous firing); all oriCs fire together once M_cell/n_oriC ≥ M_critical.",
            "With mechanistic_replisome on, a round fires only if the full 6 trimer + 2 monomer subunits per oriC are available.",
            "Two replisomes are assembled per origin, one per replichore, sharing the origin's domain index.",
            "Forks added this step are not elongated until the next step.",
            "If one dNTP is limiting, all four are scaled by the same fractional limit to preserve stoichiometry; each nucleotide polymerized releases one PPi.",
            "A domain splits into two daughter chromosomes only when both of its forks have reached the replichore length; released replisome subunits recycle to bulk.",
            "dNTPs are consumed only by replication, so the process runs as a plain Step without partitioned allocation.",
        ],
    )

    config_schema = {
        'D_period': {'_type': 'node', '_default': np.array([], dtype=float)},
        'basal_elongation_rate': {'_type': 'integer[nt/s]', '_default': 967},
        'criticalInitiationMass': {'_type': 'quantity[fg]', '_default': 975.0},
        'dntps': {'_type': 'list[string]', '_default': []},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'get_dna_critical_mass': {'_type': 'method', '_default': None},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'mechanistic_replisome': {'_type': 'boolean', '_default': True},
        'no_child_place_holder': {'_type': 'integer', '_default': -1},
        'nutrientToDoublingTime': {'_type': 'map[quantity[float,min]]', '_default': {}},
        'polymerized_dntp_weights': {'_type': 'quantity[array[float],fg]', '_default': []},
        'ppi': {'_type': 'list[string]', '_default': []},
        'replication_coordinate': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'replichore_lengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'replisome_monomers_subunits': {'_type': 'list[string]', '_default': []},
        'replisome_protein_mass': {'_type': 'float[fg]', '_default': 0},
        'replisome_trimers_subunits': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'sequences': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
    }

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'oriCs': {'_type': ORIC_ARRAY, '_default': []},
            'chromosome_domains': {'_type': CHROMOSOME_DOMAIN_ARRAY, '_default': []},
            'full_chromosomes': {'_type': FULL_CHROMOSOME_ARRAY, '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                },
            },
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'global_time': {'_type': 'float', '_default': 0.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'active_replisomes': ACTIVE_REPLISOME_ARRAY,
            'oriCs': ORIC_ARRAY,
            'chromosome_domains': CHROMOSOME_DOMAIN_ARRAY,
            'full_chromosomes': FULL_CHROMOSOME_ARRAY,
            'listeners': {
                'replication_data': {
                    # Critical initiation mass — femtograms
                    'critical_initiation_mass': {'_type': 'overwrite[float[fg]]', '_default': 0.0},
                    # Cell mass / critical mass — dimensionless ratio
                    'critical_mass_per_oriC': {'_type': 'overwrite[float]', '_default': 0.0},
                },
            },
        }



    def initialize(self, config):

        # Load parameters
        self.get_dna_critical_mass = self.parameters["get_dna_critical_mass"]
        self.criticalInitiationMass = self.parameters["criticalInitiationMass"]
        self.nutrientToDoublingTime = self.parameters["nutrientToDoublingTime"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = self.parameters["polymerized_dntp_weights"]
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        # Sim options
        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        # random state
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # mcs-06 (multiscale-complexity-showcase): stochastic D-period.
        # The active division trigger schedules division_time = global_time + D_period
        # with a FIXED D_period, so interdivision timing inherits no D-period variance
        # (the model is under-dispersed, CV ~7% vs the biological 10-30%). When
        # d_period_cv > 0 the scheduled D_period is drawn per division event from
        # Normal(1.0, d_period_cv) * D_period (seeded via self.random_state, so it is
        # reproducible per seed), routing biologically-motivated noise into the ACTIVE
        # trigger. Default 0.0 leaves baseline behaviour byte-identical; the env var
        # V2E_D_PERIOD_CV overrides it for the controlled before/after experiment.
        self.d_period_cv = float(
            os.environ.get("V2E_D_PERIOD_CV", self.parameters.get("d_period_cv", 0.0))
        )

        self.emit_unique = self.parameters.get("emit_unique", True)

        # Bulk molecule names
        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters[
            "replisome_monomers_subunits"
        ]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None

    def update(self, states, interval=None):
        self._prepare(states)
        return self._evolve(states)

    def _prepare(self, states):
        timestep = states["timestep"]
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, states["bulk"]["id"])
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, states["bulk"]["id"]
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, states["bulk"]["id"]
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, states["bulk"]["id"])
        requests = {}
        # Get total count of existing oriC's
        n_oriC = states["oriCs"]["_entryState"].sum()
        # If there are no origins, return immediately
        if n_oriC == 0:
            return

        # Get current cell mass
        cellMass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)

        # Get critical initiation mass for current simulation environment
        current_media_id = states["environment"]["media_id"]
        self.criticalInitiationMass = self.get_dna_critical_mass(
            self.nutrientToDoublingTime[current_media_id]
        )

        # Calculate mass per origin of replication, and compare to critical
        # initiation mass. If the cell mass has reached this critical mass,
        # the process will initiate a round of chromosome replication for each
        # origin of replication.
        massPerOrigin = cellMass / n_oriC
        # .to('dimensionless') forces pint to reduce mass/mass; otherwise
        # the ratio can retain residual unit scaling and the >= 1.0
        # comparison below wrongly evaluates to False.
        self.criticalMassPerOriC = (
            massPerOrigin / self.criticalInitiationMass
        ).to('dimensionless')

        # If replication should be initiated, request subunits required for
        # building two replisomes per one origin of replication, and edit
        # access to oriC and chromosome domain attributes
        requests["bulk"] = []
        if self.criticalMassPerOriC >= 1.0:
            requests["bulk"].append((self.replisome_trimers_idx, 6 * n_oriC))
            requests["bulk"].append((self.replisome_monomers_idx, 2 * n_oriC))

        # If there are no active forks return
        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        if n_active_replisomes == 0:
            return

        # Get current locations of all replication forks
        (fork_coordinates,) = attrs(states["active_replisomes"], ["coordinates"])
        sequence_length = np.abs(np.repeat(fork_coordinates, 2))

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            len(self.sequences),
            self.basal_elongation_rate,
            states["timestep"],
        )

        sequences = buildSequences(
            self.sequences,
            np.tile(np.arange(4), n_active_replisomes // 2),
            sequence_length,
            self.elongation_rates,
        )

        # Count number of each dNTP in sequences for the next timestep
        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4
        )

        # If one dNTP is limiting then limit the request for the other three by
        # the same ratio
        dNtpsTotal = counts(states["bulk"], self.dntps_idx)
        maxFractionalReactionLimit = (
            np.fmin(1, dNtpsTotal / sequenceComposition)
        ).min()

        # Request dNTPs
        requests["bulk"].append(
            (
                self.dntps_idx,
                (maxFractionalReactionLimit * sequenceComposition).astype(int),
            )
        )

        # requests no longer used in Step form; kept locally for clarity
        _ = requests

    def _evolve(self, states):
        timestep = states["timestep"]
        # Initialize the update dictionary
        update = {
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "full_chromosomes": {},
            "listeners": {"replication_data": {}},
        }

        # Module 1: Replication initiation
        # Get number of existing replisomes and oriCs
        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        n_oriC = states["oriCs"]["_entryState"].sum()

        # If there are no origins, return immediately
        if n_oriC == 0:
            return update

        # Get attributes of existing chromosome domains
        domain_index_existing_domain, child_domains = attrs(
            states["chromosome_domains"], ["domain_index", "child_domains"]
        )

        initiate_replication = False
        if self.criticalMassPerOriC >= 1.0:
            # Get number of available replisome subunits
            n_replisome_trimers = counts(states["bulk"], self.replisome_trimers_idx)
            n_replisome_monomers = counts(states["bulk"], self.replisome_monomers_idx)
            # Initiate replication only when
            # 1) The cell has reached the critical mass per oriC
            # 2) If mechanistic replisome option is on, there are enough
            # replisome subunits to assemble two replisomes per existing OriC.
            # Note that we assume asynchronous initiation does not happen.
            initiate_replication = not self.mechanistic_replisome or (
                np.all(n_replisome_trimers == 6 * n_oriC)
                and np.all(n_replisome_monomers == 2 * n_oriC)
            )

        # If all conditions are met, initiate a round of replication on every
        # origin of replication
        if initiate_replication:
            # Get attributes of existing oriCs and domains
            (domain_index_existing_oric,) = attrs(states["oriCs"], ["domain_index"])

            # Get indexes of the domains that would be getting child domains
            # (domains that contain an origin)
            new_parent_domains = np.where(
                np.isin(domain_index_existing_domain, domain_index_existing_oric)
            )[0]

            # Calculate counts of new replisomes and domains to add
            n_new_replisome = 2 * n_oriC
            n_new_domain = 2 * n_oriC

            # Calculate the domain indexes of new domains and oriC's
            max_domain_index = domain_index_existing_domain.max()
            domain_index_new = np.arange(
                max_domain_index + 1, max_domain_index + 2 * n_oriC + 1, dtype=np.int32
            )

            # Add new oriC's, and reset attributes of existing oriC's
            # All oriC's must be assigned new domain indexes
            update["oriCs"]["set"] = {"domain_index": domain_index_new[:n_oriC]}
            update["oriCs"]["add"] = {
                "domain_index": domain_index_new[n_oriC:],
            }

            # Add and set attributes of newly created replisomes.
            # New replisomes inherit the domain indexes of the oriC's they
            # were initiated from. Two replisomes are formed per oriC, one on
            # the right replichore, and one on the left.
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_oriC)
            right_replichore = right_replichore.tolist()
            domain_index_new_replisome = np.repeat(domain_index_existing_oric, 2)
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )
            update["active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            # Add and set attributes of new chromosome domains. All new domains
            # should have have no children domains.
            new_child_domains = np.full(
                (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
            )
            new_domains_update = {
                "add": {
                    "domain_index": domain_index_new,
                    "child_domains": new_child_domains,
                }
            }

            # Add new domains as children of existing domains
            child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            existing_domains_update = {"set": {"child_domains": child_domains}}
            update["chromosome_domains"].update(
                {**new_domains_update, **existing_domains_update}
            )

            # Decrement counts of replisome subunits
            if self.mechanistic_replisome:
                update["bulk"].append((self.replisome_trimers_idx, -6 * n_oriC))
                update["bulk"].append((self.replisome_monomers_idx, -2 * n_oriC))

        # Write data from this module to a listener
        update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
            self.criticalMassPerOriC.magnitude
        )
        update["listeners"]["replication_data"]["critical_initiation_mass"] = (
            self.criticalInitiationMass.to(units.fg).magnitude
        )

        # Module 2: replication elongation
        # If no active replisomes are present, return immediately
        # Note: the new replication forks added in the previous module are not
        # elongated until the next timestep.
        if n_active_replisomes == 0:
            return update

        # Get allocated counts of dNTPs
        dNtpCounts = counts(states["bulk"], self.dntps_idx)

        # Get attributes of existing replisomes
        (
            domain_index_replisome,
            right_replichore,
            coordinates_replisome,
        ) = attrs(
            states["active_replisomes"],
            ["domain_index", "right_replichore", "coordinates"],
        )

        # Build sequences to polymerize
        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
        sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

        sequences = buildSequences(
            self.sequences, sequence_indexes, sequence_length, self.elongation_rates
        )

        # Use polymerize algorithm to quickly calculate the number of
        # elongations each fork catalyzes
        reactionLimit = dNtpCounts.sum()

        active_elongation_rates = self.elongation_rates[sequence_indexes]

        result = polymerize(
            sequences,
            dNtpCounts,
            reactionLimit,
            self.random_state,
            active_elongation_rates,
        )

        sequenceElongations = result.sequenceElongation
        dNtpsUsed = result.monomerUsages

        # Compute mass increase for each elongated sequence
        mass_increase_dna = computeMassIncrease(
            sequences,
            sequenceElongations,
            self.polymerized_dntp_weights.to(units.fg).magnitude,
        )

        # Compute masses that should be added to each replisome
        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        # Update positions of each fork
        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        # Reverse signs of fork coordinates on left replichore
        updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

        # Update attributes and submasses of replisomes
        (current_dna_mass,) = attrs(states["active_replisomes"], ["massDiff_DNA"])
        update["active_replisomes"].update(
            {
                "set": {
                    "coordinates": updated_coordinates,
                    "massDiff_DNA": current_dna_mass + added_dna_mass,
                }
            }
        )

        # Update counts of polymerized metabolites
        update["bulk"].append((self.dntps_idx, -dNtpsUsed))
        update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

        # Module 3: replication termination
        # Determine if any forks have reached the end of their sequences. If
        # so, delete the replisomes and domains that were terminated.
        terminal_lengths = self.replichore_lengths[
            np.logical_not(right_replichore).astype(np.int64)
        ]
        terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

        # If any forks were terminated,
        if terminated_replisomes.sum() > 0:
            # Get domain indexes of terminated forks
            terminated_domains = np.unique(
                domain_index_replisome[terminated_replisomes]
            )

            # Get attributes of existing domains and full chromosomes
            (
                domain_index_domains,
                child_domains,
            ) = attrs(states["chromosome_domains"], ["domain_index", "child_domains"])
            (domain_index_full_chroms,) = attrs(
                states["full_chromosomes"], ["domain_index"]
            )

            # Initialize array of replisomes that should be deleted
            replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool_)

            # Count number of new full chromosomes that should be created
            n_new_chromosomes = 0

            # Initialize array for domain indexes of new full chromosomes
            domain_index_new_full_chroms = []

            for terminated_domain_index in terminated_domains:
                # Get all terminated replisomes in the terminated domain
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes,
                )

                # If both replisomes in the domain have terminated, we are
                # ready to split the chromosome and update the attributes.
                if terminated_domain_matching_replisomes.sum() == 2:
                    # Tag replisomes and domains with the given domain index
                    # for deletion
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete, terminated_domain_matching_replisomes
                    )

                    domain_mask = domain_index_domains == terminated_domain_index

                    # Get child domains of deleted domain
                    child_domains_this_domain = child_domains[
                        np.where(domain_mask)[0][0], :
                    ]

                    # Modify domain index of one existing full chromosome to
                    # index of first child domain
                    domain_index_full_chroms = domain_index_full_chroms.copy()
                    domain_index_full_chroms[
                        np.where(domain_index_full_chroms == terminated_domain_index)[0]
                    ] = child_domains_this_domain[0]

                    # Increment count of new full chromosome
                    n_new_chromosomes += 1

                    # Append chromosome index of new full chromosome
                    domain_index_new_full_chroms.append(child_domains_this_domain[1])

            # Delete terminated replisomes
            update["active_replisomes"]["delete"] = np.where(replisomes_to_delete)[0]

            # Generate new full chromosome molecules
            if n_new_chromosomes > 0:
                # mcs-06: optionally jitter the scheduled D-period so interdivision
                # timing carries biological variance (see __init__). d_period_cv == 0
                # -> scheduled_d_period == self.D_period exactly (baseline unchanged).
                scheduled_d_period = self.D_period
                if self.d_period_cv > 0.0:
                    factor = self.random_state.normal(1.0, self.d_period_cv)
                    factor = max(0.5, factor)  # floor: never an unphysically short D-period
                    scheduled_d_period = self.D_period * factor
                chromosome_add_update = {
                    "add": {
                        "domain_index": domain_index_new_full_chroms,
                        "division_time": states["global_time"] + scheduled_d_period,
                        "has_triggered_division": False,
                    }
                }

                # Reset domain index of existing chromosomes that have finished
                # replication
                chromosome_existing_update = {
                    "set": {"domain_index": domain_index_full_chroms}
                }

                update["full_chromosomes"].update(
                    {**chromosome_add_update, **chromosome_existing_update}
                )

            # Increment counts of replisome subunits
            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, replisomes_to_delete.sum())
                )

        return update


def test_chromosome_replication():
    test_config = {}
    process = ChromosomeReplication(test_config)
    assert process is not None


if __name__ == "__main__":
    test_chromosome_replication()
