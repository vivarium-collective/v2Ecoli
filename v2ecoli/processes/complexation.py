"""
============
Complexation
============

This process encodes molecular simulation of macromolecular complexation,
in which monomers are assembled into complexes. Macromolecular complexation
is done by identifying complexation reactions that are possible (which are
reactions that have sufficient counts of all sub-components), performing one
randomly chosen possible reaction, and re-identifying all possible complexation
reactions. This process assumes that macromolecular complexes form spontaneously,
and that complexation reactions are fast and complete within the time step of the
simulation.

Mathematical Model
------------------
Complexation is simulated as a continuous-time Markov chain (Gillespie
algorithm) via the ``StochasticSystem`` class from ``stochastic_arrow``.

Given a stoichiometry matrix S (reactions x molecules; the stochastic_arrow
convention where each row is a reaction) and a rate vector k, the system
evolves molecule counts x(t) over the timestep dt:

    x(t + dt) = StochasticSystem.evolve(dt, x(t), k)

Each reaction j fires stochastically with propensity:

    a_j = k_j * product_i(x_i choose |S_ji|)   for all reactant species i

The net molecule count change is:

    delta_x = x(t + dt) - x(t) = S.T @ occurrences

Each complex's reactants are complex-specific (no shared resource
competition), so this runs as a plain Step with a single Gillespie call.
"""

# TODO(wcEcoli):
# - allow for shuffling when appropriate (maybe in another process)
# - handle protein complex dissociation

import numpy as np
from stochastic_arrow import StochasticSystem

from bigraph_schema.contract import ProcessContract

# simulate_process removed

from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.ecoli_step import EcoliStep as Step

# Register default topology for this process, associating it with process name
NAME = "ecoli-complexation"
TOPOLOGY = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}


class Complexation(Step):
    """Complexation Step (Gillespie)"""

    description = (
        "Complexation — spontaneous monomer→complex assembly (Gillespie SSA).\n\n"
        "Continuous-time Markov chain over reactions (stoichiometry S, rows=reactions, rates k):\n"
        "    x(t+dt) = StochasticSystem.evolve(dt, x(t), k)\n"
        "    propensity a_j = k_j · ∏_i C(x_i, |S_ji|);   Δx = outcome − x(t) = Sᵀ·occurrences.\n"
        "Reactants are complex-specific (no shared-resource competition)."
    )

    name = NAME
    topology = TOPOLOGY

    contract = ProcessContract(
        summary=(
            "Assembles monomers into macromolecular complexes via a single "
            "Gillespie SSA (StochasticSystem) run over the whole timestep, "
            "using a stoichiometry matrix S and per-reaction rate constants k."
        ),
        symbols={
            "S": "stoichiometry matrix, reactions × molecules (stochastic_arrow convention: row j is reaction j, column i is molecule i)",
            "k": "vector of per-reaction propensity rate constants k_j (1/s)",
            "k_j": "rate constant of reaction j (1/s)",
            "x_i": "count of molecule species i (count)",
            "S_ji": "stoichiometric coefficient of molecule i in reaction j (entry at row j, column i of S)",
            "a_j": "propensity of reaction j = k_j·∏_i C(x_i, |S_ji|) (1/s)",
            "dt": "timestep over which the SSA is integrated (s)",
            "occurrences": "number of times each reaction fired over dt (count)",
            "Δx": "net molecule-count change = outcome − x(t) = Sᵀ·occurrences (count); computed directly from evolve's outcome, not a literal matrix product",
        },
        inputs={
            "bulk": "reads the counts of the monomer/complex molecules that participate in complexation reactions; these are the SSA state vector x(t)",
            "timestep": "SSA integration interval dt (s) passed to StochasticSystem.evolve",
        },
        outputs={
            "bulk": "applies the net count change Δx = x(t+dt) − x(t): consumes reactant monomers and produces complexes",
            "listeners": "writes complexation_listener.complexation_events — the per-reaction occurrence counts from the SSA run",
        },
        config={
            "stoichiometry": "stoichiometry matrix S (reactions × molecules) defining each complexation reaction",
            "rates": "per-reaction propensity rate constants k (1/s)",
            "molecule_names": "bulk ids of the molecules indexed by S (maps SSA state to bulk positions)",
            "reaction_ids": "identifiers of the complexation reactions (columns/rows of S)",
            "complex_ids": "identifiers of the assembled complex products",
        },
        assumptions=[
            "Complexes form spontaneously and complexation reactions are fast, completing within one timestep.",
            "Each complex's reactants are complex-specific, so there is no shared-resource competition — a single Gillespie call suffices and it runs as a plain Step.",
            "Complex dissociation and shuffling are not modeled here.",
        ],
    )

    config_schema = {
        'complex_ids': {'_type': 'list[string]', '_default': []},
        'molecule_names': {'_type': 'list[string]', '_default': []},
        'rates': {'_type': 'array[float[1/s]]', '_default': np.array([], dtype=float)},  # reaction propensity rate constants
        'reaction_ids': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'stoichiometry': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},  # (reactions x molecules) stoichiometry matrix S
        'time_step': {'_type': 'integer[s]', '_default': 1},
    }

    def initialize(self, config):

        self.stoichiometry = self.parameters["stoichiometry"]
        self.rates = self.parameters["rates"]
        self.molecule_names = self.parameters["molecule_names"]
        self.molecule_idx = None
        self.reaction_ids = self.parameters["reaction_ids"]
        self.complex_ids = self.parameters["complex_ids"]

        self.randomState = np.random.RandomState(seed=self.parameters["seed"])
        self.seed = self.randomState.randint(2**31)
        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'timestep': {'_type': 'integer[s]', '_default': 1},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'complexation_listener': {
                    'complexation_events': {'_type': 'overwrite[array[integer]]', '_default': [0] * 1088},
                },
            },
        }


    def update(self, states, interval=None):
        dt = states["timestep"]
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecule_names, states["bulk"]["id"]
            )

        molecule_counts = counts(states["bulk"], self.molecule_idx)

        # Single Gillespie run: x(t + dt) = StochasticSystem.evolve(dt, x(t), k)
        result = self.system.evolve(dt, molecule_counts, self.rates)
        delta_counts = result["outcome"] - molecule_counts

        return {
            "bulk": [(self.molecule_idx, delta_counts)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": result["occurrences"].astype(int)
                }
            },
        }
