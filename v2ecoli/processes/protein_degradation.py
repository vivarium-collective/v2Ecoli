"""
===================
Protein Degradation
===================

This process accounts for the degradation of protein monomers.
Specific proteins to be degraded are selected as a Poisson process.

Mathematical Model
------------------
Inputs (from stores):
    - bulk.{protein_i}                  counts (integer) of each protein species i
    - bulk.{AA_j}                       counts of each amino acid species j (unused read)
    - bulk.H2O                          water count (integer)
    - timestep                          dt in seconds

Parameters (from config):
    - k_i                               first-order degradation rate constant for
                                        protein i, in 1/s
    - a_ij                              amino-acid composition matrix: count of
                                        AA j in one copy of protein i
    - L_i = sum_j(a_ij)                 protein length (residues)
    - S                                 stoichiometry matrix (metabolites x proteins);
                                        rows = amino acids + water, columns = proteins

Calculation:
    For each protein species i, draw the number of copies that degrade this
    timestep from a Poisson distribution:

        n_degraded_i ~ Poisson(k_i * dt * n_i)

    Each degraded protein of length L_i hydrolyses (L_i - 1) water molecules,
    releasing its constituent amino acids:

        Protein_i + (L_i - 1) H2O  -->  sum_j(a_ij * AA_j)

    Stacked across all protein species, the net metabolite count change is:

        delta_metabolites = S @ n_degraded

    Water consumption per protein column equals -(sum of amino-acid counts - 1),
    matching (N-1) peptide-bond hydrolyses for a chain of length N.

Outputs (to stores):
    - bulk.{protein_i}                  -= n_degraded_i
    - bulk.{AA_j}                       += sum_i(a_ij * n_degraded_i)
    - bulk.H2O                          -= sum_i((L_i - 1) * n_degraded_i)

TODO:
 - protein complexes
 - add protease functionality
"""

import numpy as np

# simulate_process removed

from bigraph_schema.contract import ProcessContract
from v2ecoli.library.data_predicates import (
    monotonically_increasing,
    monotonically_decreasing,
    all_nonnegative,
)
from v2ecoli.library.schema import counts, bulk_name_to_idx

from v2ecoli.library.ecoli_step import EcoliStep as Step


# Register default topology for this process, associating it with process name
NAME = "ecoli-protein-degradation"
TOPOLOGY = {"bulk": ("bulk",), "timestep": ("timestep",)}


class ProteinDegradation(Step):
    """Protein Degradation Step

    Single-pass process: no resource competition with other processes,
    so the request/allocate/evolve cycle is unnecessary. Directly computes
    Poisson degradation draws and applies stoichiometric updates.
    """

    description = (
        "Protein Degradation — first-order Poisson hydrolysis of protein monomers.\n\n"
        "    nᵢ_deg ~ min(Poisson(kᵢ · dt · nᵢ), nᵢ)\n"
        "    Δmetabolites = S · n_deg\n"
        "    Proteinᵢ + (Lᵢ−1) H2O → ∑ⱼ aᵢⱼ · AAⱼ\n"
        "  kᵢ: first-order degradation rate (1/s); nᵢ: copies of protein i;\n"
        "  aᵢⱼ: count of AA j per protein i; Lᵢ = ∑ⱼ aᵢⱼ (residues);\n"
        "  S: stoichiometry (metabolites × proteins), +AA release, −(Lᵢ−1) H2O."
    )

    contract = ProcessContract(
        summary=(
            "First-order Poisson hydrolysis of protein monomers: each step draw "
            "the number of copies of each protein that degrade, then apply the "
            "stoichiometric release of amino acids and consumption of water. "
            "Single-pass Step (no resource competition, no request/allocate cycle)."
        ),
        symbols={
            "nᵢ_deg": "copies of protein i degraded this step = min(Poisson(kᵢ·dt·nᵢ), nᵢ) (count)",
            "kᵢ": "first-order degradation rate constant for protein i (1/s)",
            "nᵢ": "current copies of protein i (count)",
            "dt": "timestep (s)",
            "S": "degradation stoichiometry, metabolites × proteins (+AA release rows, −(Lᵢ−1) H2O row)",
            "n_deg": "vector of per-protein degradation counts",
            "Lᵢ": "length of protein i in residues, = Σⱼ aᵢⱼ",
            "aᵢⱼ": "count of amino acid j in one copy of protein i",
            "AAⱼ": "amino acid species j released on hydrolysis (count)",
            "H2O": "water consumed: (Lᵢ−1) molecules per degraded copy of protein i",
        },
        inputs={
            "bulk": "Reads current protein counts nᵢ to draw Poisson degradation counts; writes back −nᵢ_deg proteins, +released amino acids, and −(Lᵢ−1)·nᵢ_deg water via the stoichiometry matrix.",
            "timestep": "Provides dt scaling the per-protein Poisson degradation rate.",
        },
        outputs={
            "bulk": "Writes the stoichiometric metabolite deltas (amino-acid release, water consumption) and the per-protein count decrements.",
        },
        config={
            "raw_degradation_rate": "Per-protein first-order degradation rate constants kᵢ (1/s).",
            "protein_lengths": "Vestigial: read into self.protein_lengths but never used in update(). Water consumed per hydrolysis is NOT set from this — the −(Lᵢ−1) H2O row is derived from amino_acid_counts as −(Σⱼ aᵢⱼ − 1), i.e. Lᵢ = Σⱼ aᵢⱼ.",
            "amino_acid_counts": "Amino-acid composition matrix aᵢⱼ (AA j per protein i); forms the AA-release rows of S.",
            "protein_ids": "Protein species identities, defining the columns of the stoichiometry matrix.",
            "amino_acid_ids": "Amino-acid species identities, defining the AA rows of the stoichiometry matrix.",
            "water_id": "Bulk id of water, whose row carries the −(Lᵢ−1) peptide-bond-hydrolysis stoichiometry.",
        },
        assumptions=[
            "Degradation is a first-order Poisson process, independent per protein species, capped at the available copy count.",
            "Each degraded chain of length L consumes (L−1) water molecules (N−1 peptide-bond hydrolyses) and releases its constituent amino acids.",
            "Protein complexes and explicit protease mechanisms are not modeled (TODO in source).",
            "Single-pass: no resource competition, so no request/allocate/evolve cycle is used.",
        ],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'amino_acid_counts': {'_type': 'array[integer]', '_default': []},  # (n_proteins x n_amino_acids) composition matrix
        'amino_acid_ids': {'_type': 'list[string]', '_default': []},
        'protein_ids': {'_type': 'list[string]', '_default': []},
        'protein_lengths': {'_type': 'array[integer[aa]]', '_default': []},  # chain length per protein
        'raw_degradation_rate': {'_type': 'array[float[1/s]]', '_default': []},  # first-order rate constants k_i
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'water_id': {'_type': 'string', '_default': 'h2o'},
    }

    def initialize(self, config):

        self.raw_degradation_rate = self.parameters["raw_degradation_rate"]

        self.water_id = self.parameters["water_id"]
        self.amino_acid_ids = self.parameters["amino_acid_ids"]
        self.amino_acid_counts = self.parameters["amino_acid_counts"]

        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)

        # Build protein IDs for S matrix
        self.protein_ids = self.parameters["protein_ids"]
        self.protein_lengths = self.parameters["protein_lengths"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.metabolite_idx = None

        # Build S matrix
        self.degradation_matrix = np.zeros(
            (len(self.metabolite_ids), len(self.protein_ids)), np.int64
        )
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts
        )
        # Assuming N-1 H2O is required per peptide chain length N
        self.degradation_matrix[self.water_index, :] = -(
            np.sum(self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1
        )

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'timestep': {'_type': 'integer', '_default': 1},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
        }

    def update(self, states, interval=None):
        # At t=0, convert molecule name strings to bulk array indices
        if self.metabolite_idx is None:
            self.protein_idx = bulk_name_to_idx(self.protein_ids, states["bulk"]["id"])
            self.metabolite_idx = bulk_name_to_idx(
                self.metabolite_ids, states["bulk"]["id"]
            )

        dt = states["timestep"]
        protein_counts = counts(states["bulk"], self.protein_idx)

        # Poisson draw: expected degradation count = k_i * dt * n_i,
        # capped at available protein count
        n_to_degrade = np.fmin(
            self.random_state.poisson(
                self.raw_degradation_rate * dt * protein_counts
            ),
            protein_counts,
        )

        # Apply degradation: delta_metabolites = S @ n_degraded
        # S encodes amino acid release (+) and water consumption (-)
        delta_metabolites = np.dot(self.degradation_matrix, n_to_degrade)

        return {
            "bulk": [
                (self.metabolite_idx, delta_metabolites),
                (self.protein_idx, -n_to_degrade),
            ]
        }
