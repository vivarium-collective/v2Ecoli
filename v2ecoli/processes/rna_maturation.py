"""
===============
RNA Maturation
===============

Converts unprocessed tRNA/rRNA molecules into mature tRNA/rRNAs, and
consolidates the different variants of 23S, 16S, and 5S rRNAs into the
single variant used for ribosomal subunit assembly.

Mathematical Model
------------------
Maturation is encoded as a stoichiometric transformation. Given a sparse
stoichiometry matrix S (mature_RNAs x unprocessed_RNAs, CSR format):

    n_mature = S @ n_unprocessed

Each maturation reaction also consumes pyrophosphate (PPi) — it is added
onto the 5' ends of the mature RNAs — and may produce nucleotide fragments:

    n_ppi_consumed = ppi_per_reaction . n_unprocessed
    n_degraded_nts = degraded_nt_counts^T @ n_unprocessed

Reactions only proceed if the required maturation enzymes are present,
checked via an enzyme requirement matrix E (reactions x enzymes):

    reaction_is_off = (E @ enzyme_present) < n_required_enzymes

rRNA variant consolidation maps multiple genomic rRNA variants (e.g. 7
copies of 23S with slightly different sequences) onto a single canonical
species used downstream. The nucleotide difference between variant and
canonical sequence is balanced by adding/removing NMPs and water:

    delta_nts_variant = delta_nt_counts^T @ n_variant_copies
"""

import numpy as np

from bigraph_schema.contract import ProcessContract

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import counts, bulk_name_to_idx

# Register default topology for this process, associating it with process name
NAME = "ecoli-rna-maturation"
TOPOLOGY = {"bulk": ("bulk",), "bulk_total": ("bulk",), "listeners": ("listeners",)}


class RnaMaturation(Step):
    """RNA Maturation Step

    Stoichiometric transformation of unprocessed tRNA/rRNA into mature
    forms. No cross-process resource competition (water/NMPs consumed
    are marginal vs pools), so runs as a plain Step.
    """

    description = (
        "RNA Maturation — stoichiometric processing of unprocessed tRNA/rRNA into mature forms.\n\n"
        "    reaction_off = (E · enzyme_present) < n_required;   n_unproc[off] = 0\n"
        "    n_mature = S · n_unproc;   ppi = ppi_per_rxn · n_unproc  (PPi consumed, added to 5' ends)\n"
        "    rRNA variant consolidation: Δnts = Δnt_countsᵀ · n_variant;  NMP +Δ, H2O −Σ, H⁺ +Σ\n"
        "  S: stoich (mature × unprocessed, CSR); E: enzyme-requirement (rxns × enzymes);\n"
        "  Δnt_counts: NMP difference variant vs canonical 16S/23S/5S."
    )

    name = NAME
    topology = TOPOLOGY

    contract = ProcessContract(
        summary=(
            "Stoichiometrically converts unprocessed tRNA/rRNA into mature "
            "forms (n_mature = S·n_unproc) gated by maturation-enzyme "
            "availability, and consolidates the genomic 16S/23S/5S rRNA "
            "variants onto their single canonical species with NMP/H2O mass "
            "balance."
        ),
        symbols={
            "S": "sparse stoichiometry matrix (mature_RNAs × unprocessed_RNAs, CSR)",
            "E": "enzyme-requirement matrix (reactions × enzymes)",
            "enzyme_present": "boolean vector of maturation enzymes with nonzero count",
            "n_required": "minimum enzyme count required per maturation reaction (count)",
            "n_unproc": "counts of unprocessed RNA per reaction (count); zeroed where reaction is off",
            "n_mature": "counts of mature RNA produced = S·n_unproc (count)",
            "ppi": "pyrophosphate consumed — added onto the 5' ends of the mature RNAs per maturation reaction (count)",
            "ppi_per_rxn": "PPi consumed (incorporated) per unprocessed RNA reaction (count)",
            "n_variant": "copy counts of a genomic rRNA variant being consolidated (count)",
            "Δnt_counts": "NMP nucleotide-count difference, variant vs canonical 16S/23S/5S (nt)",
            "Δnts": "net NMPs added on consolidation = Δnt_countsᵀ·n_variant (nt)",
            "NMP": "free nucleotide monophosphates added/removed for mass balance (count)",
            "H2O": "water consumed balancing added bases (count)",
            "H⁺": "protons released balancing added bases (count)",
        },
        inputs={
            "bulk": "reads counts of unprocessed RNAs, the maturation enzymes (to gate reactions), and the 16S/23S/5S rRNA variants to consolidate",
            "bulk_total": "declared port aliased to the same ('bulk',) store as `bulk`; present in inputs()/topology for framework symmetry, but update() reads the full unpartitioned counts through the `bulk` port, not this one",
        },
        outputs={
            "bulk": "produces mature RNAs (+S·n_unproc), consumes unprocessed RNAs, consumes PPi (−ppi_update; added onto the mature-RNA 5' ends), moves each rRNA variant onto its canonical species, and adds/removes NMPs with balancing H2O (−) and H⁺ (+)",
            "listeners": "writes rna_maturation_listener: total_maturation_events, total_degraded_ntps, per-species unprocessed_rnas_consumed, mature_rnas_generated, and maturation_enzyme_counts",
        },
        config={
            "stoich_matrix": "maturation stoichiometry S (mature_RNAs × unprocessed_RNAs, CSR)",
            "enzyme_matrix": "enzyme-requirement matrix E (reactions × enzymes)",
            "n_required_enzymes": "minimum enzyme counts per reaction; reactions with (E·enzyme_present) below this are turned off",
            "n_ppi_added": "PPi consumed (added onto the 5' ends) per maturation reaction",
            "degraded_nt_counts": "nucleotide fragments released per unprocessed RNA (contributes to added bases / degraded ntps)",
            "delta_nt_counts_16s": "NMP difference between each 16S variant and the canonical 16S",
            "delta_nt_counts_23s": "NMP difference between each 23S variant and the canonical 23S",
            "delta_nt_counts_5s": "NMP difference between each 5S variant and the canonical 5S",
            "variant_16s_rRNA_ids": "bulk ids of 16S rRNA variants consolidated onto main_16s_rRNA_id",
            "variant_23s_rRNA_ids": "bulk ids of 23S rRNA variants consolidated onto main_23s_rRNA_id",
            "variant_5s_rRNA_ids": "bulk ids of 5S rRNA variants consolidated onto main_5s_rRNA_id",
            "main_16s_rRNA_id": "canonical 16S species receiving consolidated variant counts",
            "main_23s_rRNA_id": "canonical 23S species receiving consolidated variant counts",
            "main_5s_rRNA_id": "canonical 5S species receiving consolidated variant counts",
            "unprocessed_rna_ids": "bulk ids of the unprocessed RNAs consumed",
            "mature_rna_ids": "bulk ids of the mature RNAs produced",
            "rna_maturation_enzyme_ids": "bulk ids of the maturation enzymes gating reactions",
        },
        assumptions=[
            "Water and NMPs consumed are marginal relative to their pools, so there is no cross-process resource competition — it runs as a plain Step reading full bulk counts.",
            "A maturation reaction proceeds only if all its required enzymes are present at or above n_required.",
            "rRNA variant sequences differ from the canonical only in nucleotide composition; the difference is balanced by adding/removing NMPs, consuming H2O and releasing H⁺.",
        ],
    )

    config_schema = {
        'degraded_nt_counts': 'array[float[nt]]',  # nucleotides released per unprocessed RNA
        'delta_nt_counts_16s': 'array[float[nt]]',  # NMP difference: variant vs canonical 16S
        'delta_nt_counts_23s': 'array[float[nt]]',  # NMP difference: variant vs canonical 23S
        'delta_nt_counts_5s': 'array[float[nt]]',   # NMP difference: variant vs canonical 5S
        'enzyme_matrix': 'array[integer]',           # (reactions x enzymes) enzyme requirement matrix E
        'fragment_bases': 'list[string]',
        'main_16s_rRNA_id': 'string',
        'main_23s_rRNA_id': 'string',
        'main_5s_rRNA_id': 'string',
        'mature_rna_ids': 'list[string]',
        'n_ppi_added': 'array[integer]',             # PPi consumed (added to 5' ends) per maturation reaction
        'n_required_enzymes': 'array[integer]',      # minimum enzymes needed per reaction
        'nmps': 'list[string]',
        'ppi': 'string',
        'proton': 'string',
        'rna_maturation_enzyme_ids': 'list[string]',
        'stoich_matrix': 'csr_matrix',               # (mature_RNAs x unprocessed_RNAs) stoichiometry S
        'unprocessed_rna_ids': 'list[string]',
        'variant_16s_rRNA_ids': 'list[string]',
        'variant_23s_rRNA_ids': 'list[string]',
        'variant_5s_rRNA_ids': 'list[string]',
        'water': 'string',
    }

    def initialize(self, config):
        # Get matrices and vectors that describe maturation reactions
        self.stoich_matrix = self.parameters["stoich_matrix"]
        self.enzyme_matrix = self.parameters["enzyme_matrix"]
        self.n_required_enzymes = self.parameters["n_required_enzymes"]
        self.degraded_nt_counts = self.parameters["degraded_nt_counts"]
        self.n_ppi_added = self.parameters["n_ppi_added"]

        # Calculate number of NMPs that should be added when consolidating rRNA
        # molecules
        self.main_23s_rRNA_id = self.parameters["main_23s_rRNA_id"]
        self.main_16s_rRNA_id = self.parameters["main_16s_rRNA_id"]
        self.main_5s_rRNA_id = self.parameters["main_5s_rRNA_id"]

        self.variant_23s_rRNA_ids = self.parameters["variant_23s_rRNA_ids"]
        self.variant_16s_rRNA_ids = self.parameters["variant_16s_rRNA_ids"]
        self.variant_5s_rRNA_ids = self.parameters["variant_5s_rRNA_ids"]

        self.delta_nt_counts_23s = self.parameters["delta_nt_counts_23s"]
        self.delta_nt_counts_16s = self.parameters["delta_nt_counts_16s"]
        self.delta_nt_counts_5s = self.parameters["delta_nt_counts_5s"]

        # Bulk molecule IDs
        self.unprocessed_rna_ids = self.parameters["unprocessed_rna_ids"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.rna_maturation_enzyme_ids = self.parameters["rna_maturation_enzyme_ids"]
        self.fragment_bases = self.parameters["fragment_bases"]
        self.ppi = self.parameters["ppi"]
        self.water = self.parameters["water"]
        self.nmps = self.parameters["nmps"]
        self.proton = self.parameters["proton"]

        # Numpy indices for bulk molecules
        self.ppi_idx = None

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'bulk_total': {'_type': 'bulk_array', '_default': []},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'rna_maturation_listener': {
                    'total_maturation_events': {'_type': 'overwrite[integer]', '_default': 0},
                    'total_degraded_ntps': {'_type': 'overwrite[integer]', '_default': 0},
                    'unprocessed_rnas_consumed': {'_type': 'overwrite[array[integer]]', '_default': [0] * 49},
                    'mature_rnas_generated': {'_type': 'overwrite[array[integer]]', '_default': [0] * 99},
                    'maturation_enzyme_counts': {'_type': 'overwrite[array[integer]]', '_default': [0, 0, 0]},
                },
            },
        }

    def update(self, states, interval=None):
        # At t=0, convert molecule name strings to bulk array indices
        if self.ppi_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.unprocessed_rna_idx = bulk_name_to_idx(
                self.unprocessed_rna_ids, bulk_ids
            )
            self.mature_rna_idx = bulk_name_to_idx(self.mature_rna_ids, bulk_ids)
            self.rna_maturation_enzyme_idx = bulk_name_to_idx(
                self.rna_maturation_enzyme_ids, bulk_ids
            )
            self.fragment_base_idx = bulk_name_to_idx(self.fragment_bases, bulk_ids)
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water, bulk_ids)
            self.nmps_idx = bulk_name_to_idx(self.nmps, bulk_ids)
            self.proton_idx = bulk_name_to_idx(self.proton, bulk_ids)
            self.main_23s_rRNA_idx = bulk_name_to_idx(self.main_23s_rRNA_id, bulk_ids)
            self.main_16s_rRNA_idx = bulk_name_to_idx(self.main_16s_rRNA_id, bulk_ids)
            self.main_5s_rRNA_idx = bulk_name_to_idx(self.main_5s_rRNA_id, bulk_ids)
            self.variant_23s_rRNA_idx = bulk_name_to_idx(
                self.variant_23s_rRNA_ids, bulk_ids
            )
            self.variant_16s_rRNA_idx = bulk_name_to_idx(
                self.variant_16s_rRNA_ids, bulk_ids
            )
            self.variant_5s_rRNA_idx = bulk_name_to_idx(
                self.variant_5s_rRNA_ids, bulk_ids
            )

        # Read directly from bulk (no allocator — no competing processes)
        unprocessed_rna_counts = counts(states["bulk"], self.unprocessed_rna_idx)
        variant_23s_rRNA_counts = counts(states["bulk"], self.variant_23s_rRNA_idx)
        variant_16s_rRNA_counts = counts(states["bulk"], self.variant_16s_rRNA_idx)
        variant_5s_rRNA_counts = counts(states["bulk"], self.variant_5s_rRNA_idx)
        enzyme_availability = counts(
            states["bulk"], self.rna_maturation_enzyme_idx
        ).astype(bool)

        # Turn off reactions that lack sufficient enzymes:
        # reaction_is_off = (E @ enzyme_present) < n_required_enzymes
        reaction_is_off = (
            self.enzyme_matrix.dot(enzyme_availability) < self.n_required_enzymes
        )
        unprocessed_rna_counts = unprocessed_rna_counts.copy()
        unprocessed_rna_counts[reaction_is_off] = 0

        # Maturation: n_mature = S @ n_unprocessed
        n_mature_rnas = self.stoich_matrix.dot(unprocessed_rna_counts)
        n_added_bases_from_maturation = np.dot(
            self.degraded_nt_counts.T, unprocessed_rna_counts
        )
        ppi_update = self.n_ppi_added.dot(unprocessed_rna_counts)

        # rRNA variant consolidation: add NMPs to canonical, remove variants.
        # Mass balance computed from delta nucleotide counts.
        n_added_bases_from_consolidation = (
            self.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + self.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + self.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )

        n_added_bases = (
            n_added_bases_from_maturation + n_added_bases_from_consolidation
        ).astype(int)
        n_total_added_bases = int(n_added_bases.sum())

        return {
            "bulk": [
                (self.mature_rna_idx, n_mature_rnas),
                (self.unprocessed_rna_idx, -unprocessed_rna_counts),
                (self.ppi_idx, -ppi_update),
                (self.main_23s_rRNA_idx, variant_23s_rRNA_counts.sum()),
                (self.main_16s_rRNA_idx, variant_16s_rRNA_counts.sum()),
                (self.main_5s_rRNA_idx, variant_5s_rRNA_counts.sum()),
                (self.variant_23s_rRNA_idx, -variant_23s_rRNA_counts),
                (self.variant_16s_rRNA_idx, -variant_16s_rRNA_counts),
                (self.variant_5s_rRNA_idx, -variant_5s_rRNA_counts),
                (self.nmps_idx, n_added_bases),
                (self.water_idx, -n_total_added_bases),
                (self.proton_idx, n_total_added_bases),
            ],
            "listeners": {
                "rna_maturation_listener": {
                    "total_maturation_events": unprocessed_rna_counts.sum(),
                    "total_degraded_ntps": n_added_bases_from_maturation.sum(dtype=int),
                    "unprocessed_rnas_consumed": unprocessed_rna_counts,
                    "mature_rnas_generated": n_mature_rnas,
                    "maturation_enzyme_counts": counts(
                        states["bulk"], self.rna_maturation_enzyme_idx
                    ),
                }
            },
        }
