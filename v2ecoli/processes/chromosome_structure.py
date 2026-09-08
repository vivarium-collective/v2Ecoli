"""
====================
Chromosome Structure
====================

This step manages the structural state of the chromosome during replication,
handling interactions between replication forks and bound molecules.

Mathematical Model
------------------
**Collision detection and resolution**

When a replication fork advances past a bound molecule (RNAP, ribosome,
etc.), the collision must be resolved. Molecules on the template strand
are either removed (fork wins) or stall the fork (molecule wins),
depending on the molecule type and relative orientation.

**Promoter replication**

As a replisome traverses a promoter at genomic coordinate c, the
promoter is duplicated onto the newly replicated daughter strand.
Promoter replication occurs when:

    fork_position_prev < c <= fork_position_new

Both the original and replicated promoter retain their TF binding
state and associated mass.

**Chromosomal segment boundaries**

Segment boundaries (for supercoiling calculations) are reset after
replication fork passage. The linking number Lk of each segment is
maintained:

    Lk = Tw + Wr

where Tw is the twist and Wr is the writhe. When a segment is split
by a passing fork, the linking number is partitioned proportionally
to the new segment lengths.

**Molecule removal and recycling**

RNAPs and ribosomes that collide with replication forks are removed.
Their associated RNA transcripts and ribosome subunits are recycled
back to the bulk pool, and the corresponding mass adjustments are
applied to maintain conservation.
"""

import numpy as np
import numpy.typing as npt
import warnings
from bigraph_schema.contract import ProcessContract
from v2ecoli.library.ecoli_step import EcoliStep as Step
# Composer removed
# Engine removed
# topology_registry removed
from v2ecoli.library.schema import (
    attrs,
    bulk_name_to_idx,
    get_free_indices,
)
from v2ecoli.library.schema_types import (
    ACTIVE_REPLISOME_ARRAY,
    ORIC_ARRAY,
    CHROMOSOME_DOMAIN_ARRAY,
    ACTIVE_RNAP_ARRAY,
    RNA_ARRAY,
    ACTIVE_RIBOSOME_ARRAY,
    FULL_CHROMOSOME_ARRAY,
    PROMOTER_ARRAY,
    DNAA_BOX_ARRAY,
    GENE_ARRAY,
    CHROMOSOMAL_SEGMENT_ARRAY,
)
# from ecoli.library.json_state import get_state_from_file
from v2ecoli.library.polymerize import buildSequences

# Register default topology for this process, associating it with process name
NAME = "ecoli-chromosome-structure"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "active_replisomes": (
        "unique",
        "active_replisome",
    ),
    "oriCs": (
        "unique",
        "oriC",
    ),
    "chromosome_domains": (
        "unique",
        "chromosome_domain",
    ),
    "active_RNAPs": ("unique", "active_RNAP"),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
    "full_chromosomes": (
        "unique",
        "full_chromosome",
    ),
    "promoters": ("unique", "promoter"),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "genes": ("unique", "gene"),
    "chromosomal_segments": ("unique", "chromosomal_segment"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "chromosome_structure"),
}


class ChromosomeStructure(Step):
    """Chromosome Structure Process"""

    description = (
        "Chromosome Structure — resolve fork collisions, replicate promoters, repartition segments.\n\n"
        "1. Collision: fork passing a bound molecule either removes it (fork wins) or stalls (molecule wins).\n"
        "2. Promoter replication: duplicate promoter when  pos_prev < c ≤ pos_new  (TF state + mass preserved).\n"
        "3. Segment supercoiling: linking number  Lk = Tw + Wr;  on split, Lk partitioned ∝ new segment lengths.\n"
        "4. Recycling: collided RNAPs/ribosomes removed; RNA transcripts + subunits returned to bulk (mass conserved).\n"
        "  c: promoter coordinate; pos_prev/pos_new: fork position before/after step; Tw: twist; Wr: writhe."
    )

    name = NAME
    topology = TOPOLOGY

    contract = ProcessContract(
        summary=(
            "Reconcile the chromosome's structural state with replication-fork motion: resolve "
            "fork/bound-molecule collisions (removing and recycling passed RNAPs, RNAs and "
            "ribosomes) and duplicate promoters, genes and DnaA boxes onto the two daughter "
            "domains as each fork passes them."
        ),
        symbols={
            "c": "genomic coordinate of a promoter/gene/DnaA box (nt from origin)",
            "pos_prev": "fork position at the previous step (nt)",
            "pos_new": "fork position after this step (nt); a motif is passed when pos_prev < c ≤ pos_new",
            "Lk": "linking number of a chromosomal segment (dimensionless)",
            "Tw": "twist component of the linking number (dimensionless)",
            "Wr": "writhe component of the linking number (dimensionless)",
        },
        inputs={
            "bulk": (
                "Reads bulk ids once to resolve indices for every species this step recycles "
                "(inactive RNAPs, mature RNAs, fragment bases, PPi, 30S/50S subunits, amino acids, "
                "water, active TFs, DnaA-ATP/ADP)."
            ),
            "active_replisomes": (
                "Reads each fork's domain_index and coordinates to decide which bound molecules the "
                "forks have passed (the removal window per domain)."
            ),
            "chromosome_domains": (
                "Reads domain_index and child_domains to detect finished domains and to map a passed "
                "motif onto its two child domains."
            ),
            "active_RNAPs": (
                "Reads domain_index, coordinates, is_forward and unique_index to flag RNAPs the forks "
                "passed and classify head-on vs co-directional collisions."
            ),
            "RNAs": (
                "Reads TU_index, transcript_length, RNAP_index, is_full_transcript and unique_index to "
                "find transcripts of collided RNAPs and size the returned fragment bases/PPi/mature RNAs."
            ),
            "active_ribosome": (
                "Reads protein_index, peptide_length and mRNA_index to find ribosomes left on removed or "
                "degraded mRNAs and size the returned amino acids/water."
            ),
            "full_chromosomes": (
                "Reads full-chromosome domain_index to identify domains that finished replication (so all "
                "molecules on them are cleared)."
            ),
            "promoters": (
                "Reads TU_index, domain_index, coordinates and bound_TF to select passed promoters for "
                "duplication and to free their bound TFs."
            ),
            "genes": "Reads cistron_index, domain_index and coordinates to select passed genes for duplication.",
            "DnaA_boxes": (
                "Reads domain_index, coordinates, DnaA_bound, pool_label and DnaA_bound_form to select "
                "passed boxes for duplication and to release any DnaA bound on them."
            ),
            "global_time": "Reads simulation time to gate the step (update_condition) and set the next update time.",
            "timestep": "Reads the step length used to schedule next_update_time.",
            "next_update_time": "Reads the scheduled update time; the step runs only when it has arrived.",
        },
        outputs={
            "bulk": (
                "Returns recycled species to the pool: inactive RNAPs and ribosomal 30S/50S subunits, "
                "fragment bases + PPi + mature RNAs from incomplete transcripts, amino acids + water from "
                "incomplete peptides, freed active TFs, and freed DnaA-ATP/ADP from fork-crossed boxes."
            ),
            "listeners": (
                "Writes rnap_data collision statistics: n_total/head-on/co-directional collisions and their "
                "coordinates, incomplete_transcription_event per TU, and n_removed_ribosomes."
            ),
            "active_RNAPs": "Deletes RNAPs that collided with a passing fork.",
            "RNAs": "Deletes transcripts whose RNAP was removed by a collision.",
            "active_ribosome": "Deletes ribosomes bound to mRNAs that no longer exist.",
            "promoters": "Deletes each passed promoter and adds two daughter copies (one per child domain) with TF binding reset.",
            "genes": "Deletes each passed gene and adds two daughter copies on the child domains.",
            "DnaA_boxes": "Deletes each passed DnaA box and adds two daughter copies (pool_label propagated, DnaA_bound_form reset to 0).",
            "next_update_time": "Writes the next scheduled update time (global_time + timestep).",
        },
        config={
            "active_tfs": "Bulk ids of active TFs freed to the pool when a bound promoter is replicated.",
            "inactive_RNAPs": "Bulk id(s) for inactive RNAP the collided RNAPs are returned to.",
            "ribosome_30S_subunit": "Bulk id of the 30S subunit returned per removed ribosome.",
            "ribosome_50S_subunit": "Bulk id of the 50S subunit returned per removed ribosome.",
            "amino_acids": "Bulk ids of amino acids returned from incomplete polypeptides.",
            "water": "Bulk id of water consumed/released when hydrolyzing incomplete peptides.",
            "fragmentBases": "Bulk ids of the NTP fragment bases returned from incomplete transcripts.",
            "ppi": "Bulk id of pyrophosphate returned per initiated-but-incomplete transcript.",
            "rna_sequences": "RNA template sequences used to reconstruct incomplete-transcript base composition.",
            "protein_sequences": "Protein template sequences used to reconstruct incomplete-peptide amino-acid composition.",
            "rna_ids": "RNA identifiers indexing the transcript arrays.",
            "mature_rna_ids": "Bulk ids of mature RNAs that can be released when an unprocessed transcript is long enough.",
            "mature_rna_end_positions": "Per-mature-RNA end coordinates that decide which mature RNAs a truncated transcript yields.",
            "mature_rna_nt_counts": "Per-mature-RNA nucleotide counts subtracted from returned fragment bases.",
            "unprocessed_rna_index_mapping": "Maps unprocessed-RNA TU indices to their mature-RNA layout.",
            "n_mature_rnas": "Number of mature-RNA species.",
            "n_TUs": "Number of transcription units (sizes the incomplete-transcription-event listener).",
            "n_TFs": "Number of transcription factors (width of promoter bound_TF).",
            "n_amino_acids": "Number of amino-acid species.",
            "n_fragment_bases": "Number of fragment-base species.",
            "replichore_lengths": "Replichore lengths defining the min/max chromosomal coordinates.",
            "relaxed_DNA_base_pairs_per_turn": "Relaxed DNA helical repeat used in the supercoiling/linking-number calculation.",
            "terC_index": "Sentinel molecule index marking the replication terminus in segment bookkeeping.",
            "no_child_place_holder": "Sentinel domain index marking a chromosome domain with no children yet.",
        },
        assumptions=[
            "In this build the fork always wins a collision: every bound molecule the fork passes is removed and recycled; head-on vs co-directional is recorded for listeners only, not differentiated in outcome (no fork stalling).",
            "A molecule is 'passed' when it lies within the fork coordinate window of its domain, or on any domain whose children are already full chromosomes (replication finished).",
            "Passed promoters, genes and DnaA boxes are duplicated onto both child domains; daughter promoters/DnaA boxes reset their binding state and their bound TFs/DnaA are released to bulk (mass conserved).",
            "Incomplete transcripts return their NTs as fragment bases plus one PPi each, excluding any bases/PPi belonging to mature RNAs released from the truncation.",
            "Ribosomes on removed or degraded mRNAs are dropped; their partial peptides are hydrolyzed back to free amino acids and water.",
            "The chromosomal-segment / supercoiling repartition math (Lk = Tw + Wr) is implemented in _compute_new_segment_attributes but is not invoked by update() in this version, so the chromosomal_segments and oriCs ports are read-through only and left out of this contract's ports.",
        ],
        references=[
            "Katayama (2017) — DnaA dissociation from DnaA boxes after replication-fork passage.",
        ],
    )

    config_schema = {
        'active_tfs': 'list[string]',
        'amino_acids': 'list[string]',
        'emit_unique': {'_type': 'boolean{false}', '_default': False},
        'fragmentBases': 'list[string]',
        'inactive_RNAPs': 'list[string]',
        'mature_rna_end_positions': 'list[integer]',
        'mature_rna_ids': 'list[string]',
        'mature_rna_nt_counts': {'_type': 'node', '_default': []},
        'n_TFs': {'_type': 'integer{1}', '_default': 1},
        'n_TUs': {'_type': 'integer{1}', '_default': 1},
        'n_amino_acids': {'_type': 'integer{1}', '_default': 1},
        'n_fragment_bases': {'_type': 'integer{1}', '_default': 1},
        'n_mature_rnas': {'_type': 'integer{0}', '_default': 0},
        'no_child_place_holder': {'_type': 'integer{-1}', '_default': -1},
        'ppi': {'_type': 'string{ppi}', '_default': 'ppi'},
        'protein_sequences': 'list[string]',
        'relaxed_DNA_base_pairs_per_turn': {'_type': 'integer{1}', '_default': 1},
        'replichore_lengths': {'_type': 'list[integer]', '_default': [0, 0]},
        'ribosome_30S_subunit': {'_type': 'string{30S}', '_default': '30S'},
        'ribosome_50S_subunit': {'_type': 'string{50S}', '_default': '50S'},
        'rna_ids': 'list[string]',
        'rna_sequences': 'list[string]',
        'seed': {'_type': 'integer{0}', '_default': 0},
        'terC_index': {'_type': 'integer{-1}', '_default': -1},
        'time_step': {'_type': 'float{1.0}', '_default': 1.0},
        'unprocessed_rna_index_mapping': 'map[integer]',
        'water': {'_type': 'string{water}', '_default': 'water'},
    }


    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'rnap_data': {
                    'active_rnap_n_bound_ribosomes': 'array[integer]',
                },
            },
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'oriCs': {'_type': ORIC_ARRAY, '_default': []},
            'chromosome_domains': {'_type': CHROMOSOME_DOMAIN_ARRAY, '_default': []},
            'active_RNAPs': {'_type': ACTIVE_RNAP_ARRAY, '_default': []},
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
            'full_chromosomes': {'_type': FULL_CHROMOSOME_ARRAY, '_default': []},
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'genes': {'_type': GENE_ARRAY, '_default': []},
            'chromosomal_segments': {'_type': CHROMOSOMAL_SEGMENT_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'integer', '_default': 1.0},
            'next_update_time': {'_type': 'overwrite[float]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'rnap_data': {
                    'active_rnap_n_bound_ribosomes': 'array[integer]',
                },
            },
            'active_replisomes': ACTIVE_REPLISOME_ARRAY,
            'oriCs': ORIC_ARRAY,
            'chromosome_domains': CHROMOSOME_DOMAIN_ARRAY,
            'active_RNAPs': ACTIVE_RNAP_ARRAY,
            'RNAs': RNA_ARRAY,
            'active_ribosome': ACTIVE_RIBOSOME_ARRAY,
            'full_chromosomes': FULL_CHROMOSOME_ARRAY,
            'chromosomal_segments': CHROMOSOMAL_SEGMENT_ARRAY,
            'promoters': PROMOTER_ARRAY,
            'genes': GENE_ARRAY,
            'DnaA_boxes': DNAA_BOX_ARRAY,
            'next_update_time': 'overwrite[float]',
        }


    def initialize(self, config):
        self.rna_sequences = self.parameters["rna_sequences"]
        self.protein_sequences = self.parameters["protein_sequences"]
        self.n_TUs = self.parameters["n_TUs"]
        self.n_TFs = self.parameters["n_TFs"]
        self.rna_ids = self.parameters["rna_ids"]
        self.n_amino_acids = self.parameters["n_amino_acids"]
        self.n_fragment_bases = self.parameters["n_fragment_bases"]
        replichore_lengths = self.parameters["replichore_lengths"]
        self.min_coordinates = -replichore_lengths[1]
        self.max_coordinates = replichore_lengths[0]
        self.relaxed_DNA_base_pairs_per_turn = self.parameters[
            "relaxed_DNA_base_pairs_per_turn"
        ]
        self.terC_index = self.parameters["terC_index"]

        self.n_mature_rnas = self.parameters["n_mature_rnas"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.mature_rna_end_positions = self.parameters["mature_rna_end_positions"]
        self.mature_rna_nt_counts = self.parameters["mature_rna_nt_counts"]
        self.unprocessed_rna_index_mapping = self.parameters[
            "unprocessed_rna_index_mapping"
        ]

        # Get placeholder value for chromosome domains without children
        self.no_child_place_holder = self.parameters["no_child_place_holder"]

        self.inactive_RNAPs = self.parameters["inactive_RNAPs"]
        self.fragmentBases = self.parameters["fragmentBases"]
        self.ppi = self.parameters["ppi"]
        self.active_tfs = self.parameters["active_tfs"]
        self.ribosome_30S_subunit = self.parameters["ribosome_30S_subunit"]
        self.ribosome_50S_subunit = self.parameters["ribosome_50S_subunit"]
        self.amino_acids = self.parameters["amino_acids"]
        self.water = self.parameters["water"]

        self.inactive_RNAPs_idx = None

        self.emit_unique = self.parameters.get("emit_unique", True)
    def update_condition(self, timestep, states):
        """
        See :py:meth:`~ecoli.processes.partition.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def _init_bulk_indices(self, bulk_ids):
        """Resolve every molecule-name -> bulk-array-index map this Step uses.
        Called once (guarded by ``self.inactive_RNAPs_idx is None``), so it
        never runs on the per-tick path."""
        self.fragmentBasesIdx = bulk_name_to_idx(self.fragmentBases, bulk_ids)
        self.active_tfs_idx = bulk_name_to_idx(self.active_tfs, bulk_ids)
        self.ribosome_30S_subunit_idx = bulk_name_to_idx(
            self.ribosome_30S_subunit, bulk_ids)
        self.ribosome_50S_subunit_idx = bulk_name_to_idx(
            self.ribosome_50S_subunit, bulk_ids)
        self.amino_acids_idx = bulk_name_to_idx(self.amino_acids, bulk_ids)
        self.water_idx = bulk_name_to_idx(self.water, bulk_ids)
        self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
        self.inactive_RNAPs_idx = bulk_name_to_idx(self.inactive_RNAPs, bulk_ids)
        self.mature_rna_idx = bulk_name_to_idx(self.mature_rna_ids, bulk_ids)
        # dnaa-3: on fork passage, any DnaA bound on the parent DnaA box is
        # released back to bulk so mass + counts conserve. Phase 1 keeps
        # DnaA_bound_form at 0 (no binding step wired yet) so these reads are
        # no-ops; Phase 2 will exercise them.
        self.dnaa_atp_idx = bulk_name_to_idx(["MONOMER0-160[c]"], bulk_ids)[0]
        self.dnaa_adp_idx = bulk_name_to_idx(["MONOMER0-4565[c]"], bulk_ids)[0]

    def _removed_molecules_mask(self, domain_indexes, coordinates,
                                replisome_coords_by_domain, child_domains,
                                all_domain_indexes, mother_domain_indexes):
        """Boolean mask of unique molecules the replication forks have passed
        (and so must be removed from the mother strand).

        For each chromosome domain:
          * if it carries active replisomes, a molecule is removed when its
            coordinate lies within [min, max] of those replisome coordinates
            (>=/<= so molecules exactly at a fork are also removed);
          * if it has no replisomes but its children are full chromosomes
            (replication finished), every molecule on it is removed;
          * otherwise (un-replicated / interrupted) nothing is removed.
        """
        mask = np.zeros_like(domain_indexes, dtype=np.bool_)
        for domain_index in np.unique(domain_indexes):
            if domain_index in replisome_coords_by_domain:
                domain_replisome_coordinates = replisome_coords_by_domain[
                    domain_index]
                domain_mask = np.logical_and.reduce((
                    domain_indexes == domain_index,
                    coordinates >= domain_replisome_coordinates.min(),
                    coordinates <= domain_replisome_coordinates.max(),
                ))
            else:
                children_of_domain = child_domains[
                    all_domain_indexes == domain_index]
                if np.all(np.isin(children_of_domain, mother_domain_indexes)):
                    domain_mask = domain_indexes == domain_index
                else:
                    continue
            mask[domain_mask] = True
        return mask

    def _replicated_motif_attributes(self, old_coordinates, old_domain_indexes,
                                     child_domains, all_domain_indexes):
        """Attributes for a chromosomal motif (promoter/gene/DnaA box) after a
        replication fork passes it: the motif is duplicated onto the two child
        domains. Coordinates are repeated; each domain index is replaced by its
        two child-domain indexes."""
        new_coordinates = np.repeat(old_coordinates, 2)
        new_domain_indexes = child_domains[
            np.array([
                np.where(all_domain_indexes == idx)[0][0]
                for idx in old_domain_indexes
            ]),
            :,
        ].flatten()
        return new_coordinates, new_domain_indexes

    def update(self, states, interval=None):
        # At t=0, resolve molecule-name -> bulk-index maps (runs once).
        if self.inactive_RNAPs_idx is None:
            self._init_bulk_indices(states["bulk"]["id"])

        # Read unique molecule attributes
        (replisome_domain_indexes, replisome_coordinates, replisome_unique_indexes) = (
            attrs(
                states["active_replisomes"],
                ["domain_index", "coordinates", "unique_index"],
            )
        )
        (all_chromosome_domain_indexes, child_domains) = attrs(
            states["chromosome_domains"], ["domain_index", "child_domains"]
        )
        (
            RNAP_domain_indexes,
            RNAP_coordinates,
            RNAP_is_forward,
            RNAP_unique_indexes,
        ) = attrs(
            states["active_RNAPs"],
            ["domain_index", "coordinates", "is_forward", "unique_index"],
        )
        (origin_domain_indexes,) = attrs(states["oriCs"], ["domain_index"])
        (mother_domain_indexes,) = attrs(states["full_chromosomes"], ["domain_index"])
        (
            RNA_TU_indexes,
            transcript_lengths,
            RNA_RNAP_indexes,
            RNA_full_transcript,
            RNA_unique_indexes,
        ) = attrs(
            states["RNAs"],
            [
                "TU_index",
                "transcript_length",
                "RNAP_index",
                "is_full_transcript",
                "unique_index",
            ],
        )
        (ribosome_protein_indexes, ribosome_peptide_lengths, ribosome_mRNA_indexes) = (
            attrs(
                states["active_ribosome"],
                ["protein_index", "peptide_length", "mRNA_index"],
            )
        )
        (
            promoter_TU_indexes,
            promoter_domain_indexes,
            promoter_coordinates,
            promoter_bound_TFs,
        ) = attrs(
            states["promoters"], ["TU_index", "domain_index", "coordinates", "bound_TF"]
        )
        (gene_cistron_indexes, gene_domain_indexes, gene_coordinates) = attrs(
            states["genes"], ["cistron_index", "domain_index", "coordinates"]
        )
        (DnaA_box_domain_indexes, DnaA_box_coordinates, DnaA_box_bound,
         DnaA_box_pool_label, DnaA_box_bound_form) = attrs(
            states["DnaA_boxes"],
            ["domain_index", "coordinates", "DnaA_bound", "pool_label",
             "DnaA_bound_form"],
        )

        # Build dictionary of replisome coordinates with domain indexes as keys
        replisome_coordinates_from_domains = {
            domain_index: replisome_coordinates[
                replisome_domain_indexes == domain_index
            ]
            for domain_index in np.unique(replisome_domain_indexes)
        }

        # Build masks for molecules the replication forks have passed (removed
        # from the mother strand) — see _removed_molecules_mask.
        _rm_args = (replisome_coordinates_from_domains, child_domains,
                    all_chromosome_domain_indexes, mother_domain_indexes)
        removed_RNAPs_mask = self._removed_molecules_mask(
            RNAP_domain_indexes, RNAP_coordinates, *_rm_args)
        removed_promoters_mask = self._removed_molecules_mask(
            promoter_domain_indexes, promoter_coordinates, *_rm_args)
        removed_genes_mask = self._removed_molecules_mask(
            gene_domain_indexes, gene_coordinates, *_rm_args)
        removed_DnaA_boxes_mask = self._removed_molecules_mask(
            DnaA_box_domain_indexes, DnaA_box_coordinates, *_rm_args)

        # Build masks for head-on and co-directional collisions between RNAPs
        # and replication forks
        RNAP_headon_collision_mask = np.logical_and(
            removed_RNAPs_mask, np.logical_xor(RNAP_is_forward, RNAP_coordinates > 0)
        )
        RNAP_codirectional_collision_mask = np.logical_and(
            removed_RNAPs_mask, np.logical_not(RNAP_headon_collision_mask)
        )

        n_total_collisions = np.count_nonzero(removed_RNAPs_mask)
        n_headon_collisions = np.count_nonzero(RNAP_headon_collision_mask)
        n_codirectional_collisions = np.count_nonzero(RNAP_codirectional_collision_mask)

        # Write values to listeners
        update = {
            "listeners": {
                "rnap_data": {
                    "n_total_collisions": n_total_collisions,
                    "n_headon_collisions": n_headon_collisions,
                    "n_codirectional_collisions": n_codirectional_collisions,
                    "headon_collision_coordinates": RNAP_coordinates[
                        RNAP_headon_collision_mask
                    ],
                    "codirectional_collision_coordinates": RNAP_coordinates[
                        RNAP_codirectional_collision_mask
                    ],
                }
            },
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "active_RNAPs": {},
            "RNAs": {},
            "active_ribosome": {},
            "full_chromosomes": {},
            "chromosomal_segments": {},
            "promoters": {},
            "genes": {},
            "DnaA_boxes": {},
        }

        # Get mask for RNAs that are transcribed from removed RNAPs
        removed_RNAs_mask = np.isin(
            RNA_RNAP_indexes, RNAP_unique_indexes[removed_RNAPs_mask]
        )

        # Initialize counts of incomplete transcription events
        incomplete_transcription_event = np.zeros(self.n_TUs)

        # Remove RNAPs and RNAs that have collided with replisomes
        if n_total_collisions > 0:
            if removed_RNAPs_mask.sum() > 0:
                update["active_RNAPs"].update(
                    {"delete": np.where(removed_RNAPs_mask)[0]}
                )
            if removed_RNAs_mask.sum() > 0:
                update["RNAs"].update({"delete": np.where(removed_RNAs_mask)[0]})

            # Increment counts of inactive RNAPs
            update["bulk"].append((self.inactive_RNAPs_idx, n_total_collisions))

            # Get sequences of incomplete transcripts
            incomplete_sequence_lengths = transcript_lengths[removed_RNAs_mask]
            # Under resource-limited conditions, some transcripts may be
            # initiated but not elongated (zero length). Include them in the count.
            n_initiated_sequences = (~RNA_full_transcript[removed_RNAs_mask]).sum()
            n_ppi_added = n_initiated_sequences

            if n_initiated_sequences > 0:
                incomplete_rna_indexes = RNA_TU_indexes[removed_RNAs_mask]
                incomplete_transcription_event = np.bincount(
                    incomplete_rna_indexes, minlength=self.n_TUs
                )

                incomplete_sequences = buildSequences(
                    self.rna_sequences,
                    incomplete_rna_indexes,
                    np.zeros(n_total_collisions, dtype=np.int64),
                    np.full(n_total_collisions, incomplete_sequence_lengths.max()),
                )

                mature_rna_counts = np.zeros(self.n_mature_rnas, dtype=np.int64)
                base_counts = np.zeros(self.n_fragment_bases, dtype=np.int64)

                for ri, sl, seq in zip(
                    incomplete_rna_indexes,
                    incomplete_sequence_lengths,
                    incomplete_sequences,
                ):
                    # Check if incomplete RNA is an unprocessed RNA
                    if ri in self.unprocessed_rna_index_mapping:
                        # Find mature RNA molecules that would need to be added
                        # given the length of the incomplete RNA
                        mature_rna_end_pos = self.mature_rna_end_positions[
                            :, self.unprocessed_rna_index_mapping[ri]
                        ]
                        mature_rnas_produced = np.logical_and(
                            mature_rna_end_pos != 0, mature_rna_end_pos < sl
                        )

                        # Increment counts of mature RNAs
                        mature_rna_counts += mature_rnas_produced

                        # Increment counts of fragment NTPs, but exclude bases
                        # that are part of the mature RNAs generated
                        base_counts += np.bincount(
                            seq[:sl], minlength=self.n_fragment_bases
                        ) - self.mature_rna_nt_counts[mature_rnas_produced, :].sum(
                            axis=0
                        )

                        # Exclude ppi molecules that are part of mature RNAs
                        n_ppi_added -= mature_rnas_produced.sum()
                    else:
                        base_counts += np.bincount(
                            seq[:sl], minlength=self.n_fragment_bases
                        )

                # Increment counts of mature RNAs, fragment NTPs and phosphates
                update["bulk"].append((self.mature_rna_idx, mature_rna_counts))
                update["bulk"].append((self.fragmentBasesIdx, base_counts))
                update["bulk"].append((self.ppi_idx, n_ppi_added))

            assert n_initiated_sequences == incomplete_transcription_event.sum()

        update["listeners"]["rnap_data"]["incomplete_transcription_event"] = (
            incomplete_transcription_event
        )

        # Get mask for ribosomes that are bound to nonexisting mRNAs
        remaining_RNA_unique_indexes = RNA_unique_indexes[
            np.logical_not(removed_RNAs_mask)
        ]
        removed_ribosomes_mask = np.logical_not(
            np.isin(ribosome_mRNA_indexes, remaining_RNA_unique_indexes)
        )
        n_removed_ribosomes = np.count_nonzero(removed_ribosomes_mask)

        # Remove ribosomes that are bound to missing RNA molecules. This
        # includes both RNAs removed by this function and RNAs removed
        # by other processes (e.g. RNA degradation).
        if n_removed_ribosomes > 0:
            update["active_ribosome"].update(
                {"delete": np.where(removed_ribosomes_mask)[0]}
            )

            # Increment counts of inactive ribosomal subunits
            update["bulk"].extend(
                [
                    (self.ribosome_30S_subunit_idx, n_removed_ribosomes),
                    (self.ribosome_50S_subunit_idx, n_removed_ribosomes),
                ]
            )

            # Get amino acid sequences of incomplete polypeptides
            incomplete_sequence_lengths = ribosome_peptide_lengths[
                removed_ribosomes_mask
            ]
            n_initiated_sequences = np.count_nonzero(incomplete_sequence_lengths)

            if n_initiated_sequences > 0:
                incomplete_sequences = buildSequences(
                    self.protein_sequences,
                    ribosome_protein_indexes[removed_ribosomes_mask],
                    np.zeros(n_removed_ribosomes, dtype=np.int64),
                    np.full(n_removed_ribosomes, incomplete_sequence_lengths.max()),
                )

                amino_acid_counts = np.zeros(self.n_amino_acids, dtype=np.int64)

                for sl, seq in zip(incomplete_sequence_lengths, incomplete_sequences):
                    amino_acid_counts += np.bincount(
                        seq[:sl], minlength=self.n_amino_acids
                    )

                # Increment counts of free amino acids and decrease counts of
                # free water molecules
                update["bulk"].append((self.amino_acids_idx, amino_acid_counts))
                update["bulk"].append(
                    (
                        self.water_idx,
                        (n_initiated_sequences - incomplete_sequence_lengths.sum()),
                    )
                )

        # Write to listener
        update["listeners"]["rnap_data"]["n_removed_ribosomes"] = n_removed_ribosomes

        # On fork passage, promoters/genes/DnaA boxes are duplicated onto the
        # two child domains — see _replicated_motif_attributes.

        #######################
        # Replicate promoters #
        #######################
        n_new_promoters = 2 * np.count_nonzero(removed_promoters_mask)

        if n_new_promoters > 0:
            # Delete original promoters
            update["promoters"].update({"delete": np.where(removed_promoters_mask)[0]})

            # Add freed active tfs
            update["bulk"].append(
                (
                    self.active_tfs_idx,
                    promoter_bound_TFs[removed_promoters_mask, :].sum(axis=0),
                )
            )

            # Set up attributes for the replicated promoters
            promoter_TU_indexes_new = np.repeat(
                promoter_TU_indexes[removed_promoters_mask], 2
            )
            (promoter_coordinates_new, promoter_domain_indexes_new) = (
                self._replicated_motif_attributes(
                    promoter_coordinates[removed_promoters_mask],
                    promoter_domain_indexes[removed_promoters_mask],
                    child_domains, all_chromosome_domain_indexes,
                )
            )

            # Add new promoters with new domain indexes
            update["promoters"].update(
                {
                    "add": {
                        "TU_index": promoter_TU_indexes_new,
                        "coordinates": promoter_coordinates_new,
                        "domain_index": promoter_domain_indexes_new,
                        "bound_TF": np.zeros(
                            (n_new_promoters, self.n_TFs), dtype=np.bool_
                        ),
                    }
                }
            )

        # Replicate genes
        n_new_genes = 2 * np.count_nonzero(removed_genes_mask)

        if n_new_genes > 0:
            # Delete original genes
            update["genes"].update({"delete": np.where(removed_genes_mask)[0]})

            # Set up attributes for the replicated genes
            gene_cistron_indexes_new = np.repeat(
                gene_cistron_indexes[removed_genes_mask], 2
            )
            gene_coordinates_new, gene_domain_indexes_new = (
                self._replicated_motif_attributes(
                    gene_coordinates[removed_genes_mask],
                    gene_domain_indexes[removed_genes_mask],
                    child_domains, all_chromosome_domain_indexes,
                )
            )

            # Add new genes with new domain indexes
            update["genes"].update(
                {
                    "add": {
                        "cistron_index": gene_cistron_indexes_new,
                        "coordinates": gene_coordinates_new,
                        "domain_index": gene_domain_indexes_new,
                    }
                }
            )

        ########################
        # Replicate DnaA boxes #
        ########################
        n_new_DnaA_boxes = 2 * np.count_nonzero(removed_DnaA_boxes_mask)

        if n_new_DnaA_boxes > 0:
            # dnaa-3 Phase 2b: release DnaA bound to fork-crossed boxes back to
            # bulk. Without this, the bound DnaA-ATP / DnaA-ADP would be silently
            # destroyed when the parent box is deleted (the new child boxes are
            # added with DnaA_bound_form=0). Releasing to bulk both conserves
            # mass and produces the biologically-expected dissociation pulse:
            # the replication fork unwinds dsDNA → canonical DnaA-box recognition
            # is disrupted → DnaA returns to the cytoplasm (see Katayama 2017,
            # "DnaA dissociation from DnaA boxes after replication" section).
            removed_bound_form = DnaA_box_bound_form[removed_DnaA_boxes_mask]
            n_freed_atp = int(np.count_nonzero(removed_bound_form == 1))
            n_freed_adp = int(np.count_nonzero(removed_bound_form == 2))
            if n_freed_atp > 0:
                update["bulk"].append((self.dnaa_atp_idx, n_freed_atp))
            if n_freed_adp > 0:
                update["bulk"].append((self.dnaa_adp_idx, n_freed_adp))

            # Delete original DnaA boxes
            if removed_DnaA_boxes_mask.sum() > 0:
                update["DnaA_boxes"].update(
                    {"delete": np.where(removed_DnaA_boxes_mask)[0]}
                )

            # Set up attributes for the replicated boxes
            (DnaA_box_coordinates_new, DnaA_box_domain_indexes_new) = (
                self._replicated_motif_attributes(
                    DnaA_box_coordinates[removed_DnaA_boxes_mask],
                    DnaA_box_domain_indexes[removed_DnaA_boxes_mask],
                    child_domains, all_chromosome_domain_indexes,
                )
            )

            # dnaa-3: propagate pool_label from the parent box to both child
            # copies (np.repeat aligns with the x2 expansion done by
            # _replicated_motif_attributes). DnaA_bound_form resets to 0 on
            # new copies; Phase 2's binding step will write it once wired.
            DnaA_box_pool_label_new = np.repeat(
                DnaA_box_pool_label[removed_DnaA_boxes_mask], 2
            )

            # Add new DnaA boxes with new domain indexes
            dict_dna = {
                "add": {
                    "coordinates": DnaA_box_coordinates_new,
                    "domain_index": DnaA_box_domain_indexes_new,
                    "DnaA_bound": np.zeros(n_new_DnaA_boxes, dtype=np.bool_),
                    "pool_label": DnaA_box_pool_label_new,
                    "DnaA_bound_form": np.zeros(n_new_DnaA_boxes, dtype=np.int8),
                }
            }
            update["DnaA_boxes"].update(dict_dna)

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update

    def _compute_new_segment_attributes(
        self,
        old_boundary_molecule_indexes: npt.NDArray[np.int64],
        old_boundary_coordinates: npt.NDArray[np.int64],
        old_linking_numbers: npt.NDArray[np.int64],
        new_molecule_indexes: npt.NDArray[np.int64],
        new_molecule_coordinates: npt.NDArray[np.int64],
        spans_oriC: bool,
        spans_terC: bool,
    ) -> dict[str, npt.NDArray[np.int64 | np.float64]]:
        """
        Calculates the updated attributes of chromosomal segments belonging to
        a specific chromosomal domain, given the previous and current
        coordinates of molecules bound to the chromosome.

        Args:
            old_boundary_molecule_indexes: (N, 2) array of unique
                indexes of molecules that formed the boundaries of each
                chromosomal segment in the previous timestep.
            old_boundary_coordinates: (N, 2) array of chromosomal
                coordinates of molecules that formed the boundaries of each
                chromosomal segment in the previous timestep.
            old_linking_numbers: (N,) array of linking numbers of each
                chromosomal segment in the previous timestep.
            new_molecule_indexes: (N,) array of unique indexes of all
                molecules bound to the domain at the current timestep.
            new_molecule_coordinates: (N,) array of chromosomal
                coordinates of all molecules bound to the domain at the current
                timestep.
            spans_oriC: True if the domain spans the origin.
            spans_terC: True if the domain spans the terminus.

        Returns:
            Dictionary of the following format::

                {
                    'boundary_molecule_indexes': (M, 2) array of unique
                        indexes of molecules that form the boundaries of new
                        chromosomal segments,
                    'boundary_coordinates': (M, 2) array of chromosomal
                        coordinates of molecules that form the boundaries of
                        new chromosomal segments,
                    'linking_numbers': (M,) array of linking numbers of new
                        chromosomal segments
                }

        """
        # Sort old segment arrays by coordinates of left boundary
        old_coordinates_argsort = np.argsort(old_boundary_coordinates[:, 0])
        old_boundary_coordinates_sorted = old_boundary_coordinates[
            old_coordinates_argsort, :
        ]
        old_boundary_molecule_indexes_sorted = old_boundary_molecule_indexes[
            old_coordinates_argsort, :
        ]
        old_linking_numbers_sorted = old_linking_numbers[old_coordinates_argsort]

        # Sort new segment arrays by molecular coordinates
        new_coordinates_argsort = np.argsort(new_molecule_coordinates)
        new_molecule_coordinates_sorted = new_molecule_coordinates[
            new_coordinates_argsort
        ]
        new_molecule_indexes_sorted = new_molecule_indexes[new_coordinates_argsort]

        # Domain does not span the origin
        if not spans_oriC:
            # A fragment spans oriC if two boundaries have opposite signs,
            # or both are equal to zero
            oriC_fragment_counts = np.count_nonzero(
                np.logical_not(
                    np.logical_xor(
                        old_boundary_coordinates_sorted[:, 0] < 0,
                        old_boundary_coordinates_sorted[:, 1] > 0,
                    )
                )
            )

            # if oriC fragment did not exist in the domain in the previous
            # timestep, add a dummy fragment that covers the origin with
            # linking number zero. This is done to generalize the
            # implementation of this method.
            if oriC_fragment_counts == 0:
                # Index of first segment where left boundary is nonnegative
                oriC_fragment_index = np.argmax(
                    old_boundary_coordinates_sorted[:, 0] >= 0
                )

                # Get indexes of boundary molecules for this dummy segment
                oriC_fragment_boundary_molecule_indexes = np.array(
                    [
                        old_boundary_molecule_indexes_sorted[
                            oriC_fragment_index - 1, 1
                        ],
                        old_boundary_molecule_indexes_sorted[oriC_fragment_index, 0],
                    ]
                )

                # Insert dummy segment to array
                old_boundary_molecule_indexes_sorted = np.insert(
                    old_boundary_molecule_indexes_sorted,
                    oriC_fragment_index,
                    oriC_fragment_boundary_molecule_indexes,
                    axis=0,
                )
                old_linking_numbers_sorted = np.insert(
                    old_linking_numbers_sorted, oriC_fragment_index, 0
                )
            else:
                # There should not be more than one fragment that spans oriC
                assert oriC_fragment_counts == 1

        # Domain spans the terminus
        if spans_terC:
            # If the domain spans the terminus, dummy molecules are added to
            # each end of the chromosome s.t. the segment that spans terC is
            # split to two segments and we can maintain a linear representation
            # for the circular chromosome. These two segments are later
            # adjusted to have the same superhelical densities.
            new_molecule_coordinates_sorted = np.insert(
                new_molecule_coordinates_sorted,
                [0, len(new_molecule_coordinates_sorted)],
                [self.min_coordinates, self.max_coordinates],
            )

            new_molecule_indexes_sorted = np.insert(
                new_molecule_indexes_sorted,
                [0, len(new_molecule_indexes_sorted)],
                self.terC_index,
            )

            # Add dummy molecule to old segments if they do not already exist
            if old_boundary_molecule_indexes_sorted[0, 0] != self.terC_index:
                old_boundary_molecule_indexes_sorted = np.vstack(
                    (
                        np.array(
                            [
                                self.terC_index,
                                old_boundary_molecule_indexes_sorted[0, 0],
                            ]
                        ),
                        old_boundary_molecule_indexes_sorted,
                        np.array(
                            [
                                old_boundary_molecule_indexes_sorted[-1, 1],
                                self.terC_index,
                            ]
                        ),
                    )
                )
                old_linking_numbers_sorted = np.insert(
                    old_linking_numbers_sorted, [0, len(old_linking_numbers_sorted)], 0
                )

        # Recalculate linking numbers of each segment after accounting for
        # boundary molecules that were removed in the current timestep
        linking_numbers_after_removal = []
        right_boundaries_retained = np.isin(
            old_boundary_molecule_indexes_sorted[:, 1], new_molecule_indexes_sorted
        )

        # Add up linking numbers of each segment until each retained boundary
        ln_this_fragment = 0.0
        for retained, ln in zip(right_boundaries_retained, old_linking_numbers_sorted):
            ln_this_fragment += ln

            if retained:
                linking_numbers_after_removal.append(ln_this_fragment)
                ln_this_fragment = 0.0

        # Number of segments should be equal to number of retained boundaries
        assert len(linking_numbers_after_removal) == right_boundaries_retained.sum()

        # Redistribute linking numbers of the two terC segments such that the
        # segments have same superhelical densities
        if spans_terC and np.count_nonzero(right_boundaries_retained) > 1:
            # Get molecule indexes of the boundaries of the two terC segments
            # left and right of terC
            retained_boundary_indexes = np.where(right_boundaries_retained)[0]
            left_segment_boundary_index = old_boundary_molecule_indexes_sorted[
                retained_boundary_indexes[0], 1
            ]
            right_segment_boundary_index = old_boundary_molecule_indexes_sorted[
                retained_boundary_indexes[-2], 1
            ]

            # Get mapping from molecule index to chromosomal coordinates
            molecule_index_to_coordinates = {
                index: coordinates
                for index, coordinates in zip(
                    new_molecule_indexes_sorted, new_molecule_coordinates_sorted
                )
            }

            # Distribute linking number between two segments proportional to
            # the length of each segment
            left_segment_length = (
                molecule_index_to_coordinates[left_segment_boundary_index]
                - self.min_coordinates
            )
            right_segment_length = (
                self.max_coordinates
                - molecule_index_to_coordinates[right_segment_boundary_index]
            )
            full_segment_length = left_segment_length + right_segment_length
            full_linking_number = (
                linking_numbers_after_removal[0] + linking_numbers_after_removal[-1]
            )

            linking_numbers_after_removal[0] = (
                full_linking_number * left_segment_length / full_segment_length
            )
            linking_numbers_after_removal[-1] = (
                full_linking_number * right_segment_length / full_segment_length
            )

        # Get mask for molecules that already existed in the previous timestep
        existing_molecules_mask = np.isin(
            new_molecule_indexes_sorted, old_boundary_molecule_indexes_sorted
        )

        # Get numbers and lengths of new segments that each segment will be
        # split into
        segment_split_sizes = np.diff(np.where(existing_molecules_mask)[0])
        segment_lengths = np.diff(new_molecule_coordinates_sorted)

        assert len(segment_split_sizes) == len(linking_numbers_after_removal)

        # Calculate linking numbers of each segment after accounting for new
        # boundaries that were added
        new_linking_numbers = []
        i = 0

        for ln, size in zip(linking_numbers_after_removal, segment_split_sizes):
            if size == 1:
                new_linking_numbers.append(ln)
            else:
                # Split linking number proportional to length of segment
                total_length = segment_lengths[i : i + size].sum()
                new_linking_numbers.extend(
                    list(ln * segment_lengths[i : i + size] / total_length)
                )
            i += size

        # Handle edge case where a domain was just initialized, and two
        # replisomes are bound to the origin
        if len(new_linking_numbers) == 0:
            new_linking_numbers = [np.float64(0)]

        # Build Mx2 array for boundary indexes and coordinates
        new_boundary_molecule_indexes = np.hstack(
            (
                new_molecule_indexes_sorted[:-1, np.newaxis],
                new_molecule_indexes_sorted[1:, np.newaxis],
            )
        )
        new_boundary_coordinates = np.hstack(
            (
                new_molecule_coordinates_sorted[:-1, np.newaxis],
                new_molecule_coordinates_sorted[1:, np.newaxis],
            )
        )
        new_linking_numbers = np.array(new_linking_numbers)

        # If domain does not span oriC, remove new segment that spans origin
        if not spans_oriC:
            oriC_fragment_mask = np.logical_not(
                np.logical_xor(
                    new_boundary_coordinates[:, 0] < 0,
                    new_boundary_coordinates[:, 1] > 0,
                )
            )

            assert oriC_fragment_mask.sum() == 1

            new_boundary_molecule_indexes = new_boundary_molecule_indexes[
                np.logical_not(oriC_fragment_mask), :
            ]
            new_boundary_coordinates = new_boundary_coordinates[
                np.logical_not(oriC_fragment_mask), :
            ]
            new_linking_numbers = new_linking_numbers[
                np.logical_not(oriC_fragment_mask)
            ]

        return {
            "boundary_molecule_indexes": new_boundary_molecule_indexes,
            "boundary_coordinates": new_boundary_coordinates,
            "linking_numbers": new_linking_numbers,
        }
