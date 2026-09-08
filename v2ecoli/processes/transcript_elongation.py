"""
=====================
Transcript Elongation
=====================

This process models nucleotide polymerization into RNA molecules by RNA
polymerases (RNAPs).

Mathematical Model
------------------
**NTP polymerization**

Each active RNAP extends its transcript by consuming NTPs (ATP, CTP,
GTP, UTP) according to the template DNA sequence. The ``polymerize``
algorithm allocates limited NTP pools across all active RNAPs to
maximize total elongation:

    sequences = buildSequences(rna_seqs, RNAP_positions, v_elong * dt)
    result = polymerize(sequences, ntp_counts, rate_limit)

Each polymerization step consumes one NTP and releases one PPi:

    NTP  -->  NMP_incorporated + PPi

Mass increase per RNAP:

    delta_mass = sum(elongated_nt_i * weight_i)   [fg]

**Termination**

An RNAP terminates when its transcript length reaches the annotated
TU length. Completed RNAs are marked as full transcripts.

**tRNA attenuation** (optional)

For attenuated tRNA operons, RNAPs can stochastically terminate early
based on the concentration of charged tRNA. The stop probability for
each attenuated RNA depends on the (absolute) concentration of its
cognate charged tRNA:

    p_stop = f([charged_tRNA])

where f is a fitted attenuation function. RNAPs that stop early are
recycled and their partial transcripts are discarded.
"""

from os import makedirs
import numpy as np
from copy import deepcopy
import warnings

# Engine removed

from wholecell.utils.random import stochasticRound
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity

from v2ecoli.library.schema import (
    counts,
    attrs,
    bulk_name_to_idx,
)
from v2ecoli.library.data_predicates import monotonically_increasing
# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess
from ecoli.processes.unique_update import UniqueUpdate
from v2ecoli.library.schema_types import RNA_ARRAY, ACTIVE_RNAP_ARRAY
from bigraph_schema.contract import ProcessContract


# Register default topology for this process, associating it with process name
NAME = "ecoli-transcript-elongation"
TOPOLOGY = {
    "environment": ("environment",),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "attenuation_config": ("attenuation_config",),
}


def make_elongation_rates(random, rates, timestep, variable):
    return rates


def get_attenuation_stop_probabilities(trna_conc):
    return np.array([])


class TranscriptElongation(PartitionedProcess):
    """Transcript Elongation PartitionedProcess

    defaults:
        - rnaPolymeraseElongationRateDict (dict): Array with elongation rate
            set points for different media environments.
        - rnaIds (array[str]) : array of names for each TU
        - rnaLengths (array[int]) : array of lengths for each TU
            (in nucleotides?)
        - rnaSequences (2D array[int]) : Array with the nucleotide sequences
            of each TU. This is in the form of a 2D array where each row is a
            TU, and each column is a position in the TU's sequence. Nucleotides
            are stored as an index {0, 1, 2, 3}, and the row is padded with
            -1's on the right to indicate where the sequence ends.
        - ntWeights (array[float]): Array of nucleotide weights
        - endWeight (array[float]): ???,
        - replichore_lengths (array[int]): lengths of replichores
            (in nucleotides?),
        - is_mRNA (array[bool]): Mask for mRNAs
        - ppi (str): ID of PPI
        - inactive_RNAP (str): ID of inactive RNAP
        - ntp_ids list[str]: IDs of ntp's (A, C, G, U)
        - variable_elongation (bool): Whether to use variable elongation.
                                      False by default.
        - make_elongation_rates: Function to make elongation rates, of the
            form: lambda random, rates, timestep, variable: rates
    """

    description = (
        "Transcript Elongation — RNAPs extend transcripts by NTP polymerization.\n\n"
        "    seqs = buildSequences(rnaSeqs, RNAP_pos, ν·dt);  Δlen = polymerize(seqs, NTP, limit)\n"
        "    NTP → NMP_incorporated + PPi;   Δmass = ∑ᵢ Δntᵢ·wᵢ  [fg]\n"
        "  Termination: transcript len = TU_len → full transcript; RNAP recycled.\n"
        "  tRNA attenuation (optional): p_stop = f([charged_tRNA]),\n"
        "    early-terminated RNAPs recycled and partial transcripts discarded.\n"
        "  ν = elongation rate (nt/s); dt = timestep; wᵢ = nucleotide weights."
    )

    contract = ProcessContract(
        summary=(
            "PartitionedProcess: calculate_request sizes the NTP request from the "
            "sequences each active RNAP could transcribe this tick; evolve_state "
            "polymerizes with the allocated NTPs, advances RNAP coordinates, "
            "terminates finished transcripts, and (optionally) attenuates tRNA operons."
        ),
        inputs={
            'environment': (
                "reads media_id to select the media-dependent RNAP elongation rate "
                "ν used when building the transcribable sequences."
            ),
            'RNAs': (
                "reads TU_index, transcript_length and is_full_transcript of each "
                "partial RNA to build the sequences to elongate this tick."
            ),
            'active_RNAPs': (
                "reads each RNAP's coordinates, direction and unique_index to map "
                "RNAPs to their partial transcripts and advance their positions; an "
                "empty set short-circuits the tick."
            ),
            'bulk': (
                "reads available NTP (ATP/CTP/GTP/UTP) counts to bound the request "
                "(calculate_request) and the polymerization reaction limit (evolve_state)."
            ),
            'bulk_total': (
                "reads total charged-tRNA counts (for attenuation) to convert to "
                "concentration when computing stop probabilities."
            ),
            'listeners': (
                "reads listeners.mass.cell_mass to form the counts→molar factor for "
                "the tRNA-attenuation stop-probability calculation."
            ),
            'attenuation_config': (
                "reads whether tRNA attenuation is enabled and its stop-probability "
                "function / attenuated-RNA indices / location lookup."
            ),
        },
        outputs={
            'bulk': (
                "consumes NTPs, releases PPi (net = elongations − initiations, since the initiating nucleotide of each newly started transcript retains its 5′ triphosphate), returns "
                "inactive RNAP for terminated/attenuated/stalled RNAPs, and adds "
                "completed non-mRNA transcripts (and fragment bases if recycling stalls)."
            ),
            'RNAs': (
                "sets updated transcript_length and mass submasses, marks completed "
                "transcripts as full, and deletes terminated non-mRNA / attenuated / "
                "stalled partial transcripts."
            ),
            'active_RNAPs': (
                "sets advanced coordinates and deletes RNAPs that terminated, "
                "attenuated, or (if enabled) stalled."
            ),
            'listeners': (
                "writes elongation diagnostics: count_rna_synthesized, "
                "count_NTPs_used, attenuation_probability/counts_attenuated, "
                "ntp pool/used, and rnap_data (actual_elongations, did_terminate, "
                "termination_loss, did_stall)."
            ),
        },
        config={
            'rnaPolymeraseElongationRateDict': "media → RNAP elongation rate ν (nt/s) that sets how far each RNAP can move per tick.",
            'rnaSequences': "padded per-TU nucleotide-index sequences used by buildSequences/polymerize.",
            'rnaLengths': "annotated per-TU lengths (nt); termination fires when transcript_length reaches this.",
            'ntWeights': "per-nucleotide molecular weights wᵢ used to compute transcript mass increase.",
            'endWeight': "mass added once, when a transcript is first initiated (5′-end weight).",
            'ntp_ids': "bulk ids of the four NTPs consumed during elongation.",
            'ppi': "bulk id of inorganic pyrophosphate (PPi) released during elongation (net = elongations − transcript initiations).",
            'inactive_RNAP': "bulk id of the inactive-RNAP pool that terminated/attenuated/stalled RNAPs return to.",
            'is_mRNA': "per-TU mask; partial mRNAs stay as unique molecules on termination while other RNA types become bulk counts.",
            'replichore_lengths': "right/left replichore lengths (nt); wrap RNAP coordinates that cross the origin/terminus.",
            'variable_elongation': "whether per-position variable elongation rates (e.g. amplified rRNA) are used.",
            'make_elongation_rates': "builds the per-sequence elongation-rate array from the media rate and timestep.",
            'recycle_stalled_elongation': "whether RNAPs on NTP-limited (stalled) transcripts are recycled to the inactive pool with their bases returned.",
            'get_attenuation_stop_probabilities': "fitted function mapping charged-tRNA concentration to per-operon early-termination probability p_stop.",
            'attenuated_rna_indices': "TU indices eligible for tRNA-attenuation early termination.",
            'fragmentBases': "bulk ids of NTP fragments returned when a stalled transcript is discarded.",
        },
        symbols={
            'ν': "RNAP elongation rate (nucleotides/s)",
            'dt': "simulation timestep (s)",
            'wᵢ': "molecular weight of nucleotide i (fg per molecule / g·mol⁻¹)",
            'Δmass': "transcript mass increase from elongation this tick (fg)",
            'Δlen': "number of nucleotides added to a transcript this tick (nt)",
            'p_stop': "early-termination (attenuation) probability of an attenuated tRNA operon (dimensionless, 0–1)",
            'NTP': "ribonucleoside triphosphate consumed per elongation step (count)",
            'PPi': "inorganic pyrophosphate released during elongation, net of transcript-initiating nucleotides that retain their 5′ triphosphate (count)",
            'charged_tRNA': "absolute charged (aminoacylated) tRNA concentration passed directly to the attenuation stop-probability function f (mol/L)",
        },
        assumptions=[
            "NTP pools are allocated across all active RNAPs by the polymerize algorithm to maximize total elongation subject to the available-NTP reaction limit.",
            "An RNAP terminates exactly when its transcript_length reaches the annotated TU length.",
            "Partial mRNAs are already functional and are kept as unique molecules; completed non-mRNA transcripts become bulk molecules.",
            "tRNA attenuation is optional; when enabled, RNAPs stochastically terminate early with per-operon probability f([charged_tRNA]) — the fitted function of absolute charged-tRNA concentration, not a charged/total ratio — and their partial transcripts are discarded.",
            "RNAP coordinates wrap at the replichore boundaries (origin/terminus).",
        ],
        references=[],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'attenuated_rna_indices': {'_type': 'array[integer]', '_default': np.array([], dtype=int)},
        'cell_density': {'_type': 'quantity[g/L]', '_default': 1100.0},
        'charged_trnas': {'_type': 'list[string]', '_default': []},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'endWeight': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'fragmentBases': {'_type': 'list[string]', '_default': []},
        'get_attenuation_stop_probabilities': {'_type': 'method', '_default': None},
        'inactive_RNAP': {'_type': 'string', '_default': ''},
        'is_mRNA': {'_type': 'array[boolean]', '_default': np.array([], dtype=float)},
        'location_lookup': {'_type': 'map[node]', '_default': {}},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'n_avogadro': {'_type': 'quantity[float,1/mol]', '_default': 6.02214076e+23},
        'n_fragment_bases': {'_type': 'integer', '_default': 0},
        'ntWeights': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'ntp_ids': {'_type': 'list[string]', '_default': []},
        'polymerized_ntps': {'_type': 'list[string]', '_default': []},
        'ppi': {'_type': 'string', '_default': ''},
        'recycle_stalled_elongation': {'_type': 'boolean', '_default': False},
        'replichore_lengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'rnaIds': {'_type': 'list[string]', '_default': []},
        'rnaLengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'rnaPolymeraseElongationRateDict': {'_type': 'map[quantity[float,nucleotide/s]]', '_default': {}},
        'rnaSequences': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'seed': {'_type': 'integer', '_default': 0},
        'submass_indices': {'_type': 'map[integer]', '_default': {}},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'variable_elongation': {'_type': 'boolean', '_default': False},
    }

    def initialize(self, config):

        # Load parameters
        self.rnaPolymeraseElongationRateDict = self.parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.rnaIds = self.parameters["rnaIds"]
        self.rnaLengths = self.parameters["rnaLengths"]
        self.rnaSequences = self.parameters["rnaSequences"]
        self.ppi = self.parameters["ppi"]
        self.inactive_RNAP = self.parameters["inactive_RNAP"]
        self.fragmentBases = self.parameters["fragmentBases"]
        self.charged_trnas = self.parameters["charged_trnas"]
        self.ntp_ids = self.parameters["ntp_ids"]
        self.ntWeights = self.parameters["ntWeights"]
        self.endWeight = self.parameters["endWeight"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.chromosome_length = self.replichore_lengths.sum()
        self.n_fragment_bases = self.parameters["n_fragment_bases"]
        self.recycle_stalled_elongation = self.parameters["recycle_stalled_elongation"]

        # Mask for mRNAs
        self.is_mRNA = self.parameters["is_mRNA"]

        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.polymerized_ntps = self.parameters["polymerized_ntps"]
        self.charged_trna_names = self.parameters["charged_trnas"]

        # Attenuation fallback parameters (used when attenuation_config store
        # is populated by the TrnaAttenuationConfig feature module)
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.attenuated_rna_indices = self.parameters.get(
            "attenuated_rna_indices", np.array([], dtype=int))
        self.location_lookup = self.parameters.get("location_lookup", {})

        # random seed
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.bulk_RNA_idx = None

    def inputs(self):
        return {
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'active_RNAPs': {'_type': ACTIVE_RNAP_ARRAY, '_default': []},
            'bulk': {'_type': 'bulk_array', '_default': []},
            'bulk_total': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                },
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'attenuation_config': {
                'enabled': {'_type': 'boolean', '_default': False},
            },
        }

    def outputs(self):
        return (
            {
                'bulk': 'bulk_array',
                'RNAs': RNA_ARRAY,
                'active_RNAPs': ACTIVE_RNAP_ARRAY,
                'listeners':                 {
                    'transcript_elongation_listener':                     {
                        'count_rna_synthesized': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'count_NTPs_used': {'_type': 'overwrite[integer]', '_default': 0},
                        'attenuation_probability': {'_type': 'overwrite[array[float]]', '_default': []},
                        'counts_attenuated': {'_type': 'overwrite[array[integer]]', '_default': []},
                    },
                    'growth_limits':                     {
                        'ntp_used': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'ntp_pool_size': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'ntp_request_size': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'ntp_allocated': {'_type': 'overwrite[array[integer]]', '_default': []},
                    },
                    'rnap_data':                     {
                        'actual_elongations': {'_type': 'overwrite[integer]', '_default': 0},
                        'did_terminate': {'_type': 'overwrite[integer]', '_default': 0},
                        'termination_loss': {'_type': 'overwrite[integer]', '_default': 0},
                        'did_stall': {'_type': 'overwrite[integer]', '_default': 0},
                    },
                },
            }
        )
    def calculate_request(self, timestep, states):
        # At first update, convert all strings to indices
        if self.bulk_RNA_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.bulk_RNA_idx = bulk_name_to_idx(self.rnaIds, bulk_ids)
            self.ntps_idx = bulk_name_to_idx(self.ntp_ids, bulk_ids)
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)
            self.fragmentBases_idx = bulk_name_to_idx(self.fragmentBases, bulk_ids)
            self.charged_trnas_idx = bulk_name_to_idx(self.charged_trnas, bulk_ids)

        # Calculate elongation rate based on the current media
        current_media_id = states["environment"]["media_id"]

        self.rnapElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ].to(units.nt / units.s).magnitude

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnapElongationRate,
            states["timestep"],
            self.variable_elongation,
        )

        # If there are no active RNA polymerases, return immediately
        if states["active_RNAPs"]["_entryState"].sum() == 0:
            return {}

        # Determine total possible sequences of nucleotides that can be
        # transcribed in this time step for each partial transcript
        TU_indexes, transcript_lengths, is_full_transcript = attrs(
            states["RNAs"], ["TU_index", "transcript_length", "is_full_transcript"]
        )
        is_partial_transcript = np.logical_not(is_full_transcript)
        TU_indexes_partial = TU_indexes[is_partial_transcript]
        transcript_lengths_partial = transcript_lengths[is_partial_transcript]

        sequences = buildSequences(
            self.rnaSequences,
            TU_indexes_partial,
            transcript_lengths_partial,
            self.elongation_rates,
        )

        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4
        )

        # Calculate if any nucleotides are limited and request up to the number
        # in the sequences or number available
        ntpsTotal = counts(states["bulk"], self.ntps_idx)
        maxFractionalReactionLimit = np.fmin(1, ntpsTotal / sequenceComposition)

        requests = {
            "bulk": [
                (
                    self.ntps_idx,
                    (maxFractionalReactionLimit * sequenceComposition).astype(int),
                )
            ]
        }

        requests["listeners"] = {
            "growth_limits": {
                "ntp_pool_size": counts(states["bulk"], self.ntps_idx),
                "ntp_request_size": (
                    maxFractionalReactionLimit * sequenceComposition
                ).astype(int),
            }
        }

        return requests

    def evolve_state(self, timestep, states):
        ntpCounts = counts(states["bulk"], self.ntps_idx)

        # If there are no active RNA polymerases, return immediately
        if states["active_RNAPs"]["_entryState"].sum() == 0:
            return {
                "listeners": {
                    "transcript_elongation_listener": {
                        "count_NTPs_used": 0,
                        "count_rna_synthesized": np.zeros(len(self.rnaIds), dtype=int),
                    },
                    "growth_limits": {
                        "ntp_used": np.zeros(len(self.ntp_ids), dtype=int),
                        "ntp_allocated": ntpCounts,
                    },
                    "rnap_data": {
                        "actual_elongations": 0,
                        "did_terminate": 0,
                        "termination_loss": 0,
                    },
                },
                "active_RNAPs": {},
                "RNAs": {},
            }

        # Get attributes from existing RNAs
        (
            TU_index_all_RNAs,
            length_all_RNAs,
            is_full_transcript,
            is_mRNA_all_RNAs,
            RNAP_index_all_RNAs,
        ) = attrs(
            states["RNAs"],
            [
                "TU_index",
                "transcript_length",
                "is_full_transcript",
                "is_mRNA",
                "RNAP_index",
            ],
        )
        length_all_RNAs = length_all_RNAs.copy()

        update = {"listeners": {"growth_limits": {}}}

        # Determine sequences of RNAs that should be elongated
        is_partial_transcript = np.logical_not(is_full_transcript)
        partial_transcript_indexes = np.where(is_partial_transcript)[0]
        TU_index_partial_RNAs = TU_index_all_RNAs[is_partial_transcript]
        length_partial_RNAs = length_all_RNAs[is_partial_transcript]
        is_mRNA_partial_RNAs = is_mRNA_all_RNAs[is_partial_transcript]
        RNAP_index_partial_RNAs = RNAP_index_all_RNAs[is_partial_transcript]

        # Check if tRNA attenuation module is enabled
        att_cfg = states.get("attenuation_config", {})
        has_attenuation = att_cfg.get("enabled", False)

        if has_attenuation:
            cell_mass = states["listeners"]["mass"]["cell_mass"]
            cellVolume = as_quantity(cell_mass, units.fg) / self.cell_density
            counts_to_molar = 1 / (self.n_avogadro * cellVolume)
            stop_probs_fn = att_cfg.get(
                "stop_probabilities", self.parameters.get(
                    "get_attenuation_stop_probabilities"))
            att_indices = att_cfg.get(
                "attenuated_rna_indices", self.attenuated_rna_indices)
            loc_lookup = att_cfg.get(
                "location_lookup", self.location_lookup)
            attenuation_probability = stop_probs_fn(
                counts_to_molar * counts(states["bulk_total"], self.charged_trnas_idx)
            )
            prob_lookup = {
                tu: prob
                for tu, prob in zip(att_indices, attenuation_probability)
            }
            tu_stop_probability = np.array(
                [
                    prob_lookup.get(idx, 0)
                    * (length < loc_lookup.get(idx, 0))
                    for idx, length in zip(TU_index_partial_RNAs, length_partial_RNAs)
                ]
            )
            rna_to_attenuate = stochasticRound(
                self.random_state, tu_stop_probability
            ).astype(bool)
        else:
            attenuation_probability = np.zeros(len(self.attenuated_rna_indices))
            rna_to_attenuate = np.zeros(len(TU_index_partial_RNAs), bool)
        rna_to_elongate = ~rna_to_attenuate

        sequences = buildSequences(
            self.rnaSequences,
            TU_index_partial_RNAs,
            length_partial_RNAs,
            self.elongation_rates,
        )

        # Polymerize transcripts based on sequences and available nucleotides
        reactionLimit = ntpCounts.sum()
        result = polymerize(
            sequences[rna_to_elongate],
            ntpCounts,
            reactionLimit,
            self.random_state,
            self.elongation_rates[TU_index_partial_RNAs][rna_to_elongate],
            self.variable_elongation,
        )

        sequence_elongations = np.zeros_like(length_partial_RNAs)
        sequence_elongations[rna_to_elongate] = result.sequenceElongation
        ntps_used = result.monomerUsages
        did_stall_mask = result.sequences_limited_elongation

        # Calculate changes in mass associated with polymerization
        added_mass = computeMassIncrease(
            sequences, sequence_elongations, self.ntWeights
        )
        did_initialize = (length_partial_RNAs == 0) & (sequence_elongations > 0)
        added_mass[did_initialize] += self.endWeight

        # Calculate updated transcript lengths
        updated_transcript_lengths = length_partial_RNAs + sequence_elongations

        # Get attributes of active RNAPs
        coordinates, is_forward, RNAP_unique_index = attrs(
            states["active_RNAPs"], ["coordinates", "is_forward", "unique_index"]
        )

        # Active RNAP count should equal partial transcript count
        assert len(RNAP_unique_index) == len(RNAP_index_partial_RNAs)

        # All partial RNAs must be linked to an RNAP
        assert np.count_nonzero(RNAP_index_partial_RNAs == -1) == 0

        # Get mapping indexes between partial RNAs to RNAPs
        partial_RNA_to_RNAP_mapping, _ = get_mapping_arrays(
            RNAP_index_partial_RNAs, RNAP_unique_index
        )

        # Rescale boolean array of directions to an array of 1's and -1's.
        # True is converted to 1, False is converted to -1.
        direction_rescaled = (2 * (is_forward - 0.5)).astype(np.int64)

        # Compute the updated coordinates of RNAPs. Coordinates of RNAPs
        # moving in the positive direction are increased, whereas coordinates
        # of RNAPs moving in the negative direction are decreased.
        updated_coordinates = coordinates + np.multiply(
            direction_rescaled, sequence_elongations[partial_RNA_to_RNAP_mapping]
        )

        # Reset coordinates of RNAPs that cross the boundaries between right
        # and left replichores
        updated_coordinates[updated_coordinates > self.replichore_lengths[0]] -= (
            self.chromosome_length
        )
        updated_coordinates[updated_coordinates < -self.replichore_lengths[1]] += (
            self.chromosome_length
        )

        # Update transcript lengths of RNAs and coordinates of RNAPs
        length_all_RNAs[is_partial_transcript] = updated_transcript_lengths

        # Update added submasses of RNAs. Masses of partial mRNAs are counted
        # as mRNA mass as they are already functional, but the masses of other
        # types of partial RNAs are counted as nonspecific RNA mass.
        added_nsRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)
        added_mRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)

        added_nsRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, np.logical_not(is_mRNA_partial_RNAs)
        )
        added_mRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, is_mRNA_partial_RNAs
        )

        # Determine if transcript has reached the end of the sequence
        terminal_lengths = self.rnaLengths[TU_index_partial_RNAs]
        did_terminate_mask = updated_transcript_lengths == terminal_lengths
        terminated_RNAs = np.bincount(
            TU_index_partial_RNAs[did_terminate_mask],
            minlength=self.rnaSequences.shape[0],
        )

        # Update is_full_transcript attribute of RNAs
        is_full_transcript_updated = is_full_transcript.copy()
        is_full_transcript_updated[partial_transcript_indexes[did_terminate_mask]] = (
            True
        )

        n_terminated = did_terminate_mask.sum()
        n_initialized = did_initialize.sum()
        n_elongations = ntps_used.sum()

        # Get counts of new bulk RNAs
        n_new_bulk_RNAs = terminated_RNAs.copy()
        n_new_bulk_RNAs[self.is_mRNA] = 0

        update["RNAs"] = {
            "set": {
                "transcript_length": length_all_RNAs,
                "is_full_transcript": is_full_transcript_updated,
                "massDiff_nonspecific_RNA": attrs(
                    states["RNAs"], ["massDiff_nonspecific_RNA"]
                )[0]
                + added_nsRNA_mass_all_RNAs,
                "massDiff_mRNA": attrs(states["RNAs"], ["massDiff_mRNA"])[0]
                + added_mRNA_mass_all_RNAs,
            },
            "delete": partial_transcript_indexes[
                np.logical_and(did_terminate_mask, np.logical_not(is_mRNA_partial_RNAs))
            ],
        }
        update["active_RNAPs"] = {
            "set": {"coordinates": updated_coordinates},
            "delete": np.where(did_terminate_mask[partial_RNA_to_RNAP_mapping])[0],
        }

        # Attenuation removes RNAs and RNAPs
        counts_attenuated = np.zeros(len(self.attenuated_rna_indices), dtype=int)
        if np.any(rna_to_attenuate):
            for idx in TU_index_partial_RNAs[rna_to_attenuate]:
                counts_attenuated[self.attenuated_rna_indices_lookup[idx]] += 1
            update["RNAs"]["delete"] = np.append(
                update["RNAs"]["delete"], partial_transcript_indexes[rna_to_attenuate]
            )
            update["active_RNAPs"]["delete"] = np.append(
                update["active_RNAPs"]["delete"],
                np.where(rna_to_attenuate[partial_RNA_to_RNAP_mapping])[0],
            )
        n_attenuated = rna_to_attenuate.sum()

        # Handle stalled elongation
        n_total_stalled = did_stall_mask.sum()
        if self.recycle_stalled_elongation and (n_total_stalled > 0):
            # Remove RNAPs that were bound to stalled elongation transcripts
            # and increment counts of inactive RNAPs
            update["active_RNAPs"]["delete"] = np.append(
                update["active_RNAPs"]["delete"],
                np.where(did_stall_mask[partial_RNA_to_RNAP_mapping])[0],
            )
            update["bulk"].append((self.inactive_RNAP_idx, n_total_stalled))

            # Remove partial transcripts from stalled elongation
            update["RNAs"]["delete"] = np.append(
                update["RNAs"]["delete"], partial_transcript_indexes[did_stall_mask]
            )
            stalled_sequence_lengths = updated_transcript_lengths[did_stall_mask]
            n_initiated_sequences = np.count_nonzero(stalled_sequence_lengths)

            if n_initiated_sequences > 0:
                # Get the full sequence of stalled transcripts
                stalled_sequences = buildSequences(
                    self.rnaSequences,
                    TU_index_partial_RNAs[did_stall_mask],
                    np.zeros(n_total_stalled, dtype=np.int64),
                    np.full(n_total_stalled, updated_transcript_lengths.max()),
                )

                # Count the number of fragment bases in these transcripts up
                # until the stalled length
                base_counts = np.zeros(self.n_fragment_bases, dtype=np.int64)
                for sl, seq in zip(stalled_sequence_lengths, stalled_sequences):
                    base_counts += np.bincount(
                        seq[:sl], minlength=self.n_fragment_bases
                    )

                # Increment counts of fragment NTPs and phosphates
                update["bulk"].append((self.fragmentBases_idx, base_counts))
                update["bulk"].append((self.ppi_idx, n_initiated_sequences))

        update.setdefault("bulk", []).append((self.ntps_idx, -ntps_used))
        update["bulk"].append((self.bulk_RNA_idx, n_new_bulk_RNAs))
        update["bulk"].append((self.inactive_RNAP_idx, n_terminated + n_attenuated))
        update["bulk"].append((self.ppi_idx, n_elongations - n_initialized))

        # Write outputs to listeners
        update["listeners"]["transcript_elongation_listener"] = {
            "count_rna_synthesized": terminated_RNAs,
            "count_NTPs_used": n_elongations,
            "attenuation_probability": attenuation_probability,
            "counts_attenuated": counts_attenuated,
        }
        update["listeners"]["growth_limits"] = {"ntp_used": ntps_used}
        update["listeners"]["rnap_data"] = {
            "actual_elongations": sequence_elongations.sum(),
            "did_terminate": did_terminate_mask.sum(),
            "termination_loss": (terminal_lengths - length_partial_RNAs)[
                did_terminate_mask
            ].sum(),
            "did_stall": n_total_stalled,
        }

        return update


def get_mapping_arrays(x, y):
    """
    Returns the array of indexes of each element of array x in array y, and
    vice versa. Assumes that the elements of x and y are unique, and
    set(x) == set(y).
    """

    def argsort_unique(idx):
        """
        Quicker argsort for arrays that are permutations of np.arange(n).
        """
        n = idx.size
        argsort_idx = np.empty(n, dtype=np.int64)
        argsort_idx[idx] = np.arange(n)
        return argsort_idx

    x_argsort = np.argsort(x)
    y_argsort = np.argsort(y)

    x_to_y = x_argsort[argsort_unique(y_argsort)]
    y_to_x = y_argsort[argsort_unique(x_argsort)]

    return x_to_y, y_to_x
