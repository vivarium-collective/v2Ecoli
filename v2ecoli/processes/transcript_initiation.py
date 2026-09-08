"""
=====================
Transcript Initiation
=====================

This process models the binding of RNA polymerase (RNAP) to each gene.

Mathematical Model
------------------
The number of RNAPs to activate per timestep is determined so that the
average active RNAP fraction matches measured values (media-dependent).

1. **Active RNAP target**: Given the media-dependent fraction f_active:

       n_to_activate = round(f_active * n_total_RNAP) - n_currently_active

2. **Synthesis probability**: Each transcription unit (TU) i has a
   probability of initiation p_i that combines a basal probability
   with TF-mediated modulation via a sparse delta_prob matrix:

       p_i = max(0, basal_prob_i + sum_j(delta_prob[i,j] * bound_TF_j))

   The delta_prob matrix is stored in COO format as (deltaV, deltaI,
   deltaJ, shape) and reconstructed into a CSR sparse matrix.

3. **Probabilistic initiation**: The n_to_activate RNAPs are distributed
   across TUs via multinomial sampling weighted by normalized p_i,
   subject to physical constraints:

   - **Footprint constraint**: An RNAP occupies ~24 nt at the promoter.
     Overcrowded TUs (where new initiation would overlap existing RNAPs)
     have their probability set to zero.

   - **Promoter availability**: Only TUs with at least one promoter on
     an existing chromosome can initiate.

4. **ppGpp regulation** (optional): When enabled, ppGpp concentration
   modulates both basal_prob and f_active via fitted functions,
   providing growth-rate-dependent transcriptional regulation.

TODO:
  - use transcription units instead of single genes
  - match sigma factors to promoters
"""

import os

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from typing import cast

# simulate_process removed

from v2ecoli.library.schema import (
    create_unique_indices,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
)

from v2ecoli.types.quantity import ureg as units
from wholecell.utils.random import stochasticRound
from wholecell.utils.unit_struct_array import UnitStructArray

from v2ecoli.library.data_predicates import monotonically_decreasing, all_nonnegative
from scipy.stats import chisquare, poisson

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import (
    FULL_CHROMOSOME_ARRAY,
    RNA_ARRAY,
    ACTIVE_RNAP_ARRAY,
    PROMOTER_ARRAY,
)
from bigraph_schema.contract import ProcessContract


# dnaA autoregulation constants (dnaa-4 / Rashmi mechanism)
DNAA_TU_IDX = 2778        # TU00259[c] — verified against cache_dnaa4_autoreg
# s; 0 disables, 1 fully silences at f=1. Env-overridable so the no-autoreg
# control run (DNAA_AUTOREG_STRENGTH=0) shares one code path with the experiment.
AUTOREG_STRENGTH = float(os.environ.get("DNAA_AUTOREG_STRENGTH", "0.8"))
# Repression curve: "linear" = (1 - s*f); "hill" = (1 - s * f^n/(K^n+f^n)).
# Hill gives a sharper switch — LESS repression at low promoter occupancy (lifts
# the cell-cycle trough), MORE once f crosses K (caps the peak). Per Rashmi's
# handoff open question + the linear run's over-repression of the trough.
AUTOREG_FORM = os.environ.get("DNAA_AUTOREG_FORM", "linear")
AUTOREG_HILL_N = float(os.environ.get("DNAA_AUTOREG_HILL_N", "4"))
AUTOREG_HILL_K = float(os.environ.get("DNAA_AUTOREG_HILL_K", "0.5"))


def _autoreg_factor(promoter_fraction: float, strength: float,
                    form: str | None = None, n: float | None = None,
                    K: float | None = None) -> float:
    """Transcription-scaling factor from promoter occupancy f in [0,1].

    linear: ``1 - s*f``. hill: ``1 - s * f^n / (K^n + f^n)`` — sharper switch.
    """
    form = form if form is not None else AUTOREG_FORM
    if form == "hill":
        n = n if n is not None else AUTOREG_HILL_N
        K = K if K is not None else AUTOREG_HILL_K
        f = promoter_fraction
        return 1.0 - strength * (f ** n) / (K ** n + f ** n)
    return 1.0 - strength * promoter_fraction


# Register default topology for this process, associating it with process name
NAME = "ecoli-transcript-initiation"
TOPOLOGY = {
    "environment": ("environment",),
    "full_chromosomes": ("unique", "full_chromosome"),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "ppgpp_state": ("ppgpp_state",),
    "dnaa_hydrolysis": ("process_state", "dnaa_hydrolysis"),
}


class TranscriptInitiation(Step):
    """Transcript Initiation PartitionedProcess

    **Defaults:**

    - **fracActiveRnapDict** (``dict``): Dictionary with keys corresponding to
      media, values being the fraction of active RNA Polymerase (RNAP)
      for that media.
    - **rnaLengths** (``numpy.ndarray[int]``): lengths of RNAs for each transcription
      unit (TU), in nucleotides
    - **rnaPolymeraseElongationRateDict** (``dict``): Dictionary with keys
      corresponding to media, values being RNAP's elongation rate in
      that media, in nucleotides/s
    - **variable_elongation** (``bool``): Whether to add amplified elongation rates
      for rRNAs. False by default.
    - **make_elongation_rates** (``func``): Function for making elongation rates
      (see :py:meth:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.make_elongation_rates`).
      Takes PRNG, basal elongation rate, timestep, and ``variable_elongation``.
      Returns an array of elongation rates for all genes.
    - **active_rnap_footprint_size** (``int``): Maximum physical footprint of RNAP
      in nucleotides to cap initiation probabilities
    - **basal_prob** (``numpy.ndarray[float]``): Baseline probability of synthesis for
      every TU.
    - **delta_prob** (``dict``): Dictionary with four keys, used to create a matrix
      encoding the effect of transcription factors (TFs) on transcription
      probabilities::

        {'deltaV' (np.ndarray[float]): deltas associated with the effects of
            TFs on TUs,
        'deltaI' (np.ndarray[int]): index of the affected TU for each delta,
        'deltaJ' (np.ndarray[int]): index of the acting TF for each delta,
        'shape' (tuple): (m, n) = (# of TUs, # of TFs)}

    - **perturbations** (``dict``): Dictionary of genetic perturbations (optional,
      can be empty)
    - **rna_data** (``numpy.ndarray``): Structured array with an entry for each TU,
      where entries look like::

            (id, deg_rate, length (nucleotides), counts_AGCU, mw
            (molecular weight), is_mRNA, is_miscRNA, is_rRNA, is_tRNA,
            is_23S_rRNA, is_16S_rRNA, is_5S_rRNA, is_ribosomal_protein,
            is_RNAP, gene_id, Km_endoRNase, replication_coordinate,
            direction)

    - **idx_rRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to rRNAs
    - **idx_mRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to mRNAs
    - **idx_tRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to tRNAs
    - **idx_rprotein** (``numpy.ndarray[int]``): indexes of TUs corresponding ribosomal
      proteins
    - **idx_rnap** (``numpy.ndarray[int]``): indexes of TU corresponding to RNAP
    - **rnaSynthProbFractions** (``dict``): Dictionary where keys are media types,
      values are sub-dictionaries with keys 'mRna', 'tRna', 'rRna', and
      values being probabilities of synthesis for each RNA type
    - **rnaSynthProbRProtein** (``dict``): Dictionary where keys are media types,
      values are arrays storing the (fixed) probability of synthesis for
      each rProtein TU, under that media condition.
    - **rnaSynthProbRnaPolymerase** (``dict``): Dictionary where keys are media
      types, values are arrays storing the (fixed) probability of
      synthesis for each RNAP TU, under that media condition.
    - **replication_coordinate** (``numpy.ndarray[int]``): Array with chromosome
      coordinates for each TU
    - **transcription_direction** (``numpy.ndarray[bool]``): Array of transcription
      directions for each TU
    - **n_avogadro** (``unum.Unum``): Avogadro's number (constant)
    - **cell_density** (``unum.Unum``): Density of cell (constant)
    - **ppgpp** (``str``): id of ppGpp
    - **inactive_RNAP** (``str``): id of inactive RNAP
    - **synth_prob** (``Callable[[Unum, int], numpy.ndarrray[float]]``):
      Function used in model of ppGpp regulation
      (see :py:func:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.synth_prob_from_ppgpp`).
      Takes ppGpp concentration (mol/volume) and copy number, returns
      normalized synthesis probability for each gene
    - **copy_number** (``Callable[[Unum, int], numpy.ndarrray[float]]``):
      see :py:func:`~reconstruction.ecoli.dataclasses.process.replication.Replication.get_average_copy_number`.
      Takes expected doubling time in minutes and chromosome coordinates of genes,
      returns average copy number of each gene expected at doubling time
    - **ppgpp_regulation** (``bool``): Whether to include model of ppGpp regulation
    - **get_rnap_active_fraction_from_ppGpp** (``Callable[[Unum], float]``):
      Returns elongation rate for a given ppGpp concentration
    - **seed** (``int``): random seed to initialize PRNG
    """

    description = (
        "TranscriptInitiation — distributes activated RNAPs across TUs by weighted multinomial sampling.\n\n"
        "    n_to_activate = round(f_active · n_total_RNAP) - n_active\n"
        "    p_i = max(0, basal_prob_i + ∑_j delta_prob[i,j] · bound_TF_j)\n"
        "    initiations ~ Multinomial(n_to_activate, p_i / ∑_i p_i)\n"
        "  with p_i ← 0 for TUs that are footprint-crowded (~24 nt) or lack a chromosomal promoter.\n"
        "  f_active: media-dependent active RNAP fraction;  delta_prob: sparse (COO→CSR) TF effect matrix.\n"
        "  ppGpp (optional) modulates basal_prob and f_active via fitted functions."
    )

    contract = ProcessContract(
        summary=(
            "Activates inactive RNAPs and distributes them across transcription "
            "units by weighted multinomial (or Poisson tau-leap) sampling of "
            "per-promoter initiation probabilities."
        ),
        inputs={
            'environment': (
                "reads media_id to select the media-dependent active-RNAP "
                "fraction f_active, the RNAP elongation rate, and the per-RNA-type "
                "(mRNA/tRNA/rRNA) synthesis-probability fractions for this tick."
            ),
            'full_chromosomes': (
                "presence gate: if the cell has no full chromosome, every "
                "promoter initiation probability is forced to zero (no initiation)."
            ),
            'promoters': (
                "reads each promoter's TU_index and bound_TF to build the "
                "per-promoter probability p_i = basal_prob + delta_prob·bound_TF, "
                "and to map promoters↔transcription units."
            ),
            'bulk': (
                "reads the count of inactive RNAP (APORNAP-CPLX) to convert the "
                "activation probability into an integer number of RNAPs to activate."
            ),
            'timestep': (
                "reads the tick length to size the RNAP activation probability "
                "(_calculateActivationProb) and the footprint-crowding cap max_p."
            ),
            'ppgpp_state': (
                "reads optional basal_prob and frac_active_rnap written by "
                "PpgppInitiation; when present these override the media-default "
                "basal_prob and f_active and switch the TF effect to multiplicative."
            ),
            'dnaa_hydrolysis': (
                "reads promoter_fraction (bound fraction of dnaA-box sites) to "
                "scale down the dnaA TU's initiation probability (autoregulation)."
            ),
            'RNAs': (
                "reads existing RNA unique molecules only to allocate fresh, "
                "non-colliding unique indices for the RNAs/RNAPs added this tick."
            ),
        },
        outputs={
            'bulk': "decrements the inactive-RNAP pool by the number of RNAPs activated.",
            'RNAs': (
                "adds one partially-transcribed RNA unique molecule per initiation "
                "(transcript_length 0, linked to its RNAP via RNAP_index)."
            ),
            'active_RNAPs': (
                "adds the newly activated RNAP unique molecules with starting "
                "coordinates, direction and domain index."
            ),
            'listeners': (
                "writes synthesis-probability diagnostics (target/actual "
                "rna_synth_prob, max_p, tu_is_overcrowded, total_rna_init), rRNA "
                "initiation counts, and rnap_data (did_initialize, rna_init_event, "
                "log_likelihood)."
            ),
        },
        config={
            'fracActiveRnapDict': "media → active-RNAP fraction f_active lookup (media default when ppGpp regulation is off).",
            'basal_prob': "per-TU baseline synthesis probability vector basal_prob_i.",
            'delta_prob': "sparse COO TF-effect matrix (deltaV, deltaI, deltaJ, shape) on TU synthesis probabilities.",
            'get_delta_prob_matrix': "builds the dense delta_prob matrix (ppGpp-aware) at initialization.",
            'rnaSynthProbFractions': "media → target synthesis-probability fractions per RNA type (mRna/tRna/rRna) used to rescale p_i.",
            'rnaSynthProbRProtein': "media → fixed synthesis probabilities for ribosomal-protein TUs.",
            'rnaSynthProbRnaPolymerase': "media → fixed synthesis probabilities for RNAP-subunit TUs.",
            'active_rnap_footprint_size': "RNAP promoter footprint (~24 nt) that caps per-promoter init probability (max_p).",
            'rnaPolymeraseElongationRateDict': "media → RNAP elongation rate (nt/s); feeds the activation-probability and footprint cap.",
            'rnaLengths': "TU lengths (nt); set the expected transcription time and termination rate in the activation-probability calc.",
            'idx_mRNA': "TU indices classified as mRNA, for type-specific probability rescaling.",
            'idx_rRNA': "TU indices classified as rRNA, for type-specific probability rescaling.",
            'idx_tRNA': "TU indices classified as tRNA, for type-specific probability rescaling.",
            'idx_rprotein': "TU indices of ribosomal proteins, given fixed synthesis probabilities.",
            'idx_rnap': "TU indices of RNAP subunits, given fixed synthesis probabilities.",
            'replication_coordinate': "chromosomal coordinate per TU; set as the start coordinate of each new RNAP.",
            'transcription_direction': "per-TU transcription direction; set on each new RNAP.",
            'ppgpp_regulation': "whether ppGpp-dependent regulation is active (changes how the delta_prob matrix is built).",
            'inactive_RNAP': "bulk id of the inactive-RNAP (APORNAP-CPLX) pool drawn from.",
            'ppgpp': "bulk id of ppGpp.",
            'variable_elongation': "whether rRNA elongation rates are amplified when building elongation rates.",
            'pdmp_initiation_mode': "'discrete' exact-sum multinomial vs 'poisson' per-promoter tau-leap sampling of initiations.",
            'transcript_init_prob_scale': "poisson mode: scalar multiplier on the per-promoter initiation rate before drawing Poisson samples (ABC-SMC knob).",
        },
        symbols={
            'n_to_activate': "number of RNAPs to activate this tick (integer count)",
            'f_active': "media-dependent active-RNAP fraction (dimensionless, 0–1)",
            'n_total_RNAP': "total RNAP molecules (active + inactive) (count)",
            'n_active': "currently active RNAP molecules (count)",
            'p_i': "initiation probability weight of transcription unit i (dimensionless)",
            'basal_prob_i': "baseline synthesis probability of TU i (dimensionless)",
            'basal_prob': "vector of baseline per-TU synthesis probabilities (dimensionless)",
            'delta_prob[i,j]': "additive effect of bound TF j on the synthesis probability of TU i (dimensionless)",
            'bound_TF_j': "1 if TF j is bound at the promoter, else 0 (indicator)",
            'ppGpp': "guanosine tetraphosphate; its concentration modulates basal_prob and f_active (mol/L)",
        },
        assumptions=[
            "Only TUs with at least one promoter on an existing chromosome can initiate; with no full chromosome no initiation occurs.",
            "An RNAP occupies ~24 nt at the promoter; overcrowded TUs are capped at max_p and the excess probability mass is redistributed.",
            "In discrete mode initiation events are a single multinomial draw so Σ initiations equals n_to_activate exactly; poisson mode gives independent per-promoter marginals with a resource cap.",
            "rProtein and RNAP TUs carry fixed media-specific synthesis probabilities; remaining probability mass is rescaled around them.",
            "TF effect is additive on basal_prob via the sparse delta_prob matrix; under ppGpp regulation it becomes multiplicative on the ppGpp-adjusted basal via a smooth accessibility gain b²/(b+K).",
        ],
        references=[],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'active_rnap_foorprint_size': {'_type': 'integer', '_default': 1},
        'active_rnap_footprint_size': {'_type': 'quantity[nt]', '_default': 24.0},
        'attenuated_rna_indices': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'attenuation_adjustments': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'basal_prob': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'cell_density': {'_type': 'quantity[g/L]', '_default': 1100.0},
        'copy_number': {'_type': 'array[integer]', '_default': None},
        'delta_prob': {'_type': 'map[node]', '_default': {}},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'fracActiveRnapDict': {'_type': 'map[float]', '_default': {}},
        'get_delta_prob_matrix': 'method',
        'get_rnap_active_fraction_from_ppGpp': {'_type': 'method', '_default': None},
        'idx_mRNA': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_rRNA': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_rnap': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_rprotein': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_tRNA': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'inactive_RNAP': {'_type': 'string', '_default': 'APORNAP-CPLX[c]'},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'n_avogadro': {'_type': 'quantity[float,1/mol]', '_default': 6.02214076e+23},
        'perturbations': {'_type': 'map[node]', '_default': {}},
        'ppgpp': {'_type': 'string', '_default': 'ppGpp'},
        'ppgpp_regulation': {'_type': 'boolean', '_default': False},
        'replication_coordinate': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'rnaLengths': {'_type': 'quantity[array[integer],nt]', '_default': np.array([], dtype=float)},
        'rnaPolymeraseElongationRateDict': {'_type': 'map[quantity[float,nucleotide/s]]', '_default': {}},
        'rnaSynthProbFractions': {'_type': 'map[float]', '_default': {}},
        'rnaSynthProbRProtein': {'_type': 'map[float]', '_default': {}},
        'rnaSynthProbRnaPolymerase': {'_type': 'map[float]', '_default': {}},
        'rna_data': {'_type': 'units_array', '_default': {}},
        'seed': {'_type': 'integer', '_default': 0},
        # PDMP / Phase 2 — switch between the legacy discrete-time
        # multinomial event-distribution (``"discrete"``, default) and
        # per-promoter Poisson tau-leap (``"poisson"``). Both produce the
        # same expected number of initiations per tick; the Poisson
        # branch treats each promoter as an independent jump process,
        # which is what Phase 3's likelihood machinery integrates over.
        'pdmp_initiation_mode': {'_type': 'string', '_default': 'discrete'},
        # Phase-3 sprint-7 ABC-SMC knob. In poisson mode, multiplies
        # the per-promoter initiation rate by this scalar BEFORE
        # drawing Poisson samples. Default 1.0 reproduces the
        # unperturbed sampler; sweeping {0.7, 0.85, 1.0, 1.15, 1.3}
        # generates ensembles at distinguishable parameter values
        # for the ABC-SMC stub script.
        'transcript_init_prob_scale': {'_type': 'float', '_default': 1.0},
        'synth_prob': {'_type': 'array[float]', '_default': None},
        'transcription_direction': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'trna_attenuation': {'_type': 'boolean', '_default': False},
        'variable_elongation': {'_type': 'boolean', '_default': False},
    }

    def initialize(self, config):

        # Load parameters
        self.fracActiveRnapDict = self.parameters["fracActiveRnapDict"]
        self.rnaLengths = self.parameters["rnaLengths"]
        self.rnaPolymeraseElongationRateDict = self.parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]
        self.active_rnap_footprint_size = self.parameters["active_rnap_footprint_size"]

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = self.parameters["basal_prob"].copy()

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = self.parameters["delta_prob"]
        if self.parameters["get_delta_prob_matrix"] is not None:
            self.delta_prob_matrix = self.parameters["get_delta_prob_matrix"](
                dense=True, ppgpp=self.parameters.get("ppgpp_regulation", False)
            )
        else:
            # make delta_prob_matrix without adjustments
            self.delta_prob_matrix = scipy.sparse.csr_matrix(
                (
                    self.delta_prob["deltaV"],
                    (self.delta_prob["deltaI"], self.delta_prob["deltaJ"]),
                ),
                shape=self.delta_prob["shape"],
            ).toarray()

        # Determine changes from genetic perturbations
        self.genetic_perturbations = {}
        self.perturbations = self.parameters["perturbations"]
        self.rna_data = self.parameters["rna_data"]

        if len(self.perturbations) > 0:
            probability_indexes = [
                (index, self.perturbations[rna_data["id"]])
                for index, rna_data in enumerate(self.rna_data)
                if rna_data["id"] in self.perturbations
            ]

            self.genetic_perturbations = {
                "fixedRnaIdxs": [pair[0] for pair in probability_indexes],
                "fixedSynthProbs": [pair[1] for pair in probability_indexes],
            }

        # ID Groups
        self.idx_rRNA = self.parameters["idx_rRNA"]
        self.idx_mRNA = self.parameters["idx_mRNA"]
        self.idx_tRNA = self.parameters["idx_tRNA"]
        self.idx_rprotein = self.parameters["idx_rprotein"]
        self.idx_rnap = self.parameters["idx_rnap"]

        # Synthesis probabilities for different categories of genes
        self.rnaSynthProbFractions = self.parameters["rnaSynthProbFractions"]
        self.rnaSynthProbRProtein = self.parameters["rnaSynthProbRProtein"]
        self.rnaSynthProbRnaPolymerase = self.parameters["rnaSynthProbRnaPolymerase"]

        # Coordinates and transcription directions of transcription units
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.transcription_direction = self.parameters["transcription_direction"]

        self.inactive_RNAP = self.parameters["inactive_RNAP"]

        # Constants for volume calculations
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]
        self.ppgpp = self.parameters["ppgpp"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.pdmp_initiation_mode = str(
            self.parameters.get("pdmp_initiation_mode", "discrete"))
        self.transcript_init_prob_scale = float(
            self.parameters.get("transcript_init_prob_scale", 1.0))
        if self.transcript_init_prob_scale <= 0:
            raise ValueError(
                f"transcript_init_prob_scale must be > 0; got "
                f"{self.transcript_init_prob_scale!r}")
        if self.pdmp_initiation_mode not in ("discrete", "poisson"):
            raise ValueError(
                f"pdmp_initiation_mode must be 'discrete' or 'poisson'; "
                f"got {self.pdmp_initiation_mode!r}"
            )

        # Helper indices for Numpy indexing
        self.ppgpp_idx = None

    def inputs(self):
        return {
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'full_chromosomes': {'_type': FULL_CHROMOSOME_ARRAY, '_default': []},
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'bulk': {'_type': 'bulk_array', '_default': []},
            # active_RNAPs and listeners are OUTPUT-only: this process adds RNAPs
            # and publishes observations but never reads them (Phase-2 removal of
            # two dead input ports; both remain in outputs()/TOPOLOGY).
            'timestep': {'_type': 'integer', '_default': 1.0},
            'ppgpp_state': {
                'basal_prob': {'_type': 'array[float]', '_default': []},
                'frac_active_rnap': {'_type': 'float', '_default': 0.0},
            },
            'dnaa_hydrolysis': {
                'promoter_fraction': {'_type': 'float', '_default': 0.0},
            },
        }

    def outputs(self):
        return (
            {
                'bulk': 'bulk_array',
                'RNAs': RNA_ARRAY,
                'active_RNAPs': ACTIVE_RNAP_ARRAY,
                'listeners':                 {
                    'rna_synth_prob':                     {
                        'target_rna_synth_prob': {'_type': 'overwrite[array[float]]', '_default': []},
                        'actual_rna_synth_prob': {'_type': 'overwrite[array[float]]', '_default': []},
                        'max_p': {'_type': 'overwrite[float]', '_default': 0.0},
                        'tu_is_overcrowded': {'_type': 'overwrite[array[boolean]]', '_default': []},
                        'total_rna_init': {'_type': 'overwrite[integer]', '_default': 0},
                    },
                    'ribosome_data':                     {
                        'rRNA_initiated_TU': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'rRNA_init_prob_TU': {'_type': 'overwrite[array[float]]', '_default': []},
                        'total_rna_init': {'_type': 'overwrite[integer]', '_default': 0},
                    },
                    'rnap_data':                     {
                        'did_initialize': {'_type': 'overwrite[integer]', '_default': 0},
                        'rna_init_event': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'log_likelihood': {'_type': 'overwrite[float]', '_default': 0.0},
                    },
                },
            }
        )
    def update(self, states, interval=None):
        self._prepare(states)
        return self._evolve(states)

    def _prepare(self, states):
        # At first update, convert all strings to indices
        if self.ppgpp_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)

        # Read current environment
        current_media_id = states["environment"]["media_id"]

        if states["full_chromosomes"]["_entryState"].sum() > 0:
            # Get attributes of promoters
            TU_index, bound_TF = attrs(states["promoters"], ["TU_index", "bound_TF"])

            # Check if ppGpp regulation module has written state
            ppgpp = states.get("ppgpp_state", {})
            ppgpp_basal = ppgpp.get("basal_prob")
            has_ppgpp = (ppgpp_basal is not None
                         and hasattr(ppgpp_basal, '__len__')
                         and len(ppgpp_basal) > 0)

            if has_ppgpp:
                basal_prob = ppgpp_basal
                self.fracActiveRnap = ppgpp["frac_active_rnap"]
                ppgpp_scale = basal_prob[TU_index]
                # TF effect is multiplicative on the gene's ppGpp-adjusted basal
                # expression: p = basal * (1 + scale-weighted delta). A gene with
                # zero basal can never be induced, so the original code used a
                # hard `ppgpp_scale[basal == 0] = 1` switch to let a silent gene
                # take the TF delta additively at full strength. That `== 0`
                # switch is a floating-point knife-edge: whether a near-zero
                # basal lands at exactly 0 (full delta) or denormal-tiny (delta
                # suppressed) is a BLAS/underflow accident that differs across
                # numpy builds, so one truly-null gene (TU0-14529) reached ~19%
                # of all transcription on carbon-poor media in one engine and ~0
                # in the other. Replace the switch with a smooth Hill/Michaelis-
                # Menten accessibility gain — scale = b^2/(b+K): ~b for expressed
                # genes (multiplicative regime unchanged), smoothly -> 0 for
                # silent genes (b << K). This is the cooperative-threshold
                # biology (a promoter needs a minimum RNAP-recruitment competence
                # before a bound TF can act); it suppresses truly-null genes
                # consistently and is identical across underflow regimes, so the
                # engines agree. K sits in the empty gap between the numerical
                # noise floor (~1e-13) and the smallest real expression (~1e-9).
                _PPGPP_SCALE_K = 1e-11
                ppgpp_scale = ppgpp_scale ** 2 / (ppgpp_scale + _PPGPP_SCALE_K)
            else:
                basal_prob = self.basal_prob
                self.fracActiveRnap = self.fracActiveRnapDict[current_media_id]
                ppgpp_scale = 1

            # Calculate probabilities of the RNAP binding to each promoter
            self.promoter_init_probs = basal_prob[TU_index] + ppgpp_scale * np.multiply(
                self.delta_prob_matrix[TU_index, :], bound_TF
            ).sum(axis=1)

            if len(self.genetic_perturbations) > 0:
                self._rescale_initiation_probs(
                    self.genetic_perturbations["fixedRnaIdxs"],
                    self.genetic_perturbations["fixedSynthProbs"],
                    TU_index,
                )

            # Adjust probabilities to not be negative
            self.promoter_init_probs[self.promoter_init_probs < 0] = 0.0
            self.promoter_init_probs /= self.promoter_init_probs.sum()

            if not has_ppgpp:
                # Adjust synthesis probabilities depending on environment
                synthProbFractions = self.rnaSynthProbFractions[current_media_id]

                # Create masks for different types of RNAs
                is_mrna = np.isin(TU_index, self.idx_mRNA)
                is_trna = np.isin(TU_index, self.idx_tRNA)
                is_rrna = np.isin(TU_index, self.idx_rRNA)
                is_rprotein = np.isin(TU_index, self.idx_rprotein)
                is_rnap = np.isin(TU_index, self.idx_rnap)
                is_fixed = is_trna | is_rrna | is_rprotein | is_rnap

                # Rescale initiation probabilities based on type of RNA
                self.promoter_init_probs[is_mrna] *= (
                    synthProbFractions["mRna"] / self.promoter_init_probs[is_mrna].sum()
                )
                self.promoter_init_probs[is_trna] *= (
                    synthProbFractions["tRna"] / self.promoter_init_probs[is_trna].sum()
                )
                self.promoter_init_probs[is_rrna] *= (
                    synthProbFractions["rRna"] / self.promoter_init_probs[is_rrna].sum()
                )

                # Set fixed synthesis probabilities for RProteins and RNAPs
                self._rescale_initiation_probs(
                    np.concatenate((self.idx_rprotein, self.idx_rnap)),
                    np.concatenate(
                        (
                            self.rnaSynthProbRProtein[current_media_id],
                            self.rnaSynthProbRnaPolymerase[current_media_id],
                        )
                    ),
                    TU_index,
                )

                assert self.promoter_init_probs[is_fixed].sum() < 1.0

                # Scale remaining synthesis probabilities accordingly
                scaleTheRestBy = (
                    1.0 - self.promoter_init_probs[is_fixed].sum()
                ) / self.promoter_init_probs[~is_fixed].sum()
                self.promoter_init_probs[~is_fixed] *= scaleTheRestBy

            # dnaA autoregulation (dnaa-4 / Rashmi mechanism):
            # scale dnaA promoter init-probs by (1 - s*f) where f = bound fraction
            # of dnaA-box sites from the dnaa_hydrolysis port.
            promoter_fraction = float(
                states.get("dnaa_hydrolysis", {}).get("promoter_fraction", 0.0))
            if promoter_fraction > 0.0 and AUTOREG_STRENGTH > 0.0:
                dnaa_promoters = (TU_index == DNAA_TU_IDX)
                if dnaa_promoters.any():
                    self.promoter_init_probs[dnaa_promoters] *= _autoreg_factor(
                        promoter_fraction, AUTOREG_STRENGTH)
                    # Renormalize: promoter_init_probs is a distribution consumed by
                    # multinomial (which forces the last element to the remainder), so
                    # the repressed dnaA mass must redistribute proportionally, not pile
                    # onto the last promoter. Repression of dnaA is preserved (p is tiny).
                    self.promoter_init_probs /= self.promoter_init_probs.sum()

        # If there are no chromosomes in the cell, set all probs to zero
        else:
            self.promoter_init_probs = np.zeros(
                states["promoters"]["_entryState"].sum()
            )

        self.rnaPolymeraseElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ]
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnaPolymeraseElongationRate.to(units.nt / units.s).magnitude,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation,
        )

    # ---- _evolve, decomposed by concern -------------------------------------
    # Phase 1 functional regrouping (behaviour-preserving): the initiation
    # tick is a short composition of named functions, grouped as
    #   compute  - synthesis probabilities -> footprint cap -> sampling
    #   biology  - apply_initiations (the only state-changing step)
    #   observe  - initiation_listeners (derived/logged data; Phase 2 will
    #              lift this into a separate listener step, removing the
    #              `listeners` output port from this process)
    # RNG call order is preserved exactly (synthesis -> cap -> sample -> apply),
    # so the run is bit-identical to the monolithic version.

    def _evolve(self, states):
        update = self._zero_update()

        # no synthesis if no chromosome
        if len(states["full_chromosomes"]) == 0:
            return update

        # promoter attributes + the promoter<->TU incidence matrix (shared setup)
        TU_index, domain_index_promoters = attrs(
            states["promoters"], ["TU_index", "domain_index"]
        )
        TU_to_promoter = self._tu_to_promoter(states, TU_index)

        # compute: target synthesis probabilities -> RNAPs to activate
        target_TU_synth_probs, n_RNAPs_to_activate = self._synthesis_probabilities(
            states, TU_to_promoter
        )
        update["listeners"]["rna_synth_prob"]["target_rna_synth_prob"] = (
            target_TU_synth_probs
        )
        if n_RNAPs_to_activate == 0:
            return update

        # compute: footprint-crowding cap -> actual probabilities
        max_p, actual_TU_synth_probs, tu_is_overcrowded = self._cap_overcrowding(
            TU_to_promoter, n_RNAPs_to_activate, states["timestep"]
        )

        # compute: sample per-promoter initiation counts
        n_initiations, poisson_means = self._sample_initiations(
            states, n_RNAPs_to_activate
        )
        # The realised event count can differ from the target (Poisson tau-leap
        # fluctuates; multinomial always matches). Rebind to the observed sum.
        n_RNAPs_to_activate = int(n_initiations.sum())

        # biology: the only state change - new RNAs + RNAPs, inactive-RNAP draw
        update.update(
            self._apply_initiations(
                states, TU_index, domain_index_promoters, n_initiations
            )
        )

        # observe: derived/logged data (candidate for a separate listener step)
        update["listeners"] = self._initiation_listeners(
            TU_to_promoter, n_initiations, poisson_means, n_RNAPs_to_activate,
            target_TU_synth_probs, actual_TU_synth_probs, max_p, tu_is_overcrowded,
        )
        return update

    def _zero_update(self):
        """Base update: zeroed listeners + empty biological slots. Returned as-is
        when there is no chromosome / nothing to activate (with the one field
        the early paths compute set by the caller)."""
        return {
            "listeners": {
                "rna_synth_prob": {
                    "target_rna_synth_prob": np.zeros(self.n_TUs),
                    "actual_rna_synth_prob": np.zeros(self.n_TUs),
                    "tu_is_overcrowded": np.zeros(self.n_TUs, dtype=np.bool_),
                    "total_rna_init": 0,
                    "max_p": 0.0,
                },
                "ribosome_data": {"total_rna_init": 0},
                "rnap_data": {
                    "did_initialize": 0,
                    "rna_init_event": np.zeros(self.n_TUs, dtype=np.int64),
                },
            },
            "active_RNAPs": {},
            "full_chromosomes": {},
            "promoters": {},
            "RNAs": {},
        }

    def _tu_to_promoter(self, states, TU_index):
        """Sparse promoter->TU incidence matrix (maps per-promoter quantities to
        per-transcription-unit totals)."""
        n_promoters = states["promoters"]["_entryState"].sum()
        return scipy.sparse.csr_matrix(
            (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
            shape=(self.n_TUs, n_promoters),
            dtype=np.int8,
        )

    def _synthesis_probabilities(self, states, TU_to_promoter):
        """compute: per-TU target synthesis probabilities and the integer number
        of RNAPs to activate this tick (activation probability x inactive pool).
        Note: activation prob uses the target (not the yet-unknown actual)
        probabilities - the difference is small and this matches the original."""
        target_TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
        self.activationProb = self._calculateActivationProb(
            states["timestep"],
            self.fracActiveRnap,
            self.rnaLengths,
            (units.nt / units.s) * self.elongation_rates,
            target_TU_synth_probs,
        )
        n_RNAPs_to_activate = np.int64(
            self.activationProb * counts(states["bulk"], self.inactive_RNAP_idx)
        )
        return target_TU_synth_probs, n_RNAPs_to_activate

    def _cap_overcrowding(self, TU_to_promoter, n_RNAPs_to_activate, timestep):
        """compute: cap each promoter's init probability at the physically
        allowed max_p (from the RNAP footprint), redistributing the excess mass;
        returns max_p, the post-cap actual per-TU probabilities, and which TUs
        were overcrowded. Mutates self.promoter_init_probs in place (as before)."""
        max_p = (
            self.rnaPolymeraseElongationRate
            / self.active_rnap_footprint_size
            * (units.s)
            * timestep
            / n_RNAPs_to_activate
        ).magnitude
        is_overcrowded = self.promoter_init_probs > max_p
        while np.any(self.promoter_init_probs > max_p):
            self.promoter_init_probs[is_overcrowded] = max_p
            scale_the_rest_by = (
                1.0 - self.promoter_init_probs[is_overcrowded].sum()
            ) / self.promoter_init_probs[~is_overcrowded].sum()
            self.promoter_init_probs[~is_overcrowded] *= scale_the_rest_by
            is_overcrowded |= self.promoter_init_probs > max_p
        actual_TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
        tu_is_overcrowded = TU_to_promoter.dot(is_overcrowded).astype(bool)
        return max_p, actual_TU_synth_probs, tu_is_overcrowded

    def _sample_initiations(self, states, n_RNAPs_to_activate):
        """compute (stochastic): draw per-promoter initiation counts. Two modes -
        ``discrete`` is a single multinomial (exact sum = n_RNAPs_to_activate);
        ``poisson`` is per-promoter Poisson tau-leap with a resource cap
        (independent marginals, sum fluctuates then is capped to the inactive
        pool). Returns (n_initiations, poisson_means) - poisson_means is None in
        discrete mode (used only for the log-likelihood listener)."""
        if self.pdmp_initiation_mode == "poisson":
            poisson_means = (self.transcript_init_prob_scale
                             * n_RNAPs_to_activate
                             * self.promoter_init_probs)
            n_initiations = self.random_state.poisson(poisson_means).astype(np.int64)
            # Resource cap: total activations cannot exceed the actual inactive
            # pool. Capping the (symmetric-variance) Poisson sum only from above
            # would systematically undercount, so subsample events weighted by
            # per-promoter counts.
            n_inactive_RNAPs = int(counts(states["bulk"], self.inactive_RNAP_idx))
            if int(n_initiations.sum()) > n_inactive_RNAPs:
                event_promoters = np.repeat(
                    np.arange(n_initiations.size), n_initiations,
                )
                keep_idx = self.random_state.choice(
                    event_promoters.size, size=n_inactive_RNAPs, replace=False,
                )
                n_initiations = np.bincount(
                    event_promoters[keep_idx], minlength=n_initiations.size,
                ).astype(np.int64)
            return n_initiations, poisson_means
        n_initiations = self.random_state.multinomial(
            n_RNAPs_to_activate, self.promoter_init_probs
        )
        return n_initiations, None

    def _apply_initiations(self, states, TU_index, domain_index_promoters,
                           n_initiations):
        """biology: the tick's only state change. Add one partially-transcribed
        RNA + one active RNAP per initiation event (linked via RNAP_index) and
        decrement the inactive-RNAP bulk pool. Returns the RNAs/active_RNAPs/bulk
        update fragments."""
        n = int(n_initiations.sum())
        TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
        domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)
        coordinates = self.replication_coordinate[TU_index_partial_RNAs]
        is_forward = self.transcription_direction[TU_index_partial_RNAs]

        RNAP_indexes = create_unique_indices(n, states["RNAs"])
        is_mRNA = np.isin(TU_index_partial_RNAs, self.idx_mRNA)
        return {
            "active_RNAPs": {"add": {
                "unique_index": RNAP_indexes,
                "domain_index": domain_index_rnap,
                "coordinates": coordinates,
                "is_forward": is_forward,
            }},
            "bulk": [(self.inactive_RNAP_idx, -n_initiations.sum())],
            "RNAs": {"add": {
                "TU_index": TU_index_partial_RNAs,
                "transcript_length": np.zeros(cast(int, n)),
                "is_mRNA": is_mRNA,
                "is_full_transcript": np.zeros(cast(int, n), dtype=bool),
                "can_translate": is_mRNA,
                "RNAP_index": RNAP_indexes,
            }},
        }

    def _initiation_listeners(self, TU_to_promoter, n_initiations, poisson_means,
                              n_RNAPs_to_activate, target_TU_synth_probs,
                              actual_TU_synth_probs, max_p, tu_is_overcrowded):
        """observe: derived/logged diagnostics - synthesis-probability targets vs
        actuals + crowding, rRNA initiation counts, and rnap_data (did_initialize,
        per-TU init events, and the Poisson log-likelihood of the draw). Pure
        function of the tick's results - the Phase 2 externalization target."""
        rna_init_event = TU_to_promoter.dot(n_initiations)
        rRNA_initiations = rna_init_event[self.idx_rRNA]
        # Per-tick log-likelihood of the observed counts under the Poisson rates
        # that drove the sampler (0.0 sentinel in discrete mode until wired).
        if self.pdmp_initiation_mode == "poisson":
            log_lik = float(poisson.logpmf(n_initiations, poisson_means).sum())
        else:
            log_lik = 0.0
        return {
            "rna_synth_prob": {
                "target_rna_synth_prob": target_TU_synth_probs,
                "actual_rna_synth_prob": actual_TU_synth_probs,
                "tu_is_overcrowded": tu_is_overcrowded,
                "max_p": max_p,
                "total_rna_init": n_RNAPs_to_activate,
            },
            "ribosome_data": {
                "rRNA_initiated_TU": rRNA_initiations.astype(int),
                "rRNA_init_prob_TU": rRNA_initiations / float(n_RNAPs_to_activate),
                "total_rna_init": n_RNAPs_to_activate,
            },
            "rnap_data": {
                "did_initialize": n_RNAPs_to_activate,
                "rna_init_event": rna_init_event.astype(np.int64),
                "log_likelihood": log_lik,
            },
        }

    def _calculateActivationProb(
        self,
        timestep,
        fracActiveRnap,
        rnaLengths,
        rnaPolymeraseElongationRates,
        synthProb,
    ):
        """
        Calculate expected RNAP termination rate based on RNAP elongation rate
        - allTranscriptionTimes: Vector of times required to transcribe each
        transcript
        - allTranscriptionTimestepCounts: Vector of numbers of timesteps
        required to transcribe each transcript
        - averageTranscriptionTimeStepCounts: Average number of timesteps
        required to transcribe a transcript, weighted by synthesis
        probabilities of each transcript
        - expectedTerminationRate: Average number of terminations in one
        timestep for one transcript
        """
        allTranscriptionTimes = 1.0 / rnaPolymeraseElongationRates * rnaLengths
        timesteps = (1.0 / (timestep * units.s) * allTranscriptionTimes).magnitude
        allTranscriptionTimestepCounts = np.ceil(timesteps)
        averageTranscriptionTimestepCounts = np.dot(
            synthProb, allTranscriptionTimestepCounts
        )
        expectedTerminationRate = 1.0 / averageTranscriptionTimestepCounts

        """
        Modify given fraction of active RNAPs to take into account early
        terminations in between timesteps
        - allFractionTimeInactive: Vector of probabilities an "active" RNAP
        will in effect be "inactive" because it has terminated during a
        timestep
        - averageFractionTimeInactive: Average probability of an "active" RNAP
        being in effect "inactive", weighted by synthesis probabilities
        - effectiveFracActiveRnap: New higher "goal" for fraction of active
        RNAP, considering that the "effective" fraction is lower than what the
        listener sees
        """
        allFractionTimeInactive = (
            1
            - (1.0 / (timestep * units.s) * allTranscriptionTimes).magnitude
            / allTranscriptionTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, synthProb)
        effectiveFracActiveRnap = fracActiveRnap / (1 - averageFractionTimeInactive)

        # Return activation probability that will balance out the expected termination rate
        activation_prob = (
            effectiveFracActiveRnap
            * expectedTerminationRate
            / (1 - effectiveFracActiveRnap)
        )

        if activation_prob > 1:
            activation_prob = 1

        return activation_prob

    def _rescale_initiation_probs(self, fixed_indexes, fixed_synth_probs, TU_index):
        """
        Rescales the initiation probabilities of each promoter such that the
        total synthesis probabilities of certain types of RNAs are fixed to
        a predetermined value. For instance, if there are two copies of
        promoters for RNA A, whose synthesis probability should be fixed to
        0.1, each promoter is given an initiation probability of 0.05.
        """
        for idx, synth_prob in zip(fixed_indexes, fixed_synth_probs):
            fixed_mask = TU_index == idx
            self.promoter_init_probs[fixed_mask] = synth_prob / fixed_mask.sum()
