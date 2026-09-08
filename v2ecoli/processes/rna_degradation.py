"""
===============
RNA Degradation
===============

Mathematical Model
------------------
RNA degradation is modeled as two sequential enzymatic steps:

**Step 1 -- Endonucleolytic cleavage (endoRNases)**

The steady-state degradation rate for each RNA species i is given by
competitive Michaelis-Menten kinetics across all RNA substrates:

    v_i = kcat_endo * [EndoRNase] * ([r_i]/Km_i) / (1 + sum_j([r_j]/Km_j))

where:
    - kcat_endo (1/s): catalytic rate constant of endoRNases
    - [r_i] (mol/L): concentration of RNA species i
    - Km_i (mol/L): Michaelis-Menten constant for species i, fitted so that
      the steady-state rate recapitulates first-order decay:
      kd_i * r_i = kcat_endo * [EndoRNase] * [r_i]/Km_i / (1 + sum([r_j]/Km_j))

The fraction of endoRNase capacity allocated to each RNA:

    f_i = ([r_i]/Km_i) / (1 + sum_j([r_j]/Km_j))

Total RNAs to degrade per class (mRNA, tRNA, rRNA) are computed from the
class-specific sum of f_i, then distributed to individual RNAs via
multinomial sampling weighted by f_i within each class.

Cleaved RNAs produce polymerized nucleotide fragments (fragment bases)
plus pyrophosphate (PPi), modeled via a stoichiometric matrix:

    delta_fragments = S_endo @ n_degraded

**Step 2 -- Exonucleolytic digestion (exoRNases)**

Fragment bases from step 1 are hydrolyzed into free NMPs:

    FragmentBase + H2O  -->  NMP + H+

The exoRNase capacity per timestep:

    C_exo = sum(n_exoRNases) * kcat_exo * dt

If C_exo >= total fragments, all are digested. Otherwise, fragments are
partially digested via multinomial sampling proportional to fragment
base composition.
"""

import numpy as np

from bigraph_schema.contract import ProcessContract

from v2ecoli.library.schema import (
    bulk_name_to_idx,
    counts,
    attrs,
)

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity

# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess
from v2ecoli.library.schema_types import RNA_ARRAY, ACTIVE_RIBOSOME_ARRAY


# Register default topology for this process, associating it with process name
NAME = "ecoli-rna-degradation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}


class RnaDegradation(PartitionedProcess):
    """RNA Degradation PartitionedProcess"""

    description = (
        "RNA Degradation — endo cleavage (competitive Michaelis–Menten) then exo digestion.\n\n"
        "    fᵢ = ([rᵢ]/Kmᵢ) / (1 + ∑ⱼ [rⱼ]/Kmⱼ);   νᵢ = kcat_endo · [EndoRNase] · fᵢ\n"
        "    per class c: n_deg_c = Draw(∑_{i∈c} νᵢ · dt · V · N_A), multinomial∝fᵢ, ≤ supply\n"
        "    Δfragments = S_endo · n_deg;   exo: C_exo = ∑n_exo · kcat_exo · dt\n"
        "    FragmentBase + H2O → NMP + H⁺   (full digest if C_exo ≥ Σbases, else multinomial)\n"
        "  [rᵢ]: RNA conc (mol/L); Kmᵢ: MM constant; νᵢ: endo flux; S_endo: cleavage stoich (NMP+PPi)."
    )

    name = NAME
    topology = TOPOLOGY

    contract = ProcessContract(
        summary=(
            "Partitioned process. calculate_request sizes RNA degradation from "
            "competitive Michaelis–Menten endoRNase kinetics (per-class "
            "stochastic draws distributed by saturation fraction fᵢ, bounded by "
            "supply) and reserves the RNAs, water and fragment bases needed; "
            "evolve_state applies endonucleolytic cleavage (Δfragments = "
            "S_endo·n_deg) then exonucleolytic digestion of fragment bases into "
            "NMPs."
        ),
        symbols={
            "rᵢ": "concentration [rᵢ] of RNA species i (mol/L)",
            "Kmᵢ": "fitted Michaelis–Menten constant for RNA species i (mol/L)",
            "fᵢ": "endoRNase saturation fraction for RNA i = ([rᵢ]/Kmᵢ)/(1+∑ⱼ[rⱼ]/Kmⱼ) (dimensionless)",
            "kcat_endo": "catalytic rate constant of endoRNases (1/s)",
            "EndoRNase": "endoRNase enzyme; [EndoRNase] its concentration/count",
            "νᵢ": "endonucleolytic degradation flux of RNA i = kcat_endo·[EndoRNase]·fᵢ",
            "n_deg_c": "total RNAs to degrade for class c ∈ {mRNA, tRNA, rRNA} (count)",
            "dt": "timestep (s)",
            "V": "cell volume from cell_mass/cell_density (L)",
            "N_A": "Avogadro constant (1/mol), for counts↔concentration conversion",
            "n_deg": "per-species vector of RNAs degraded this tick (count)",
            "S_endo": "endo cleavage stoichiometry matrix (metabolites × RNAs): NMP fragments + PPi",
            "C_exo": "exoRNase digestion capacity this tick = ∑n_exo·kcat_exo·dt (bases)",
            "n_exo": "exoRNase enzyme counts (count)",
            "kcat_exo": "catalytic rate constant of exoRNases (1/s)",
            "FragmentBase": "polymerized nucleotide fragment released by endo cleavage",
            "H2O": "water consumed hydrolyzing fragment bases in exonucleolytic digestion (one per base; endonucleolytic cleavage consumes none) (count)",
            "NMP": "free nucleotide monophosphate released by exo digestion (count)",
            "H⁺": "proton released per fragment base digested (count)",
        },
        inputs={
            "bulk": "reads RNA counts, charged-tRNA counts (added to uncharged), endoRNase and exoRNase enzyme counts, fragment-base counts, and water; endo/exo counts and Kms set the degradation rates, and after partitioning the allocated RNA/water/fragment counts drive the reactions",
            "RNAs": "reads unique mRNA entries (TU_index, can_translate, is_full_transcript) to add translatable mRNAs to the degradation pool, deactivate cleaved mRNAs, and delete full inactive transcripts",
            "listeners": "reads mass.cell_mass to convert molecule counts to concentrations (V = cell_mass/cell_density) for the Michaelis–Menten rates",
            "timestep": "tick length dt (s) scaling the per-class degradation draws and the exoRNase capacity",
        },
        outputs={
            "bulk": "decrements degraded bulk RNAs, adds endo-cleavage fragment bases + PPi (endo cleavage consumes no water — its stoich water row is 0), then digests fragment bases into NMPs (+H⁺), consuming one H2O per fragment base in the exonucleolytic step",
            "RNAs": "sets can_translate=False on endonucleolytically cleaved unique mRNAs and deletes full inactive mRNAs that are degraded",
            "listeners": "writes rna_degradation_listener: per-cistron and (optionally) per-RNA degraded counts, nucleotides_from_degradation, fraction_active_endoRNases (∑fᵢ), diff_relative_first_order_decay, and fragment_bases_digested",
        },
        config={
            "Kcat_endoRNases": "per-endoRNase catalytic rate constants kcat_endo (1/s)",
            "Kms": "per-RNA Michaelis–Menten constants Kmᵢ (mol/L), fitted to recapitulate first-order decay",
            "kcat_exoRNase": "exoRNase catalytic rate constant kcat_exo (1/s) setting exo digestion capacity",
            "rna_deg_rates": "first-order RNA degradation rates (1/s) used to report divergence from first-order decay",
            "is_mRNA": "boolean mask selecting mRNA-class RNAs (degraded as unique/deactivated)",
            "is_tRNA": "boolean mask selecting tRNA-class RNAs",
            "is_rRNA": "boolean mask selecting rRNA-class RNAs",
            "is_miscRNA": "boolean mask selecting miscRNAs",
            "degrade_misc": "when true, degrade miscRNAs together with mRNAs (as one transient class)",
            "cell_density": "cell density ρ (g/L) converting cell_mass → volume for concentrations",
            "n_avogadro": "Avogadro constant N_A (1/mol) for counts↔concentration conversion",
            "nt_counts": "per-RNA nucleotide composition, forming the NMP rows of S_endo",
            "rna_lengths": "per-RNA length (nt) setting water needed for hydrolysis and total NMP yield",
            "endoRNase_ids": "bulk ids of endoRNases whose counts drive endonucleolytic flux",
            "exoRNase_ids": "bulk ids of exoRNases whose counts set exonucleolytic capacity",
            "charged_trna_names": "charged tRNA ids added to uncharged tRNA counts when sizing tRNA degradation",
            "uncharged_trna_indexes": "indices where charged-tRNA counts are folded into the RNA pool",
            "cistron_tu_mapping_matrix": "maps degraded transcription-unit counts to per-cistron degraded counts",
            "polymerized_ntp_ids": "fragment-base (polymerized NTP) ids produced by endo cleavage and digested by exo",
            "nmp_ids": "NMP ids produced by exonucleolytic digestion",
        },
        assumptions=[
            "Km values are fitted so the steady-state Michaelis–Menten rate recapitulates first-order RNA decay kd_i·r_i.",
            "Bulk RNAs (tRNA/rRNA) are degraded immediately; unique mRNAs are deactivated on cleavage but not deleted until they are full transcripts, simplifying transcript elongation.",
            "Cleaved RNA nucleotides are lumped as one linear chain of fragment bases (5' phosphate removed as PPi); exo digestion has no sequence specificity — one H2O consumed and one H⁺ released per base, and if C_exo ≥ total bases all fragments are digested, else a composition-weighted multinomial subsample.",
        ],
    )

    config_schema = {
        'Kcat_endoRNases': {'_type': 'quantity[array[float],1/s]', '_default': np.array([], dtype=float)},
        'Kms': {'_type': 'quantity[array[float],mol/L]', '_default': np.array([], dtype=float)},
        'all_rna_ids': {'_type': 'list[string]', '_default': []},
        'cell_density': {'_type': 'quantity[g/L]', '_default': 1100.0},
        'charged_trna_names': {'_type': 'list[string]', '_default': []},
        'cistron_ids': {'_type': 'list[string]', '_default': []},
        'cistron_tu_mapping_matrix': {'_type': 'csr_matrix', '_default': []},
        'degrade_misc': {'_type': 'boolean', '_default': False},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'endoRNase_ids': {'_type': 'list[string]', '_default': []},
        'exoRNase_ids': {'_type': 'list[string]', '_default': []},
        'is_mRNA': {'_type': 'array[boolean]', '_default': np.array([], dtype=float)},
        'is_miscRNA': {'_type': 'array[boolean]', '_default': np.array([], dtype=float)},
        'is_rRNA': {'_type': 'array[boolean]', '_default': np.array([], dtype=float)},
        'is_tRNA': {'_type': 'array[boolean]', '_default': np.array([], dtype=float)},
        'kcat_exoRNase': {'_type': 'quantity[float,1/s]', '_default': np.array([], dtype=float)},
        'mature_rna_cistron_indexes': {'_type': 'array[integer]', '_default': []},
        'mature_rna_ids': {'_type': 'list[string]', '_default': []},
        'n_avogadro': {'_type': 'quantity[float,1/mol]', '_default': 0.0},
        'n_total_RNAs': {'_type': 'integer', '_default': 0},
        'nmp_ids': {'_type': 'list[string]', '_default': []},
        'nt_counts': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'polymerized_ntp_ids': {'_type': 'list[string]', '_default': []},
        'ppi_id': {'_type': 'string', '_default': 'ppi'},
        'proton_id': {'_type': 'string', '_default': 'h+'},
        'ribosome30S': {'_type': 'string', '_default': 'ribosome30S'},
        'ribosome50S': {'_type': 'string', '_default': 'ribosome50S'},
        'rna_deg_rates': {'_type': 'quantity[array[float],1/s]', '_default': []},
        'rna_ids': {'_type': 'list[string]', '_default': []},
        'rna_lengths': {'_type': 'array[integer[nt]]', '_default': np.array([], dtype=float)},
        'rrfa_idx': {'_type': 'integer', '_default': 0},
        'rrla_idx': {'_type': 'integer', '_default': 0},
        'rrsa_idx': {'_type': 'integer', '_default': 0},
        'seed': {'_type': 'integer', '_default': 0},
        'uncharged_trna_indexes': {'_type': 'array[integer]', '_default': []},
        'water_id': {'_type': 'string', '_default': 'h2o'},
        # Per-RNA degradation counts emit ~50KB/ts (mostly zeros).
        # Off by default; aggregate stats (nucleotides_from_degradation,
        # count_RNA_degraded_per_cistron) are always emitted.
        'emit_per_rna_degradation_counts': {'_type': 'boolean', '_default': False},
    }

    def initialize(self, config):
        self.rna_ids = self.parameters["rna_ids"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.n_transcribed_rnas = len(self.rna_ids)
        self.mature_rna_exists = len(self.mature_rna_ids) > 0
        self.cistron_ids = self.parameters["cistron_ids"]
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]
        self.mature_rna_cistron_indexes = self.parameters["mature_rna_cistron_indexes"]
        self.all_rna_ids = self.parameters["all_rna_ids"]
        self.n_total_RNAs = self.parameters["n_total_RNAs"]

        # Load constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Load RNase kinetic data
        self.endoRNase_ids = self.parameters["endoRNase_ids"]
        self.exoRNase_ids = self.parameters["exoRNase_ids"]
        self.kcat_exoRNase = self.parameters["kcat_exoRNase"]
        self.Kcat_endoRNases = self.parameters["Kcat_endoRNases"]

        # Load information about uncharged/charged tRNA
        self.uncharged_trna_indexes = self.parameters["uncharged_trna_indexes"]
        self.charged_trna_names = self.parameters["charged_trna_names"]

        # Load first-order RNA degradation rates
        # (estimated by mRNA half-life data)
        self.rna_deg_rates = self.parameters["rna_deg_rates"]

        self.is_mRNA = self.parameters["is_mRNA"]
        self.is_rRNA = self.parameters["is_rRNA"]
        self.is_tRNA = self.parameters["is_tRNA"]

        # NEW to vivarium-ecoli
        self.is_miscRNA = self.parameters["is_miscRNA"]
        self.degrade_misc = self.parameters["degrade_misc"]

        self.rna_lengths = self.parameters["rna_lengths"]
        self.nt_counts = self.parameters["nt_counts"]

        # Build stoichiometric matrix
        self.polymerized_ntp_ids = self.parameters["polymerized_ntp_ids"]
        self.nmp_ids = self.parameters["nmp_ids"]
        self.water_id = self.parameters["water_id"]
        self.ppi_id = self.parameters["ppi_id"]
        self.proton_id = self.parameters["proton_id"]

        self.end_cleavage_metabolite_ids = self.polymerized_ntp_ids + [
            self.water_id,
            self.ppi_id,
            self.proton_id,
        ]
        nmp_idx = list(range(4))
        water_idx = self.end_cleavage_metabolite_ids.index(self.water_id)
        ppi_idx = self.end_cleavage_metabolite_ids.index(self.ppi_id)
        proton_idx = self.end_cleavage_metabolite_ids.index(self.proton_id)
        self.endo_degradation_stoich_matrix = np.zeros(
            (len(self.end_cleavage_metabolite_ids), self.n_total_RNAs), np.int64
        )
        self.endo_degradation_stoich_matrix[nmp_idx, :] = self.nt_counts.T
        self.endo_degradation_stoich_matrix[water_idx, :] = 0
        self.endo_degradation_stoich_matrix[ppi_idx, :] = 1
        self.endo_degradation_stoich_matrix[proton_idx, :] = 0

        # Load Michaelis-Menten constants fitted to recapitulate
        # first-order RNA decay model
        self.Kms = self.parameters["Kms"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_per_rna = self.parameters.get(
            "emit_per_rna_degradation_counts", False)

        # Numpy indices for bulk molecules
        self.water_idx = None

    def inputs(self):
        return (
            {
                'bulk': {'_type': 'bulk_array', '_default': []},
                'RNAs': {'_type': RNA_ARRAY, '_default': []},
                'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
                'listeners':                 {
                    'mass':                     {
                        'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                    },
                },
                'timestep': {'_type': 'integer[s]', '_default': 1},
            }
        )

    def outputs(self):
        return (
            {
                'bulk': {'_type': 'bulk_array', '_default': []},
                'RNAs': {'_type': RNA_ARRAY, '_default': []},
                'listeners':                 {
                    'rna_degradation_listener':                     {
                        'count_rna_degraded': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'nucleotides_from_degradation': {'_type': 'overwrite[integer[nt]]', '_default': 0},
                        'count_RNA_degraded_per_cistron': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'fraction_active_endoRNases': {'_type': 'overwrite[float]', '_default': 0.0},  # sum(f_i), dimensionless
                        'diff_relative_first_order_decay': {'_type': 'overwrite[float]', '_default': 0.0},
                        'fract_endo_rrna_counts': {'_type': 'overwrite[array[float]]', '_default': []},
                        'fragment_bases_digested': {'_type': 'overwrite[integer[nt]]', '_default': 0},
                    },
                },
            }
        )

    def calculate_request(self, timestep, states):
        """Reserve bulk counts needed for this timestep's degradation.

        Mathematical Model
        ------------------
        Inputs (read):
            - bulk.{RNA_i}                     RNA counts, per species (integer)
            - bulk.{tRNA_i}                    charged tRNA counts (integer)
            - bulk.{EndoRNase}                 active endoRNase counts (integer)
            - bulk.{ExoRNase}, bulk.H2O        exo capacity substrates
            - listeners.mass.cell_mass         fg (to convert counts → concentrations)
            - timestep                         dt (s)

        Calculation:
            1. Endoribonucleolytic rate for each RNA species i (Michaelis-Menten):

                f_i = ([r_i]/Km_i) / (1 + sum_j([r_j]/Km_j))
                v_endo_i = kcat_endo * [EndoRNase] * f_i

            2. Expected degradation draws over dt, per class c ∈ {mRNA, tRNA, rRNA}:

                n_requested_c = StochasticDraw( sum_{i∈c}(v_endo_i * dt * V * N_A) )

            3. Within each class, distribute to individual RNAs via
               multinomial sampling weighted by f_i, then take minimum
               with current count so the request is bounded by supply.

            4. Water / fragment-base requests for the exoribonucleolytic
               step are computed from the stoichiometric S_endo matrix
               applied to the tentative per-RNA degradation vector.

        Outputs (request port):
            - bulk.request.{RNA_i}             copies of RNA i we want to consume
            - bulk.request.{H2O}               water needed for (N_i - 1) hydrolyses
            - bulk.request.{FragmentBase_k}    fragment bases the exo step may touch
        """
        if self.water_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.charged_trna_idx = bulk_name_to_idx(self.charged_trna_names, bulk_ids)
            self.bulk_rnas_idx = bulk_name_to_idx(self.all_rna_ids, bulk_ids)
            self.nmps_idx = bulk_name_to_idx(self.nmp_ids, bulk_ids)
            self.fragment_metabolites_idx = bulk_name_to_idx(
                self.end_cleavage_metabolite_ids, bulk_ids
            )
            self.fragment_bases_idx = bulk_name_to_idx(
                self.polymerized_ntp_ids, bulk_ids
            )
            self.endoRNase_idx = bulk_name_to_idx(self.endoRNase_ids, bulk_ids)
            self.exoRNase_idx = bulk_name_to_idx(self.exoRNase_ids, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water_id, bulk_ids)
            self.proton_idx = bulk_name_to_idx(self.proton_id, bulk_ids)

        # counts_to_molar = 1 / (N_A * V)  [1/mol * 1/L = mol/L per count]
        # .to(units.L) forces dimensional reduction; without it pint keeps
        # cell_mass/cell_density as `femtogram·liter/gram` (fg/g = 1e-15
        # dimensionless but pint doesn't simplify mass/mass across prefixes),
        # which cascades to wrong magnitudes in rna_conc_molar/Kms.
        cell_mass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        cell_volume = (cell_mass / self.cell_density).to(units.L)
        counts_to_molar = (1 / (self.n_avogadro * cell_volume)).to(units.mol / units.L)

        # Get total counts of RNAs including free rRNAs, uncharged and charged tRNAs, and
        # active (translatable) unique mRNAs
        bulk_RNA_counts = counts(states["bulk"], self.bulk_rnas_idx)
        bulk_RNA_counts[self.uncharged_trna_indexes] += counts(
            states["bulk"], self.charged_trna_idx
        )

        TU_index, can_translate, is_full_transcript = attrs(
            states["RNAs"], ["TU_index", "can_translate", "is_full_transcript"]
        )

        TU_index_translatable_mRNAs = TU_index[can_translate]
        unique_RNA_counts = np.bincount(
            TU_index_translatable_mRNAs, minlength=self.n_total_RNAs
        )
        total_RNA_counts = bulk_RNA_counts + unique_RNA_counts

        # Compute RNA concentrations
        rna_conc_molar = counts_to_molar * total_RNA_counts

        # Get counts of endoRNases
        endoRNase_counts = counts(states["bulk"], self.endoRNase_idx)
        total_kcat_endoRNase = np.dot(self.Kcat_endoRNases, endoRNase_counts)

        # Michaelis-Menten saturation fraction per RNA:
        # f_i = ([r_i]/Km_i) / (1 + sum_j([r_j]/Km_j))
        frac_endoRNase_saturated = (
            rna_conc_molar / self.Kms / (1 + np.sum(rna_conc_molar / self.Kms))
        ).magnitude

        # Calculate difference in degradation rates from first-order decay
        # and the number of EndoRNases per one molecule of RNA
        total_endoRNase_counts = np.sum(endoRNase_counts)
        diff_relative_first_order_decay = np.sum(
            np.abs(
                self.rna_deg_rates * total_RNA_counts
                - total_kcat_endoRNase * frac_endoRNase_saturated
            )
        )
        endoRNase_per_rna = total_endoRNase_counts / np.sum(total_RNA_counts)

        requests = {"listeners": {"rna_degradation_listener": {}}}
        requests["listeners"]["rna_degradation_listener"][
            "fraction_active_endoRNases"
        ] = np.sum(frac_endoRNase_saturated)
        requests["listeners"]["rna_degradation_listener"][
            "diff_relative_first_order_decay"
        ] = diff_relative_first_order_decay.magnitude
        requests["listeners"]["rna_degradation_listener"]["fract_endo_rrna_counts"] = (
            endoRNase_per_rna
        )

        # Dissect RNAse specificity into mRNA, tRNA, and rRNA
        # NEW to vivarium-ecoli: Degrade miscRNAs and mRNAs together
        if self.degrade_misc:
            is_transient_rna = self.is_mRNA | self.is_miscRNA
            mrna_specificity = np.dot(frac_endoRNase_saturated, is_transient_rna)
        else:
            mrna_specificity = np.dot(frac_endoRNase_saturated, self.is_mRNA)
        trna_specificity = np.dot(frac_endoRNase_saturated, self.is_tRNA)
        rrna_specificity = np.dot(frac_endoRNase_saturated, self.is_rRNA)

        n_total_mrnas_to_degrade = self._calculate_total_n_to_degrade(
            states["timestep"], mrna_specificity, total_kcat_endoRNase
        )
        n_total_trnas_to_degrade = self._calculate_total_n_to_degrade(
            states["timestep"], trna_specificity, total_kcat_endoRNase
        )
        n_total_rrnas_to_degrade = self._calculate_total_n_to_degrade(
            states["timestep"], rrna_specificity, total_kcat_endoRNase
        )

        # Compute RNAse specificity
        rna_specificity = frac_endoRNase_saturated / np.sum(frac_endoRNase_saturated)

        # Boolean variable that tracks existence of each RNA
        rna_exists = (total_RNA_counts > 0).astype(np.int64)

        # Compute degradation probabilities of each RNA: for mRNAs and rRNAs, this
        # is based on the specificity of each mRNA. For tRNAs and rRNAs,
        # this is distributed evenly.
        if self.degrade_misc:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, is_transient_rna * rna_exists)
                * rna_specificity
                * is_transient_rna
                * rna_exists
            )
        else:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, self.is_mRNA * rna_exists)
                * rna_specificity
                * self.is_mRNA
                * rna_exists
            )
        rrna_deg_probs = (
            1.0
            / np.dot(rna_specificity, self.is_rRNA * rna_exists)
            * rna_specificity
            * self.is_rRNA
            * rna_exists
        )
        trna_deg_probs = (
            1.0 / np.dot(self.is_tRNA, rna_exists) * self.is_tRNA * rna_exists
        )

        # Mask RNA counts into each class of RNAs
        if self.degrade_misc:
            mrna_counts = total_RNA_counts * is_transient_rna
        else:
            mrna_counts = total_RNA_counts * self.is_mRNA
        trna_counts = total_RNA_counts * self.is_tRNA
        rrna_counts = total_RNA_counts * self.is_rRNA

        # Determine number of individual RNAs to be degraded for each class
        # of RNA.
        n_mrnas_to_degrade = self._get_rnas_to_degrade(
            n_total_mrnas_to_degrade, mrna_deg_probs, mrna_counts
        )
        n_trnas_to_degrade = self._get_rnas_to_degrade(
            n_total_trnas_to_degrade, trna_deg_probs, trna_counts
        )
        n_rrnas_to_degrade = self._get_rnas_to_degrade(
            n_total_rrnas_to_degrade, rrna_deg_probs, rrna_counts
        )
        n_RNAs_to_degrade = n_mrnas_to_degrade + n_trnas_to_degrade + n_rrnas_to_degrade

        # Bulk RNAs (tRNAs and rRNAs) are degraded immediately. Unique RNAs
        # (mRNAs) are immediately deactivated (becomes unable to bind
        # ribosomes), but not degraded until transcription is finished and the
        # mRNA becomes a full transcript to simplify the transcript elongation
        # process.
        n_bulk_RNAs_to_degrade = n_RNAs_to_degrade.copy()
        n_bulk_RNAs_to_degrade[self.is_mRNA.astype(bool)] = 0
        self.n_unique_RNAs_to_deactivate = n_RNAs_to_degrade.copy()
        self.n_unique_RNAs_to_deactivate[np.logical_not(self.is_mRNA.astype(bool))] = 0

        requests.setdefault("bulk", []).extend(
            [
                (self.bulk_rnas_idx, n_bulk_RNAs_to_degrade),
                (
                    self.fragment_bases_idx,
                    counts(states["bulk"], self.fragment_bases_idx),
                ),
            ]
        )

        # Calculate the amount of water required for total RNA hydrolysis by
        # endo and exonucleases. We first calculate the number of unique RNAs
        # that should be degraded at this timestep.
        self.unique_mRNAs_to_degrade = np.logical_and(
            np.logical_not(can_translate), is_full_transcript
        )
        self.n_unique_RNAs_to_degrade = np.bincount(
            TU_index[self.unique_mRNAs_to_degrade], minlength=self.n_total_RNAs
        )

        # Assuming complete hydrolysis for now. Note that one additional water
        # molecule is needed for each RNA to hydrolyze the 5' diphosphate.
        water_for_degraded_rnas = np.dot(
            n_bulk_RNAs_to_degrade + self.n_unique_RNAs_to_degrade, self.rna_lengths
        )
        water_for_fragments = counts(states["bulk"], self.fragment_bases_idx).sum()
        requests["bulk"].append(
            (self.water_idx, water_for_degraded_rnas + water_for_fragments)
        )
        return requests

    def evolve_state(self, timestep, states):
        """Apply the degradation reaction to the allocated counts.

        Mathematical Model
        ------------------
        Inputs (read):
            - bulk.{RNA_i}                     allocated RNA counts to degrade
            - bulk.{H2O}                       allocated water
            - bulk.{FragmentBase_k}            fragments available for exo digestion
            - bulk.{ExoRNase}                  exoRNase enzyme counts
            - unique.RNA                       active nascent / mature RNA entries
            - timestep                         dt (s)

        Calculation:
            1. Endoribonucleolytic cleavage of the allocated n_degraded RNAs:

                delta_fragments = S_endo @ n_degraded

               Each cleaved RNA of length L_i consumes (L_i - 1) H2O and
               emits its nucleotide composition as polymerized fragments + PPi.

            2. Exoribonucleolytic capacity for this tick:

                C_exo = sum(n_exoRNases) * kcat_exo * dt

               If C_exo ≥ total fragment bases, fully digest; else
               digest a multinomial subsample weighted by composition.
               Each base digested yields one NMP and one H+, consuming
               one H2O:

                FragmentBase + H2O  →  NMP + H+

            3. Unique RNA entries whose TU_index was scheduled for
               degradation are deactivated and removed.

        Outputs (written):
            - bulk.{RNA_i}                     -= n_degraded_i
            - bulk.{NMP_k}                     += n_bases_digested_k
            - bulk.{FragmentBase_k}            += Δ from endo, -= from exo
            - bulk.H2O                         -= hydrolysis water (endo + exo)
            - bulk.PPi, bulk.Hplus             += reaction byproducts
            - unique.RNA                       deactivate / remove degraded RNAs
            - listeners.rna_degradation_listener.*  counts per class, NMP yield
        """
        # Get vector of numbers of RNAs to degrade for each RNA species
        n_degraded_bulk_RNA = counts(states["bulk"], self.bulk_rnas_idx)
        n_degraded_unique_RNA = self.n_unique_RNAs_to_degrade
        n_degraded_RNA = n_degraded_bulk_RNA + n_degraded_unique_RNA

        # Deactivate and degrade unique RNAs
        TU_index, can_translate = attrs(states["RNAs"], ["TU_index", "can_translate"])
        can_translate = can_translate.copy()
        n_deactivated_unique_RNA = self.n_unique_RNAs_to_deactivate

        # Deactive unique RNAs
        non_zero_deactivation = n_deactivated_unique_RNA > 0

        for index, n_degraded in zip(
            np.arange(n_deactivated_unique_RNA.size)[non_zero_deactivation],
            n_deactivated_unique_RNA[non_zero_deactivation],
        ):
            # Get mask for translatable mRNAs belonging to the degraded species
            mask = np.logical_and(TU_index == index, can_translate)

            # Choose n_degraded indexes randomly to deactivate
            can_translate[
                self.random_state.choice(
                    size=n_degraded, a=np.where(mask)[0], replace=False
                )
            ] = False

        count_RNA_degraded_per_cistron = self.cistron_tu_mapping_matrix.dot(
            n_degraded_RNA[: self.n_transcribed_rnas]
        )
        # Add degraded counts from mature RNAs
        if self.mature_rna_exists:
            count_RNA_degraded_per_cistron[self.mature_rna_cistron_indexes] += (
                n_degraded_RNA[self.n_transcribed_rnas :]
            )

        # Per-RNA count is large (~50KB); only emit when flag is on
        per_rna_counts = (n_degraded_RNA if self.emit_per_rna
                          else np.zeros(0, dtype=n_degraded_RNA.dtype))
        update = {
            "listeners": {
                "rna_degradation_listener": {
                    "count_rna_degraded": per_rna_counts,
                    "nucleotides_from_degradation": np.dot(
                        n_degraded_RNA, self.rna_lengths
                    ),
                    "count_RNA_degraded_per_cistron": count_RNA_degraded_per_cistron,
                }
            },
            # Degrade bulk RNAs
            "bulk": [(self.bulk_rnas_idx, -n_degraded_bulk_RNA)],
            "RNAs": {
                "set": {"can_translate": can_translate},
                # Degrade full mRNAs that are inactive
                "delete": np.where(self.unique_mRNAs_to_degrade)[0],
            },
        }

        # Modeling assumption: Once a RNA is cleaved by an endonuclease its
        # resulting nucleotides are lumped together as "polymerized fragments".
        # These fragments can carry over from previous timesteps. We are also
        # assuming that during endonucleolytic cleavage the 5'terminal
        # phosphate is removed. This is modeled as all of the fragments being
        # one long linear chain of "fragment bases".

        # Example:
        # PPi-Base-PO4(-)-Base-PO4(-)-Base-OH
        # ==>
        # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + PPi
        # Note: Lack of -OH on 3' end of chain
        metabolites_endo_cleavage = np.dot(
            self.endo_degradation_stoich_matrix, n_degraded_RNA
        )

        # Increase polymerized fragment counts
        update["bulk"].append(
            (self.fragment_metabolites_idx, metabolites_endo_cleavage)
        )
        # fragment_metabolites overlaps with fragment_bases
        bulk_count_copy = states["bulk"].copy()
        if len(bulk_count_copy.dtype) > 1:
            bulk_count_copy = bulk_count_copy["count"]
        bulk_count_copy[self.fragment_metabolites_idx] += metabolites_endo_cleavage
        fragment_bases = bulk_count_copy[self.fragment_bases_idx]

        # Check if exonucleolytic digestion can happen
        if fragment_bases.sum() == 0:
            return update

        # Calculate exolytic cleavage events

        # Modeling assumption: We model fragments as one long fragment chain of
        # polymerized nucleotides. We are also assuming that there is no
        # sequence specificity or bias towards which nucleotides are
        # hydrolyzed.

        # Example:
        # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + 3 H2O
        # ==>
        # 3 NMP + 3 H(+)
        # Note: Lack of -OH on 3' end of chain

        n_exoRNases = counts(states["bulk"], self.exoRNase_idx)
        n_fragment_bases = fragment_bases
        n_fragment_bases_sum = n_fragment_bases.sum()

        exornase_capacity = (
            n_exoRNases.sum() * self.kcat_exoRNase * (units.s * states["timestep"])
        )

        if exornase_capacity >= n_fragment_bases_sum:
            update["bulk"].extend(
                [
                    (self.nmps_idx, n_fragment_bases),
                    (self.water_idx, -n_fragment_bases_sum),
                    (self.proton_idx, n_fragment_bases_sum),
                    (self.fragment_bases_idx, -n_fragment_bases),
                ]
            )
            total_fragment_bases_digested = n_fragment_bases_sum

        else:
            fragment_specificity = n_fragment_bases / n_fragment_bases_sum
            possible_bases_to_digest = self.random_state.multinomial(
                exornase_capacity, fragment_specificity
            )
            n_fragment_bases_digested = n_fragment_bases - np.fmax(
                n_fragment_bases - possible_bases_to_digest, 0
            )

            total_fragment_bases_digested = n_fragment_bases_digested.sum()

            update["bulk"].extend(
                [
                    (self.nmps_idx, n_fragment_bases_digested),
                    (self.water_idx, -total_fragment_bases_digested),
                    (self.proton_idx, total_fragment_bases_digested),
                    (self.fragment_bases_idx, -n_fragment_bases_digested),
                ]
            )

        update["listeners"]["rna_degradation_listener"]["fragment_bases_digested"] = (
            total_fragment_bases_digested
        )

        # Note that once mRNAs have been degraded,
        # chromosome_structure.py will handle deleting the active
        # ribosomes that were translating those mRNAs.

        return update

    def _calculate_total_n_to_degrade(
        self, timestep, specificity, total_kcat_endornase
    ):
        """
        Calculate the total number of RNAs to degrade for a specific class of
        RNAs, based on the specificity of endoRNases on that specific class and
        the total kcat value of the endoRNases.

        Args:
            specificity: Sum of fraction of active endoRNases for all RNAs
                in a given class
            total_kcat_endornase: The summed kcat of all existing endoRNases
        Returns:
            Total number of RNAs to degrade for the given class of RNAs
        """
        return np.round(
            (specificity * total_kcat_endornase * (units.s * timestep)).magnitude
        )

    def _get_rnas_to_degrade(self, n_total_rnas_to_degrade, rna_deg_probs, rna_counts):
        """
        Distributes the total count of RNAs to degrade for each class of RNAs
        into individual RNAs, based on the given degradation probabilities
        of individual RNAs. The upper bound is set by the current count of the
        specific RNA.

        Args:
            n_total_rnas_to_degrade: Total number of RNAs to degrade for the
                given class of RNAs (integer, scalar)
            rna_deg_probs: Degradation probabilities of each RNA (vector of
                equal length to the total number of different RNAs)
            rna_counts: Current counts of each RNA molecule (vector of equal
                length to the total number of different RNAs)
        Returns:
            Vector of equal length to rna_counts, specifying the number of
            molecules to degrade for each RNA
        """
        n_rnas_to_degrade = np.zeros_like(rna_counts)
        remaining_rna_counts = rna_counts

        while (
            n_rnas_to_degrade.sum() < n_total_rnas_to_degrade
            and remaining_rna_counts.sum() != 0
        ):
            n_rnas_to_degrade += np.fmin(
                self.random_state.multinomial(
                    n_total_rnas_to_degrade - n_rnas_to_degrade.sum(), rna_deg_probs
                ),
                remaining_rna_counts,
            )
            remaining_rna_counts = rna_counts - n_rnas_to_degrade

        return n_rnas_to_degrade
