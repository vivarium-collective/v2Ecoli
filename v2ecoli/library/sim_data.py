import re
import binascii
from itertools import chain
import numpy as np
import pandas as pd
import pickle
import os
from typing import Any, Optional, TYPE_CHECKING
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint, pint_to_unum
vivunits = units
from v2ecoli.library.unit_struct_array import UnitStructArray
from v2ecoli.library.fitting import normalize
from v2ecoli.library.filepath import ROOT_PATH

from v2ecoli.processes.polypeptide_elongation import MICROMOLAR_UNITS
from v2ecoli.library.parameters import param_store
from v2ecoli.library.initial_conditions import (
    calculate_cell_mass,
    initialize_bulk_counts,
    initialize_trna_charging,
    initialize_unique_molecules,
    set_small_molecule_counts,
)

if TYPE_CHECKING:
    from v2ecoli.reconstruction.ecoli.simulation_data import SimulationDataEcoli

RAND_MAX = 2**31

# --- Injected bulk-species seeding (drug-agnostic) -------------------------
# Species an injected subsystem needs (e.g. an antibiotic arm's drug + complex
# species) are NOT in the ParCa cache; they are seeded at composite-generate
# time by :func:`seed_bulk_species` below, from a per-species molar-mass
# declaration the fork supplies through ``injected_processes`` — so the engine
# carries no knowledge of any particular drug.


def seed_bulk_species(bulk_state, specs):
    """Drug-agnostic bulk-species seeding for the generic injection seam.

    The engine-neutral bulk-species seeder for the injection seam: an injected
    subsystem (antibiotic or otherwise) declares the bulk species it
    needs seeded — with their molar masses — through ``injected_processes`` /
    the run config, so the engine carries ZERO knowledge of any particular drug.

    ``specs`` is a list of dicts, each:
        ``{"id": str,
           "molar_mass_g_per_mol": float,     # the species' own molar mass,
                                               # placed in the metabolite submass slot
           "complex_with": Optional[str]}``    # id of an EXISTING bulk species whose
                                               # molar-mass vector is ADDED (drug-target
                                               # complex); the partner's reconstruction
                                               # mass need not be known by the caller.

    Appends each species to the columnar ``bulk_state`` at ``count == 0`` with the
    correct fg submass columns (the same ``molar_mass -> fg`` conversion
    ``initialize_bulk_counts`` applies to every bulk molecule). Idempotent
    (species already present are skipped); a no-op (empty / falsy ``specs``)
    returns ``bulk_state`` unchanged. This is the exact arithmetic the retired
    drug-specific ``*_species_masses`` helpers used — a hydrolysed form simply
    passes its already-summed molar mass (drug + water), and a complex passes the
    drug part plus ``complex_with``.
    """
    if not specs:
        return bulk_state

    submass_cols = [n for n in bulk_state.dtype.names if n.endswith("_submass")]
    n_submass = len(submass_cols)
    metabolite_index = submass_cols.index("metabolite_submass")
    n_avogadro = (1 * units.avogadro_constant).to("1/mol").magnitude
    molar_to_fg = 1e15 / n_avogadro

    existing_ids = set(bulk_state["id"].tolist())
    new_rows = []
    for spec in specs:
        sid = spec["id"]
        if sid in existing_ids:
            continue  # idempotent
        molar = np.zeros(n_submass)
        molar[metabolite_index] = float(spec["molar_mass_g_per_mol"])
        partner = spec.get("complex_with")
        if partner:
            rows = bulk_state[bulk_state["id"] == partner]
            if len(rows) == 0:
                raise ValueError(
                    f"seed_bulk_species: complex_with={partner!r} (for {sid!r}) "
                    "not found in the bulk store; cannot build the complex mass.")
            # Recover the partner's molar-mass (g/mol) vector from its fg columns
            # and add it, matching the retired complex arithmetic exactly.
            partner_fg = np.array([rows[0][c] for c in submass_cols], dtype=float)
            molar = molar + partner_fg / molar_to_fg
        new_rows.append((sid, 0, *(molar * molar_to_fg)))

    if not new_rows:
        return bulk_state

    appended = np.array(new_rows, dtype=bulk_state.dtype)
    return np.append(bulk_state, appended)


class LoadSimData:
    def __init__(
        self,
        sim_data_path: Optional[str] = None,
        seed: int = 0,
        sim_data: Optional["SimulationDataEcoli"] = None,
        max_duration: int = 10,
        fixed_media: Optional[str] = None,
        media_timeline: tuple[tuple[int, str]] = (
            (0, "minimal"),
        ),  # have to change both media_timeline and condition
        condition: str = "basal",
        trna_charging: bool = True,
        ppgpp_regulation: bool = True,
        mar_regulon: bool = False,
        process_configs: Optional[dict[str, Any]] = None,
        initial_state_gaussian: bool = True,
        superhelical_density: bool = False,
        recycle_stalled_elongation: bool = False,
        mechanistic_replisome: bool = False,
        trna_attenuation: bool = True,
        variable_elongation_transcription: bool = True,
        variable_elongation_translation: bool = False,
        mechanistic_translation_supply: bool = True,
        mechanistic_aa_transport: bool = True,
        translation_supply: bool = True,
        aa_supply_in_charging: bool = True,
        disable_ppgpp_elongation_inhibition: bool = False,
        emit_unique: bool = False,
        **kwargs,
    ):
        """
        Loads simulation data generated by the ParCa
        (:py:func:`~reconstruction.ecoli.fit_sim_data_1.fitSimData_1`,
        runscript located at :py:mod:`runscripts.parca`) and extracts
        parameters for each process. Typically instantiated by
        :py:class:`~ecoli.composites.ecoli_master.Ecoli` with
        keyword arguments given by the config loaded by
        :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim`.

        Args:
            sim_data_path: Path to simulation data pickle file
            seed: Used to deterministically seed all random number
                generators. Simulations with the same seed will yield
                the same output.
            max_duration: Maximum duration of single cell simulation (seconds)
            media_timeline: Iterable of tuples where the first element of
                each tuple is the time to start using a certain media and the
                second element is a string corresponding to one of the media
                options in ``self.sim_data.external_state.saved_media``
            trna_charging: Use steady-state charging model
                (:py:class:`~ecoli.processes.polypeptide_elongation.SteadyStatePolypeptideElongation`)
                in :py:class:`~ecoli.processes.polypeptide_elongation.SteadyStatePolypeptideElongation`
            ppgpp_regulation: Enable growth rate control using ppGpp in
                polypeptide elongation and transcript initiation
            mar_regulon: Enable tetracycline-related transcriptional regulation
                of antibiotic resistance genes by the mar operon
            process_configs: Mapping of process names to config dictionaries,
                currently only used to configure :py:class:`~ecoli.processes.rna_interference.RnaInterference`
            initial_state_gaussian: If the simulation is configured to generate an
                initial state from pickled simulation data (see option 3 in
                :py:meth:`~ecoli.composites.ecoli_master.Ecoli.initial_state`),
                enabling this adds Gaussian noise to the generated state
            superhelical_density: Enables superhelical density calculations on
                ``('unique', 'chromosomal_segment)'`` molecules in
                :py:class:`~ecoli.processes.chromosome_structure.ChromosomeStructure`
            mechanistic_replisome: Ensures that there are adequate replisome
                subunits to initiate each round of chromosome replication in
                :py:class:`~ecoli.processes.chromosome_replication.ChromosomeReplication`
            recycle_stalled_elongation: Free up RNAPs and nucleotides for
                stalled transcripts in
                :py:class:`~ecoli.processes.transcript_elongation.TranscriptElongation`
            trna_attenuation: Implements `attenuation <https://en.wikipedia.org/wiki/Attenuator_(genetics)>`_
                in :py:class:`~ecoli.processes.transcript_initiation.TranscriptInitiation`
                and :py:class:`~ecoli.processes.transcript_elongation.TranscriptElongation`
            variable_elongation_transcription: Allow different elongation rate for different transcripts
                (currently increases rates for rRNA, see
                :py:meth:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.make_elongation_rates`).
                Usually set this consistently for ParCa and simulation.
                :py:class:`~ecoli.processes.transcript_initiation.TranscriptInitiation`
                Usually set this consistently for ParCa and simulation.
            variable_elongation_translation: Allow different polypeptides to
                have different translation rates (currently increases rates for
                ribosomal proteins, see
                :py:meth:`~reconstruction.ecoli.dataclasses.process.translation.Translation.make_elongation_rates`).
                Usually set this consistently for ParCa and simulation.
            mechanistic_translation_supply: Calculate charged tRNA supply using
                starting amino acid concentration only based on mechanistic
                synthesis and supply in
                :py:class:`~ecoli.processes.polypeptide_elongation.SteadyStatePolypeptideElongation`
                when ``trna_charging`` is ``True``
            mechanistic_aa_transport: Constrain amino acid uptake based on
                external concentrations and exchange rates in
                :py:class:`~ecoli.processes.metabolism.Metabolism`
            translation_supply: Use :py:class:`~ecoli.processes.polypeptide_elongation.TranslationSupplyPolypeptideElongation`
                instead of :py:class:`~ecoli.processes.polypeptide_elongation.SteadyStatePolypeptideElongation`.
                Superseded by ``trna_charging``
            aa_supply_in_charging: Calculate charged tRNA supply from each sub
                time step while solving the charging steady state in
                :py:class:`~ecoli.processes.polypeptide_elongation.SteadyStatePolypeptideElongation`
                when ``trna_charging`` is ``True``
            disable_ppgpp_elongation_inhibition: Turn off ppGpp-mediated
                inhibition in :py:class:`~ecoli.processes.polypeptide_elongation.SteadyStatePolypeptideElongation`
                when ``trna_charging`` is ``True``
        """
        self.seed = seed
        self.max_duration = max_duration
        self.random_state = np.random.RandomState(seed=seed)
        # Iterable of tuples with the format (time, media_id)
        if condition is not None:
            self.condition = condition

        if fixed_media is not None and media_timeline is not None:
            media_timeline = ((0, fixed_media),)

        self.media_timeline = media_timeline

        self.trna_charging = trna_charging
        self.ppgpp_regulation = ppgpp_regulation
        self.initial_state_gaussian = initial_state_gaussian
        self.superhelical_density = superhelical_density
        self.mechanistic_replisome = mechanistic_replisome
        self.trna_attenuation = trna_attenuation
        self.variable_elongation_transcription = variable_elongation_transcription
        self.variable_elongation_translation = variable_elongation_translation
        self.mechanistic_translation_supply = mechanistic_translation_supply
        self.mechanistic_aa_transport = mechanistic_aa_transport
        self.translation_supply = translation_supply
        self.aa_supply_in_charging = aa_supply_in_charging
        self.disable_ppgpp_elongation_inhibition = disable_ppgpp_elongation_inhibition
        self.recycle_stalled_elongation = recycle_stalled_elongation
        self.emit_unique = emit_unique

        # NEW to vivarium-ecoli: Whether to lump miscRNA with mRNAs
        # when calculating degradation
        self.degrade_misc = False

        # load sim_data — either from a pickle path or a pre-hydrated object.
        # Skipping the pickle round-trip is the v2ecoli ParCa fast path: the
        # composite produces a live ``SimulationDataEcoli`` we can hand in
        # directly, which saves a ~300 MB dump + reload.
        if sim_data is not None:
            self.sim_data: "SimulationDataEcoli" = sim_data
        elif sim_data_path is not None:
            with open(sim_data_path, "rb") as sim_data_file:
                self.sim_data: "SimulationDataEcoli" = pickle.load(sim_data_file)
        else:
            raise ValueError(
                "LoadSimData requires either sim_data_path or sim_data")

        if condition is not None:
            self.sim_data.condition = condition

        # Derive the runtime media from the CONDITION's nutrients. media_timeline
        # defaults to ((0,'minimal'),) and setting `condition` alone never updated
        # it (the "have to change both" footgun), so every non-basal condition
        # silently ran on 'minimal' media — e.g. no_oxygen ran AEROBIC instead of
        # minimal_minus_oxygen (O2=0), and with_aa's media_id was mislabelled.
        # Pull it straight from sim_data unless the caller set fixed_media or an
        # explicit (non-default) timeline. Basal -> 'minimal' (unchanged).
        if fixed_media is None and self.media_timeline == ((0, "minimal"),):
            cond_nutrients = self.sim_data.conditions.get(
                self.sim_data.condition, {}).get("nutrients")
            if cond_nutrients:
                self.media_timeline = ((0, cond_nutrients),)

        # Used by processes to apply submass updates to correct unique attr
        self.submass_indices = {
            f"massDiff_{submass}": idx
            for submass, idx in self.sim_data.submass_name_to_index.items()
        }

        # Logic to handle internal shifts
        if "agent_id" in kwargs and hasattr(self.sim_data, "internal_shift_dict"):
            generation = len(kwargs["agent_id"])
            func_to_apply = None
            func_params = ()
            for shift_gen, (
                shift_func,
                shift_params,
            ) in self.sim_data.internal_shift_dict.items():
                if generation >= shift_gen:
                    func_to_apply = shift_func
                    func_params = shift_params
            if func_to_apply is not None:
                func_to_apply(self.sim_data, *func_params)

        # NEW to vivarium-ecoli
        # Changes gene expression upon tetracycline exposure
        # Note: Incompatible with operons because there are genes
        # that are part of the same operon but have different changes
        # in expression under tetracycline exposure (e.g. marRAB)
        if mar_regulon:
            # Define aliases to reduce code verbosity
            treg_alias = self.sim_data.process.transcription_regulation
            bulk_mol_alias = self.sim_data.internal_state.bulk_molecules
            eq_alias = self.sim_data.process.equilibrium

            # Assume marA (PD00365) controls the entire tetracycline
            # gene expression program and marR (CPLX0-7710) is inactivated
            # by complexation with tetracycline
            treg_alias.tf_ids += ["PD00365", "CPLX0-7710"]
            treg_alias.delta_prob["shape"] = (
                treg_alias.delta_prob["shape"][0],
                treg_alias.delta_prob["shape"][1] + 2,
            )
            treg_alias.tf_to_tf_type["PD00365"] = "0CS"
            treg_alias.tf_to_tf_type["CPLX0-7710"] = "1CS"
            treg_alias.active_to_bound["CPLX0-7710"] = "marR-tet"

            # Add marA and marR to pPromoterBound dict for initial state generation
            for bound_probs in self.sim_data.pPromoterBound.values():
                bound_probs["CPLX0-7710"] = 1
                bound_probs["PD00365"] = 1

            # Expand promoter bound_TF field for marA and marR
            curr_size = self.sim_data.internal_state.unique_molecule.unique_molecule_definitions[
                "promoter"
            ]["bound_TF"][1]
            self.sim_data.internal_state.unique_molecule.unique_molecule_definitions[
                "promoter"
            ]["bound_TF"] = ("?", curr_size + 2)

            # TU index of genes for outer membrane proteins, regulators,
            # and inner membrane transporters. The gene_fc.csv file is
            # generated by ``data/marA_binding/get_TU_ID.py`` — an
            # optional tetracycline-modelling setup step. If the file
            # is absent, warn + skip the entire marA_binding patch
            # (the de_genes-dependent setup below is wrapped in
            # ``if _have_gene_fc:`` so the rest of LoadSimData
            # continues normally).
            _gene_fc_path = os.path.join(ROOT_PATH, "data/marA_binding/gene_fc.csv")
            _have_gene_fc = os.path.exists(_gene_fc_path)
            if not _have_gene_fc:
                import warnings as _warnings
                _warnings.warn(
                    "sim_data.LoadSimData: mar_regulon=True was requested but "
                    f"{_gene_fc_path!r} is missing — skipping the marA_binding "
                    "transcription-regulation patch. Generate it via "
                    "data/marA_binding/get_TU_ID.py if you need tetracycline "
                    "modelling.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if _have_gene_fc:
                de_genes = pd.read_csv(_gene_fc_path)
                new_deltaI = de_genes["TU_idx"].to_numpy()
                new_deltaJ = np.array([24] * 24)
            # Values were chosen to recapitulate mRNA fold change when exposed
            # to 1.5 mg/L tetracycline (Viveiros et al. 2007)
                new_deltaV = np.array(
                    [
                        1.76e-03,
                        2.21e-05,
                        2.44e-05,
                        2.10e-06,
                        4.11e-06,
                        7.80e-06,
                        5.40e-04,
                        7.42e-06,
                        1.51e-06,
                        2.95e-05,
                        2.02e-05,
                        1.96e-04,
                        5.77e-05,
                        2.34e-04,
                        2.04e-06,
                        1.58e-07,
                        8.89e-08,
                        8.52e-07,
                        8.09e-06,
                        1.68e-08,
                        7.17e-08,
                        8.08e-06,
                        1.40e-08,
                        -5.30e-07,
                    ]
                )

                treg_alias.delta_prob["deltaI"] = np.concatenate(
                    [treg_alias.delta_prob["deltaI"], new_deltaI]
                )
                treg_alias.delta_prob["deltaJ"] = np.concatenate(
                    [treg_alias.delta_prob["deltaJ"], new_deltaJ]
                )
                treg_alias.delta_prob["deltaV"] = np.concatenate(
                    [treg_alias.delta_prob["deltaV"], new_deltaV]
                )

                # Add mass data for tetracycline, marR-tet, and 30s-tet
                bulk_data = bulk_mol_alias.bulk_data.fullArray()
                marR_mass = np.array(bulk_data[bulk_data["id"] == "CPLX0-7710[c]"][0][1])
                free_30s_mass = np.array(
                    bulk_data[bulk_data["id"] == "CPLX0-3953[c]"][0][1]
                )
                tet_mass = param_store.get(("tetracycline", "mass")).magnitude
                tet_mass = np.array([0, 0, 0, 0, 0, 0, tet_mass, 0, 0])
                bulk_data = np.append(
                    bulk_data,
                    np.array(
                        [
                            ("marR-tet[c]",) + (marR_mass + tet_mass,),
                            ("tetracycline[p]",) + (tet_mass,),
                            ("tetracycline[c]",) + (tet_mass,),
                            ("CPLX0-3953-tetracycline[c]",) + (free_30s_mass + tet_mass,),
                        ],
                        dtype=bulk_data.dtype,
                    ),
                )
                bulk_units = bulk_mol_alias.bulk_data.fullUnits()
                bulk_mol_alias.bulk_data = UnitStructArray(bulk_data, bulk_units)

                # Add equilibrium reaction for marR-tetracycline and
                # reinitialize self.sim_data.process.equilibrium variables
                stoich_matrix_shape = eq_alias._stoichMatrix.shape
                eq_alias._stoichMatrixI = np.concatenate(
                    [
                        eq_alias._stoichMatrixI,
                        np.arange(stoich_matrix_shape[0], stoich_matrix_shape[0] + 3),
                    ]
                )
                eq_alias._stoichMatrixJ = np.concatenate(
                    [eq_alias._stoichMatrixJ, np.array([stoich_matrix_shape[1]] * 3)]
                )
                eq_alias._stoichMatrixV = np.concatenate(
                    [eq_alias._stoichMatrixV, np.array([-1, -1, 1])]
                )
                eq_alias.molecule_names += [
                    "CPLX0-7710[c]",
                    "tetracycline[c]",
                    "marR-tet[c]",
                ]
                eq_alias.ids_complexes = [
                    eq_alias.molecule_names[i]
                    for i in np.where(np.any(eq_alias.stoich_matrix() > 0, axis=1))[0]
                ]
                eq_alias.rxn_ids += ["marR-tet"]
                # All existing equilibrium rxns use a forward rate of 1
                eq_alias.rates_fwd = np.concatenate([eq_alias.rates_fwd, np.array([1])])
                # Existing equilibrium rxns use a default reverse rate of 1e-6
                # This happens to nearly perfectly yield full MarR inactivation
                # at 1.5 mg/L external tetracycline
                eq_alias.rates_rev = np.concatenate([eq_alias.rates_rev, np.array([1e-6])])

                # Mass balance matrix
                eq_alias._stoichMatrixMass = np.concatenate(
                    [
                        eq_alias._stoichMatrixMass,
                        np.array(
                            [marR_mass.sum(), tet_mass.sum(), (marR_mass + tet_mass).sum()]
                        ),
                    ]
                )
                eq_alias.balance_matrix = eq_alias.stoich_matrix() * eq_alias.mass_matrix()

                # Find the mass balance of each equation in the balanceMatrix
                massBalanceArray = eq_alias.mass_balance()

                # The stoichometric matrix should balance out to numerical zero.
                assert np.max(np.absolute(massBalanceArray)) < 1e-9

                # Build matrices
                eq_alias._populateDerivativeAndJacobian()
                eq_alias._stoichMatrix = eq_alias.stoich_matrix()

        # NEW to vivarium-ecoli
        # Append new RNA IDs and degradation rates for sRNA-mRNA duplexes
        if isinstance(process_configs, dict):
            rnai_data = process_configs.get("ecoli-rna-interference", False)
            if rnai_data:
                # Define aliases to reduce code verbosity
                ts_alias = self.sim_data.process.transcription
                bulk_mol_alias = self.sim_data.internal_state.bulk_molecules
                treg_alias = self.sim_data.process.transcription_regulation

                self.duplex_ids = np.array(rnai_data["duplex_ids"])
                n_duplex_rnas = len(self.duplex_ids)
                duplex_deg_rates = np.array(rnai_data["duplex_deg_rates"])
                duplex_km = np.array(rnai_data["duplex_km"])
                duplex_na = np.zeros(n_duplex_rnas)
                # Mark duplexes as miscRNAs so they are degraded appropriately
                duplex_is_miscRNA = np.ones(n_duplex_rnas, dtype=np.bool_)

                self.srna_ids = np.array(rnai_data["srna_ids"])
                target_ids = np.array(rnai_data["target_ids"])
                self.target_tu_ids = np.zeros(len(target_ids), dtype=int)

                self.binding_probs = np.array(rnai_data["binding_probs"])

                # Get duplex length, ACGU content, molecular weight, and sequence
                duplex_lengths = np.zeros(n_duplex_rnas)
                duplex_ACGU = np.zeros((n_duplex_rnas, 4))
                duplex_mw = np.zeros(n_duplex_rnas)
                cistron_data = ts_alias.cistron_data.fullArray()
                cistron_tu_mapping = ts_alias.cistron_tu_mapping_matrix
                rna_data = ts_alias.rna_data.fullArray()
                rna_units = ts_alias.rna_data.fullUnits()
                rna_sequences = ts_alias.transcription_sequences
                duplex_sequences = np.full((n_duplex_rnas, rna_sequences.shape[1]), -1)
                for i, (srna_id, target_id) in enumerate(
                    zip(self.srna_ids, target_ids)
                ):
                    # Use first match for each sRNA and target mRNA
                    srna_tu_id = cistron_tu_mapping[
                        cistron_data["id"] == srna_id[:-3]
                    ].nonzero()[1][0]
                    self.target_tu_ids[i] = cistron_tu_mapping[
                        cistron_data["id"] == target_id[:-3]
                    ].nonzero()[1][0]
                    duplex_ACGU[i] = (
                        rna_data["counts_ACGU"][srna_tu_id]
                        + rna_data["counts_ACGU"][self.target_tu_ids[i]]
                    )
                    duplex_mw[i] = (
                        rna_data["mw"][srna_tu_id]
                        + rna_data["mw"][self.target_tu_ids[i]]
                    )
                    srna_length = rna_data["length"][srna_tu_id]
                    target_length = rna_data["length"][self.target_tu_ids[i]]
                    duplex_lengths[i] = srna_length + target_length
                    if duplex_lengths[i] > duplex_sequences.shape[1]:
                        # Extend columns in sequence arrays to accomodate duplexes
                        # where the sum of the RNA lengths > # of columns
                        extend_length = duplex_lengths[i] - duplex_sequences.shape[1]
                        extend_duplex_sequences = np.full(
                            (duplex_sequences.shape[0], extend_length),
                            -1,
                            dtype=duplex_sequences.dtype,
                        )
                        duplex_sequences = np.append(
                            duplex_sequences, extend_duplex_sequences, axis=1
                        )
                        extend_rna_sequences = np.full(
                            (rna_sequences.shape[0], extend_length),
                            -1,
                            dtype=rna_sequences.dtype,
                        )
                        rna_sequences = np.append(
                            rna_sequences, extend_rna_sequences, axis=1
                        )
                    duplex_sequences[i, :srna_length] = rna_sequences[srna_tu_id][
                        :srna_length
                    ]
                    duplex_sequences[i, srna_length : srna_length + target_length] = (
                        rna_sequences[self.target_tu_ids[i]][:target_length]
                    )

                # Make duplex metadata visible to all RNA-related processes
                old_n_rnas = rna_data.shape[0]
                rna_data = np.resize(rna_data, old_n_rnas + n_duplex_rnas)
                rna_sequences = np.resize(
                    rna_sequences, (old_n_rnas + n_duplex_rnas, rna_sequences.shape[1])
                )
                for i, new_rna in enumerate(
                    zip(
                        self.duplex_ids,
                        duplex_deg_rates,
                        duplex_na,
                        duplex_lengths,
                        duplex_ACGU,
                        duplex_mw,
                        duplex_na,
                        duplex_na,
                        duplex_km,
                        duplex_na,
                        duplex_na,
                        duplex_na,
                        duplex_na,
                        duplex_is_miscRNA,
                        duplex_na,
                        duplex_na,
                        duplex_na,
                        duplex_na,
                        duplex_na,
                        duplex_na,
                    )
                ):
                    rna_data[old_n_rnas + i] = new_rna
                    rna_sequences[old_n_rnas + i] = duplex_sequences[i]
                ts_alias.transcription_sequences = rna_sequences
                ts_alias.rna_data = UnitStructArray(rna_data, rna_units)

                # Add bulk mass data for duplexes
                bulk_data = bulk_mol_alias.bulk_data.fullArray()
                bulk_units = bulk_mol_alias.bulk_data.fullUnits()
                old_n_bulk = bulk_data.shape[0]
                bulk_data = np.resize(bulk_data, bulk_data.shape[0] + n_duplex_rnas)
                for i, duplex in enumerate(self.duplex_ids):
                    duplex_submasses = np.zeros(9)
                    duplex_submasses[2] = duplex_mw[i]
                    bulk_data[old_n_bulk + i] = (duplex, duplex_submasses)
                bulk_mol_alias.bulk_data = UnitStructArray(bulk_data, bulk_units)

                # Add filler transcription data for duplex RNAs to prevent errors
                treg_alias.basal_prob = np.append(treg_alias.basal_prob, 0)
                treg_alias.delta_prob["shape"] = (
                    treg_alias.delta_prob["shape"][0] + 1,
                    treg_alias.delta_prob["shape"][1],
                )

                # Set flag so miscRNA duplexes are degraded together with mRNAs
                self.degrade_misc = True

                # Resize cistron-TU mapping matrix
                curr_shape = ts_alias.cistron_tu_mapping_matrix._shape
                ts_alias.cistron_tu_mapping_matrix._shape = (
                    curr_shape[0],
                    curr_shape[1] + n_duplex_rnas,
                )

                # Add duplexes to RNA synth prob calculations
                ts_alias.exp_free = np.concatenate(
                    [ts_alias.exp_free, [0] * n_duplex_rnas]
                )
                ts_alias.exp_ppgpp = np.concatenate(
                    [ts_alias.exp_ppgpp, [0] * n_duplex_rnas]
                )

    def get_monomer_counts_indices(self, names):
        """Given a list of monomer names without location tags, this returns
        the indices of those monomers in the monomer_counts listener array.
        The "id" column of reconstruction/ecoli/flat/proteins.tsv contains
        nearly all supported monomer names."""
        monomer_ids = self.sim_data.process.translation.monomer_data["id"]
        # Strip location string (e.g. [c])
        monomer_ids = np.array(
            [re.split(r"\[.\]", monomer)[0] for monomer in monomer_ids]
        )
        return [int(np.where(monomer_ids == name)[0][0]) for name in names]

    def get_mrna_counts_indices(self, names):
        """Given a list of mRNA names without location tags, this returns
        the indices of those mRNAs in the mRNA_counts listener array.
        The "id" column of reconstruction/ecoli/flat/rnas.tsv contains
        nearly all supported mRNA names."""
        is_mrna = self.sim_data.process.transcription.rna_data["is_mRNA"]
        mrna_ids = self.sim_data.process.transcription.rna_data["id"][is_mrna]
        # Strip location string (e.g. [c])
        mrna_ids = np.array([re.split(r"\[.\]", mrna)[0] for mrna in mrna_ids])
        return [int(np.where(mrna_ids == name)[0][0]) for name in names]

    def get_rna_indices(self, names):
        """Given a list of RNA names without location tags, this returns
        the TU indices of those RNAs (for rna_init_events and rna_synth_prob).
        The "id" column of reconstruction/ecoli/flat/rnas.tsv contains
        nearly all supported RNA names."""
        rna_ids = self.sim_data.process.transcription.rna_data["id"]
        # Strip location string (e.g. [c])
        rna_ids = np.array([re.split(r"\[.\]", mrna)[0] for mrna in rna_ids])
        return [int(np.where(rna_ids == name)[0][0]) for name in names]

    def _seedFromName(self, name):
        return binascii.crc32(name.encode("utf-8"), self.seed) & 0xFFFFFFFF

    def _precompute_biomass_concentrations(self):
        """Precompute biomass concentrations for all unique doubling times."""
        seen = {}
        for dt in self.sim_data.condition_to_doubling_time.values():
            dt_min = float(unum_to_pint(dt).to(units.min).magnitude)
            if dt_min not in seen:
                result = self.sim_data.mass.getBiomassAsConcentrations(dt)
                seen[dt_min] = {
                    k: float(unum_to_pint(v).to(units.mol / units.L).magnitude)
                    for k, v in result.items()
                }
        return seen

    def get_config_by_name(self, name, time_step=1):
        name_config_mapping = {
            "ecoli-tf-binding": self.get_tf_config,
            "ecoli-transcript-initiation": self.get_transcript_initiation_config,
            "ecoli-transcript-elongation": self.get_transcript_elongation_config,
            "ecoli-rna-degradation": self.get_rna_degradation_config,
            "ecoli-polypeptide-initiation": self.get_polypeptide_initiation_config,
            "ecoli-polypeptide-elongation": self.get_polypeptide_elongation_config,
            "ecoli-complexation": self.get_complexation_config,
            "ecoli-two-component-system": self.get_two_component_system_config,
            "ecoli-equilibrium": self.get_equilibrium_config,
            "ecoli-protein-degradation": self.get_protein_degradation_config,
            "ecoli-metabolism": self.get_metabolism_config,
            "ecoli-metabolism-redux": self.get_metabolism_redux_config,
            "ecoli-metabolism-redux-classic": self.get_metabolism_redux_config,
            "ecoli-chromosome-replication": self.get_chromosome_replication_config,
            "ecoli-mass": self.get_mass_config,
            "ecoli-mass-listener": self.get_mass_listener_config,
            "post-division-mass-listener": self.get_mass_listener_config,
            "RNA_counts_listener": self.get_rna_counts_listener_config,
            "monomer_counts_listener": self.get_monomer_counts_listener_config,
            "rna_synth_prob_listener": self.get_rna_synth_prob_listener_config,
            "allocator": self.get_allocator_config,
            "ecoli-chromosome-structure": self.get_chromosome_structure_config,
            "ecoli-rna-interference": self.get_rna_interference_config,
            "tetracycline-ribosome-equilibrium": self.get_tetracycline_ribosome_equilibrium_config,
            "ecoli-rna-maturation": self.get_rna_maturation_config,
            "ecoli-tf-unbinding": self.get_tf_unbinding_config,
            "dna_supercoiling_listener": self.get_dna_supercoiling_listener_config,
            "ribosome_data_listener": self.get_ribosome_data_listener_config,
            "rnap_data_listener": self.get_rnap_data_listener_config,
            "unique_molecule_counts": self.get_unique_molecule_counts_config,
            "exchange_data": self.get_exchange_data_config,
            "media_update": self.get_media_update_config,
            "bulk-timeline": self.get_bulk_timeline_config,
        }

        try:
            return name_config_mapping[name](time_step=time_step)
        except KeyError:
            raise KeyError(
                f"Process of name {name} is not known to LoadSimData.get_config_by_name"
            )

    def get_chromosome_replication_config(self, time_step=1):
        get_dna_critical_mass = self.sim_data.mass.get_dna_critical_mass
        doubling_time = self.sim_data.condition_to_doubling_time[
            self.sim_data.condition
        ]

        # get_submass_array returns Unum in g/mol; convert to fg/count by
        # going through g/mol then dividing by Avogadro's number. Pint has
        # no implicit mol↔count equivalence (unlike wholecell's Unum), so
        # we do the Avogadro division explicitly rather than using a
        # direct fg/count asNumber/to conversion.
        n_avogadro = unum_to_pint(
            self.sim_data.constants.n_avogadro
        ).to(1 / units.mol).magnitude
        replisome_trimer_subunit_masses = np.vstack(
            [
                unum_to_pint(self.sim_data.getter.get_submass_array(x))
                .to(units.fg / units.mol).magnitude / n_avogadro
                for x in self.sim_data.molecule_groups.replisome_trimer_subunits
            ]
        )
        replisome_monomer_subunit_masses = np.vstack(
            [
                unum_to_pint(self.sim_data.getter.get_submass_array(x))
                .to(units.fg / units.mol).magnitude / n_avogadro
                for x in self.sim_data.molecule_groups.replisome_monomer_subunits
            ]
        )
        replisome_mass_array = 3 * replisome_trimer_subunit_masses.sum(
            axis=0
        ) + replisome_monomer_subunit_masses.sum(axis=0)

        chromosome_replication_config = {
            "time_step": time_step,
            "get_dna_critical_mass": {
                "_function": "mass.get_dna_critical_mass",
                "_data": {
                    "dry_mass_params": [float(x) for x in self.sim_data.mass._dryMassParams],
                    "cell_dry_mass_fraction": float(self.sim_data.mass.cell_dry_mass_fraction),
                },
            },
            "criticalInitiationMass": unum_to_pint(get_dna_critical_mass(doubling_time)),
            "nutrientToDoublingTime": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.nutrient_to_doubling_time.items()
            },
            "replichore_lengths": self.sim_data.process.replication.replichore_lengths,
            "sequences": self.sim_data.process.replication.replication_sequences,
            "polymerized_dntp_weights": unum_to_pint(
                self.sim_data.process.replication.replication_monomer_weights
            ),
            "replication_coordinate": self.sim_data.process.transcription.rna_data[
                "replication_coordinate"
            ],
            "D_period": unum_to_pint(self.sim_data.process.replication.d_period).to(units.s).magnitude,
            "replisome_protein_mass": replisome_mass_array.sum(),
            "no_child_place_holder": self.sim_data.process.replication.no_child_place_holder,
            "basal_elongation_rate": self.sim_data.process.replication.basal_elongation_rate,
            "make_elongation_rates": {
                "_function": "replication.make_elongation_rates",
            },
            # sim options
            "mechanistic_replisome": self.mechanistic_replisome,
            # molecules
            "replisome_trimers_subunits": self.sim_data.molecule_groups.replisome_trimer_subunits,
            "replisome_monomers_subunits": self.sim_data.molecule_groups.replisome_monomer_subunits,
            "dntps": self.sim_data.molecule_groups.dntps,
            "ppi": [self.sim_data.molecule_ids.ppi],
            # random state
            "seed": self._seedFromName("ChromosomeReplication"),
            "submass_indices": self.submass_indices,
        }

        return chromosome_replication_config

    def get_tf_config(self, time_step=1):
        tf_binding_config = {
            "time_step": time_step,
            "tf_ids": self.sim_data.process.transcription_regulation.tf_ids,
            "rna_ids": self.sim_data.process.transcription.rna_data["id"],
            "delta_prob": self.sim_data.process.transcription_regulation.delta_prob,
            "n_avogadro": unum_to_pint(self.sim_data.constants.n_avogadro),
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density),
            "p_promoter_bound_tf": self.sim_data.process.transcription_regulation.p_promoter_bound_tf,
            "tf_to_tf_type": self.sim_data.process.transcription_regulation.tf_to_tf_type,
            "active_to_bound": self.sim_data.process.transcription_regulation.active_to_bound,
            # Translation layer: vEcoli processes still expect a callable;
            # we supply the bound method from simData alongside the precomputed dict.
            "get_unbound": self.sim_data.process.equilibrium.get_unbound,
            # Precomputed unbound TF mapping (replaces get_unbound bound method)
            "tf_to_unbound": {
                tf + "[c]": self.sim_data.process.equilibrium.get_unbound(tf + "[c]")
                for tf in self.sim_data.process.transcription_regulation.tf_ids
                if (self.sim_data.process.transcription_regulation.tf_to_tf_type.get(tf) == "1CS"
                    and tf == self.sim_data.process.transcription_regulation.active_to_bound.get(tf))
            },
            "active_to_inactive_tf": self.sim_data.process.two_component_system.active_to_inactive_tf,
            "bulk_molecule_ids": self.sim_data.internal_state.bulk_molecules.bulk_data[
                "id"
            ],
            "bulk_mass_data": unum_to_pint(self.sim_data.internal_state.bulk_molecules.bulk_data[
                "mass"
            ]),
            "seed": self._seedFromName("TfBinding"),
            "submass_indices": self.submass_indices,
            "emit_unique": self.emit_unique,
        }

        return tf_binding_config

    def get_transcript_initiation_config(self, time_step=1):
        transcript_initiation_config = {
            "time_step": time_step,
            "fracActiveRnapDict": self.sim_data.process.transcription.rnapFractionActiveDict,
            "rnaLengths": unum_to_pint(self.sim_data.process.transcription.rna_data["length"]),
            "rnaPolymeraseElongationRateDict": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.process.transcription.rnaPolymeraseElongationRateDict.items()
            },
            "variable_elongation": self.variable_elongation_transcription,
            "make_elongation_rates": {
                "_function": "transcription.make_elongation_rates",
                "_data": {
                    "n_TUs": self.sim_data.process.transcription.rna_data["id"].shape[0],
                    "rRNA_indexes": self.sim_data.process.transcription.rRNA_indexes.tolist(),
                    "stable_RNA_elongation_rate": int(self.sim_data.process.transcription.stable_RNA_elongation_rate),
                },
            },
            "active_rnap_footprint_size": unum_to_pint(self.sim_data.process.transcription.active_rnap_footprint_size),
            "basal_prob": self.sim_data.process.transcription_regulation.basal_prob,
            "delta_prob": self.sim_data.process.transcription_regulation.delta_prob,
            # Translation layer: vEcoli processes still expect the bound method.
            "get_delta_prob_matrix": self.sim_data.process.transcription_regulation.get_delta_prob_matrix,
            # Precomputed delta_prob_matrix (replaces get_delta_prob_matrix bound method)
            "delta_prob_matrix": self.sim_data.process.transcription_regulation.get_delta_prob_matrix(
                dense=True, ppgpp=self.ppgpp_regulation
            ),
            "perturbations": getattr(self.sim_data, "genetic_perturbations", {}),
            "rna_data": self.sim_data.process.transcription.rna_data,
            "idx_rRNA": np.where(
                self.sim_data.process.transcription.rna_data["is_rRNA"]
            )[0],
            "idx_mRNA": np.where(
                self.sim_data.process.transcription.rna_data["is_mRNA"]
            )[0],
            "idx_tRNA": np.where(
                self.sim_data.process.transcription.rna_data["is_tRNA"]
            )[0],
            "idx_rprotein": np.where(
                self.sim_data.process.transcription.rna_data[
                    "includes_ribosomal_protein"
                ]
            )[0],
            "idx_rnap": np.where(
                self.sim_data.process.transcription.rna_data["includes_RNAP"]
            )[0],
            "rnaSynthProbFractions": self.sim_data.process.transcription.rnaSynthProbFraction,
            "rnaSynthProbRProtein": self.sim_data.process.transcription.rnaSynthProbRProtein,
            "rnaSynthProbRnaPolymerase": self.sim_data.process.transcription.rnaSynthProbRnaPolymerase,
            "replication_coordinate": self.sim_data.process.transcription.rna_data[
                "replication_coordinate"
            ],
            "transcription_direction": self.sim_data.process.transcription.rna_data[
                "is_forward"
            ],
            "n_avogadro": unum_to_pint(self.sim_data.constants.n_avogadro),
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density),
            "inactive_RNAP": "APORNAP-CPLX[c]",
            "ppgpp": self.sim_data.molecule_ids.ppGpp,
            "synth_prob": {
                "_function": "transcription.synth_prob_from_ppgpp",
                "_data": {
                    "exp_free": self.sim_data.process.transcription.exp_free.tolist(),
                    "exp_ppgpp": self.sim_data.process.transcription.exp_ppgpp.tolist(),
                    "ppgpp_growth_parameters": list(self.sim_data.process.transcription._ppgpp_growth_parameters),
                    "rna_deg_rates": unum_to_pint(self.sim_data.process.transcription.rna_data["deg_rate"]).to(1 / units.s).magnitude.tolist(),
                    "wt_replication_coordinates": self.sim_data.process.transcription.rna_data["wt_replication_coordinate"].tolist(),
                    "is_rRNA": self.sim_data.process.transcription.rna_data["is_rRNA"].tolist(),
                    "ppgpp_km_squared": float(self.sim_data.process.transcription._ppgpp_km_squared),
                },
            },
            "copy_number": {
                "_function": "replication.get_average_copy_number",
                "_data": {
                    "replichore_lengths": self.sim_data.process.replication.replichore_lengths.tolist(),
                    "c_period_in_mins": float(unum_to_pint(self.sim_data.process.replication.c_period).to(units.min).magnitude),
                    "d_period_in_mins": float(unum_to_pint(self.sim_data.process.replication.d_period).to(units.min).magnitude),
                },
            },
            "ppgpp_regulation": self.ppgpp_regulation,
            "get_rnap_active_fraction_from_ppGpp": {
                "_function": "transcription.get_rnap_active_fraction_from_ppGpp",
                "_data": {
                    "fraction_active_rnap_bound": float(self.sim_data.process.transcription.fraction_active_rnap_bound),
                    "fraction_active_rnap_free": float(self.sim_data.process.transcription.fraction_active_rnap_free),
                    "ppgpp_km_squared": float(self.sim_data.process.transcription._ppgpp_km_squared),
                },
            },
            # attenuation
            "trna_attenuation": self.trna_attenuation,
            "attenuated_rna_indices": self.sim_data.process.transcription.attenuated_rna_indices,
            "attenuation_adjustments": self.sim_data.process.transcription.attenuation_basal_prob_adjustments,
            # random seed
            "seed": self._seedFromName("TranscriptInitiation"),
            "emit_unique": self.emit_unique,
        }

        return transcript_initiation_config

    def get_transcript_elongation_config(self, time_step=1):
        transcript_elongation_config = {
            "time_step": time_step,
            "rnaPolymeraseElongationRateDict": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.process.transcription.rnaPolymeraseElongationRateDict.items()
            },
            "rnaIds": self.sim_data.process.transcription.rna_data["id"],
            "rnaLengths": unum_to_pint(self.sim_data.process.transcription.rna_data[
                "length"
            ]).magnitude,
            "rnaSequences": self.sim_data.process.transcription.transcription_sequences,
            "ntWeights": self.sim_data.process.transcription.transcription_monomer_weights,
            "endWeight": self.sim_data.process.transcription.transcription_end_weight,
            "replichore_lengths": self.sim_data.process.replication.replichore_lengths,
            "is_mRNA": self.sim_data.process.transcription.rna_data["is_mRNA"],
            "ppi": self.sim_data.molecule_ids.ppi,
            "inactive_RNAP": "APORNAP-CPLX[c]",
            "ntp_ids": ["ATP[c]", "CTP[c]", "GTP[c]", "UTP[c]"],
            "variable_elongation": self.variable_elongation_transcription,
            "make_elongation_rates": {
                "_function": "transcription.make_elongation_rates",
                "_data": {
                    "n_TUs": self.sim_data.process.transcription.rna_data["id"].shape[0],
                    "rRNA_indexes": self.sim_data.process.transcription.rRNA_indexes.tolist(),
                    "stable_RNA_elongation_rate": int(self.sim_data.process.transcription.stable_RNA_elongation_rate),
                },
            },
            "fragmentBases": self.sim_data.molecule_groups.polymerized_ntps,
            "charged_trnas": self.sim_data.process.transcription.charged_trna_names,
            # attenuation
            "trna_attenuation": self.trna_attenuation,
            "polymerized_ntps": self.sim_data.molecule_groups.polymerized_ntps,
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density),
            "n_avogadro": unum_to_pint(self.sim_data.constants.n_avogadro),
            "get_attenuation_stop_probabilities": {
                "_function": "transcription.get_attenuation_stop_probabilities",
                "_data": {
                    "aa_from_trna": self.sim_data.process.transcription.aa_from_trna.tolist(),
                    "attenuation_k": unum_to_pint(self.sim_data.process.transcription.attenuation_k).magnitude.tolist()
                        if hasattr(self.sim_data.process.transcription, 'attenuation_k')
                        else [],
                },
            },
            "attenuated_rna_indices": self.sim_data.process.transcription.attenuated_rna_indices,
            "location_lookup": self.sim_data.process.transcription.attenuation_location,
            "recycle_stalled_elongation": self.recycle_stalled_elongation,
            # random seed
            "seed": self._seedFromName("TranscriptElongation"),
            "submass_indices": self.submass_indices,
            "emit_unique": self.emit_unique,
        }

        return transcript_elongation_config

    def get_rna_degradation_config(self, time_step=1):
        transcription = self.sim_data.process.transcription
        rna_ids = list(transcription.rna_data["id"])
        mature_rna_ids = list(transcription.mature_rna_data["id"])
        all_rna_ids = rna_ids + mature_rna_ids
        rna_id_to_index = {rna_id: i for (i, rna_id) in enumerate(all_rna_ids)}
        cistron_ids = transcription.cistron_data["id"]
        cistron_id_to_index = {
            cistron_id: i for (i, cistron_id) in enumerate(cistron_ids)
        }

        rna_degradation_config = {
            "time_step": time_step,
            "rna_ids": rna_ids,
            "mature_rna_ids": mature_rna_ids,
            "cistron_ids": cistron_ids,
            "cistron_tu_mapping_matrix": transcription.cistron_tu_mapping_matrix,
            "mature_rna_cistron_indexes": np.array(
                [cistron_id_to_index[rna_id[:-3]] for rna_id in mature_rna_ids]
            ),
            "all_rna_ids": all_rna_ids,
            "n_total_RNAs": len(all_rna_ids),
            "n_avogadro": unum_to_pint(self.sim_data.constants.n_avogadro),
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density),
            "endoRNase_ids": self.sim_data.process.rna_decay.endoRNase_ids,
            "exoRNase_ids": self.sim_data.molecule_groups.exoRNases,
            "kcat_exoRNase": unum_to_pint(self.sim_data.constants.kcat_exoRNase),
            "Kcat_endoRNases": unum_to_pint(self.sim_data.process.rna_decay.kcats),
            "charged_trna_names": transcription.charged_trna_names,
            "uncharged_trna_indexes": np.array(
                [
                    rna_id_to_index[trna_id]
                    for trna_id in transcription.uncharged_trna_names
                ]
            ),
            "rna_deg_rates": (1 / units.s)
            * np.concatenate(
                (
                    unum_to_pint(transcription.rna_data["deg_rate"]).to(1 / units.s).magnitude,
                    unum_to_pint(transcription.mature_rna_data["deg_rate"]).to(1 / units.s).magnitude,
                )
            ),
            # All mature RNAs are not mRNAs
            "is_mRNA": np.concatenate(
                (
                    transcription.rna_data["is_mRNA"].astype(np.int64),
                    np.zeros(len(transcription.mature_rna_data), np.int64),
                )
            ),
            "is_rRNA": np.concatenate(
                (
                    transcription.rna_data["is_rRNA"].astype(np.int64),
                    transcription.mature_rna_data["is_rRNA"].astype(np.int64),
                )
            ),
            "is_tRNA": np.concatenate(
                (
                    transcription.rna_data["is_tRNA"].astype(np.int64),
                    transcription.mature_rna_data["is_tRNA"].astype(np.int64),
                )
            ),
            # NEW to vivarium-ecoli, used to degrade duplexes from RNAi
            "is_miscRNA": np.concatenate(
                (
                    transcription.rna_data["is_miscRNA"].astype(np.int64),
                    np.array(
                        [False] * len(transcription.mature_rna_data), dtype=np.int64
                    ),
                )
            ),
            "degrade_misc": self.degrade_misc,
            # End of new code
            # Load lengths and nucleotide counts for all degradable RNAs
            "rna_lengths": np.concatenate(
                (
                    unum_to_pint(transcription.rna_data["length"]).magnitude,
                    unum_to_pint(transcription.mature_rna_data["length"]).magnitude,
                )
            ),
            "nt_counts": np.concatenate(
                (
                    unum_to_pint(transcription.rna_data["counts_ACGU"]).to(units.nt).magnitude,
                    unum_to_pint(transcription.mature_rna_data["counts_ACGU"]).to(units.nt).magnitude,
                )
            ),
            # Load bulk molecule names
            "polymerized_ntp_ids": self.sim_data.molecule_groups.polymerized_ntps,
            "water_id": self.sim_data.molecule_ids.water,
            "ppi_id": self.sim_data.molecule_ids.ppi,
            "proton_id": self.sim_data.molecule_ids.proton,
            "nmp_ids": self.sim_data.molecule_groups.nmps,
            "rrfa_idx": rna_id_to_index["RRFA-RRNA[c]"],
            "rrla_idx": rna_id_to_index["RRLA-RRNA[c]"],
            "rrsa_idx": rna_id_to_index["RRSA-RRNA[c]"],
            "ribosome30S": self.sim_data.molecule_ids.s30_full_complex,
            "ribosome50S": self.sim_data.molecule_ids.s50_full_complex,
            # Load Michaelis-Menten constants fitted to recapitulate
            # first-order RNA decay model
            "Kms": (units.mol / units.L)
            * np.concatenate(
                (
                    unum_to_pint(transcription.rna_data["Km_endoRNase"]).to(units.mol / units.L).magnitude,
                    unum_to_pint(transcription.mature_rna_data["Km_endoRNase"]).to(units.mol / units.L).magnitude,
                )
            ),
            "seed": self._seedFromName("RnaDegradation"),
            "emit_unique": self.emit_unique,
        }

        return rna_degradation_config

    def get_polypeptide_initiation_config(self, time_step=1):
        polypeptide_initiation_config = {
            "time_step": time_step,
            "protein_lengths": unum_to_pint(self.sim_data.process.translation.monomer_data[
                "length"
            ]).magnitude,
            "translation_efficiencies": normalize(
                self.sim_data.process.translation.translation_efficiencies_by_monomer
            ),
            "active_ribosome_fraction": self.sim_data.process.translation.ribosomeFractionActiveDict,
            "elongation_rates": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.process.translation.ribosomeElongationRateDict.items()
            },
            "variable_elongation": self.variable_elongation_translation,
            "make_elongation_rates": {
                "_function": "translation.make_elongation_rates",
                "_data": {
                    "n_monomers": self.sim_data.process.translation.n_monomers,
                    "ribosomal_protein_indexes": self.sim_data.process.translation.ribosomal_protein_indexes.tolist(),
                    "max_elongation_rate": int(self.sim_data.process.translation.max_elongation_rate),
                },
            },
            "rna_id_to_cistron_indexes": self.sim_data.process.transcription.rna_id_to_cistron_indexes,
            "cistron_start_end_pos_in_tu": self.sim_data.process.transcription.cistron_start_end_pos_in_tu,
            "tu_ids": self.sim_data.process.transcription.rna_data["id"],
            "active_ribosome_footprint_size": unum_to_pint(self.sim_data.process.translation.active_ribosome_footprint_size),
            "cistron_to_monomer_mapping": self.sim_data.relation.cistron_to_monomer_mapping,
            "cistron_tu_mapping_matrix": self.sim_data.process.transcription.cistron_tu_mapping_matrix,
            "monomer_index_to_cistron_index": {
                i: self.sim_data.process.transcription._cistron_id_to_index[
                    monomer["cistron_id"]
                ]
                for (i, monomer) in enumerate(
                    self.sim_data.process.translation.monomer_data
                )
            },
            "monomer_index_to_tu_indexes": self.sim_data.relation.monomer_index_to_tu_indexes,
            "ribosome30S": self.sim_data.molecule_ids.s30_full_complex,
            "ribosome50S": self.sim_data.molecule_ids.s50_full_complex,
            "seed": self._seedFromName("PolypeptideInitiation"),
            "monomer_ids": self.sim_data.process.translation.monomer_data["id"],
            "emit_unique": self.emit_unique,
        }

        return polypeptide_initiation_config

    def get_polypeptide_elongation_config(self, time_step=1):
        constants = self.sim_data.constants
        molecule_ids = self.sim_data.molecule_ids
        translation = self.sim_data.process.translation
        transcription = self.sim_data.process.transcription
        metabolism = self.sim_data.process.metabolism

        polypeptide_elongation_config = {
            "time_step": time_step,
            # simulation options
            "aa_supply_in_charging": self.aa_supply_in_charging,
            "mechanistic_translation_supply": self.mechanistic_translation_supply,
            "mechanistic_aa_transport": self.mechanistic_aa_transport,
            "ppgpp_regulation": self.ppgpp_regulation,
            "disable_ppgpp_elongation_inhibition": self.disable_ppgpp_elongation_inhibition,
            "variable_elongation": self.variable_elongation_translation,
            # base parameters
            "n_avogadro": unum_to_pint(constants.n_avogadro),
            "proteinIds": translation.monomer_data["id"],
            "proteinLengths": unum_to_pint(translation.monomer_data["length"]).magnitude,
            "proteinSequences": translation.translation_sequences,
            "aaWeightsIncorporated": translation.translation_monomer_weights,
            "endWeight": translation.translation_end_weight,
            "make_elongation_rates": {
                "_function": "translation.make_elongation_rates",
                "_data": {
                    "n_monomers": translation.n_monomers,
                    "ribosomal_protein_indexes": translation.ribosomal_protein_indexes.tolist(),
                    "max_elongation_rate": int(translation.max_elongation_rate),
                },
            },
            "next_aa_pad": translation.next_aa_pad,
            "ribosomeElongationRate": float(
                unum_to_pint(self.sim_data.growth_rate_parameters.ribosomeElongationRate).to(units.aa / units.s).magnitude
            ),
            # Amino acid supply calculations
            "translation_aa_supply": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.translation_supply_rate.items()
            },
            "import_threshold": self.sim_data.external_state.import_constraint_threshold,
            # Data structures for charging
            "aa_from_trna": transcription.aa_from_trna,
            # Growth associated maintenance energy requirements for elongations
            "gtpPerElongation": constants.gtp_per_translation,
            # Bulk molecules
            "ribosome30S": self.sim_data.molecule_ids.s30_full_complex,
            "ribosome50S": self.sim_data.molecule_ids.s50_full_complex,
            "amino_acids": self.sim_data.molecule_groups.amino_acids,
            # parameters for specific elongation models
            "aa_exchange_names": np.array(
                [
                    self.sim_data.external_state.env_to_exchange_map[aa[:-3]]
                    for aa in self.sim_data.molecule_groups.amino_acids
                ]
            ),
            "basal_elongation_rate": unum_to_pint(self.sim_data.constants.ribosome_elongation_rate_basal).to(units.aa / units.s).magnitude,
            "ribosomeElongationRateDict": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.process.translation.ribosomeElongationRateDict.items()
            },
            "uncharged_trna_names": self.sim_data.process.transcription.uncharged_trna_names,
            "proton": self.sim_data.molecule_ids.proton,
            "water": self.sim_data.molecule_ids.water,
            "cellDensity": unum_to_pint(constants.cell_density),
            "elongation_max": unum_to_pint(
                constants.ribosome_elongation_rate_max
                if self.variable_elongation_translation
                else constants.ribosome_elongation_rate_basal
            ),
            "aa_from_synthetase": transcription.aa_from_synthetase,
            "charging_stoich_matrix": transcription.charging_stoich_matrix(),
            "charged_trna_names": transcription.charged_trna_names,
            "charging_molecule_names": transcription.charging_molecules,
            "synthetase_names": transcription.synthetase_names,
            "ppgpp_reaction_metabolites": metabolism.ppgpp_reaction_metabolites,
            "elong_rate_by_ppgpp": {
                "_function": "growth_rate.get_ribosome_elongation_rate_by_ppgpp",
                "_data": {
                    "ppgpp_units_str": "umol/L",
                    "rate_units_str": "aa/s",
                    "fit_vmax": float(self.sim_data.growth_rate_parameters._ribosome_elongation_rate_by_ppgpp[2]),
                    "KI": float(self.sim_data.growth_rate_parameters._ribosome_elongation_rate_by_ppgpp[3]),
                    "H": float(self.sim_data.growth_rate_parameters._ribosome_elongation_rate_by_ppgpp[4]),
                    "charging_fraction_of_max_elong_rate": float(self.sim_data.growth_rate_parameters._charging_fraction_of_max_elong_rate),
                },
            },
            "rela": molecule_ids.RelA,
            "spot": molecule_ids.SpoT,
            "ppgpp": molecule_ids.ppGpp,
            "kS": unum_to_pint(constants.synthetase_charging_rate).to(1 / units.s).magnitude,
            "KMaa": unum_to_pint(transcription.aa_kms).to(MICROMOLAR_UNITS).magnitude,
            "KMtf": unum_to_pint(transcription.trna_kms).to(MICROMOLAR_UNITS).magnitude,
            "krta": unum_to_pint(constants.Kdissociation_charged_trna_ribosome).to(MICROMOLAR_UNITS).magnitude,
            "krtf": unum_to_pint(constants.Kdissociation_uncharged_trna_ribosome).to(MICROMOLAR_UNITS).magnitude,
            # get_amino_acid_conc_conversion is upstream Unum-native; convert
            # our pint MICROMOLAR_UNITS at the boundary.
            "unit_conversion": metabolism.get_amino_acid_conc_conversion(
                pint_to_unum(MICROMOLAR_UNITS)
            ),
            "KD_RelA": unum_to_pint(transcription.KD_RelA).to(MICROMOLAR_UNITS).magnitude,
            "k_RelA": unum_to_pint(constants.k_RelA_ppGpp_synthesis).to(1 / units.s).magnitude,
            "k_SpoT_syn": unum_to_pint(constants.k_SpoT_ppGpp_synthesis).to(1 / units.s).magnitude,
            "k_SpoT_deg": unum_to_pint(constants.k_SpoT_ppGpp_degradation).to(1 / (MICROMOLAR_UNITS * units.s)).magnitude,
            "KI_SpoT": unum_to_pint(transcription.KI_SpoT).to(MICROMOLAR_UNITS).magnitude,
            "ppgpp_reaction_stoich": metabolism.ppgpp_reaction_stoich,
            "synthesis_index": metabolism.ppgpp_reaction_names.index(
                metabolism.ppgpp_synthesis_reaction
            ),
            "degradation_index": metabolism.ppgpp_reaction_names.index(
                metabolism.ppgpp_degradation_reaction
            ),
            "aa_supply_scaling": metabolism.aa_supply_scaling,
            "aa_enzymes": metabolism.aa_enzymes,
            "amino_acid_synthesis": metabolism.amino_acid_synthesis,
            "amino_acid_import": metabolism.amino_acid_import,
            "amino_acid_export": metabolism.amino_acid_export,
            "aa_importers": metabolism.aa_importer_names,
            "aa_exporters": metabolism.aa_exporter_names,
            "get_pathway_enzyme_counts_per_aa": {
                "_function": "metabolism.get_pathway_enzyme_counts_per_aa",
                "_data": {
                    "enzyme_to_amino_acid_fwd": metabolism.enzyme_to_amino_acid_fwd.tolist(),
                    "enzyme_to_amino_acid_rev": metabolism.enzyme_to_amino_acid_rev.tolist(),
                },
            },
            "import_constraint_threshold": self.sim_data.external_state.import_constraint_threshold,
            "seed": self._seedFromName("PolypeptideElongation"),
            "emit_unique": self.emit_unique,
        }

        return polypeptide_elongation_config

    def get_complexation_config(self, time_step=1):
        complexation_config = {
            "time_step": time_step,
            "stoichiometry": self.sim_data.process.complexation.stoich_matrix()
            .astype(np.int64)
            .T,
            "rates": self.sim_data.process.complexation.rates,
            "molecule_names": self.sim_data.process.complexation.molecule_names,
            "seed": self._seedFromName("Complexation"),
            "reaction_ids": self.sim_data.process.complexation.ids_reactions,
            "complex_ids": self.sim_data.process.complexation.ids_complexes,
            "emit_unique": self.emit_unique,
        }

        return complexation_config

    def get_two_component_system_config(self, time_step=1):
        two_component_system_config = {
            "time_step": time_step,
            "jit": False,
            # TODO -- wcEcoli has this in 1/mmol, why?
            "n_avogadro": unum_to_pint(self.sim_data.constants.n_avogadro).to(1 / units.mmol).magnitude,
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density).to(units.g / units.L).magnitude,
            "moleculesToNextTimeStep": {
                "_function": "two_component_system.ode_solver",
                "_data": {
                    "stoich_matrix_I": self.sim_data.process.two_component_system._stoichMatrixI.tolist(),
                    "stoich_matrix_J": self.sim_data.process.two_component_system._stoichMatrixJ.tolist(),
                    "stoich_matrix_V": self.sim_data.process.two_component_system._stoichMatrixV.tolist(),
                    "rates_fwd": self.sim_data.process.two_component_system.rates_fwd.tolist(),
                    "rates_rev": self.sim_data.process.two_component_system.rates_rev.tolist(),
                    "independent_molecule_indexes": self.sim_data.process.two_component_system.independent_molecule_indexes.tolist(),
                    "atp_reaction_reactant_mask": self.sim_data.process.two_component_system.atp_reaction_reactant_mask.tolist(),
                    "independent_molecules_atp_index": int(self.sim_data.process.two_component_system.independent_molecules_atp_index),
                    "dependency_matrix": self.sim_data.process.two_component_system.dependency_matrix.tolist(),
                    "rates_fn_dill": __import__('base64').b64encode(
                        __import__('dill').dumps(self.sim_data.process.two_component_system._rates)
                    ).decode('ascii'),
                    "rates_jac_fn_dill": __import__('base64').b64encode(
                        __import__('dill').dumps(self.sim_data.process.two_component_system._rates_jacobian)
                    ).decode('ascii'),
                },
            },
            "moleculeNames": self.sim_data.process.two_component_system.molecule_names,
            "seed": self._seedFromName("TwoComponentSystem"),
            "emit_unique": self.emit_unique,
        }

        # return two_component_system_config, stoichI, stoichJ, stoichV
        return two_component_system_config

    def get_equilibrium_config(self, time_step=1):
        equilibrium_config = {
            "time_step": time_step,
            "jit": False,
            "n_avogadro": unum_to_pint(self.sim_data.constants.n_avogadro).to(1 / units.mol).magnitude,
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density).to(units.g / units.L).magnitude,
            "stoichMatrix": self.sim_data.process.equilibrium.stoich_matrix().astype(
                np.int64
            ),
            "fluxesAndMoleculesToSS": {
                "_function": "equilibrium.ode_solver",
                "_data": {
                    "stoich_matrix": self.sim_data.process.equilibrium._stoichMatrix.tolist(),
                    "rates_fwd": self.sim_data.process.equilibrium.rates_fwd.tolist(),
                    "rates_rev": self.sim_data.process.equilibrium.rates_rev.tolist(),
                    "integrate_dt_mask": self.sim_data.process.equilibrium.integrate_dt_mask.tolist(),
                    "mets_to_rxn_fluxes": self.sim_data.process.equilibrium.mets_to_rxn_fluxes.tolist(),
                    "Rp": self.sim_data.process.equilibrium.Rp.tolist(),
                    "Pp": self.sim_data.process.equilibrium.Pp.tolist(),
                    "rates_fn_dill": __import__('base64').b64encode(
                        __import__('dill').dumps(self.sim_data.process.equilibrium._rates)
                    ).decode('ascii'),
                    "rates_jac_fn_dill": __import__('base64').b64encode(
                        __import__('dill').dumps(self.sim_data.process.equilibrium._rates_jacobian)
                    ).decode('ascii'),
                },
            },
            "moleculeNames": self.sim_data.process.equilibrium.molecule_names,
            "seed": self._seedFromName("Equilibrium"),
            "complex_ids": self.sim_data.process.equilibrium.ids_complexes,
            "reaction_ids": self.sim_data.process.equilibrium.rxn_ids,
            "emit_unique": self.emit_unique,
        }

        return equilibrium_config

    def get_protein_degradation_config(self, time_step=1):
        protein_degradation_config = {
            "time_step": time_step,
            "raw_degradation_rate": unum_to_pint(self.sim_data.process.translation.monomer_data[
                "deg_rate"
            ]).to(1 / units.s).magnitude,
            "water_id": self.sim_data.molecule_ids.water,
            "amino_acid_ids": self.sim_data.molecule_groups.amino_acids,
            "amino_acid_counts": unum_to_pint(self.sim_data.process.translation.monomer_data[
                "aa_counts"
            ]).magnitude,
            "protein_ids": self.sim_data.process.translation.monomer_data["id"],
            "protein_lengths": unum_to_pint(self.sim_data.process.translation.monomer_data[
                "length"
            ]).magnitude,
            "seed": self._seedFromName("ProteinDegradation"),
            "emit_unique": self.emit_unique,
        }

        return protein_degradation_config

    def get_metabolism_redux_config(self, time_step=1):
        metabolism = self.sim_data.process.metabolism
        aa_names = self.sim_data.molecule_groups.amino_acids
        aa_exchange_names = np.array(
            [
                self.sim_data.external_state.env_to_exchange_map[aa[:-3]]
                for aa in aa_names
            ]
        )
        aa_targets_not_updated = {"L-SELENOCYSTEINE[c]"}

        # Setup for variant that has a fixed change in ppGpp until a concentration is reached
        if hasattr(self.sim_data, "ppgpp_ramp"):
            self.sim_data.ppgpp_ramp.set_time(self.max_duration)
            self.sim_data.growth_rate_parameters.get_ppGpp_conc = (
                self.sim_data.ppgpp_ramp.get_ppGpp_conc
            )

        # if current_timeline_id is specified by a variant in sim_data, look it up in saved_timelines.
        if self.sim_data.external_state.current_timeline_id:
            current_timeline = self.sim_data.external_state.saved_timelines[
                self.sim_data.external_state.current_timeline_id
            ]
        else:
            current_timeline = self.media_timeline
        saved_media = self.sim_data.external_state.saved_media
        current_concentrations = saved_media[current_timeline[0][1]]

        # Get import molecules
        exch_from_conc = self.sim_data.external_state.exchange_data_from_concentrations
        exchange_data = exch_from_conc(current_concentrations)
        unconstrained = exchange_data["importUnconstrainedExchangeMolecules"]
        constrained = exchange_data["importConstrainedExchangeMolecules"]
        imports = set(unconstrained) | set(constrained)

        metabolism_config = {
            "time_step": time_step,
            # stoich
            "stoich_dict": metabolism.reaction_stoich,
            "maintenance_reaction": metabolism.maintenance_reaction,
            "reaction_catalysts": metabolism.reaction_catalysts,
            "get_mass": {
                "_function": "getter.get_mass",
                "_data": {
                    "all_total_masses": {k: float(v) for k, v in self.sim_data.getter._all_total_masses.items()},
                    "mass_units_value": 1.0,
                },
            },
            # wcEcoli parameters
            "get_import_constraints": {
                "_function": "external_state.get_import_constraints",
                "_data": {
                    "all_external_exchange_molecules": list(self.sim_data.external_state.all_external_exchange_molecules),
                },
            },
            "aa_targets_not_updated": aa_targets_not_updated,
            "import_constraint_threshold": self.sim_data.external_state.import_constraint_threshold,
            "exchange_molecules": self.sim_data.external_state.all_external_exchange_molecules,
            # these are options given to the wholecell.sim.simulation
            "use_trna_charging": self.trna_charging,
            "include_ppgpp": (not self.ppgpp_regulation)
            or (not self.trna_charging)
            or getattr(metabolism, "force_constant_ppgpp", False),
            "mechanistic_aa_transport": self.mechanistic_aa_transport,
            # variables
            "avogadro": unum_to_pint(self.sim_data.constants.n_avogadro),
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density),
            "nutrient_to_doubling_time": self.sim_data.nutrient_to_doubling_time,
            "dark_atp": self.sim_data.constants.darkATP,
            "non_growth_associated_maintenance": self.sim_data.constants.non_growth_associated_maintenance,
            "cell_dry_mass_fraction": self.sim_data.mass.cell_dry_mass_fraction,
            "ppgpp_id": self.sim_data.molecule_ids.ppGpp,
            "get_ppGpp_conc": {
                "_function": "growth_rate.get_ppGpp_conc",
                "_data": {
                    "x_units_str": "min",
                    "y_units_str": "pmol/L",
                    "fit_params": list(self.sim_data.growth_rate_parameters._ppGpp_concentration[2]),
                },
            },
            "get_masses": {
                "_function": "getter.get_masses",
                "_data": {
                    "all_total_masses": {k: float(v) for k, v in self.sim_data.getter._all_total_masses.items()},
                    "mass_units_value": 1.0,
                },
            },
            "kinetic_constraint_reactions": metabolism.kinetic_constraint_reactions,
            "doubling_time": self.sim_data.condition_to_doubling_time[
                self.sim_data.condition
            ],
            # Bind the sim_data method directly. The factory variant used to
            # precompute a doubling-time-indexed lookup table, which dropped
            # the rp_ratio argument on the floor — that scales glycogen /
            # murein / wall-component mass fractions and mattered to
            # metabolism's per-tick targets. Dropping it left v2ecoli
            # targeting ~15% more mass on those submasses, which FBA then
            # produced — the small-molecule submass grew ~13% faster than
            # vEcoli over a cell cycle.
            "get_biomass_as_concentrations": self.sim_data.mass.getBiomassAsConcentrations,
            "aa_names": self.sim_data.molecule_groups.amino_acids,
            "linked_metabolites": metabolism.concentration_updates.linked_metabolites,
            "aa_exchange_names": aa_exchange_names,
            "removed_aa_uptake": np.array(
                [aa in aa_targets_not_updated for aa in aa_exchange_names]
            ),
            "constraints_to_disable": metabolism.constraints_to_disable,
            "secretion_penalty_coeff": metabolism.secretion_penalty_coeff,
            "kinetic_objective_weight": metabolism.kinetic_objective_weight,
            "kinetic_objective_weight_in_range": metabolism.kinetic_objective_weight_in_range,
            # these values came from the initialized environment state
            "current_timeline": current_timeline,
            "media_id": current_timeline[0][1],
            "imports": imports,
            # methods
            "concentration_updates": {
                "_function": "metabolism.concentration_updates",
                "_data": {
                    "default_concentrations_dict": {k: float(v) for k, v in metabolism.concentration_updates.default_concentrations_dict.items()},
                    "exchange_fluxes": {k: list(v) if isinstance(v, set) else v for k, v in metabolism.concentration_updates.exchange_fluxes.items()},
                    "relative_changes": dict(metabolism.concentration_updates.relative_changes),
                    "molecule_set_amounts": {k: float(unum_to_pint(v).magnitude) if hasattr(unum_to_pint(v), 'magnitude') else float(v) for k, v in metabolism.concentration_updates.molecule_set_amounts.items()},
                    "molecule_scale_factors": dict(metabolism.concentration_updates.molecule_scale_factors),
                    "linked_metabolites": dict(metabolism.concentration_updates.linked_metabolites),
                },
            },
            "exchange_data_from_media": {
                "_function": "external_state.exchange_data_from_media",
                "_data": {
                    "saved_media": {k: dict(v) for k, v in self.sim_data.external_state.saved_media.items()},
                    "env_to_exchange_map": dict(self.sim_data.external_state.env_to_exchange_map),
                    "secretion_exchange_molecules": list(self.sim_data.external_state.secretion_exchange_molecules),
                    "import_constraint_threshold": float(self.sim_data.external_state.import_constraint_threshold),
                    "carbon_sources": list(self.sim_data.external_state.carbon_sources),
                },
            },
            "get_kinetic_constraints": {
                "_function": "metabolism.get_kinetic_constraints",
                "_data": {
                    "enzymes_expr": metabolism._enzymes,
                    "saturations_expr": metabolism._saturations,
                    "kcats": metabolism._kcats.tolist(),
                },
            },
            # exchange_constraints is built at init time from resolved concentration_updates
            # ports schema
            "catalyst_ids": metabolism.catalyst_ids,
            "kinetic_constraint_enzymes": metabolism.kinetic_constraint_enzymes,
            "kinetic_constraint_substrates": metabolism.kinetic_constraint_substrates,
            # Used to create conversion matrix that compiles individual fluxes
            # in the FBA solution to the fluxes of base reactions
            "base_reaction_ids": metabolism.base_reaction_ids,
            "fba_reaction_ids_to_base_reaction_ids": metabolism.reaction_id_to_base_reaction_id,
        }

        # TODO Create new config-get with only necessary parts.

        return metabolism_config

    def get_metabolism_config(self, time_step=1):
        # bad_rxns = ["RXN-12440", "TRANS-RXN-121", "TRANS-RXN-300"]
        # for rxn in bad_rxns:
        #     self.sim_data.process.metabolism.reaction_stoich.pop(rxn, None)
        #     self.sim_data.process.metabolism.reaction_catalysts.pop(rxn, None)
        #     self.sim_data.process.metabolism.reactions_with_catalyst.remove(rxn) \
        #         if rxn in self.sim_data.process.metabolism.reactions_with_catalyst else None
        metabolism = self.sim_data.process.metabolism
        aa_names = self.sim_data.molecule_groups.amino_acids
        aa_exchange_names = np.array(
            [
                self.sim_data.external_state.env_to_exchange_map[aa[:-3]]
                for aa in aa_names
            ]
        )
        aa_targets_not_updated = {"L-SELENOCYSTEINE[c]"}

        # Setup for variant that has a fixed change in ppGpp until a concentration is reached
        if hasattr(self.sim_data, "ppgpp_ramp"):
            self.sim_data.ppgpp_ramp.set_time(self.max_duration)
            self.sim_data.growth_rate_parameters.get_ppGpp_conc = (
                self.sim_data.ppgpp_ramp.get_ppGpp_conc
            )

        # if current_timeline_id is specified by a variant in sim_data, look it up in saved_timelines.
        if self.sim_data.external_state.current_timeline_id:
            current_timeline = self.sim_data.external_state.saved_timelines[
                self.sim_data.external_state.current_timeline_id
            ]
        else:
            current_timeline = self.media_timeline
        saved_media = self.sim_data.external_state.saved_media
        current_concentrations = saved_media[current_timeline[0][1]]

        # Get import molecules
        exch_from_conc = self.sim_data.external_state.exchange_data_from_concentrations
        exchange_data = exch_from_conc(current_concentrations)
        unconstrained = exchange_data["importUnconstrainedExchangeMolecules"]
        constrained = exchange_data["importConstrainedExchangeMolecules"]
        imports = set(unconstrained) | set(constrained)

        metabolism_config = {
            "time_step": time_step,
            # metabolism parameters
            "stoichiometry": metabolism.reaction_stoich,
            "catalyst_ids": metabolism.catalyst_ids,
            "concentration_updates": {
                "_function": "metabolism.concentration_updates",
                "_data": {
                    "default_concentrations_dict": {k: float(v) for k, v in metabolism.concentration_updates.default_concentrations_dict.items()},
                    "exchange_fluxes": {k: list(v) if isinstance(v, set) else v for k, v in metabolism.concentration_updates.exchange_fluxes.items()},
                    "relative_changes": dict(metabolism.concentration_updates.relative_changes),
                    "molecule_set_amounts": {k: float(unum_to_pint(v).magnitude) if hasattr(unum_to_pint(v), 'magnitude') else float(v) for k, v in metabolism.concentration_updates.molecule_set_amounts.items()},
                    "molecule_scale_factors": dict(metabolism.concentration_updates.molecule_scale_factors),
                    "linked_metabolites": dict(metabolism.concentration_updates.linked_metabolites),
                },
            },
            "maintenance_reaction": metabolism.maintenance_reaction,
            # wcEcoli parameters
            "get_import_constraints": {
                "_function": "external_state.get_import_constraints",
                "_data": {
                    "all_external_exchange_molecules": list(self.sim_data.external_state.all_external_exchange_molecules),
                },
            },
            "nutrientToDoublingTime": self.sim_data.nutrient_to_doubling_time,
            "aa_names": aa_names,
            "aa_targets_not_updated": aa_targets_not_updated,
            "import_constraint_threshold": self.sim_data.external_state.import_constraint_threshold,
            "exchange_molecules": self.sim_data.external_state.all_external_exchange_molecules,
            # these are options given to the wholecell.sim.simulation
            "use_trna_charging": self.trna_charging,
            "include_ppgpp": (not self.ppgpp_regulation)
            or (not self.trna_charging)
            or getattr(metabolism, "force_constant_ppgpp", False),
            "mechanistic_aa_transport": self.mechanistic_aa_transport,
            # these values came from the initialized environment state
            "current_timeline": current_timeline,
            "media_id": current_timeline[0][1],
            "imports": imports,
            # Translation layer: vEcoli's FluxBalanceAnalysisModel.__init__ reads
            # attributes directly off the metabolism instance (reaction_stoich,
            # maintenance_reaction, exchange_constraints, concentration_updates...).
            # We supply the instance as well as the extracted fields so either
            # access pattern works.
            "metabolism": metabolism,
            # Extracted from metabolism instance (replaces "metabolism": metabolism)
            "reaction_stoich": metabolism.reaction_stoich,
            "reactions_with_catalyst": metabolism.reactions_with_catalyst,
            "catalysis_matrix_I": metabolism.catalysis_matrix_I,
            "catalysis_matrix_J": metabolism.catalysis_matrix_J,
            "catalysis_matrix_V": metabolism.catalysis_matrix_V,
            "kinetic_constraint_reactions": metabolism.kinetic_constraint_reactions,
            "constraints_to_disable": metabolism.constraints_to_disable,
            "kinetic_constraint_enzymes": metabolism.kinetic_constraint_enzymes,
            "kinetic_constraint_substrates": metabolism.kinetic_constraint_substrates,
            "solver": metabolism.solver,
            "secretion_penalty_coeff": metabolism.secretion_penalty_coeff,
            "kinetic_objective_weight": metabolism.kinetic_objective_weight,
            "kinetic_objective_weight_in_range": metabolism.kinetic_objective_weight_in_range,
            "reaction_catalysts": metabolism.reaction_catalysts,
            "get_kinetic_constraints": {
                "_function": "metabolism.get_kinetic_constraints",
                "_data": {
                    "enzymes_expr": metabolism._enzymes,
                    "saturations_expr": metabolism._saturations,
                    "kcats": metabolism._kcats.tolist(),
                },
            },
            "exchange_constraints": metabolism.exchange_constraints,  # TODO: last bound method
            "ngam": unum_to_pint(self.sim_data.constants.non_growth_associated_maintenance),
            "avogadro": unum_to_pint(self.sim_data.constants.n_avogadro),
            "cell_density": unum_to_pint(self.sim_data.constants.cell_density),
            "dark_atp": unum_to_pint(self.sim_data.constants.darkATP),
            "cell_dry_mass_fraction": self.sim_data.mass.cell_dry_mass_fraction,
            # Bind the sim_data method directly. The factory variant used to
            # precompute a doubling-time-indexed lookup table, which dropped
            # the rp_ratio argument on the floor — that scales glycogen /
            # murein / wall-component mass fractions and mattered to
            # metabolism's per-tick targets. Dropping it left v2ecoli
            # targeting ~15% more mass on those submasses, which FBA then
            # produced — the small-molecule submass grew ~13% faster than
            # vEcoli over a cell cycle.
            "get_biomass_as_concentrations": self.sim_data.mass.getBiomassAsConcentrations,
            "ppgpp_id": self.sim_data.molecule_ids.ppGpp,
            "get_ppGpp_conc": {
                "_function": "growth_rate.get_ppGpp_conc",
                "_data": {
                    "x_units_str": "min",
                    "y_units_str": "pmol/L",
                    "fit_params": list(self.sim_data.growth_rate_parameters._ppGpp_concentration[2]),
                },
            },
            "exchange_data_from_media": {
                "_function": "external_state.exchange_data_from_media",
                "_data": {
                    "saved_media": {k: dict(v) for k, v in self.sim_data.external_state.saved_media.items()},
                    "env_to_exchange_map": dict(self.sim_data.external_state.env_to_exchange_map),
                    "secretion_exchange_molecules": list(self.sim_data.external_state.secretion_exchange_molecules),
                    "import_constraint_threshold": float(self.sim_data.external_state.import_constraint_threshold),
                    "carbon_sources": list(self.sim_data.external_state.carbon_sources),
                },
            },
            "get_masses": {
                "_function": "getter.get_masses",
                "_data": {
                    "all_total_masses": {k: float(v) for k, v in self.sim_data.getter._all_total_masses.items()},
                    "mass_units_value": 1.0,
                },
            },
            "doubling_time": self.sim_data.condition_to_doubling_time[
                self.sim_data.condition
            ],
            "amino_acid_ids": sorted(
                self.sim_data.amino_acid_code_to_id_ordered.values()
            ),
            "seed": self._seedFromName("Metabolism"),
            "linked_metabolites": metabolism.concentration_updates.linked_metabolites,
            "aa_exchange_names": aa_exchange_names,
            "removed_aa_uptake": np.array(
                [aa in aa_targets_not_updated for aa in aa_exchange_names]
            ),
            # TODO: testing, remove later (perhaps after moving change to sim_data)
            "reduce_murein_objective": False,
            # Used to create conversion matrix that compiles individual fluxes
            # in the FBA solution to the fluxes of base reactions
            "base_reaction_ids": self.sim_data.process.metabolism.base_reaction_ids,
            "fba_reaction_ids_to_base_reaction_ids": self.sim_data.process.metabolism.reaction_id_to_base_reaction_id,
        }

        return metabolism_config

    def get_mass_config(self, time_step=1):
        bulk_ids = self.sim_data.internal_state.bulk_molecules.bulk_data["id"]
        molecular_weights = {}
        for molecule_id in bulk_ids:
            molecular_weights[molecule_id] = unum_to_pint(self.sim_data.getter.get_mass(
                molecule_id
            )).to(units.fg / units.mol).magnitude

        # unique molecule masses
        unique_masses = {}
        uniqueMoleculeMasses = (
            self.sim_data.internal_state.unique_molecule.unique_molecule_masses
        )
        for id_, mass in zip(uniqueMoleculeMasses["id"], uniqueMoleculeMasses["mass"]):
            unique_masses[id_] = unum_to_pint((mass / self.sim_data.constants.n_avogadro)).to(units.fg).magnitude

        mass_config = {
            "molecular_weights": molecular_weights,
            "unique_masses": unique_masses,
            "cellDensity": unum_to_pint(self.sim_data.constants.cell_density).to(units.g / units.L).magnitude,
            "water_id": "WATER[c]",
            "emit_unique": self.emit_unique,
        }
        return mass_config

    def get_mass_listener_config(self, time_step=1):
        u_masses = self.sim_data.internal_state.unique_molecule.unique_molecule_masses
        molecule_ids = tuple(sorted(u_masses["id"]))
        molecule_id_to_mass = {}
        for id_, mass in zip(u_masses["id"], u_masses["mass"]):
            molecule_id_to_mass[id_] = unum_to_pint((
                mass / self.sim_data.constants.n_avogadro
            )).to(units.fg).magnitude
        molecule_masses = np.array([molecule_id_to_mass[x] for x in molecule_ids])

        mass_config = {
            "cellDensity": unum_to_pint(self.sim_data.constants.cell_density).to(units.g / units.L).magnitude,
            "bulk_ids": self.sim_data.internal_state.bulk_molecules.bulk_data["id"],
            "bulk_masses": unum_to_pint(self.sim_data.internal_state.bulk_molecules.bulk_data[
                "mass"
            ]).to(units.fg / units.mol).magnitude
            / unum_to_pint(self.sim_data.constants.n_avogadro).to(1 / units.mol).magnitude,
            "unique_ids": molecule_ids,
            "unique_masses": molecule_masses,
            "compartment_abbrev_to_index": self.sim_data.compartment_abbrev_to_index,
            "expectedDryMassIncreaseDict": self.sim_data.expectedDryMassIncreaseDict,
            "compartment_indices": {
                "projection": self.sim_data.compartment_id_to_index[
                    "CCO-CELL-PROJECTION"
                ],
                "cytosol": self.sim_data.compartment_id_to_index["CCO-CYTOSOL"],
                "extracellular": self.sim_data.compartment_id_to_index[
                    "CCO-EXTRACELLULAR"
                ],
                "flagellum": self.sim_data.compartment_id_to_index["CCO-FLAGELLUM"],
                "membrane": self.sim_data.compartment_id_to_index["CCO-MEMBRANE"],
                "outer_membrane": self.sim_data.compartment_id_to_index[
                    "CCO-OUTER-MEM"
                ],
                "periplasm": self.sim_data.compartment_id_to_index["CCO-PERI-BAC"],
                "pilus": self.sim_data.compartment_id_to_index["CCO-PILUS"],
                "inner_membrane": self.sim_data.compartment_id_to_index[
                    "CCO-PM-BAC-NEG"
                ],
            },
            "compartment_id_to_index": self.sim_data.compartment_id_to_index,
            "n_avogadro": self.sim_data.constants.n_avogadro,  # 1/mol
            "time_step": time_step,
            "submass_to_idx": self.sim_data.submass_name_to_index,
            "condition_to_doubling_time": {
                k: unum_to_pint(v)
                for k, v in self.sim_data.condition_to_doubling_time.items()
            },
            "condition": self.sim_data.condition,
            "emit_unique": self.emit_unique,
        }

        return mass_config

    def get_rna_counts_listener_config(self, time_step=1):
        counts_config = {
            "time_step": time_step,
            "all_TU_ids": self.sim_data.process.transcription.rna_data["id"],
            "mRNA_indexes": np.where(
                self.sim_data.process.transcription.rna_data["is_mRNA"]
            )[0],
            "rRNA_indexes": np.where(
                self.sim_data.process.transcription.rna_data["is_rRNA"]
            )[0],
            "all_cistron_ids": self.sim_data.process.transcription.cistron_data["id"],
            "cistron_is_mRNA": self.sim_data.process.transcription.cistron_data[
                "is_mRNA"
            ],
            "cistron_is_rRNA": self.sim_data.process.transcription.cistron_data[
                "is_rRNA"
            ],
            "cistron_tu_mapping_matrix": self.sim_data.process.transcription.cistron_tu_mapping_matrix,
            "emit_unique": self.emit_unique,
        }
        counts_config["mRNA_TU_ids"] = counts_config["all_TU_ids"][
            counts_config["mRNA_indexes"]
        ]
        counts_config["rRNA_TU_ids"] = counts_config["all_TU_ids"][
            counts_config["rRNA_indexes"]
        ]
        counts_config["mRNA_cistron_ids"] = counts_config["all_cistron_ids"][
            counts_config["cistron_is_mRNA"]
        ]
        counts_config["rRNA_cistron_ids"] = counts_config["all_cistron_ids"][
            counts_config["cistron_is_rRNA"]
        ]

        return counts_config

    def get_monomer_counts_listener_config(self, time_step=1):
        monomer_counts_config = {
            "time_step": time_step,
            # Get IDs of all bulk molecules
            "bulk_molecule_ids": self.sim_data.internal_state.bulk_molecules.bulk_data[
                "id"
            ],
            "unique_ids": self.sim_data.internal_state.unique_molecule.unique_molecule_masses[
                "id"
            ],
            # Get IDs of molecules involved in complexation and equilibrium
            "complexation_molecule_ids": self.sim_data.process.complexation.molecule_names,
            "complexation_complex_ids": self.sim_data.process.complexation.ids_complexes,
            "equilibrium_molecule_ids": self.sim_data.process.equilibrium.molecule_names,
            "equilibrium_complex_ids": self.sim_data.process.equilibrium.ids_complexes,
            "monomer_ids": self.sim_data.process.translation.monomer_data[
                "id"
            ].tolist(),
            # Get IDs of complexed molecules monomers involved in two component system
            # `modified_molecules` exists only on ParCa fixtures rebuilt after the
            # TCS port; fall back to molecule_names for older cached sim_data.
            "two_component_system_molecule_ids": list(
                getattr(
                    self.sim_data.process.two_component_system,
                    "modified_molecules",
                    self.sim_data.process.two_component_system.molecule_names,
                )
            ),
            "two_component_system_complex_ids": list(
                self.sim_data.process.two_component_system.complex_to_monomer.keys()
            ),
            # Get IDs of ribosome subunits
            "ribosome_50s_subunits": self.sim_data.process.complexation.get_monomers(
                self.sim_data.molecule_ids.s50_full_complex
            ),
            "ribosome_30s_subunits": self.sim_data.process.complexation.get_monomers(
                self.sim_data.molecule_ids.s30_full_complex
            ),
            # Get IDs of RNA polymerase subunits
            "rnap_subunits": self.sim_data.process.complexation.get_monomers(
                self.sim_data.molecule_ids.full_RNAP
            ),
            # Get IDs of replisome subunits
            "replisome_trimer_subunits": self.sim_data.molecule_groups.replisome_trimer_subunits,
            "replisome_monomer_subunits": self.sim_data.molecule_groups.replisome_monomer_subunits,
            # Get the IDs of transcription factors (compartment-tagged), so the
            # listener can fold bound-TF subunit counts back into the monomer total
            "tf_ids": [
                tf_id + f"[{self.sim_data.getter.get_compartment(tf_id)[0]}]"
                for tf_id in self.sim_data.process.transcription_regulation.tf_ids
            ],
            # Get stoichiometric matrices for complexation, equilibrium, two component system and the
            # assembly of unique molecules
            "complexation_stoich": self.sim_data.process.complexation.stoich_matrix_monomers(),
            "equilibrium_stoich": self.sim_data.process.equilibrium.stoich_matrix_monomers(),
            "two_component_system_stoich": self.sim_data.process.two_component_system.stoich_matrix_monomers(),
            "emit_unique": self.emit_unique,
        }

        return monomer_counts_config

    def get_allocator_config(self, time_step=1, process_names=None):
        if not process_names:
            process_names = []
        allocator_config = {
            "time_step": time_step,
            "molecule_names": self.sim_data.internal_state.bulk_molecules.bulk_data[
                "id"
            ],
            # Allocator is built into BulkMolecules container in wcEcoli
            "seed": self._seedFromName("BulkMolecules"),
            "process_names": process_names,
            "custom_priorities": {
                "ecoli-rna-degradation": 10,
                "ecoli-protein-degradation": 10,
                "ecoli-two-component-system": -5,
                "ecoli-tf-binding": -10,
                "ecoli-metabolism": -10,
            },
            "emit_unique": self.emit_unique,
        }
        return allocator_config

    def get_chromosome_structure_config(self, time_step=1):
        transcription = self.sim_data.process.transcription
        mature_rna_ids = transcription.mature_rna_data["id"]
        unprocessed_rna_indexes = np.where(transcription.rna_data["is_unprocessed"])[0]

        chromosome_structure_config = {
            "time_step": time_step,
            # Load parameters
            "rna_sequences": transcription.transcription_sequences,
            "protein_sequences": self.sim_data.process.translation.translation_sequences,
            "n_TUs": len(transcription.rna_data),
            "n_TFs": len(self.sim_data.process.transcription_regulation.tf_ids),
            "rna_ids": transcription.rna_data["id"],
            "n_amino_acids": len(self.sim_data.molecule_groups.amino_acids),
            "n_fragment_bases": len(self.sim_data.molecule_groups.polymerized_ntps),
            "replichore_lengths": self.sim_data.process.replication.replichore_lengths,
            "relaxed_DNA_base_pairs_per_turn": self.sim_data.process.chromosome_structure.relaxed_DNA_base_pairs_per_turn,
            "terC_index": self.sim_data.process.chromosome_structure.terC_dummy_molecule_index,
            "n_mature_rnas": len(mature_rna_ids),
            "mature_rna_ids": mature_rna_ids,
            "mature_rna_end_positions": transcription.mature_rna_end_positions,
            "mature_rna_nt_counts": unum_to_pint(
                transcription.mature_rna_data["counts_ACGU"]
            ).to(units.nt).magnitude.astype(int),
            "unprocessed_rna_index_mapping": {
                rna_index: i for (i, rna_index) in enumerate(unprocessed_rna_indexes)
            },
            "calculate_superhelical_densities": self.superhelical_density,
            # Get placeholder value for chromosome domains without children
            "no_child_place_holder": self.sim_data.process.replication.no_child_place_holder,
            # Load bulk molecule views
            "inactive_RNAPs": self.sim_data.molecule_ids.full_RNAP,
            "fragmentBases": self.sim_data.molecule_groups.polymerized_ntps,
            "ppi": self.sim_data.molecule_ids.ppi,
            "active_tfs": [
                x + "[c]" for x in self.sim_data.process.transcription_regulation.tf_ids
            ],
            "ribosome_30S_subunit": self.sim_data.molecule_ids.s30_full_complex,
            "ribosome_50S_subunit": self.sim_data.molecule_ids.s50_full_complex,
            "amino_acids": self.sim_data.molecule_groups.amino_acids,
            "water": self.sim_data.molecule_ids.water,
            "seed": self._seedFromName("ChromosomeStructure"),
            "emit_unique": self.emit_unique,
        }
        return chromosome_structure_config

    def get_rna_interference_config(self, time_step=1):
        rna_interference_config = {
            "time_step": time_step,
            "srna_ids": self.srna_ids,
            "target_tu_ids": self.target_tu_ids,
            "binding_probs": self.binding_probs,
            "duplex_ids": self.duplex_ids,
            "ribosome30S": self.sim_data.molecule_ids.s30_full_complex,
            "ribosome50S": self.sim_data.molecule_ids.s50_full_complex,
            "seed": self.random_state.randint(RAND_MAX),
            "emit_unique": self.emit_unique,
        }
        return rna_interference_config

    def get_tetracycline_ribosome_equilibrium_config(self, time_step=1):
        rna_ids = self.sim_data.process.transcription.rna_data["id"]
        is_trna = self.sim_data.process.transcription.rna_data["is_tRNA"].astype(
            np.bool_
        )
        tetracycline_ribosome_equilibrium_config = {
            "time_step": time_step,
            "trna_ids": rna_ids[is_trna],
            # Ensure that a new seed is set upon division
            "seed": self.random_state.randint(RAND_MAX),
            "emit_unique": self.emit_unique,
        }
        return tetracycline_ribosome_equilibrium_config

    def get_rna_maturation_config(self, time_step=1):
        transcription = self.sim_data.process.transcription
        rna_data = transcription.rna_data
        mature_rna_data = transcription.mature_rna_data

        config = {
            "time_step": time_step,
            # Get matrices and vectors that describe maturation reactions
            "stoich_matrix": transcription.rna_maturation_stoich_matrix,
            "enzyme_matrix": transcription.rna_maturation_enzyme_matrix.astype(int),
            "degraded_nt_counts": transcription.rna_maturation_degraded_nt_counts,
            # Get rRNA IDs
            "main_23s_rRNA_id": self.sim_data.molecule_groups.s50_23s_rRNA[0],
            "main_16s_rRNA_id": self.sim_data.molecule_groups.s30_16s_rRNA[0],
            "main_5s_rRNA_id": self.sim_data.molecule_groups.s50_5s_rRNA[0],
            "variant_23s_rRNA_ids": self.sim_data.molecule_groups.s50_23s_rRNA[1:],
            "variant_16s_rRNA_ids": self.sim_data.molecule_groups.s30_16s_rRNA[1:],
            "variant_5s_rRNA_ids": self.sim_data.molecule_groups.s50_5s_rRNA[1:],
            "unprocessed_rna_ids": rna_data["id"][rna_data["is_unprocessed"]],
            "mature_rna_ids": transcription.mature_rna_data["id"],
            "rna_maturation_enzyme_ids": transcription.rna_maturation_enzymes,
            # Other bulk IDs
            "fragment_bases": self.sim_data.molecule_groups.polymerized_ntps,
            "ppi": self.sim_data.molecule_ids.ppi,
            "water": self.sim_data.molecule_ids.water,
            "nmps": self.sim_data.molecule_groups.nmps,
            "proton": self.sim_data.molecule_ids.proton,
            "emit_unique": self.emit_unique,
        }
        config["n_required_enzymes"] = config["enzyme_matrix"].sum(axis=1)
        config["n_ppi_added"] = config["stoich_matrix"].toarray().sum(axis=0) - 1

        # Calculate number of NMPs that should be added when consolidating rRNA
        # molecules
        counts_ACGU = np.vstack(
            (
                unum_to_pint(rna_data["counts_ACGU"]).to(units.nt).magnitude,
                unum_to_pint(mature_rna_data["counts_ACGU"]).to(units.nt).magnitude,
            )
        )
        rna_id_to_index = {
            rna_id: i
            for i, rna_id in enumerate(chain(rna_data["id"], mature_rna_data["id"]))
        }

        def calculate_delta_nt_counts(main_id, variant_ids):
            main_index = rna_id_to_index[main_id]
            variant_indexes = np.array(
                [rna_id_to_index[rna_id] for rna_id in variant_ids]
            )

            delta_nt_counts = (
                counts_ACGU[variant_indexes, :] - counts_ACGU[main_index, :]
            )
            return delta_nt_counts

        config["delta_nt_counts_23s"] = calculate_delta_nt_counts(
            config["main_23s_rRNA_id"], config["variant_23s_rRNA_ids"]
        )
        config["delta_nt_counts_16s"] = calculate_delta_nt_counts(
            config["main_16s_rRNA_id"], config["variant_16s_rRNA_ids"]
        )
        config["delta_nt_counts_5s"] = calculate_delta_nt_counts(
            config["main_5s_rRNA_id"], config["variant_5s_rRNA_ids"]
        )

        return config

    def get_tf_unbinding_config(self, time_step=1):
        config = {
            "time_step": time_step,
            "tf_ids": self.sim_data.process.transcription_regulation.tf_ids,
            "submass_indices": self.submass_indices,
            "emit_unique": self.emit_unique,
        }
        # Build array of active TF masses
        bulk_ids = self.sim_data.internal_state.bulk_molecules.bulk_data["id"]
        tf_indexes = [
            np.where(bulk_ids == tf_id + "[c]")[0][0] for tf_id in config["tf_ids"]
        ]
        config["active_tf_masses"] = unum_to_pint((
            self.sim_data.internal_state.bulk_molecules.bulk_data["mass"][tf_indexes]
            / self.sim_data.constants.n_avogadro
        )).to(units.fg).magnitude
        return config

    def get_rna_synth_prob_listener_config(self, time_step=1):
        return {
            "time_step": time_step,
            "gene_ids": self.sim_data.process.transcription.cistron_data["gene_id"],
            "rna_ids": self.sim_data.process.transcription.rna_data["id"],
            "tf_ids": self.sim_data.process.transcription_regulation.tf_ids,
            "cistron_ids": self.sim_data.process.transcription.cistron_data["gene_id"],
            "cistron_tu_mapping_matrix": self.sim_data.process.transcription.cistron_tu_mapping_matrix,
            "emit_unique": self.emit_unique,
        }

    def get_dna_supercoiling_listener_config(self, time_step=1):
        return {
            "time_step": time_step,
            "relaxed_DNA_base_pairs_per_turn": self.sim_data.process.chromosome_structure.relaxed_DNA_base_pairs_per_turn,
            "emit_unique": self.emit_unique,
        }

    def get_unique_molecule_counts_config(self, time_step=1):
        return {
            "time_step": time_step,
            "unique_ids": self.sim_data.internal_state.unique_molecule.unique_molecule_masses[
                "id"
            ],
            "emit_unique": self.emit_unique,
        }

    def get_ribosome_data_listener_config(self, time_step=1):
        return {
            "time_step": time_step,
            "monomer_ids": self.sim_data.process.translation.monomer_data[
                "id"
            ].tolist(),
            "rRNA_cistron_tu_mapping_matrix": self.sim_data.process.transcription.rRNA_cistron_tu_mapping_matrix,
            "rRNA_is_5S": self.sim_data.process.transcription.cistron_data[
                "is_5S_rRNA"
            ][self.sim_data.process.transcription.cistron_data["is_rRNA"]],
            "rRNA_is_16S": self.sim_data.process.transcription.cistron_data[
                "is_16S_rRNA"
            ][self.sim_data.process.transcription.cistron_data["is_rRNA"]],
            "rRNA_is_23S": self.sim_data.process.transcription.cistron_data[
                "is_23S_rRNA"
            ][self.sim_data.process.transcription.cistron_data["is_rRNA"]],
            "emit_unique": self.emit_unique,
        }

    def get_rnap_data_listener_config(self, time_step=1):
        return {
            "time_step": time_step,
            "stable_RNA_indexes": np.where(
                np.logical_or(
                    self.sim_data.process.transcription.rna_data["is_rRNA"],
                    self.sim_data.process.transcription.rna_data["is_tRNA"],
                )
            )[0],
            "cistron_ids": self.sim_data.process.transcription.cistron_data["id"],
            "cistron_tu_mapping_matrix": self.sim_data.process.transcription.cistron_tu_mapping_matrix,
            "emit_unique": self.emit_unique,
        }

    def get_exchange_data_config(self, time_step=1):
        return {
            "time_step": time_step,
            "external_state": self.sim_data.external_state,
            "environment_molecules": list(
                self.sim_data.external_state.env_to_exchange_map.keys()
            ),
        }

    def get_media_update_config(self, time_step=1):
        # if current_timeline_id is specified by a variant in sim_data,
        # look it up in saved_timelines.
        if self.sim_data.external_state.current_timeline_id:
            current_timeline = self.sim_data.external_state.saved_timelines[
                self.sim_data.external_state.current_timeline_id
            ]
        else:
            current_timeline = self.media_timeline
        return {
            "time_step": time_step,
            "saved_media": self.sim_data.external_state.saved_media,
            "media_id": current_timeline[0][1],
        }

    def get_bulk_timeline_config(self, time_step=1):
        # if current_timeline_id is specified by a variant in sim_data, look it up in saved_timelines.
        if self.sim_data.external_state.current_timeline_id:
            current_timeline = self.sim_data.external_state.saved_timelines[
                self.sim_data.external_state.current_timeline_id
            ]
        else:
            current_timeline = self.media_timeline
        return {
            "time_step": time_step,
            "timeline": {
                time: {("media_id",): media_id} for time, media_id in current_timeline
            },
        }

    def generate_initial_state(self):
        """
        Calculate the initial conditions for a new cell without inherited state
        from a parent cell.
        """
        mass_coeff = 1.0
        if self.initial_state_gaussian:
            mass_coeff = self.random_state.normal(loc=1.0, scale=0.1)

        # if current_timeline_id is specified by a variant in sim_data,
        # look it up in saved_timelines.
        if self.sim_data.external_state.current_timeline_id:
            current_timeline = self.sim_data.external_state.saved_timelines[
                self.sim_data.external_state.current_timeline_id
            ]
        else:
            current_timeline = self.media_timeline
        media_id = current_timeline[0][1]
        current_concentrations = self.sim_data.external_state.saved_media[media_id]
        exch_from_conc = self.sim_data.external_state.exchange_data_from_concentrations
        exchange_data = exch_from_conc(current_concentrations)
        unconstrained = exchange_data["importUnconstrainedExchangeMolecules"]
        constrained = exchange_data["importConstrainedExchangeMolecules"]
        import_molecules = set(unconstrained) | set(constrained)

        bulk_state = initialize_bulk_counts(
            self.sim_data,
            media_id,
            import_molecules,
            self.random_state,
            mass_coeff,
            self.ppgpp_regulation,
            self.trna_attenuation,
        )
        cell_mass = calculate_cell_mass(bulk_state, {}, self.sim_data)
        # Create new PRNG for unique ID generation so self.random_state
        # can be used to faithfully replicate wcEcoli behavior
        unique_id_rng = np.random.RandomState(seed=self.seed + 100)
        unique_molecules = initialize_unique_molecules(
            bulk_state,
            self.sim_data,
            cell_mass,
            self.random_state,
            unique_id_rng,
            self.superhelical_density,
            self.ppgpp_regulation,
            self.trna_attenuation,
            self.mechanistic_replisome,
        )

        if self.trna_charging:
            initialize_trna_charging(
                bulk_state,
                unique_molecules,
                self.sim_data,
                self.variable_elongation_translation,
            )

        cell_mass = calculate_cell_mass(bulk_state, unique_molecules, self.sim_data)
        set_small_molecule_counts(
            bulk_state["count"],
            self.sim_data,
            media_id,
            import_molecules,
            mass_coeff,
            cell_mass,
        )

        # Numpy arrays are read-only outside of updaters for safety
        bulk_state.flags.writeable = False
        for unique_state in unique_molecules.values():
            unique_state.flags.writeable = False

        return {
            "bulk": bulk_state,
            "unique": unique_molecules,
            "environment": {
                "exchange": {mol: 0 for mol in current_concentrations},
                "exchange_data": {
                    "unconstrained": sorted(unconstrained),
                    "constrained": constrained,
                },
                "media_id": media_id,
            },
            "boundary": {
                "external": {
                    mol: conc * vivunits.mM
                    for mol, conc in current_concentrations.items()
                }
            },
        }
