"""
KnowledgeBase for Ecoli
Whole-cell knowledge base for Ecoli. Contains all raw, un-fit data processed
directly from CSV flat files.

"""

import io
import os
import json
from typing import List, Dict
import warnings

from v2ecoli.processes.parca.reconstruction.spreadsheets import read_tsv
from v2ecoli.processes.parca.wholecell.io import tsv
from v2ecoli.processes.parca.wholecell.utils import units  # used by eval()
from v2ecoli.processes.parca.reconstruction.ecoli.sources import relpath_to_key

FLAT_DIR = os.path.join(os.path.dirname(__file__), "flat")

#: ecoli-sources canonical keys for the long-form TPM tier, and the ONLY way to
#: address them. They are deliberately not reachable through
#: ``list_of_dict_filenames``:
#:
#: * the key is not derived from the path — ``relpath_to_key`` maps the shipped
#:   file to ``rnaseq_experimental__vecoli_m9_glucose_minus_aas``, not to
#:   ``rnaseq_experimental_tpms``; and
#: * ``_load_tsv`` names the attribute after the file's BASENAME, so a
#:   ``knockdown()`` variant bundle (which points the key at
#:   ``rnaseq_experimental_tpms__kd.tsv``) would land the table on a
#:   per-variant attribute name that nothing reads — a build that validates,
#:   hashes stably and changes nothing.
#:
#: So these load by canonical key onto a FIXED attribute
#: (``rnaseq_tpm_tables``), outside the flat-file loop.
RNASEQ_EXPERIMENTAL_KEY = "rnaseq_experimental_tpms"
RNASEQ_BASAL_KEY = "rnaseq_basal_tpms"
RNASEQ_TPM_KEYS = (RNASEQ_EXPERIMENTAL_KEY, RNASEQ_BASAL_KEY)


def load_tpm_table(path):
    """Read a long-form ``(gene_id, tpm_mean[, tpm_std])`` TPM table.

    Deliberately NOT ``spreadsheets.read_tsv``. That reader is a ``JsonReader``
    which ``json.loads`` every cell, and works only because every flat KB file
    is JSON-quoted (``"EG10001"``). The ecoli-sources TPM tier is plain
    unquoted TSV, so ``read_tsv`` raises
    ``ValueError: failed to parse json string:EG10001`` on it. The two tiers
    have different on-disk conventions and only pandas reads both.

    ``RnaseqTpmTableSchema`` (ecoli-sources' own Pandera schema for this tier)
    is applied when importable, so a malformed experimental table fails at load
    with a column/dtype error rather than deep in expression fitting. The
    guarded import mirrors ``sources.SourceBundle._validate``: an install
    without the ``schemas`` package degrades to unvalidated rather than failing.
    """
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    try:
        from schemas import RnaseqTpmTableSchema  # ecoli-sources package
    except ImportError:
        return df
    return RnaseqTpmTableSchema.validate(df)
LIST_OF_DICT_FILENAMES = [
    "amino_acid_export_kms.tsv",
    "amino_acid_export_kms_removed.tsv",
    "amino_acid_pathways.tsv",
    "amino_acid_uptake_rates.tsv",
    "amino_acid_uptake_rates_removed.tsv",
    "biomass.tsv",
    "compartments.tsv",
    "complexation_reactions.tsv",
    "complexation_reactions_added.tsv",
    "complexation_reactions_modified.tsv",
    "complexation_reactions_removed.tsv",
    "disabled_kinetic_reactions.tsv",
    "dna_sites.tsv",
    "dry_mass_composition.tsv",
    "endoRNases.tsv",
    "equilibrium_reaction_rates.tsv",
    "equilibrium_reactions.tsv",
    "equilibrium_reactions_added.tsv",
    "equilibrium_reactions_removed.tsv",
    "fold_changes.tsv",
    "fold_changes_nca.tsv",
    "fold_changes_removed.tsv",
    "footprint_sizes.tsv",
    "genes.tsv",
    "growth_rate_dependent_parameters.tsv",
    "linked_metabolites.tsv",
    "metabolic_reactions.tsv",
    "metabolic_reactions_added.tsv",
    "metabolic_reactions_modified.tsv",
    "metabolic_reactions_removed.tsv",
    "metabolism_kinetics.tsv",
    "metabolite_concentrations.tsv",
    "metabolite_concentrations_added.tsv",
    "metabolite_concentrations_removed.tsv",
    "metabolites.tsv",
    "metabolites_added.tsv",
    "modified_proteins.tsv",
    "molecular_weight_keys.tsv",
    "ppgpp_fc.tsv",
    "ppgpp_regulation.tsv",
    "ppgpp_regulation_added.tsv",
    "ppgpp_regulation_removed.tsv",
    "protein_half_lives_measured.tsv",
    "protein_half_lives_modified.tsv",
    "protein_half_lives_n_end_rule.tsv",
    "protein_half_lives_pulsed_silac.tsv",
    "proteins.tsv",
    "relative_metabolite_concentrations.tsv",
    "rna_half_lives.tsv",
    "rna_half_lives_removed.tsv",
    "rna_maturation_enzymes.tsv",
    "rnas.tsv",
    "secretions.tsv",
    "sequence_motifs.tsv",
    "transcription_factors.tsv",
    # "transcription_units.tsv",  # special cased in the constructor
    "transcription_units_added.tsv",
    "transcription_units_removed.tsv",
    "transcription_units_modified.tsv",
    "transcriptional_attenuation.tsv",
    "transcriptional_attenuation_removed.tsv",
    "tf_one_component_bound.tsv",
    "translation_efficiency.tsv",
    "trna_charging_reactions.tsv",
    "trna_charging_reactions_added.tsv",
    "trna_charging_reactions_removed.tsv",
    "two_component_systems.tsv",
    "two_component_system_templates.tsv",
    os.path.join("mass_fractions", "glycogen_fractions.tsv"),
    os.path.join("mass_fractions", "ion_fractions.tsv"),
    os.path.join("mass_fractions", "LPS_fractions.tsv"),
    os.path.join("mass_fractions", "lipid_fractions.tsv"),
    os.path.join("mass_fractions", "murein_fractions.tsv"),
    os.path.join("mass_fractions", "soluble_fractions.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_0p4.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_0p7.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_1p6.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_1p07.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_2p5.tsv"),
    os.path.join("trna_data", "trna_growth_rates.tsv"),
    os.path.join("rna_seq_data", "rnaseq_rsem_tpm_mean.tsv"),
    os.path.join("rna_seq_data", "rnaseq_rsem_tpm_std.tsv"),
    os.path.join("rna_seq_data", "rnaseq_seal_rpkm_mean.tsv"),
    os.path.join("rna_seq_data", "rnaseq_seal_rpkm_std.tsv"),
    os.path.join("rrna_options", "remove_rrff", "genes_removed.tsv"),
    os.path.join("rrna_options", "remove_rrff", "rnas_removed.tsv"),
    os.path.join("rrna_options", "remove_rrff", "transcription_units_modified.tsv"),
    os.path.join(
        "rrna_options", "remove_rrna_operons", "transcription_units_added.tsv"
    ),
    os.path.join(
        "rrna_options", "remove_rrna_operons", "transcription_units_removed.tsv"
    ),
    os.path.join("condition", "tf_condition.tsv"),
    os.path.join("condition", "condition_defs.tsv"),
    os.path.join("condition", "environment_molecules.tsv"),
    os.path.join("condition", "timelines_def.tsv"),
    os.path.join("condition", "media_recipes.tsv"),
    os.path.join("condition", "media", "5X_supplement_EZ.tsv"),
    os.path.join("condition", "media", "MIX0-55.tsv"),
    os.path.join("condition", "media", "MIX0-57.tsv"),
    os.path.join("condition", "media", "MIX0-58.tsv"),
    os.path.join("condition", "media", "MIX0-844.tsv"),
    # Finite tryptophan supplement (ecoli-sources#18): minimal_plus_tryptophan is
    # MIX0-57 x0.9 + TRP_supplement x0.1, matching vEcoli-private; the recipe
    # references it, so the loader must read it (test_media_registration).
    os.path.join("condition", "media", "TRP_supplement.tsv"),
    os.path.join("base_codes", "amino_acids.tsv"),
    os.path.join("base_codes", "dntp.tsv"),
    os.path.join("base_codes", "nmp.tsv"),
    os.path.join("base_codes", "ntp.tsv"),
    os.path.join("adjustments", "amino_acid_pathways.tsv"),
    os.path.join("adjustments", "balanced_translation_efficiencies.tsv"),
    os.path.join("adjustments", "translation_efficiencies_adjustments.tsv"),
    os.path.join("adjustments", "rna_expression_adjustments.tsv"),
    os.path.join("adjustments", "rna_deg_rates_adjustments.tsv"),
    os.path.join("adjustments", "protein_deg_rates_adjustments.tsv"),
    os.path.join("adjustments", "relative_metabolite_concentrations_changes.tsv"),
]
SEQUENCE_FILE = "sequence.fasta"
LIST_OF_PARAMETER_FILENAMES = [
    "dna_supercoiling.tsv",
    "parameters.tsv",
    "mass_parameters.tsv",
    os.path.join("new_gene_data", "new_gene_baseline_expression_parameters.tsv"),
]

REMOVED_DATA = {
    "amino_acid_export_kms": "amino_acid_export_kms_removed",
    "amino_acid_uptake_rates": "amino_acid_uptake_rates_removed",
    "complexation_reactions": "complexation_reactions_removed",
    "equilibrium_reactions": "equilibrium_reactions_removed",
    "fold_changes": "fold_changes_removed",
    "fold_changes_nca": "fold_changes_removed",
    "metabolic_reactions": "metabolic_reactions_removed",
    "metabolite_concentrations": "metabolite_concentrations_removed",
    "ppgpp_regulation": "ppgpp_regulation_removed",
    "transcriptional_attenuation": "transcriptional_attenuation_removed",
    "trna_charging_reactions": "trna_charging_reactions_removed",
    "rna_half_lives": "rna_half_lives_removed",
}
MODIFIED_DATA = {
    "complexation_reactions": "complexation_reactions_modified",
    "metabolic_reactions": "metabolic_reactions_modified",
}

def media_registration_gaps(bundle=None, list_of_dict_filenames=None):
    """Media files a recipe references that the KB loader would never read.

    ``LIST_OF_DICT_FILENAMES`` hardcodes which media ingredient files the
    loader reads. A media file can be shipped, declared in a bundle manifest
    and resolved by :class:`SourceBundle`, and still never be loaded, because
    membership of that list is maintained by hand. The build does not object:
    the recipe's ``base media`` / ``added media`` reference simply resolves
    against a table that was never populated.

    ``bundle`` is a parameter rather than a default lookup because the failure
    this guards against arrives through the override chain. A recipe supplied
    by ``--bundle-overrides`` can name a medium the public reference bundle
    never had, and a check hardcoded to the reference bundle could not see it.
    Pass the same ``SourceBundle`` the build will use.

    Returns:
        dict mapping media id -> reason, empty when the invariant holds.
        ``"not in LIST_OF_DICT_FILENAMES"`` means the loader has no entry for
        it; ``"not resolvable in the bundle"`` means the loader would look for
        a file the bundle cannot supply.
    """
    from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

    if bundle is None:
        bundle = SourceBundle()
    if list_of_dict_filenames is None:
        list_of_dict_filenames = LIST_OF_DICT_FILENAMES

    registered = {
        os.path.splitext(os.path.basename(f))[0]
        for f in list_of_dict_filenames
        if os.path.dirname(f) == os.path.join("condition", "media")
    }

    referenced = set()
    for row in read_tsv(str(bundle.path("condition__media_recipes"))):
        for column in ("base media", "added media"):
            media_id = (row.get(column) or "").strip()
            if media_id:
                referenced.add(media_id)

    gaps = {}
    for media_id in sorted(referenced):
        if media_id not in registered:
            gaps[media_id] = "not in LIST_OF_DICT_FILENAMES"
        elif not bundle.has_key(relpath_to_key(
                os.path.join("condition", "media", media_id + ".tsv"))):
            gaps[media_id] = "not resolvable in the bundle"
    return gaps


ADDED_DATA = {
    "complexation_reactions": "complexation_reactions_added",
    "equilibrium_reactions": "equilibrium_reactions_added",
    "metabolic_reactions": "metabolic_reactions_added",
    "metabolites": "metabolites_added",
    "ppgpp_regulation": "ppgpp_regulation_added",
    "trna_charging_reactions": "trna_charging_reactions_added",
    "metabolite_concentrations": "metabolite_concentrations_added",
}


class DataStore(object):
    def __init__(self):
        pass


#: Tables whose row identity is NOT a single ``id`` column, and the columns
#: that stand in for one. Without an entry here ``_join_data``'s collision
#: guard skips such a table entirely, so an added row naming a host entity is
#: joined silently -- the exact failure the guard exists to prevent, in the
#: tables where it is structurally blind.
COMPOSITE_ID_COLUMNS = {
    # metabolism.py keys kinetic constraints on (reaction, enzyme) and
    # accumulates rather than replaces, so a collision SHIFTS host chemistry.
    "metabolism_kinetics": ("reactionID", "enzymeID"),
}


class KnowledgeBaseEcoli(object):
    """KnowledgeBaseEcoli"""

    # COORDINATE CONVENTION: left_end_pos / right_end_pos are 1-based and
    # INCLUSIVE, matching the EcoCyc-derived flat files. A feature spanning
    # [L, R] occupies genome_sequence[L - 1 : R] (see
    # v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/
    # getter_functions.py:194, which slices exactly that way) and has length
    # R - L + 1. Every slice and comparison throughout this class assumes
    # this; getting it wrong is silent, because feature LENGTHS stay correct
    # under an off-by-one while the SEQUENCE shifts.

    def __init__(
        self,
        operons_on: bool,
        remove_rrna_operons: bool,
        remove_rrff: bool,
        stable_rrna: bool,
        new_genes_option: str = "off",
        bundle=None,
    ):
        if bundle is None:
            from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
            bundle = SourceBundle()
        self._bundle = bundle
        self.operons_on = operons_on
        self.stable_rrna = stable_rrna
        self.new_genes_option = new_genes_option

        if not operons_on and remove_rrna_operons:
            warnings.warn(
                "Setting the 'remove_rrna_operons' option to 'True'"
                " has no effect on the simulations when the 'operon'"
                " option is set to 'off'."
            )

        self.compartments: List[
            dict
        ] = []  # mypy can't track setattr(self, attr_name, rows)
        self.transcription_units: List[dict] = []

        # Make copies to prevent issues with sticky global variables when
        # running multiple operon workflows through Fireworks
        self.list_of_dict_filenames: List[str] = LIST_OF_DICT_FILENAMES.copy()
        self.list_of_parameter_filenames: List[str] = LIST_OF_PARAMETER_FILENAMES.copy()
        self.removed_data: Dict[str, str] = REMOVED_DATA.copy()
        self.modified_data: Dict[str, str] = MODIFIED_DATA.copy()
        self.added_data: Dict[str, str] = ADDED_DATA.copy()

        # Keyed by insertion subdirectory ('' for a single contiguous
        # cassette) so each cassette's file map is joined on its own
        # ``_join_data`` call rather than merged into one.
        self.new_gene_added_data: Dict[str, Dict[str, str]] = {}
        # (insertion_subdir, nested_attr) in discovery order; the splice
        # loop re-sorts these by declared position.
        self._new_gene_nested_attrs: list[tuple[str, str]] = []
        self.parameter_file_attribute_names: List[str] = [
            os.path.splitext(os.path.basename(filename))[0]
            for filename in self.list_of_parameter_filenames
        ]

        if self.operons_on:
            self.list_of_dict_filenames.append("transcription_units.tsv")
            if remove_rrna_operons:
                # Use alternative file with all rRNA transcription units if
                # remove_rrna_operons option was used
                self.removed_data.update(
                    {
                        "transcription_units": "rrna_options.remove_rrna_operons.transcription_units_removed",
                    }
                )
                self.added_data.update(
                    {
                        "transcription_units": "rrna_options.remove_rrna_operons.transcription_units_added",
                    }
                )
            else:
                self.removed_data.update(
                    {
                        "transcription_units": "transcription_units_removed",
                    }
                )
                self.added_data.update(
                    {
                        "transcription_units": "transcription_units_added",
                    }
                )
                self.modified_data.update(
                    {
                        "transcription_units": "transcription_units_modified",
                    }
                )

        if remove_rrff:
            self.list_of_parameter_filenames.append(
                os.path.join(
                    "rrna_options", "remove_rrff", "mass_parameters_modified.tsv"
                )
            )
            self.removed_data.update(
                {
                    "genes": "rrna_options.remove_rrff.genes_removed",
                    "rnas": "rrna_options.remove_rrff.rnas_removed",
                }
            )
            self.modified_data.update(
                {
                    "mass_parameters": "rrna_options.remove_rrff.mass_parameters_modified",
                }
            )
            if self.operons_on:
                self.modified_data.update(
                    {
                        "transcription_units": "rrna_options.remove_rrff.transcription_units_modified",
                    }
                )

        if self.new_genes_option != "off":
            # ⭐ MULTI-CASSETTE. A payload is either one contiguous insertion
            # or several noncontiguous ones, each in its own subdirectory with
            # its own insertion_location.tsv. ``[""]`` is the contiguous case,
            # which keeps the paths below identical to the single-insertion
            # form rather than special-casing it.
            _subdirs = self._new_gene_insertion_subdirs(new_genes_option) or [""]
            for insertion_subdir in _subdirs:
                new_genes_option_dir = os.path.join(
                    new_genes_option, insertion_subdir
                ) if insertion_subdir else new_genes_option
                self.new_gene_added_data.setdefault(insertion_subdir, {})
                new_gene_subdir = new_genes_option_dir
                new_gene_path = os.path.join("new_gene_data", new_gene_subdir)
                if self._bundle is not None:
                    # ⚠ A bundle is keyed, not pathed. ``new_gene_subdir`` is a
                    # FILESYSTEM path and gains a separator once cassettes are
                    # nested, so it cannot be interpolated into a key prefix --
                    # ``relpath_to_key`` is what maps one to the other.
                    assert self._bundle.keys_with_prefix(
                        relpath_to_key(new_gene_path) + "__"
                    ), "This new_genes_data subdirectory is invalid."
                else:
                    assert os.path.isdir(os.path.join(FLAT_DIR, new_gene_path)), (
                        "This new_genes_data subdirectory is invalid."
                    )
                # ⚠ ATTRIBUTE path, not a filesystem path: dots traverse the
                # loaded-data tree, so a nested cassette is
                # ``new_gene_data.<option>.<subdir>.`` -- NOT the slashed
                # ``new_gene_subdir`` used for file lookup above.
                nested_attr = "new_gene_data." + new_genes_option + "."
                if insertion_subdir:
                    nested_attr += insertion_subdir + "."
                self._new_gene_nested_attrs.append(
                    (insertion_subdir, nested_attr)
                )

                # These files do not need to be joined to existing files
                self.list_of_dict_filenames.append(
                    os.path.join(new_gene_path, "insertion_location.tsv")
                )
                self.list_of_dict_filenames.append(
                    os.path.join(new_gene_path, "gene_sequences.tsv")
                )

                # These files need to be joined to existing files
                new_gene_shared_files = [
                    "genes",
                    "rnas",
                    "proteins",
                    "rna_half_lives",
                    "protein_half_lives_measured",
                ]
                for f in new_gene_shared_files:
                    file_path = os.path.join(new_gene_path, f + ".tsv")
                    # If these files are empty, fill in with default values at a
                    # later point
                    if self._bundle is not None:
                        present = self._bundle.has_key(relpath_to_key(file_path))
                    else:
                        present = os.path.isfile(os.path.join(FLAT_DIR, file_path))
                    assert present, (
                        f"File {f}.tsv must be present in the new_genes_data"
                        f" subdirectory {new_gene_subdir}."
                    )
                    self.list_of_dict_filenames.append(file_path)
                    self.new_gene_added_data[insertion_subdir].update({f: nested_attr + f})

                # OPTIONAL joins. These are deliberately not in
                # ``new_gene_shared_files`` above: that list is asserted present,
                # and neither ``gfp`` nor ``template`` ships any of these files, so
                # requiring them would break every existing new-gene build.
                def _new_gene_file_present(filename):
                    """Whether the insertion ships ``filename``, bundle or flat."""
                    rel = os.path.join(new_gene_path, filename + ".tsv")
                    return (
                        self._bundle.has_key(relpath_to_key(rel))
                        if self._bundle is not None
                        else os.path.isfile(os.path.join(FLAT_DIR, rel))
                    )

                def _join_if_present(filename, key=None):
                    """Join one optional new-gene file, if the insertion ships it.

                    Rows are appended to the base attribute of the same name;
                    ``key`` overrides that when the base attribute lives under a
                    nested path (as ``rna_seq_data`` does).
                    """
                    rel = os.path.join(new_gene_path, filename + ".tsv")
                    if not _new_gene_file_present(filename):
                        return
                    self.list_of_dict_filenames.append(rel)
                    self.new_gene_added_data[insertion_subdir].update(
                        {key or filename: nested_attr + filename}
                    )

                # Why each of these is needed at all:
                #
                # ``metabolites`` -- a heterologous pathway's product and
                # intermediates are molecules the base flat files know nothing
                # about, and without them the product has no entry in the bulk
                # store to accumulate into.
                _join_if_present("metabolites")

                # ``complexation_reactions`` -- an insertion whose enzymes act as
                # protein COMPLEXES (e.g. a homodimer) names the complex as the
                # catalyst, and without complexation the complex is never formed:
                # the monomers accumulate with nothing to do and any consumer
                # looking up the catalyst id fails.
                _join_if_present("complexation_reactions")

                # ``metabolic_reactions`` -- THE REACTIONS THE INSERTED ENZYMES
                # CATALYSE. Without them the pathway's enzymes are expressed, its
                # product has a bulk entry, and nothing connects the two: the
                # product sits at its initial count for the whole simulation and
                # every flux and yield readout is a structural zero rather than a
                # measurement. ``metabolism.py`` builds ``reaction_stoich`` from
                # ``raw_data.metabolic_reactions``, so an insertion that declares
                # its own reactions must have them joined here or they do not
                # exist as far as the model is concerned.
                _join_if_present("metabolic_reactions")

                # ``metabolism_kinetics`` -- kcat/KM constraints for those
                # reactions. Absent, they are unconstrained rather than wrong, but
                # an insertion that ships measured kinetics means them to apply.
                _join_if_present("metabolism_kinetics")

                # ``transcription_units`` -- the insertion's OWN operon structure.
                # Absent, every inserted gene becomes its own transcription unit,
                # which is a different genetic construct from the one declared: it
                # changes transcription initiation, mRNA counts and the coupling
                # between the genes. ⚠ Positions in this file are RELATIVE to the
                # insertion and are converted to genome coordinates in
                # ``_update_gene_locations``; joining the file without that
                # conversion would place the operon at the wrong locus.
                #
                # ⚠ GATED ON ``operons_on``, and the gate is not cosmetic. With
                # operons off the BASE transcription_units table is never loaded
                # (see the constructor above), so ``self.transcription_units`` is
                # empty and ``_join_data`` -- which guards the added side but reads
                # ``data[0]`` on the base side -- raises a bare IndexError naming
                # nothing. It is also semantically pointless: ``transcription.py``
                # consumes TUs only when ``sim_data.operons_on``. The two options
                # are independent flags on the CLI, so this combination is
                # reachable rather than theoretical.
                if self.operons_on:
                    _join_if_present("transcription_units")
                elif _new_gene_file_present("transcription_units"):
                    warnings.warn(
                        f"new-gene insertion {new_gene_subdir!r} ships "
                        "transcription_units.tsv, but operons are disabled for "
                        "this build; its operon structure will be ignored and "
                        "each inserted gene becomes its own transcription unit."
                    )

                # ``environment_molecules`` / ``secretions`` -- THE PRODUCT'S WAY
                # OUT, and without it the pathway cannot run at all.
                #
                # This is not a reporting concern. FBA is a steady-state mass
                # balance: a metabolite that is produced, consumed by nothing, and
                # absent from the biomass objective has no sink, so the only
                # feasible flux through its producing reaction is ZERO. An
                # insertion can therefore have every gene expressed, every enzyme
                # complexed and every reaction joined, and still carry no flux --
                # reproducing exactly the silent zero the reaction join was added
                # to eliminate, one layer further on.
                #
                # ``environment_molecules`` makes the product an external exchange
                # molecule (id + compartment); ``secretions`` puts it in the
                # secretion set. Which compartment the exchange happens in is the
                # PAYLOAD's decision and deliberately not encoded here: a pathway
                # whose product is transported to the periplasm declares it there,
                # while one modelling export as a direct cytosolic exchange
                # declares it in the cytosol. Both are expressible; the base tables
                # already carry entries of each kind.
                _join_if_present(
                    "environment_molecules", key="condition.environment_molecules"
                )
                _join_if_present("secretions")

                _join_if_present(
                    "rnaseq_rsem_tpm_mean",
                    key="rna_seq_data.rnaseq_rsem_tpm_mean",
                )

        # Load raw data from TSV files
        for filename in self.list_of_dict_filenames:
            self._load_tsv(filename, self._resolve(filename))

        for filename in self.list_of_parameter_filenames:
            self._load_parameters(filename, self._resolve(filename))

        self._load_rnaseq_tpm_tables()

        self.genome_sequence = self._load_sequence(self._resolve(SEQUENCE_FILE))

        self._prune_data()

        self._join_data()
        self._modify_data()

        # Gene insertion is applied here. Gene DELETION is not: chromosome-level
        # knockouts are produced upstream by the ecoli-sources knockout generator
        # (`processing.genotypes.knockout`) and reach ParCa as a variant bundle,
        # not as a constructor option.
        if self.new_genes_option != "off":
            # ⭐⭐ SPLICE ORDER IS LOAD-BEARING, AND DESCENDING IS THE CORRECT ONE.
            #
            # ``insertion_pos`` in each cassette's insertion_location.tsv is a
            # coordinate in the ORIGINAL, UNSPLICED genome. (The convention is
            # asserted below, and it is the only one that makes the declared
            # loci meaningful -- payloads name their site by the real E. coli
            # locus it corresponds to.)
            #
            # Each splice shifts every coordinate above it by the cassette
            # length. Splicing HIGH-TO-LOW means a later (lower) splice moves an
            # already-placed cassette together with all of its neighbours, so
            # its biological locus is preserved. Splicing low-to-high instead
            # invalidates every declared position after the first: the next
            # cassette's raw coordinate is then read in the shifted frame.
            #
            # ⛔ THIS IS NOT THEORETICAL AND THE OBVIOUS ORDER IS THE WRONG ONE.
            # Measured 2026-08-31 against the upstream fork, which sorts
            # ascending: the SAME cassette resolved to two different loci --
            # 400925 alone, 398966 when a 1259 bp cassette was inserted below it
            # first -- 1959 bp apart, differing only in whether another
            # insertion was present.
            #
            # ⚠ AND THE LOADER'S OWN "has been shifted" DIAGNOSTIC CANNOT SEE
            # THIS. ``_update_gene_insertion_location`` moves a position only on
            # COLLISION, so a wrong-frame coordinate that lands in an intergenic
            # gap is placed silently at the wrong locus, and a collision shift
            # that does occur has no relation to the cassette length. The only
            # sound check is to resolve a cassette ALONE and compare loci --
            # which is what this module's regression test does.
            _insertions = []
            for insertion_subdir, nested_attr in self._new_gene_nested_attrs:
                _loc = self._resolve_nested(nested_attr + "insertion_location")
                assert len(_loc) == 1, (
                    "each noncontiguous insertion should be in its own"
                    f" directory; {insertion_subdir or new_genes_option!r} has"
                    f" {len(_loc)}"
                )
                _insertions.append((_loc[0]["insertion_pos"], insertion_subdir, nested_attr))
            # DESCENDING. See above -- reversing this is the defect.
            _insertions.sort(key=lambda t: t[0], reverse=True)

            for _declared_pos, insertion_subdir, nested_attr in _insertions:
                self._check_new_gene_ids(nested_attr)

                insert_pos = self._update_gene_insertion_location(nested_attr)

                insertion_sequence = self._get_new_gene_sequence(nested_attr)

                insert_end = self._update_gene_locations(nested_attr, insert_pos)
                self.new_gene_added_data[insertion_subdir].update(
                    {"genes": nested_attr + "genes"}
                )

                self.genome_sequence = (
                    self.genome_sequence[:insert_pos]
                    + insertion_sequence
                    + self.genome_sequence[insert_pos:]
                )
                assert (
                    self.genome_sequence[insert_pos:insert_end] == insertion_sequence
                )

                # Joined PER INSERTION: ``added_data`` is a per-iteration scratch
                # slot, not a set that accumulates. ``_join_data`` appends each
                # cassette's rows to the base tables on its own call.
                # ⚠ Its duplicate-id guard is a real N>1 interaction: two
                # cassettes that each declare the same row (a shared host
                # metabolite, say) collide on the SECOND join and fail loudly
                # there rather than in the payload that caused it.
                self.added_data = self.new_gene_added_data[insertion_subdir]
                self._join_data()

    def _load_rnaseq_tpm_tables(self):
        """Load the long-form TPM tier by CANONICAL KEY onto fixed attributes.

        The KB loads these; it does not GATE on them. Which tier a build
        actually fits against is decided once, on ``sim_data``
        (``sim_data.rnaseq_source``), so both entry points — the ``v2ecoli-parca``
        CLI, which injects a KB, and the composite path, which builds one — reach
        the same decision through the same field. Loading here unconditionally is
        what makes that possible: nothing downstream has to ask the KB to have
        been constructed differently.

        Cost is ~4.6k rows per key. Absent keys are simply absent — a hand-cut
        bundle without them still builds, and the reference path never reads them.
        """
        self.rnaseq_tpm_tables: Dict[str, object] = {}
        self.rnaseq_tpm_sources: Dict[str, str] = {}
        if self._bundle is None:
            return
        for key in RNASEQ_TPM_KEYS:
            if not self._bundle.has_key(key):
                continue
            path = self._bundle.path(key)
            self.rnaseq_tpm_tables[key] = load_tpm_table(path)
            self.rnaseq_tpm_sources[key] = str(path)

    def _resolve(self, rel_path):
        return self._bundle.resolve_relpath(rel_path)

    def _load_tsv(self, rel_path, abs_path):
        path = self
        parts = rel_path.replace(os.sep, "/").split("/")
        for sub_path in parts[:-1]:
            if not hasattr(path, sub_path):
                setattr(path, sub_path, DataStore())
            path = getattr(path, sub_path)
        attr_name = parts[-1].split(".")[0]
        setattr(path, attr_name, [])

        rows = read_tsv(str(abs_path))
        setattr(path, attr_name, rows)

    def _load_sequence(self, file_path):
        from Bio import SeqIO

        with open(file_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                return record.seq

    def _load_parameters(self, rel_path, abs_path):
        path = self
        parts = rel_path.replace(os.sep, "/").split("/")
        for sub_path in parts[:-1]:
            if not hasattr(path, sub_path):
                setattr(path, sub_path, DataStore())
            path = getattr(path, sub_path)
        attr_name = parts[-1].split(".")[0]
        param_dict = {}

        with io.open(str(abs_path), "rb") as csvfile:
            reader = tsv.dict_reader(csvfile)
            for row in reader:
                value = json.loads(row["value"])
                if row["units"] != "":
                    unit = eval(row["units"])  # risky!
                    unit = units.getUnit(unit)  # strip
                    value = value * unit
                param_dict[row["name"]] = value

        setattr(path, attr_name, param_dict)

    def _prune_data(self):
        """
        Remove rows that are specified to be removed. Data will only be removed
        if all data in a row in the file specifying rows to be removed matches
        the same columns in the raw data file.
        """

        # Check each pair of files to be removed
        for data_attr, attr_to_remove in self.removed_data.items():
            # Build the set of data to identify rows to be removed
            data_to_remove = getattr(self, attr_to_remove.split(".")[0])
            for attr in attr_to_remove.split(".")[1:]:
                data_to_remove = getattr(data_to_remove, attr)
            removed_cols = list(data_to_remove[0].keys())
            ids_to_remove = set()
            for row in data_to_remove:
                ids_to_remove.add(tuple([row[col] for col in removed_cols]))

            # Remove any matching rows
            data = getattr(self, data_attr)
            n_entries = len(data)
            removed_ids = set()
            for i, row in enumerate(data[::-1]):
                checked_id = tuple([row[col] for col in removed_cols])
                if checked_id in ids_to_remove:
                    data.pop(n_entries - i - 1)
                    removed_ids.add(checked_id)

            # Print warnings for entries that were marked to be removed that
            # does not exist in the original data file. Fold changes are
            # excluded since the original entries are split between two files.
            if not data_attr.startswith("fold_changes"):
                for unremoved_id in ids_to_remove - removed_ids:
                    print(
                        f"Warning: Could not remove row {unremoved_id} "
                        f"in flat file {data_attr} because the row does not "
                        f"exist."
                    )

    def _resolve_nested(self, dotted: str):
        """Walk a dotted attribute path on the loaded-data tree."""
        parts = dotted.split(".")
        obj = getattr(self, parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

    def _new_gene_insertion_subdirs(self, new_gene_subdir: str) -> list[str]:
        """Cassette subdirectories under one ``new_genes_option``, or ``[]``.

        A new-gene payload is either ONE contiguous cassette (files directly
        under ``new_gene_data/<option>/``) or SEVERAL noncontiguous cassettes,
        each in its own subdirectory with its own ``insertion_location.tsv``.
        Returns the subdirectory names for the second shape and ``[]`` for the
        first, so the caller can treat both uniformly.

        ⚠ Both source shapes must be supported and they enumerate differently.
        The flat tree can be scanned with ``os.scandir``; a bundle has no
        directories at all, only canonical keys, so the subdirectory is
        recovered from the key SEGMENTS:
        ``new_gene_data__<option>__<subdir>__<file>`` is nested, whereas
        ``new_gene_data__<option>__<file>`` is contiguous. Scanning the
        filesystem when a bundle is in use would silently find nothing and
        demote a multi-cassette payload to a single-cassette one.
        """
        if self._bundle is not None:
            # ⚠ Same path-vs-key hazard as the caller's presence check: build
            # the prefix through ``relpath_to_key`` rather than interpolating.
            # Harmless while this is only ever called with a top-level option
            # (no separator to mangle), but it fails silently -- matching
            # nothing -- the moment one contains a path separator.
            prefix = relpath_to_key(
                os.path.join("new_gene_data", new_gene_subdir)
            ) + "__"
            subdirs = set()
            for key in self._bundle.keys_with_prefix(prefix):
                rest = key[len(prefix) :].split("__")
                if len(rest) > 1:  # <subdir>__<file> => nested
                    subdirs.add(rest[0])
            return sorted(subdirs)
        base = os.path.join(FLAT_DIR, "new_gene_data", new_gene_subdir)
        if not os.path.isdir(base):
            return []
        with os.scandir(base) as entries:
            return sorted(e.name for e in entries if e.is_dir())

    def _join_data(self):
        """
        Add rows that are specified in additional files. Data will only be added
        if all the loaded columns from both datasets match.
        """

        # Join data for each file with data to be added
        for data_attr, attr_to_add in self.added_data.items():
            # Get datasets to join
            data = getattr(self, data_attr.split(".")[0])
            for attr in data_attr.split(".")[1:]:
                data = getattr(data, attr)

            added_data = getattr(self, attr_to_add.split(".")[0])
            for attr in attr_to_add.split(".")[1:]:
                added_data = getattr(added_data, attr)

            if added_data:  # Some new gene additional files may be empty
                # Check columns are the same for each dataset
                col_diff = set(data[0].keys()).symmetric_difference(
                    added_data[0].keys()
                )
                if col_diff:
                    raise ValueError(
                        f"Could not join datasets {data_attr} and {attr_to_add} "
                        f"because columns do not match (different columns: {col_diff})."
                    )

                # An added row whose id already exists in the base table does
                # not merge -- both rows are kept and every downstream consumer
                # that builds an id-keyed dict silently takes the LAST one
                # (e.g. molecular weights and charges in getter_functions /
                # metabolism). For a new-gene insertion that means a payload
                # re-declaring a HOST molecule would quietly redefine the
                # host's chemistry: a heterologous pathway consumes host
                # metabolites, so its own tables can plausibly name one.
                # Fail loudly instead -- a redefinition may well be intended,
                # but it must be deliberate rather than a silent last-write.
                #
                # Guarding the GENERIC join rather than only the two new
                # optional tables was checked by enumerating every shipped
                # (base <- added) pair and comparing raw id sets: the live
                # id-keyed joins (complexation_reactions, equilibrium_reactions,
                # metabolic_reactions, metabolites, trna_charging_reactions,
                # transcription_units) all carry ZERO collisions, and
                # ppgpp_regulation has no id column so the guard skips it.
                # ⚠ A new-gene insertion can now contribute to several of these
                # AND to metabolism_kinetics, which is keyed compositely rather
                # than by id -- see COMPOSITE_ID_COLUMNS below.
                # ⚠ The test suite alone does NOT establish this: the 81-row
                # transcription_units join comes from the remove_rrna_operons
                # option, which nothing sets True (every call site hardcodes
                # False), so no test exercises that path.
                # ⚠ Nor does the NG- naming convention: a payload is not bound
                # by it. That is an argument FOR guarding generically rather
                # than trusting the prefix.
                # ⚠ Guarding on ``id`` alone leaves a table with no id column
                # UNGUARDED, not merely unchecked: an added row naming a HOST
                # entity joins in silence. ``metabolism_kinetics`` is the live
                # case -- it is keyed on (reactionID, enzymeID), and
                # ``metabolism.py`` ACCUMULATES constraints per that pair
                # rather than replacing them, so a payload row naming a host
                # pair shifts that host reaction's constraint with nothing
                # raised anywhere.
                key_columns = COMPOSITE_ID_COLUMNS.get(data_attr)
                if key_columns and all(c in data[0] for c in key_columns):
                    def _identity(row):
                        return tuple(row[c] for c in key_columns)
                    label = "+".join(key_columns)
                elif "id" in data[0]:
                    def _identity(row):
                        return row["id"]
                    label = "id"
                else:
                    _identity = None

                if _identity is not None:
                    base_ids = {_identity(row) for row in data}
                    clashes = sorted(
                        {_identity(row) for row in added_data
                         if _identity(row) in base_ids},
                        key=str,
                    )
                    if clashes:
                        raise ValueError(
                            f"Cannot join {attr_to_add} into {data_attr}: "
                            f"{len(clashes)} {label}(s) already exist in the "
                            f"base table and would silently redefine it "
                            f"{clashes[:5]}. Rename the added rows, or remove "
                            f"the colliding rows from the base table via the "
                            f"corresponding *_removed.tsv."
                        )

            # Join datasets
            for row in added_data:
                data.append(row)

    def _modify_data(self):
        """
        Modify entires in rows that are specified to be modified. Rows must be
        identified by their entries in the first column (usually the ID column).
        """
        # Check each pair of files to be modified
        for data_attr, modify_attr in self.modified_data.items():
            # Build the set of data to identify rows to be modified
            data_to_modify = getattr(self, modify_attr.split(".")[0])
            for attr in modify_attr.split(".")[1:]:
                data_to_modify = getattr(data_to_modify, attr)
            data = getattr(self, data_attr)

            # If modifying a parameter file, replace values in dictionary
            if data_attr in self.parameter_file_attribute_names:
                for key, value in data_to_modify.items():
                    if key not in data:
                        raise ValueError(
                            f"Could not modify data {data_attr}"
                            f"with {modify_attr} because the name {key} does "
                            f"not exist in {data_attr}."
                        )

                    data[key] = value

            # If modifying a table file, replace rows
            else:
                id_col_name = list(data_to_modify[0].keys())[0]

                id_to_modified_cols = {}
                for row in data_to_modify:
                    id_to_modified_cols[row[id_col_name]] = row

                # Modify any matching rows with identical IDs
                if list(data[0].keys())[0] != id_col_name:
                    raise ValueError(
                        f"Could not modify data {data_attr} with "
                        f"{modify_attr} because the names of the first columns "
                        f"do not match."
                    )

                modified_entry_ids = set()
                for i, row in enumerate(data):
                    if row[id_col_name] in id_to_modified_cols:
                        modified_cols = id_to_modified_cols[row[id_col_name]]
                        for col_name in data[i]:
                            if col_name in modified_cols:
                                data[i][col_name] = modified_cols[col_name]
                        modified_entry_ids.add(row[id_col_name])

                # Check for entries in modification data that do not exist in
                # original data
                id_diff = set(id_to_modified_cols.keys()).symmetric_difference(
                    modified_entry_ids
                )
                if id_diff:
                    raise ValueError(
                        f"Could not modify data {data_attr} with "
                        f"{modify_attr} because of one or more entries in "
                        f"{modify_attr} that do not exist in {data_attr} "
                        f"(nonexistent entries: {id_diff})."
                    )

    def _check_new_gene_ids(self, nested_attr):
        """
        Check to ensure each new gene, RNA, and protein id starts with NG.
        """
        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        new_genes_data = getattr(nested_data, "genes")
        new_RNA_data = getattr(nested_data, "rnas")
        new_protein_data = getattr(nested_data, "proteins")

        for row in new_genes_data:
            assert row["id"].startswith("NG"), "ids of new genes must start with NG"
        for row in new_RNA_data:
            assert row["id"].startswith("NG"), "ids of new gene rnas must start with NG"
        for row in new_protein_data:
            assert row["id"].startswith("NG"), (
                "ids of new gene proteins must start with NG"
            )
        return

    def _update_gene_insertion_location(self, nested_attr):
        """
        Update insertion location of new genes to prevent conflicts.
        """

        genes_data = getattr(self, "genes")
        tu_data = getattr(self, "transcription_units")
        dna_sites_data = getattr(self, "dna_sites")

        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        insert_loc_data = getattr(nested_data, "insertion_location")

        assert len(insert_loc_data) == 1, (
            "each noncontiguous insertion should be in its own directory"
        )
        insert_pos = insert_loc_data[0]["insertion_pos"]

        if not tu_data:
            # Check if specified insertion location is in another gene
            data_to_check = genes_data.copy()
        else:
            # Check if specified insertion location is in a transcription unit
            data_to_check = tu_data.copy()

        # Add important DNA sites to the list of locations to check
        # TODO: Check for other DNA sites if we include any in the future
        sites_data_to_check = [
            site
            for site in dna_sites_data
            if site["common_name"] == "oriC" or site["common_name"] == "TerC"
        ]
        data_to_check += sites_data_to_check

        conflicts = [
            row
            for row in data_to_check
            if ((row["left_end_pos"] is not None) and (row["left_end_pos"] != ""))
            and ((row["right_end_pos"] is not None) and (row["left_end_pos"] != ""))
            and (row["left_end_pos"] < insert_pos)
            and (row["right_end_pos"] >= insert_pos)
        ]
        # Change insertion location to after conflicts
        if conflicts:
            shift = max([sub["right_end_pos"] for sub in conflicts]) - insert_pos + 1
            insert_pos = insert_pos + shift

        return insert_pos

    def _update_global_coordinates(self, data, insert_pos, insert_len):
        """
        Updates the left and right end positions for all elements in data if
        their positions will be impacted by the new gene insertion.

        Args:
            data: Data attribute to update
            insert_pos: Location of new gene insertion
            insert_len: Length of new gene insertion

        """
        for row in data:
            left = row["left_end_pos"]
            right = row["right_end_pos"]
            if (
                (left is not None and left != "")
                and (right is not None and right != "")
                and left >= insert_pos
            ):
                row.update({"left_end_pos": left + insert_len})
                row.update({"right_end_pos": right + insert_len})

    def _update_gene_locations(self, nested_attr, insert_pos):
        """
        Modify positions of original genes based upon the insertion location
        of new genes. Returns end position of the gene insertion.
        """

        genes_data = getattr(self, "genes")
        tu_data = getattr(self, "transcription_units")
        dna_sites_data = getattr(self, "dna_sites")

        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        new_genes_data = getattr(nested_data, "genes")
        new_genes_data = sorted(new_genes_data, key=lambda d: d["left_end_pos"])

        for i in range(len(new_genes_data) - 1):
            assert (
                new_genes_data[i + 1]["left_end_pos"]
                == new_genes_data[i]["right_end_pos"] + 1
            ), "gaps in new gene insertions are not supported at this time"

        insert_end = new_genes_data[-1]["right_end_pos"] + insert_pos
        insert_len = insert_end - insert_pos

        # Update global positions of original genes
        self._update_global_coordinates(genes_data, insert_pos, insert_len)

        # Update global positions of transcription units
        if tu_data:
            self._update_global_coordinates(tu_data, insert_pos, insert_len)

        # Update DNA site positions
        # (including the origin and terminus of replication)
        self._update_global_coordinates(dna_sites_data, insert_pos, insert_len)

        # Change relative insertion positions to global in reference genome
        for row in new_genes_data:
            left = row["left_end_pos"]
            right = row["right_end_pos"]
            row.update({"left_end_pos": left + insert_pos})
            row.update({"right_end_pos": right + insert_pos})

        # If the new genes are in operons, change relative positions to global.
        # ⚠ ORDER IS LOAD-BEARING: the loop above has already converted
        # ``new_genes_data`` to genome coordinates, and the upper-bound
        # assertion below compares against it, so both sides are global. Moving
        # this block above that loop would compare a global position against a
        # relative one and pass or fail for the wrong reason.
        new_genes_tu_data = getattr(nested_data, "transcription_units", None)
        if new_genes_tu_data:
            new_genes_tu_data = sorted(
                new_genes_tu_data, key=lambda d: d["left_end_pos"]
            )
            for row in new_genes_tu_data:
                left = row["left_end_pos"]
                right = row["right_end_pos"]

                # An added transcription unit must lie entirely within the
                # added genes: it may not reach back into the original genome
                # on either side. Checked rather than clamped, because a TU
                # that silently spanned the insertion boundary would transcribe
                # native genes as part of the construct.
                #
                # Bounds are compared in the insertion's own RELATIVE frame
                # against ``insert_len`` -- deliberately, so this block does not
                # depend on whether the gene rows above have been converted yet.
                # Comparing a relative bound against an already-converted gene
                # row would make the check order-sensitive, and an order-
                # sensitive assertion is one that passes for the wrong reason
                # the first time somebody moves it.
                assert left >= 1, (
                    "added transcription unit start positions cannot overlap "
                    "original genes at this time"
                )
                assert right <= insert_len, (
                    "added transcription unit end positions cannot exceed new "
                    "gene end position at this time"
                )
                # A reversed span passes both bounds and emerges as a TU whose
                # end precedes its start. Never present in the base data;
                # guarded because this block exists to catch payload-authoring
                # mistakes, and this is one.
                assert left <= right, (
                    "added transcription unit start position "
                    f"({left}) must not follow its end position ({right})"
                )

                row.update({"left_end_pos": left + insert_pos})
                row.update({"right_end_pos": right + insert_pos})

        return insert_end

    def _get_new_gene_sequence(self, nested_attr):
        """
        Determine genome sequnce for insertion using the sequences and
        relative locations of the new genes.
        """
        from Bio import Seq

        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        new_genes_data = getattr(nested_data, "genes")
        seq_data = getattr(nested_data, "gene_sequences")

        insertion_seq = Seq.Seq("")
        new_genes_data = sorted(new_genes_data, key=lambda d: d["left_end_pos"])
        assert new_genes_data[0]["left_end_pos"] == 1, (
            "first gene in new sequence must start at relative coordinate 1"
        )

        for gene in new_genes_data:
            if gene["direction"] == "+":
                seq_row = next(
                    (row for row in seq_data if row["id"] == gene["id"]), None
                )
                seq_string = seq_row["gene_seq"]
                seq_addition = Seq.Seq(seq_string)
                insertion_seq += seq_addition
            else:
                seq_row = next(
                    (row for row in seq_data if row["id"] == gene["id"]), None
                )
                seq_string = seq_row["gene_seq"]
                seq_addition = Seq.Seq(seq_string)
                insertion_seq += seq_addition.reverse_complement()

            assert len(seq_addition) == (
                gene["right_end_pos"] - gene["left_end_pos"] + 1
            ), (
                "left and right end positions must agree with actual "
                "sequence length for " + gene["id"]
            )

        return insertion_seq
