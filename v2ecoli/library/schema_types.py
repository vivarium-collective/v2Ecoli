"""
=====================
E. coli Schema Types
=====================

Bigraph-schema type expressions for the E. coli whole-cell model's
structured numpy arrays and common port types.

These type strings can be used directly in ``inputs()`` and ``outputs()``
methods, and are parsed by bigraph-schema into proper ``Array`` schemas
with structured numpy dtypes.
"""

# ---------------------------------------------------------------------------
# Mass submass fields shared across all unique molecule types
# ---------------------------------------------------------------------------
_SUBMASS_FIELDS = (
    'massDiff_rRNA:float|massDiff_tRNA:float|massDiff_mRNA:float|'
    'massDiff_miscRNA:float|massDiff_nonspecific_RNA:float|massDiff_protein:float|'
    'massDiff_metabolite:float|massDiff_water:float|massDiff_DNA:float'
)

_UNIQUE_TAIL = f'{_SUBMASS_FIELDS}|_entryState:integer|unique_index:integer'

# ---------------------------------------------------------------------------
# Bulk molecule array
# ---------------------------------------------------------------------------
BULK_ARRAY = (
    'array[id:string|count:integer|'
    'rRNA_submass:float|tRNA_submass:float|mRNA_submass:float|'
    'miscRNA_submass:float|nonspecific_RNA_submass:float|protein_submass:float|'
    'metabolite_submass:float|water_submass:float|DNA_submass:float]'
)

# ---------------------------------------------------------------------------
# Unique molecule arrays
# ---------------------------------------------------------------------------
# init_prob_override (flagella-cascade investigation): per-promoter transcription
# initiation probability written by ecoli-flagella-transcription-regulation (the
# Kalir & Alon SUM-gate). Defaults to 0.0; transcript_initiation only substitutes
# it where > 0, so it is a no-op for every promoter when the flagella feature is off.
_PROMOTER_STRUCT = f'unique_array[TU_index:integer|coordinates:integer|domain_index:integer|bound_TF:array[23,boolean]|init_prob_override:float|{_UNIQUE_TAIL}]'

_RNA_STRUCT = f'unique_array[TU_index:integer|transcript_length:integer|is_mRNA:boolean|is_full_transcript:boolean|can_translate:boolean|RNAP_index:integer|{_UNIQUE_TAIL}]'

_ACTIVE_RNAP_STRUCT = f'unique_array[domain_index:integer|coordinates:integer|is_forward:boolean|{_UNIQUE_TAIL}]'

_ACTIVE_RIBOSOME_STRUCT = f'unique_array[protein_index:integer|peptide_length:integer|mRNA_index:integer|pos_on_mRNA:integer|{_UNIQUE_TAIL}]'

_ACTIVE_REPLISOME_STRUCT = f'unique_array[domain_index:integer|right_replichore:boolean|coordinates:integer|{_UNIQUE_TAIL}]'

_FULL_CHROMOSOME_STRUCT = f'unique_array[division_time:float|has_triggered_division:boolean|domain_index:integer|{_UNIQUE_TAIL}]'

_ORIC_STRUCT = f'unique_array[domain_index:integer|{_UNIQUE_TAIL}]'

_CHROMOSOME_DOMAIN_STRUCT = f'unique_array[domain_index:integer|child_domains:array[2,integer]|{_UNIQUE_TAIL}]'

_CHROMOSOMAL_SEGMENT_STRUCT = f'unique_array[boundary_molecule_indexes:array[2,integer]|boundary_coordinates:array[2,integer]|domain_index:integer|linking_number:float|{_UNIQUE_TAIL}]'

_GENE_STRUCT = f'unique_array[cistron_index:integer|coordinates:integer|domain_index:integer|{_UNIQUE_TAIL}]'

_DNAA_BOX_STRUCT = f'unique_array[coordinates:integer|domain_index:integer|DnaA_bound:boolean|pool_label:integer|DnaA_bound_form:integer|{_UNIQUE_TAIL}]'

# flagella-cascade investigation (2026-08-06): a growing flagellar filament,
# tracked per-instance like active_ribosome's peptide_length, rather than as
# a single bulk complexation-reaction coefficient (a real ~20,000-subunit
# FliC stoichiometry makes Gillespie SSA's combinatorial propensity
# calculation blow up -- see flagella_filament_elongation.py). filament_length
# is the number of FliC subunits incorporated so far.
_NASCENT_FLAGELLUM_STRUCT = f'unique_array[filament_length:integer|{_UNIQUE_TAIL}]'

# ---------------------------------------------------------------------------
# Biological named types
# ---------------------------------------------------------------------------
# The public *_ARRAY constants used by every process's inputs()/outputs() are
# biological NAMES; each name is registered (v2ecoli/types) to the structural
# expression above, so a port advertises e.g. 'rna' instead of the raw
# unique_array[17 fields]. Same resolved schema -> behaviour-preserving.
BIOLOGICAL_UNIQUE_TYPES = {
    'promoter': _PROMOTER_STRUCT,
    'rna': _RNA_STRUCT,
    'active_RNAP': _ACTIVE_RNAP_STRUCT,
    'active_ribosome': _ACTIVE_RIBOSOME_STRUCT,
    'active_replisome': _ACTIVE_REPLISOME_STRUCT,
    'full_chromosome': _FULL_CHROMOSOME_STRUCT,
    'oriC': _ORIC_STRUCT,
    'chromosome_domain': _CHROMOSOME_DOMAIN_STRUCT,
    'chromosomal_segment': _CHROMOSOMAL_SEGMENT_STRUCT,
    'gene': _GENE_STRUCT,
    'DnaA_box': _DNAA_BOX_STRUCT,
    'nascent_flagellum': _NASCENT_FLAGELLUM_STRUCT,
}
PROMOTER_ARRAY = 'promoter'
RNA_ARRAY = 'rna'
ACTIVE_RNAP_ARRAY = 'active_RNAP'
ACTIVE_RIBOSOME_ARRAY = 'active_ribosome'
ACTIVE_REPLISOME_ARRAY = 'active_replisome'
FULL_CHROMOSOME_ARRAY = 'full_chromosome'
ORIC_ARRAY = 'oriC'
CHROMOSOME_DOMAIN_ARRAY = 'chromosome_domain'
CHROMOSOMAL_SEGMENT_ARRAY = 'chromosomal_segment'
GENE_ARRAY = 'gene'
DNAA_BOX_ARRAY = 'DnaA_box'
NASCENT_FLAGELLUM_ARRAY = 'nascent_flagellum'

# ---------------------------------------------------------------------------
# Convenience mapping: port name → type expression
# ---------------------------------------------------------------------------
UNIQUE_TYPES = {
    'promoters': PROMOTER_ARRAY,
    'promoter': PROMOTER_ARRAY,
    'RNAs': RNA_ARRAY,
    'RNA': RNA_ARRAY,
    'active_RNAPs': ACTIVE_RNAP_ARRAY,
    'active_RNAP': ACTIVE_RNAP_ARRAY,
    'active_ribosome': ACTIVE_RIBOSOME_ARRAY,
    'active_ribosomes': ACTIVE_RIBOSOME_ARRAY,
    'active_replisomes': ACTIVE_REPLISOME_ARRAY,
    'active_replisome': ACTIVE_REPLISOME_ARRAY,
    'full_chromosomes': FULL_CHROMOSOME_ARRAY,
    'full_chromosome': FULL_CHROMOSOME_ARRAY,
    'oriCs': ORIC_ARRAY,
    'oriC': ORIC_ARRAY,
    'chromosome_domains': CHROMOSOME_DOMAIN_ARRAY,
    'chromosome_domain': CHROMOSOME_DOMAIN_ARRAY,
    'chromosomal_segments': CHROMOSOMAL_SEGMENT_ARRAY,
    'chromosomal_segment': CHROMOSOMAL_SEGMENT_ARRAY,
    'genes': GENE_ARRAY,
    'gene': GENE_ARRAY,
    'DnaA_boxes': DNAA_BOX_ARRAY,
    'DnaA_box': DNAA_BOX_ARRAY,
    'nascent_flagella': NASCENT_FLAGELLUM_ARRAY,
    'nascent_flagellum': NASCENT_FLAGELLUM_ARRAY,
}
