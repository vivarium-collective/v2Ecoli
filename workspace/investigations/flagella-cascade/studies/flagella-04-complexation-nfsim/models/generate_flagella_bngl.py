"""
Generate a BioNetGen model for E. coli flagella complexation.

Reads the complexation stoichiometry (see below) and generates a BNGL file
with sequential bimolecular binding rules for each assembly step.

Each multi-subunit complex is modeled as a scaffold molecule with counter
states that track how many of each subunit have been incorporated. Monomers
bind one at a time via bimolecular reactions. When all subunits are bound,
a final rule converts the scaffold into the completed complex.
"""
import os

# ---------------------------------------------------------------------------
# Stoichiometry -- UPDATED 2026-08-12 to match v2ecoli's corrected reaction
# network (see v2ecoli's flagella-cascade investigation,
# workspace/investigations/flagella-cascade/studies/
# flagella-02-transcription-regulation/study.yaml, findings
# flagella-02-export-apparatus-flip-flop-stoichiometry-fixed,
# flagella-02-cring-export-apparatus-hierarchy-fix,
# flagella-02-ms-ring-ordering-fix). Original values here were sourced from
# an older, unrelated codebase (vivarium-chemotaxis flagella_chromosome.py)
# and had NOT been checked against real structural literature -- most
# multi-subunit coefficients were simple placeholders (mostly -1), and the
# assembly ORDER (which complex depends on which) didn't match real biology
# either. Both are fixed here. Negative values = consumed, positive =
# produced.
#
# STOICHIOMETRY FIXES (real cryo-EM literature, same citations as v2ecoli):
#   fliN: -1 -> -111 (was off by two orders of magnitude; PMC10128058,
#     "Precise Measurement of the Stoichiometry of the Adaptive Bacterial
#     Flagellar Switch")
#   fliG: -26 -> -34 (C-ring 34-fold rotational symmetry; two independent
#     literature searches this session both gave 34, not 26)
#   fliP:fliQ:fliR: -1:-1:-1 -> -5:-4:-1 (real cryo-EM-determined
#     stoichiometry; Kuhlen et al. 2018 Nat Struct Mol Biol, building on
#     Fukumura et al. 2017 PLOS Biol 15:e2002281/PMC5542437)
#   flhA: -1 -> -9 (homo-nonameric ring in the export gate; PLOS One
#     10.1371/journal.pone.0252800 + corroborating literature)
#   flgH, flgI (L-ring, P-ring): -1 -> -26 each (C26 symmetry, cryo-EM)
#   motA, motB (stator): -1 -> -55, -22 (derived estimate, medium
#     confidence -- 5:2 per-unit ratio is solid, x~11 stator units/motor is
#     the estimated part, flagged in v2ecoli's own docs too)
#   flgB, flgC, flgF (proximal rod): -1 -> -5, -6, -5
#   flgG (distal rod): -1 -> -24
#   fliE: -1 -> -6
#   flgK, flgL (hook-filament junction): -1 -> -11 each (cryo-EM)
#   fliF (MS-ring): was MISSING from the motor-switch reaction entirely (only
#     consumed, wrongly, at -1 in the final motor reaction) -> -34, moved to
#     the motor-switch reaction (see HIERARCHY FIX below)
#   fliC: REMOVED from this model entirely (2026-08-12) -- see FLIC REMOVAL
#     note below.
#
# HIERARCHY FIXES (real assembly order, Minamino & Namba 2008 Nature;
# Chevance & Hughes 2008 Nat Rev Microbiol -- MS-ring forms first, C-ring
# assembles onto it, THEN the export apparatus inserts into the C-ring's
# pore, THEN the export apparatus is incorporated into the motor complex):
#   'flagellar motor switch reaction' (C-ring) now also consumes fliF
#     (MS-ring) -- represents the merged MS-ring + C-ring structure, same
#     simplification v2ecoli uses (single-stage merge, not a third complex).
#   'flagellar export apparatus reaction 1' now ALSO consumes
#     'flagellar motor switch' (-1) -- the export apparatus can't form until
#     the C-ring already exists (was previously two independent branches
#     only merging at the motor reaction).
#   'flagellar motor reaction' no longer consumes 'flagellar motor switch'
#     or fliF directly (both moved upstream, see above) -- avoids double-
#     counting the C-ring/MS-ring for one motor complex.
#   'flagellar motor reaction' now consumes 'flagellar export apparatus'
#     (-1) instead of the final 'flagellum reaction' consuming it -- matches
#     v2ecoli's flagella_motor_complex_assembly.py, which consumes CPLX0-7451
#     to build FLAGELLAR-MOTOR-COMPLEX, not at filament completion.
#
# NOT changed here (logged as an open question in v2ecoli, not resolved):
#   fliO stays at -1, still a genuine consumed reactant in the export-
#   apparatus reaction, even though real literature describes it as a
#   transient assembly scaffold not necessarily part of the final mature
#   complex. Maya's explicit direction (2026-08-12): keep it at -1 for now;
#   this is exactly the kind of role-vs-copy-number question NFsim rules
#   should eventually constrain properly, not something to hand-fix here.
#
# REAL BULK MOLECULE IDS (2026-08-12, same day, later): species keys below
# renamed from generic placeholder names (fliF, flgH, ...) to the ACTUAL
# v2ecoli WCM bulk molecule IDs (FLIF-FLAGELLAR-MS-RING[i],
# FLGH-FLAGELLAR-L-RING[j], ...) -- part of NFSIM_WCM_WIRING_PLAN.md step 1,
# so this model can eventually read/write the SAME shared bulk pool the real
# WCM Steps use, with no translation layer needed. Every ID cross-checked
# directly against the real, currently-running Step files
# (flagella_motor_switch_assembly.py, flagella_export_apparatus_assembly.py,
# flagella_motor_complex_assembly.py, flagella_filament_nucleation.py,
# flagella_filament_elongation.py) and/or reconstruction/ecoli/flat/*.tsv,
# not guessed from cistron IDs -- caught one real mismatch this way: FlhC's
# actual bulk protein ID is MONOMER0-2488[c], NOT EG10319-MONOMER[c] (EG10319
# is the gene/cistron ID referenced in flagella_transcription_regulation.py's
# docstring, a different EcoCyc accession from the protein monomer itself;
# confirmed via CPLX0-3930_RXN's real definition in complexation_reactions.tsv:
# {"CPLX0-3930":1, "EG10320-MONOMER":-4, "MONOMER0-2488":-2}).
#
# REAL BUG FOUND IN THE ACTUAL WCM CODE (not an NFsim issue, flagged here
# since this cross-referencing work is what surfaced it): the canonical
# reaction spec (complexation_reactions_modified.tsv,
# "FLAGELLAR-MOTOR-COMPLEX_RXN") includes FLGI-FLAGELLAR-P-RING at -26,
# matching flagella_motor_complex_assembly.py's OWN docstring ("FlgH/FlgI=26
# (L-ring/P-ring)") -- but that Step's actual _REQUIREMENTS dict only
# consumes FlgH, never FlgI. The real bulk ID exists and resolves fine
# (FLGI-FLAGELLAR-P-RING[j], confirmed in reconstruction/ecoli/flat/
# proteins.tsv) -- this looks like a real, if minor, omission in the
# currently-running Step, not a deliberate simplification (nothing in that
# file's docstring explains dropping it). Still NOT fixed in the real WCM
# Step -- out of scope for this investigation to fix there.
#
# REVISED 2026-08-17: this NFsim model no longer includes flgI either
# (reverted from the earlier decision to include it per the canonical
# spec). NFsim's model is meant to eventually REPLACE
# flagella_motor_complex_assembly.py -- keeping NFsim consistent with what
# that Step ACTUALLY does today (excludes FlgI) avoids a real mass-balance
# mismatch between the two pipelines while this is unresolved, rather than
# having NFsim match an aspirational spec nothing currently implements. If
# the real Step's FlgI omission is ever fixed, this should be revisited and
# re-added here too.
#
# THREE SPECIES KEPT AS DESCRIPTIVE NAMES, NOT RENAMED -- no real WCM bulk
# molecule corresponds to them:
#   'flagellar export apparatus subunit' -- purely this generator's own
#     internal bookkeeping (splits the 9-reactant export-apparatus assembly
#     into two sub-reactions; the real WCM Step does this in one shot).
#   'flagellar hook' -- the real WCM has NO discrete "hook complete" bulk
#     molecule at all; flagella_filament_nucleation.py merges hook
#     completion and nucleation into a single Step/event (consumes motor
#     complex + 120x FlgE + 11x FlgK + 11x FlgL directly, creating a
#     nascent_flagellum UNIQUE molecule, not a bulk one).
#   'flagella' (final observable) -- corresponds to creating a
#     nascent_flagellum UNIQUE molecule in the real pipeline (via
#     flagella_filament_nucleation.py today), not a bulk-store molecule --
#     bridging this is the wrapper Step's job (NFSIM_WCM_WIRING_PLAN.md
#     step 3), not a rename.
#
# FLIC REMOVAL (2026-08-12): fliC was briefly set to -5000 (matching
# v2ecoli's current target_length) and immediately caused the SAME
# combinatorial/file-size explosion v2ecoli itself hit and already solved --
# this generator represents "N copies bound" as N explicit numbered scaffold
# states, so a single coefficient of 5000 expanded into 5000 individual BNGL
# rules (237 -> 5,588 total rules, 486 KB file) for FliC alone. v2ecoli's own
# fix for the identical problem was to pull filament elongation OUT of the
# combinatorial engine entirely (Gillespie SSA there, this rule-matching
# network here) and run it as a separate incremental process
# (flagella_filament_elongation.py: dL/dt = a/(b+L), Renault et al. 2017,
# adding FliC one subunit at a time outside the reaction network). Rather
# than reimplementing that incremental logic a second time inside this
# standalone BNGL/NFsim example -- untested against the real WCM, and
# redundant with code that already works -- fliC and filament growth are
# EXCLUDED from this model entirely. The 'flagellum reaction' below now
# represents assembly complete through the HOOK-BASAL-BODY stage only (motor
# + hook + hook-filament junction + cap machinery) -- the real endpoint of
# what rule-based complexation was ever meant to validate here. Filament
# elongation (FliC incorporation) is explicitly out of scope for this
# standalone example; when this couples to the real WCM (task: assess NFsim
# coupling vs. this session's custom v2ecoli assembly Steps), v2ecoli's
# existing, working flagella_filament_elongation.py should be reused
# directly rather than reimplemented in BNGL. The final species is still
# named 'flagella' for backward compatibility with run_nfsim_assembly.py's
# existing observable name -- see that script's own updated comment.
# ---------------------------------------------------------------------------
# Old placeholder-name version kept per standing preserve-old-code rule:
# COMPLEXATION_STOICHIOMETRY = {
#     'flhDC': {
#         'flhD': -4.0,
#         'flhC': -2.0,
#         'flhDC': 1.0,
#     },
#     'flagellar motor switch reaction': {
#         'flagellar motor switch': 1.0,
#         'fliF': -34.0, 'fliG': -34.0, 'fliM': -34.0, 'fliN': -111.0,
#     },
#     'flagellar export apparatus reaction 1': {
#         'flagellar export apparatus subunit': 1.0,
#         'flagellar motor switch': -1.0,
#         'flhA': -9.0, 'flhB': -1.0, 'fliO': -1.0, 'fliP': -5.0,
#         'fliQ': -4.0, 'fliR': -1.0, 'fliJ': -1.0, 'fliI': -6.0,
#     },
#     'flagellar export apparatus reaction 2': {
#         'flagellar export apparatus': 1.0,
#         'flagellar export apparatus subunit': -1.0,
#         'fliH': -12.0,
#     },
#     'flagellar motor reaction': {
#         'flagellar motor': 1.0,
#         'flagellar export apparatus': -1.0,
#         'fliL': -2.0, 'flgH': -26.0, 'motA': -55.0, 'motB': -22.0,
#         'flgB': -5.0, 'flgC': -6.0, 'flgF': -5.0, 'flgG': -24.0,
#         'flgI': -26.0, 'fliE': -6.0,
#     },
#     'flagellar hook reaction': {
#         'flagellar hook': 1,
#         'flgE': -120.0,
#     },
#     'flagellum reaction': {
#         'flagella': 1.0,
#         'flagellar motor': -1.0,
#         'flgL': -11.0, 'flgK': -11.0, 'fliD': -5.0, 'flagellar hook': -1,
#     },
# }
COMPLEXATION_STOICHIOMETRY = {
    'flhDC': {
        'EG10320-MONOMER[c]': -4.0,   # was flhD
        'MONOMER0-2488[c]': -2.0,     # was flhC -- NOT EG10319-MONOMER, see note above
        'CPLX0-3930[c]': 1.0,         # was flhDC
    },
    'flagellar motor switch reaction': {
        'CPLX0-7450[i]': 1.0,                    # was 'flagellar motor switch'
        'FLIF-FLAGELLAR-MS-RING[i]': -34.0,       # was fliF
        'FLIG-FLAGELLAR-SWITCH-PROTEIN[i]': -34.0,  # was fliG
        'FLIM-FLAGELLAR-C-RING-SWITCH[i]': -34.0,   # was fliM
        'FLIN-FLAGELLAR-C-RING-SWITCH[m]': -111.0,  # was fliN
    },
    'flagellar export apparatus reaction 1': {
        'flagellar export apparatus subunit': 1.0,  # no real bulk ID -- generator-internal, see note above
        'CPLX0-7450[i]': -1.0,        # was 'flagellar motor switch'
        'G370-MONOMER[i]': -9.0,      # was flhA
        'G7028-MONOMER[i]': -1.0,     # was flhB
        'EG11224-MONOMER[j]': -1.0,   # was fliO
        'EG11975-MONOMER[i]': -5.0,   # was fliP
        'EG11976-MONOMER[j]': -4.0,   # was fliQ
        'EG11977-MONOMER[i]': -1.0,   # was fliR
        'G378-MONOMER[c]': -1.0,      # was fliJ
        'G377-MONOMER[c]': -6.0,      # was fliI
    },
    'flagellar export apparatus reaction 2': {
        'CPLX0-7451[j]': 1.0,                     # was 'flagellar export apparatus'
        'flagellar export apparatus subunit': -1.0,  # no real bulk ID -- generator-internal
        'EG11656-MONOMER[c]': -12.0,  # was fliH
    },
    # OLD single-step version (pre-2026-08-27), kept per standing
    # preserve-old-code rule. This step built the rod, the L-ring, and the
    # stator (MotA/MotB) all at once, straight off the export apparatus, and
    # skipped the P-ring entirely. Real biology (see module docstring's
    # 2026-08-27 "HOOK DEPENDENCY FIX" note) says: (1) the rod finishes
    # first, (2) the P-ring (FlgI) forms next, (3) the L-ring (FlgH) forms
    # last and is what actually opens the door for the hook, and (4) the
    # stator is NOT required to finish the structure at all -- a cell can
    # build a complete, non-motile flagellum with no MotA/MotB. Bundling the
    # stator into this step wrongly made structural completion depend on
    # genes that real biology says are only needed for movement, not
    # building.
    # 'flagellar motor reaction': {
    #     'FLAGELLAR-MOTOR-COMPLEX[j]': 1.0,   # was 'flagellar motor'
    #     'CPLX0-7451[j]': -1.0,               # was 'flagellar export apparatus'
    #     'EG10322-MONOMER[j]': -2.0,          # was fliL
    #     'FLGH-FLAGELLAR-L-RING[j]': -26.0,   # was flgH
    #     'MOTA-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]': -55.0,  # was motA
    #     'MOTB-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]': -22.0,  # was motB
    #     'FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -5.0,   # was flgB
    #     'FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -6.0,   # was flgC
    #     'FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -5.0,   # was flgF
    #     'FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]': -24.0,  # was flgG
    #     # 'FLGI-FLAGELLAR-P-RING[j]': -26.0,  # REMOVED 2026-08-17 -- matches
    #     # flagella_motor_complex_assembly.py's actual (real-bug) omission,
    #     # see "REAL BUG FOUND" / "REVISED 2026-08-17" notes above
    #     'EG11346-MONOMER[p]': -6.0,          # was fliE
    # },

    # NEW 2026-08-27, three real, ordered steps replacing the one above.
    # HOOK DEPENDENCY FIX -- see module docstring. Real order: rod, then
    # P-ring, then L-ring (Cohen & Hughes 2014, J Bacteriol 196:2387 for the
    # L-ring step specifically triggering rod-to-hook transition; FlgA/FlgI
    # P-ring literature for P-ring forming before L-ring). Stator (MotA,
    # MotB) removed entirely from this chain -- real biology says it is not
    # required for structural completion, only for movement, so it is left
    # untouched as free bulk pool, tracked separately if ever needed.
    'flagellar rod reaction': {
        'flagellar rod': 1.0,   # no real bulk ID -- new intermediate
        'CPLX0-7451[j]': -1.0,               # was 'flagellar export apparatus'
        'EG10322-MONOMER[j]': -2.0,          # was fliL
        'FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -5.0,   # was flgB
        'FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -6.0,   # was flgC
        'FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -5.0,   # was flgF
        'FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]': -24.0,  # was flgG
        'EG11346-MONOMER[p]': -6.0,          # was fliE
    },
    'flagellar p-ring reaction': {
        'flagellar rod with p-ring': 1.0,   # no real bulk ID -- new intermediate
        'flagellar rod': -1.0,
        'FLGI-FLAGELLAR-P-RING[j]': -26.0,   # was flgI, 26 copies (real,
        # confirmed via literature search 2026-08-27 -- same 26-fold
        # symmetry as the L-ring). Previously omitted entirely (see removed
        # line above); now included as a real, separate, ordered step.
    },
    'flagellar l-ring reaction': {
        'FLAGELLAR-MOTOR-COMPLEX[j]': 1.0,   # real bulk ID -- now represents
        # the finished rod+P-ring+L-ring base, ready for the hook (NOT the
        # stator -- see note above).
        'flagellar rod with p-ring': -1.0,
        'FLGH-FLAGELLAR-L-RING[j]': -26.0,   # was flgH
    },
    'flagellar hook reaction': {
        'flagellar hook': 1,           # no real bulk ID -- see note above
        # NEW 2026-08-27: hook now requires the finished base (see module
        # docstring's "HOOK DEPENDENCY FIX"), not just free FlgE alone.
        'FLAGELLAR-MOTOR-COMPLEX[j]': -1.0,
        'G361-MONOMER[c]': -120.0,     # was flgE
    },
    'flagellum reaction': {
        'flagella': 1.0,   # no real bulk ID (maps to nascent_flagellum creation) -- see note above
        # OLD (pre-2026-08-27): this step, not the hook step, used to consume
        # FLAGELLAR-MOTOR-COMPLEX[j] directly. Moved to the hook step above
        # so hook -> flagellum is one continuous, non-conflicting chain
        # (each item consumed exactly once by the next real step), kept per
        # standing preserve-old-code rule:
        # 'FLAGELLAR-MOTOR-COMPLEX[j]': -1.0,  # was 'flagellar motor'
        'EG11545-MONOMER[e]': -11.0,   # was flgL
        'EG11967-MONOMER[e]': -11.0,   # was flgK
        # 'EG10841-MONOMER[e]': -5.0,    # was fliD -- REMOVED 2026-08-21, see
        # "FLID DOUBLE-CONSUMPTION" note below. Kept per standing
        # preserve-old-code rule.
        'flagellar hook': -1,          # no real bulk ID -- see note above,
        # now carries the whole rod+P-ring+L-ring+hook history forward.
    },
}

# FLID DOUBLE-CONSUMPTION FIX (2026-08-21): the 'flagellum reaction' above
# used to also consume 5x FliD (EG10841-MONOMER[e]) at this point (hook-
# basal-body + cap + hook-filament-junction complete, immediately BEFORE
# filament elongation begins). But v2ecoli's flagella_filament_elongation.py
# -- the unchanged, shared Step this model's docstring already says should
# own filament completion (see "'flagellum reaction' below now represents
# assembly complete through the HOOK-BASAL-BODY stage" above) -- ALSO
# consumes 5x FliD (fliD_per_completion=5) at its own completion event
# (filament reaching target_length), deliberately matching CPLX0-7452_RXN's
# real coefficient for mass conservation. That elongation-side consumption
# is the ORIGINAL, validated design shared by BOTH the old deterministic
# pipeline and this NFsim pipeline -- it's what still runs FliD accounting
# for the currently-default flagella_regulation feature, so it can't be
# removed there without breaking that pipeline's only FliD consumption
# point. This model's own inclusion of FliD in 'flagellum reaction' was
# added independently (cross-referencing real cryo-EM/structural
# stoichiometry for the final assembled structure) without checking that
# the shared downstream Step already accounts for it -- a real double-count
# introduced specifically by enabling NFsim (5 FliD consumed twice per
# completed flagellum instead of once).
#
# Real biology confirms only ONE consumption event should exist, and it
# happens BEFORE elongation, not after: FliD assembles into a pentameric
# cap at the hook-filament junction before flagellin polymerization begins,
# and stays at the growing tip for the entire elongation process -- "once
# the FliD pentamer is completed, it cannot accept another monomer" (Song
# et al. 2017, J Mol Biol 429:847, "Self-Oligomerizing Structure of the
# Flagellar Cap Protein FliD and Its Implication in Filament Assembly";
# corroborated by the cryo-EM mechanism paper, Postel et al. 2020, Nat
# Commun 11:1965). So NFsim's OWN timing (at hook-basal-body-cap
# completion, before elongation) is actually the biologically correct
# point -- it's flagella_filament_elongation.py's completion-time charge
# that's mistimed (a pre-NFsim bookkeeping simplification, from when there
# was no earlier "cap complete" event to hang the cost on). The elongation
# Step is intentionally left as-is here (fixing timing there would need
# re-validating the old deterministic pipeline's own mass conservation,
# out of scope for this NFsim-side fix) -- removing FliD from THIS
# reaction is the surgical fix: it stops the double-count for the NFsim
# pipeline specifically, without touching the shared Step the old pipeline
# still depends on.

# ---------------------------------------------------------------------------
# Rate constants -- UPDATED 2026-08-12. K_BIND was previously an arbitrary
# placeholder (5e-1, no citation). Replaced with a real, literature-measured
# association rate constant, converted to the per-molecule stochastic units
# BNGL/NFsim expects.
#
# FlhA<->FlhB binding (flagellar export apparatus): k_on = 8.5e4 M^-1 s^-1,
# k_off = 0.09 s^-1, biosensor/SPR measurement, Salmonella enterica.
# McMurry, Sampson, Case & Hughes (2015), "Weak Interactions between
# Salmonella enterica FlhB and Other Flagellar Export Apparatus Proteins
# Govern Type III Secretion Dynamics," PLOS One,
# doi:10.1371/journal.pone.0134884 (PMC4526367).
#
# Deterministic (M^-1 s^-1) -> stochastic (molecules^-1 s^-1) conversion:
#   k_stoch = k_on / (N_A * V)
# using V = 1 femtoliter (1e-15 L), the standard assumed E. coli cell volume
# in stochastic bacterial modeling (v2ecoli itself derives volume dynamically
# from dry_mass rather than a fixed constant, so no repo-internal value to
# match here -- this is the field's usual convention instead).
#
# This SAME converted value is also used as the default k_bind for every
# OTHER binding reaction in this model that lacks its own measured rate
# constant (the C-ring: FliG/FliM/FliN/FliF -- searched, only structural
# characterization exists in the literature, no kinetics; and the rest of
# the export apparatus and motor-complex components). This is a real,
# same-organelle, same-protein-family adjacent proxy -- not a generic guess
# -- since it's measured from a direct physical neighbor within this exact
# assembly (FlhA/FlhB are both export-apparatus subunits), rather than an
# unrelated system's "typical" protein-protein rate. Flagged explicitly:
# this is a proxy, not a per-interaction measurement, for every reaction
# except the one it was actually measured for.
N_AVOGADRO = 6.022e23         # /mol
CELL_VOLUME_L = 1e-15         # 1 fL, standard assumed E. coli cell volume
K_ON_FLHA_FLHB_MOLAR = 8.5e4  # M^-1 s^-1, McMurry et al. 2015 (real, cited)
K_OFF_FLHA_FLHB = 0.09        # s^-1, McMurry et al. 2015 -- NOTE: not
# currently used anywhere; this generator only emits irreversible ("->")
# binding rules, so there is no reverse reaction to attach this to yet. Real
# value, kept here for whenever reversibility is added.
K_BIND = K_ON_FLHA_FLHB_MOLAR / (N_AVOGADRO * CELL_VOLUME_L)  # ~1.412e-4 /molecule/s
K_NUCLEATION = 5e-2   # unused now -- see NUCLEATION_SUPPRESSION_FACTOR below
K_COMPLETION = 10.0   # still an unconverted placeholder -- no literature search done for this rate

# NUCLEATION FIX (2026-08-12): the original per-reaction nucleation rate was
# computed from a "target_propensity" formula (see generate_bngl below,
# original: nuc_rate = (n_flagella/50) / combinatorial) that was INDEPENDENT
# of k_bind -- with the corrected, much larger real stoichiometry, this let
# new scaffolds nucleate far faster than any existing scaffold could finish
# (4,530 real reaction events fired over a 2400s test run, species diversity
# growing throughout, yet ZERO complexes ever completed -- confirmed by
# actually running the model, not assumed). Real E. coli flagellar assembly
# does the opposite: existing structures preferentially absorb material over
# nucleating new ones (Chang, Sung & Hong 2025, Biochem Biophys Reports
# 42:102051, "Intrinsic clustering of flagellar basal body proteins in E.
# coli" -- the SAME citation v2ecoli's own flagella_filament_nucleation.py
# uses for the identical principle at the whole-flagellum level).
#
# Fixed by tying nucleation rate to a small FRACTION of k_bind instead of an
# independent formula, so that once any scaffold already exists, adding to
# it (propensity = k_bind * scaffold_count * free_monomer) overwhelmingly
# outcompetes starting a new one (propensity = k_nuc * free_monA * free_monB)
# -- the qualitative mechanism real biology uses, not a specific literature
# rate (no such per-sub-assembly nucleation rate was found in literature;
# this is a design-level ratio choice, analogous in spirit to v2ecoli's own
# ~1000x-scale gap between its measured nucleation_rate=0.00167/s and its
# elongation rate, not a directly measured value).
# Old value kept per standing preserve-old-code rule:
# NUCLEATION_SUPPRESSION_FACTOR = 1000.0
#
# RESCALED 2026-08-12 (NFSIM_WCM_WIRING_PLAN.md step 2): 1000 was calibrated
# against the standalone demo's ARTIFICIAL seed counts (n_flagella=5 scaled,
# e.g. FliF=170). Once seeded from REAL WCM ambient bulk counts instead
# (diagnostic_real_bulk_seeding.py), that factor was nowhere near enough:
# nucleation propensity is proportional to monomer_count^2 (homodimer
# nucleation for most reactions here), and real ambient counts run
# meaningfully higher than the demo's -- FliF real=657 vs demo=170 (3.86x
# count -> ~14.9x propensity), FlgE real=3508 vs demo=600 (5.85x count ->
# ~34.2x propensity, the worst case that was actually observed blocking
# hook progress: confirmed directly, ~67-70 distinct partial scaffolds
# nucleated in parallel and NONE completed across 8 simulated hours, free
# FliF driven from 657 down to a floor of 14 split across ~70 competitors).
# Rescaled by ~35x (to 35,000) to cover that worst observed case with some
# margin. This is a single GLOBAL constant applied uniformly to every
# reaction's nucleation rate -- a real, accepted limitation flagged
# explicitly (chosen deliberately over reworking the rate law to scale
# per-reaction with each reaction's own ambient monomer count, a bigger
# change deferred for now): different reactions have different real:demo
# ratios (the flhDC reaction's nucleating species, MONOMER0-2488/FlhC, real
# ratio is ~65x count / ~4200x propensity -- FAR more suppressed than
# FliF's or FlgE's case, but this doesn't block downstream assembly since
# nothing else in this model consumes CPLX0-3930/FlhDC). Revisit if a
# future reaction's nucleating species has a real:demo ratio outside the
# FliF-FlgE range this was tuned against.
NUCLEATION_SUPPRESSION_FACTOR = 35000.0

# REAL NUCLEATION RATE (2026-08-17, NFSIM_WCM_WIRING_PLAN.md follow-up):
# NUCLEATION_SUPPRESSION_FACTOR above was a reactively-tuned global constant
# (chosen only to stop the "many parallel scaffolds, none finish" failure
# mode at real WCM scale -- see the RESCALED comment above), never derived
# from an actual literature rate. Replaced here, for the C-ring/MS-ring
# nucleation reaction specifically ('flagellar motor switch reaction' --
# the real first physical step of starting a whole new flagellum), with a
# properly derived rate:
#
# Sim et al. 2017 (Sci Rep 7:41189, "Growth rate control of flagellar
# assembly in E. coli strain RP437") measured, at fast growth (mu=0.6/hr,
# 1.2hr doubling time), a mean of 7.8 flagella/cell (n=1,099 cells,
# chemostat steady state). Back-calculating a per-cell nucleation rate:
# 7.8 flagella / (1.2*3600 s) ~= 1.81e-3/s -- matching (to within 8%) the
# SAME rate v2ecoli's OWN flagella_filament_nucleation.py already derives
# and uses from the SAME paper (0.00167/s, that Step's own citation).
# Using the already-established 0.00167/s value here for consistency
# across both pipelines rather than introducing a second, slightly
# different number from the same source.
#
# UNIT CONVERSION CAVEAT, stated plainly: 0.00167/s is a per-cell EVENT
# rate (matching flagella_filament_nucleation.py's own fixed-interval,
# concentration-INDEPENDENT trigger design -- fires once every ~600s
# regardless of monomer supply). NFsim's nucleation is a bimolecular
# mass-action reaction instead (FliF()+FliF()->Growing_CPLX0_7450_i(...),
# propensity = k_nuc * [FliF]^2), which is NOT concentration-independent by
# construction -- an inherent mismatch between "true Poisson process" (real
# biology, and the custom Step's design) and "mass-action kinetics" (what
# BNGL/NFsim's generator produces). Resolved by calibrating k_nuc so the
# propensity equals the real 0.00167/s rate AT THE REAL AMBIENT FliF
# concentration this investigation has used throughout (657, from
# diagnostic_real_bulk_seeding.py's live-composite read):
#   k_nuc_cring = 0.00167 / (657 * 656 / 2) ~= 7.75e-9 /molecule/s
# Flagged explicitly: this matches the real rate only AT that reference
# concentration -- as real FliF rises or falls (e.g. across growth
# conditions, or over time as Class II expression changes), the mass-action
# propensity will scale with [FliF]^2, not stay pinned to 0.00167/s the way
# the true concentration-independent process would. A real, accepted
# limitation of representing this specific kind of biology in NFsim's
# reaction-rate framework, not something resolved by a bigger/smaller
# constant.
REAL_NUCLEATION_RATE_PER_S = 0.00167  # Sim et al. 2017, same citation as
                                        # flagella_filament_nucleation.py

# GENERALIZED 2026-08-17 (real bug found and fixed same day): initially
# applied K_NUC_CRING only to the C-ring reaction, leaving every OTHER
# nucleation reaction at plain k_bind on the reasoning "downstream stages
# should be fast once a flagellum has committed." That reasoning is WRONG
# for hook specifically -- confirmed directly, real WCM run: 1,226 parallel
# Growing_flagellar_hook scaffolds nucleated in a single 1200s chunk, all
# stuck at 2-6 of 120 needed subunits, reproducing the exact "many parallel
# scaffolds, none finish" failure mode already fixed once for C-ring. Root
# cause: hook nucleates from two free FlgE monomers directly -- a real
# ambient pool of ~3,500 -- the SAME KIND of event as C-ring (starting a
# brand-new scaffold from an abundant free-monomer pool), not a
# downstream/precursor-gated step. flhDC has the identical structural risk
# (nucleates from free FlhC, real ambient ~649) even though it doesn't
# block the rest of the network (nothing else consumes CPLX0-3930).
#
# Real ambient counts below are this investigation's own
# diagnostic_real_bulk_seeding.py live-composite reading (2026-08-12/17).
# Any nucleation reaction whose nucleating species has a REAL bulk-ID
# ambient count gets the SAME calibrated-real-rate treatment as C-ring
# (rate = REAL_NUCLEATION_RATE_PER_S / (N*(N-1)/2) at that species' own
# real count) -- NOT a single shared constant, since different monomers'
# real counts differ by ~5x (FliF 657 vs FlgE 3508), and using one
# reference concentration for all of them was exactly the earlier mistake.
# Reactions nucleating from a SCARCE, dynamically-produced intermediate
# (export apparatus subunit from CPLX0-7450, motor reaction from
# CPLX0-7451) correctly keep plain k_bind -- those are naturally
# self-limiting by their own low, slowly-built-up availability, and don't
# need an artificial rate limit at all (matches the custom Steps pipeline's
# own architecture: only whole-flagellum nucleation is rate-limited,
# everything already gated by a scarce precursor is not).
REAL_AMBIENT_MONOMER_COUNTS = {
    'FLIF-FLAGELLAR-MS-RING[i]': 657,   # C-ring nucleating species
    'G361-MONOMER[c]': 3508,            # FlgE, hook nucleating species
    'MONOMER0-2488[c]': 649,            # FlhC, flhDC nucleating species
}

# NOT CONDITION-AWARE -- FLAGGED FOR REVISIT (2026-08-21, Maya's explicit
# call after tracing a real acetate-condition test). These three counts are
# a single snapshot from the BASAL condition (diagnostic_real_bulk_seeding.py,
# 2026-08-12/17) -- they are NOT the cell's actual current ambient count
# under any other growth condition. Since nucleation propensity scales with
# count^2 (this is a homodimer-type binding reaction: propensity = rate *
# n*(n-1)/2, i.e. the number of possible pairs among n identical molecules)
# while the calibrated `rate` constant below is fixed once against these
# hardcoded numbers, a cell with a genuinely different ambient monomer count
# (e.g. a smaller, slower-growing cell under a different media condition --
# real biology, not a bug) gets a propensity that no longer matches
# REAL_NUCLEATION_RATE_PER_S, even though the rate LAW itself is still
# correctly derived.
#
# Confirmed directly, not hypothetical: a real acetate-condition test
# (out/cache_full_flit_v12_acetate, media_condition='acetate', doubling_time
# 136min) has FliF=206 vs this dict's hardcoded 657 (~3.2x smaller, a real,
# correctly-scaled smaller cell) and FlgE=1050 vs 3508 (~3.3x smaller). Using
# the SAME rate constant calibrated for 657/3508 against these actual, much
# smaller counts gives propensity = rate * 206*205/2 = rate * 21,115, vs the
# rate * 657*656/2 = rate * 215,496 it was calibrated for -- ~9.8% of the
# intended REAL_NUCLEATION_RATE_PER_S. Result: under that acetate cache, the
# ENTIRE upstream assembly chain (C-ring through motor-complex) never fired
# even once across a full 120-minute run -- not because acetate-grown cells
# should nucleate 10x slower (no such claim exists in the literature this
# investigation has cited), but because the rate constant was calibrated
# against a monomer count this smaller cell doesn't have.
#
# NOT FIXED HERE: this module now accepts an optional
# `real_ambient_monomer_counts` override (see generate_bngl/write_bngl/
# get_model_path below) so a caller building a condition-specific model CAN
# pass the cell's actual live counts instead of this hardcoded snapshot --
# but nothing currently DOES that automatically. The real fix -- having
# flagella_nfsim_complexation.py's initialize() read the live/cached ambient
# counts for whatever condition it's actually running under and pass them
# through before calling get_model_path() -- is not yet built. Revisit
# before trusting any non-basal-condition NFsim result quantitatively.
_DEFAULT_REAL_AMBIENT_MONOMER_COUNTS = REAL_AMBIENT_MONOMER_COUNTS


def _nucleating_species(consumed):
    """Return (nuc_species_1, nuc_species_2) for a reaction's consumed dict
    -- the exact same species-selection logic the rule-writing loop below
    uses, factored out so the parameter-writing loop can compute a
    per-reaction nucleation rate using the SAME nucleating pair."""
    species_by_count = sorted(consumed.keys(), key=lambda s: consumed[s])
    nuc_species_1 = species_by_count[0]
    if consumed[nuc_species_1] >= 2:
        nuc_species_2 = nuc_species_1
    else:
        nuc_species_2 = species_by_count[1]
    return nuc_species_1, nuc_species_2


def _calibrated_nucleation_rate(consumed, k_bind, real_ambient_monomer_counts=None):
    """Real, per-reaction nucleation rate: if the nucleating species has a
    known real ambient count, calibrate so propensity matches
    REAL_NUCLEATION_RATE_PER_S at that real count. Otherwise (nucleating
    from a scarce, dynamically-produced precursor), plain k_bind -- no
    artificial suppression needed.

    `real_ambient_monomer_counts` defaults to the module-level (basal-only)
    REAL_AMBIENT_MONOMER_COUNTS snapshot -- see that dict's own "NOT
    CONDITION-AWARE" comment. Pass a condition-specific dict here (same keys,
    real live counts) to calibrate correctly for a non-basal cell; nothing
    does this automatically yet.
    """
    if real_ambient_monomer_counts is None:
        real_ambient_monomer_counts = _DEFAULT_REAL_AMBIENT_MONOMER_COUNTS
    nuc_species_1, _ = _nucleating_species(consumed)
    real_count = real_ambient_monomer_counts.get(nuc_species_1)
    if real_count is None:
        return k_bind
    return REAL_NUCLEATION_RATE_PER_S / (real_count * (real_count - 1) / 2)

# Number of flagella worth of monomers to provide
N_FLAGELLA = 5


def _safe_name(name):
    """Convert a name to a valid BNG identifier.

    Extended 2026-08-12 to also strip '[' / ']' -- species keys are now real
    v2ecoli bulk molecule IDs (e.g. 'FLIF-FLAGELLAR-MS-RING[i]'), which carry
    a compartment suffix BNGL identifiers can't contain. This mapping is
    deterministic and one-way by design: a future wrapper Step that needs to
    go from a safe name back to the real bulk ID should keep its OWN
    {_safe_name(real_id): real_id} lookup built from the known real IDs
    (e.g. via real_bulk_ids() below), not try to invert this string
    transform, since '-' and '[' both collapse to the same '_' here.
    """
    return name.replace(' ', '_').replace('-', '_').replace('[', '_').replace(']', '')


# Species keys in COMPLEXATION_STOICHIOMETRY that ARE real v2ecoli bulk
# molecule IDs (i.e. everything except the three generator-internal-only
# names documented above: 'flagellar export apparatus subunit',
# 'flagellar hook', 'flagella'). Added 2026-08-12 for the future wrapper
# Step (NFSIM_WCM_WIRING_PLAN.md step 3) to build its safe-name -> real-ID
# lookup from, rather than guessing/inverting _safe_name()'s output.
_NON_BULK_SPECIES = {
    'flagellar export apparatus subunit', 'flagellar hook', 'flagella',
    # New 2026-08-27 (HOOK DEPENDENCY FIX): the two new in-between items on
    # the way to a finished base -- rod alone, then rod with the P-ring.
    # Neither has a real bulk ID, same as the three above.
    'flagellar rod', 'flagellar rod with p-ring',
}


def real_bulk_ids():
    """Return the set of species names in COMPLEXATION_STOICHIOMETRY that are
    real v2ecoli bulk molecule IDs (excludes the 3 generator-internal-only
    names -- see _NON_BULK_SPECIES)."""
    names = set()
    for stoich in COMPLEXATION_STOICHIOMETRY.values():
        names.update(stoich.keys())
    return names - _NON_BULK_SPECIES


def _parse_reaction(rxn_name, stoich):
    """Parse a reaction into consumed monomers and produced complex."""
    consumed = {}
    product = None
    for species, count in stoich.items():
        if count < 0:
            consumed[species] = int(abs(count))
        elif count > 0:
            product = species
    return consumed, product


def default_production_rates():
    """Compute default rates: produce enough monomers for 1 flagellum per 100s."""
    duration = 100.0

    demand = {}
    complex_names = set()
    for stoich in COMPLEXATION_STOICHIOMETRY.values():
        for species, count in stoich.items():
            if count > 0:
                complex_names.add(species)
            else:
                demand[species] = demand.get(species, 0) + abs(count)

    rates = {}
    for species, count in demand.items():
        if species not in complex_names:
            rates[f'Free_{_safe_name(species)}'] = count / duration

    return rates


# Ordered reactions (assembly hierarchy). Extracted to a module constant
# 2026-08-12 so both generate_bngl() and bulk_id_to_observable_name() (added
# for NFSIM_WCM_WIRING_PLAN.md step 3) share the exact same reaction/species
# derivation instead of duplicating it.
REACTION_ORDER = [
    'flhDC',
    'flagellar motor switch reaction',
    'flagellar export apparatus reaction 1',
    'flagellar export apparatus reaction 2',
    # 'flagellar motor reaction',  # REPLACED 2026-08-27 by the 3 real,
    # ordered steps below -- see COMPLEXATION_STOICHIOMETRY's "HOOK
    # DEPENDENCY FIX" note. Kept per standing preserve-old-code rule.
    'flagellar rod reaction',
    'flagellar p-ring reaction',
    'flagellar l-ring reaction',
    'flagellar hook reaction',
    'flagellum reaction',
]


def _parse_all_reactions():
    """Parse every reaction in REACTION_ORDER; return (reactions,
    monomer_names, complex_names_ordered, complex_names, all_consumed) --
    the same derivation generate_bngl() needs, factored out so
    bulk_id_to_observable_name() can reuse it exactly rather than
    duplicating the monomer-vs-complex split."""
    reactions = {}
    for rxn_name in REACTION_ORDER:
        consumed, product = _parse_reaction(rxn_name, COMPLEXATION_STOICHIOMETRY[rxn_name])
        reactions[rxn_name] = {'consumed': consumed, 'product': product}

    complex_names = set()
    for rxn in reactions.values():
        complex_names.add(rxn['product'])

    all_consumed = set()
    for rxn in reactions.values():
        all_consumed.update(rxn['consumed'].keys())

    monomer_names = sorted(all_consumed - complex_names)
    complex_names_ordered = [reactions[r]['product'] for r in REACTION_ORDER]
    return reactions, monomer_names, complex_names_ordered, complex_names, all_consumed


def bulk_id_to_observable_name():
    """Return {real_bulk_id: NFsim_observable_name} for every real v2ecoli
    bulk molecule ID this model uses (real_bulk_ids()) -- monomers are
    observed as 'Free_{safe_name}', complex products as bare '{safe_name}'
    (matches generate_bngl()'s own observables-block emission exactly, see
    the 'begin observables' section below). Added 2026-08-12,
    NFSIM_WCM_WIRING_PLAN.md step 3, so the wrapper Step doesn't need to
    re-derive which real IDs are monomers vs. products -- reuses
    _parse_all_reactions() directly rather than guessing."""
    _, monomer_names, complex_names_ordered, _, _ = _parse_all_reactions()
    real_ids = real_bulk_ids()
    mapping = {}
    for name in monomer_names:
        if name in real_ids:
            mapping[name] = f'Free_{_safe_name(name)}'
    for name in complex_names_ordered:
        if name in real_ids:
            mapping[name] = _safe_name(name)
    return mapping


def generate_bngl(n_flagella=N_FLAGELLA, k_bind=K_BIND, k_nucleation=K_NUCLEATION,
                   k_completion=K_COMPLETION, real_ambient_monomer_counts=None):
    """Generate the complete BNGL model string.

    `real_ambient_monomer_counts`: optional override for
    REAL_AMBIENT_MONOMER_COUNTS (same keys: real bulk IDs -> real ambient
    count), forwarded to _calibrated_nucleation_rate. Defaults to the
    basal-only hardcoded snapshot -- see that dict's "NOT CONDITION-AWARE"
    comment. Pass the cell's actual live counts here to get a correctly
    calibrated nucleation rate for a non-basal condition.
    """

    reaction_order = REACTION_ORDER
    reactions, monomer_names, complex_names_ordered, complex_names, all_consumed = _parse_all_reactions()

    # Calculate initial monomer counts
    monomer_counts = {}
    for rxn in reactions.values():
        for species, count in rxn['consumed'].items():
            if species in monomer_names:
                needed = count * n_flagella
                monomer_counts[species] = max(monomer_counts.get(species, 0), needed)

    # ---- Build BNGL ----
    lines = []
    lines.append('begin model')
    lines.append('')

    # -- Parameters --
    lines.append('begin parameters')
    lines.append(f'    n_flagella  {n_flagella}')
    lines.append(f'    k_bind      {k_bind}')
    lines.append(f'    k_nucleation {k_nucleation}')
    lines.append(f'    k_completion {k_completion}')
    lines.append('')
    # NUCLEATION FIX (2026-08-12), SUPERSEDED 2026-08-17 for the C-ring
    # reaction specifically -- see K_NUC_CRING module docstring for the full
    # derivation (real Sim et al. 2017 rate, calibrated to real ambient
    # FliF concentration). Old uniform-suppression version (same rate for
    # EVERY reaction, k_bind/NUCLEATION_SUPPRESSION_FACTOR) kept per
    # standing preserve-old-code rule:
    #   nuc_rate = k_bind / NUCLEATION_SUPPRESSION_FACTOR
    #   for rxn_name in reaction_order:
    #       ...
    #       lines.append(f'    k_nuc_{safe_product}  {nuc_rate:.6e}')
    # Even older target_propensity-based formula (independent of k_bind,
    # also kept per standing preserve-old-code rule):
    #   target_propensity = n_flagella / 50.0
    #   ...
    #   if combinatorial > 0:
    #       nuc_rate = target_propensity / combinatorial
    #   else:
    #       nuc_rate = k_nucleation
    #
    # Current (generalized 2026-08-17, see _calibrated_nucleation_rate()
    # docstring for the real bug this replaced): every nucleation reaction
    # gets its OWN calibrated rate based on whether its nucleating species
    # is a real, abundant free monomer (C-ring/FliF, hook/FlgE, flhDC/FlhC
    # -- each calibrated to ITS OWN real ambient count) or a scarce,
    # dynamically-produced precursor (export apparatus subunit, motor
    # reaction -- plain k_bind, naturally self-limiting, matches the custom
    # deterministic Steps pipeline's own architecture: only whole-flagellum
    # nucleation is rate-limited, everything already gated by a scarce
    # precursor is not).
    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        total = sum(consumed.values())
        if total > 2:
            safe_product = _safe_name(product)
            this_rate = _calibrated_nucleation_rate(
                consumed, k_bind, real_ambient_monomer_counts)
            lines.append(f'    k_nuc_{safe_product}  {this_rate:.6e}')
    lines.append('')

    for monomer in sorted(monomer_counts.keys()):
        safe = _safe_name(monomer)
        lines.append(f'    {safe}_0  {monomer_counts[monomer]}')
    lines.append('end parameters')
    lines.append('')

    # -- Molecule Types --
    lines.append('begin molecule types')

    for monomer in monomer_names:
        lines.append(f'    {_safe_name(monomer)}()')

    intermediate_complexes = sorted(complex_names & all_consumed)
    for cx in intermediate_complexes:
        lines.append(f'    {_safe_name(cx)}()')

    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        safe_product = _safe_name(product)

        total_subunits = sum(consumed.values())

        if total_subunits <= 2:
            continue

        state_parts = []
        for species in sorted(consumed.keys()):
            count = consumed[species]
            safe_species = _safe_name(species)
            states = '~'.join(str(i) for i in range(count + 1))
            state_parts.append(f'{safe_species}~{states}')

        scaffold_name = f'Growing_{safe_product}'
        lines.append(f'    {scaffold_name}({",".join(state_parts)})')

    final_complexes = sorted(complex_names - all_consumed)
    for cx in final_complexes:
        lines.append(f'    {_safe_name(cx)}()')

    lines.append('end molecule types')
    lines.append('')

    # -- Seed Species --
    lines.append('begin seed species')
    for monomer in sorted(monomer_counts.keys()):
        safe = _safe_name(monomer)
        lines.append(f'    {safe}()  {safe}_0')
    lines.append('end seed species')
    lines.append('')

    # -- Observables --
    lines.append('begin observables')

    for monomer in monomer_names:
        safe = _safe_name(monomer)
        lines.append(f'    Molecules  Free_{safe}  {safe}()')

    for cx_name in complex_names_ordered:
        safe = _safe_name(cx_name)
        lines.append(f'    Molecules  {safe}  {safe}()')

    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        safe_product = _safe_name(product)
        total_subunits = sum(consumed.values())
        if total_subunits > 2:
            scaffold_name = f'Growing_{safe_product}'
            lines.append(f'    Molecules  {scaffold_name}_total  {scaffold_name}()')

    lines.append('end observables')
    lines.append('')

    # -- Reaction Rules --
    lines.append('begin reaction rules')

    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        safe_product = _safe_name(product)
        total_subunits = sum(consumed.values())

        lines.append(f'')
        lines.append(f'    # === {rxn_name} ===')
        lines.append(f'    # Product: {product}')
        lines.append(f'    # Subunits: {", ".join(f"{c}x {s}" for s, c in consumed.items())}')

        if total_subunits == 1:
            species = list(consumed.keys())[0]
            safe_species = _safe_name(species)
            lines.append(f'    {safe_species}() -> {safe_product}()  k_bind')

        elif total_subunits == 2:
            species_list = []
            for species, count in consumed.items():
                for _ in range(count):
                    species_list.append(species)

            if len(species_list) == 2:
                s1, s2 = species_list
                lines.append(f'    {_safe_name(s1)}() + {_safe_name(s2)}() -> {safe_product}()  k_bind')

        else:
            scaffold_name = f'Growing_{safe_product}'
            sorted_species = sorted(consumed.keys())

            species_by_count = sorted(consumed.keys(), key=lambda s: consumed[s])
            nuc_species_1 = species_by_count[0]
            if consumed[nuc_species_1] >= 2:
                nuc_species_2 = nuc_species_1
            else:
                nuc_species_2 = species_by_count[1]

            init_states = []
            nuc_counts = {}
            for species in sorted_species:
                if species == nuc_species_1:
                    nuc_counts[species] = nuc_counts.get(species, 0) + 1
                if species == nuc_species_2:
                    nuc_counts[species] = nuc_counts.get(species, 0) + 1

            for species in sorted_species:
                safe_sp = _safe_name(species)
                c = nuc_counts.get(species, 0)
                init_states.append(f'{safe_sp}~{c}')

            safe_nuc1 = _safe_name(nuc_species_1)
            safe_nuc2 = _safe_name(nuc_species_2)

            nuc_rate_name = f'k_nuc_{safe_product}'
            lines.append(f'    # Nucleation (rate scaled by 1/total_subunits)')
            lines.append(f'    {safe_nuc1}() + {safe_nuc2}() -> '
                         f'{scaffold_name}({",".join(init_states)})  {nuc_rate_name}')

            for species in sorted_species:
                safe_sp = _safe_name(species)
                count = consumed[species]

                start = nuc_counts.get(species, 0)

                for i in range(start, count):
                    lines.append(
                        f'    {scaffold_name}({safe_sp}~{i}) + {_safe_name(species)}() -> '
                        f'{scaffold_name}({safe_sp}~{i + 1})  k_bind')

            complete_states = []
            for species in sorted_species:
                safe_sp = _safe_name(species)
                complete_states.append(f'{safe_sp}~{consumed[species]}')

            lines.append(f'    # Completion')
            lines.append(f'    {scaffold_name}({",".join(complete_states)}) -> '
                         f'{safe_product}()  k_completion')

    lines.append('')
    lines.append('end reaction rules')
    lines.append('')
    lines.append('end model')

    return '\n'.join(lines)


def write_bngl(output_path=None, **kwargs):
    """Generate and write the BNGL model file."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), 'flagella_complexation.bngl')

    bngl_text = generate_bngl(**kwargs)

    with open(output_path, 'w') as f:
        f.write(bngl_text)

    return output_path


def get_model_path(real_ambient_monomer_counts=None):
    """Return path to the generated BNGL model, generating it if needed.

    `real_ambient_monomer_counts`: forwarded to generate_bngl -- see that
    function's docstring and REAL_AMBIENT_MONOMER_COUNTS's "NOT
    CONDITION-AWARE" comment. NOTE: if the file already exists on disk (the
    common case -- flagella_nfsim_complexation.py calls this with no
    arguments), it is reused AS-IS regardless of this argument -- passing a
    condition-specific override here only takes effect when the file doesn't
    exist yet, or the caller deletes it first / calls write_bngl directly.
    Nothing currently regenerates a condition-specific model automatically;
    this is the wiring gap flagged in REAL_AMBIENT_MONOMER_COUNTS's comment.
    """
    path = os.path.join(
        os.path.dirname(__file__), 'flagella_complexation.bngl')
    if not os.path.exists(path):
        write_bngl(path, real_ambient_monomer_counts=real_ambient_monomer_counts)
    return path


def make_production_document(
    model_file=None,
    n_steps=100,
    complexation_interval=50.0,
    production_interval=1.0,
    production_rate_scale=1.0,
):
    """v2ecoli-local equivalent of pbg_nfsim.composites.make_production_document.

    Added 2026-08-12, part of Maya Abdalla's flagella-cascade investigation.
    pbg_nfsim's own make_production_document hardcodes get_model_path() and
    default_production_rates() from ITS OWN bundled (deliberately left
    generic/uncorrected) generate_flagella_bngl.py -- there's no way to pass
    a custom model_file through it and get matching production rates, since
    the rates always come from pbg_nfsim's own stock stoichiometry
    regardless of which model_file is supplied. This function is otherwise
    identical in structure, but sources BOTH the model file and the
    production rates from THIS module -- the v2ecoli-owned, corrected
    stoichiometry (see COMPLEXATION_STOICHIOMETRY above for the full
    provenance/citations) -- so the two stay consistent. pbg_nfsim is still
    used for the actual runtime process classes (NFSimProcess,
    MonomerProduction) via flagella_nfsim_assembly.py's core_extensions --
    those are genuinely generic engine code, not investigation-specific
    science, and stay as an external dependency.
    """
    if model_file is None:
        model_file = get_model_path()

    rates = default_production_rates()
    scaled_rates = {
        name: rate * production_rate_scale
        for name, rate in rates.items()
    }

    return {
        'production': {
            '_type': 'process',
            'address': 'local:monomer-production',
            'config': {
                'production_rates': scaled_rates,
            },
            'outputs': {
                'monomers': ['species'],
            },
            'interval': production_interval,
        },
        'complexation': {
            '_type': 'process',
            'address': 'local:nfsim',
            'config': {
                'model_file': model_file,
                'n_steps': n_steps,
            },
            'inputs': {
                'observables': ['species'],
                # Added 2026-08-12: without this, NFSimProcess's scaffold
                # persistence fix (pbg_nfsim/processes.py) has nowhere to
                # round-trip through -- the port existed on the process but
                # was never connected to a store, so Growing_X scaffold
                # state was still silently dropped every interval despite
                # the wrapper itself now being capable of carrying it.
                'scaffold_species': ['scaffold'],
            },
            'outputs': {
                'observables': ['species'],
                'scaffold_species': ['scaffold'],
            },
            'interval': complexation_interval,
        },
        'species': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'species': 'map[float]',
                    'time': 'float',
                },
            },
            'inputs': {
                'species': ['species'],
                'time': ['global_time'],
            },
        },
    }


if __name__ == '__main__':
    path = write_bngl()
    print(f'Model written to: {path}')

    with open(path) as f:
        text = f.read()
    n_rules = text.count(' -> ')
    print(f'Reaction rules: {n_rules}')
