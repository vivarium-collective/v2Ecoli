# Archive: FliT-mediated FlhD4C2 checkpoint (removed 2026-08-10)

This folder preserves the code, reaction-network entries, and config wiring
for the FliT:FlhD4C2 negative-feedback checkpoint that was part of the
`v2ecoli` flagella-cascade investigation (`workspace/investigations/
flagella-cascade/`) from 2026-08-05/06 through 2026-08-10. It was removed
from the active codebase on 2026-08-10, at Maya Abdalla's explicit
instruction, in favor of a planned NFsim rule-based representation. Kept
here for a clean before/after comparison, not because the mechanism was
wrong or the work was wasted -- see "Why removed" below.

## What this was

The flagellar master regulator FlhD4C2 (`CPLX0-3930[c]`) needed SOME active
shutdown mechanism -- without one, flagella count ran away unboundedly (the
original `flagella-02-flagella-count-unbounded-runaway` finding). Two Steps
were added to address this:

1. **`flagella_flhdc_degradation.py`** (`ecoli-flhdc-degradation`) -- basal
   ClpXP-mediated turnover of the assembled FlhD4C2 complex. The standard
   v2ecoli `protein_degradation` process only degrades individual monomers,
   never assembled complexes, so without this Step FlhD4C2 has literally no
   decay pathway at all. This part is **not** FliT-specific -- it's basal
   turnover (Tomoyasu et al. 2003; Claret & Hughes 2000 for the rate
   estimate). **Removing it (see below) reopens the unbounded-runaway risk
   until the NFsim replacement is built** -- a known, accepted tradeoff of
   this reversion, not an oversight.

2. **`flagella_flit_flhdc_checkpoint.py`** (`ecoli-flit-flhdc-checkpoint`)
   -- the actual FliT-mediated regulatory mechanism: free FliT-dimer
   (released once a flagellum's FliD cap is exported, via the
   `FLIT-FLID-CPLX_RXN` equilibrium below) binds FlhD4C2 and enhances its
   degradation, using a fast-equilibrium reduction from Utsey & Keener
   (2020, PLOS Comput Biol 16:e1007689).

Supporting reaction-network infrastructure (also removed, see
`reaction_network_snapshot.md` for exact text):
- `FLIT-FLID-CPLX_RXN` (`equilibrium_reactions.tsv` /
  `equilibrium_reaction_rates.tsv`) -- the FliT:FliD binding equilibrium
  that fed the checkpoint's "free FliT" signal.
- A historical comment in `complexation_reactions_modified.tsv` documenting
  an earlier, reverted attempt to have `CPLX0-7452_RXN` itself release FliT
  directly (blocked by `molecule_groups.py`'s one-product constraint on
  complexation reactions).

**Kept, not archived**: `FLIT-DIMER_RXN` (FliT homodimer formation,
`complexation_reactions_added.tsv`) stays in the live reaction network.
FliT dimerization is real, independent biology (Yamamoto & Kutsukake 2006)
that doesn't depend on this checkpoint -- it's simply unconsumed by
anything else right now. Left in place as a building block for the planned
NFsim work ("FliT only being there," per Maya's framing).

## Why removed

The mechanism is real, biochemically-confirmed biology -- **in Salmonella**.
Yamamoto & Kutsukake (2006, *J Bacteriol* 188:5124) showed FliT binds the
FlhC subunit of FlhD4C2 and blocks it from occupying class-2 promoters,
confirmed by pull-down/far-Western. FliD acts as the counterbalance,
titrating FliT away while still being exported.

**But when Albanna et al. (2018, *Sci Rep* 8:16705) directly tested a
Delta-fliT mutant in E. coli MG1655 -- the exact K-12 reference strain this
WCM models -- side-by-side with S. Typhimurium LT2, they found no
significant phenotypic difference in E. coli, vs. a clear increase in
flagellar number in the Salmonella Delta-fliT mutant.** Their interpretation:
FliT's regulatory role may be Salmonella-specific/adaptive, with E. coli's
FlhD4C2 relatively insensitive to it (or E. coli uses some other,
unidentified brake instead).

So: the mechanism was correctly implemented and literature-grounded, but the
one study that directly tested it in K-12 found little-to-no phenotypic
effect there. Rather than keep a mechanism whose real-world significance in
this specific organism is doubtful, the decision was made to remove it and
move toward representing FliT's role (chaperone + any regulatory
consequence) via an NFsim rule instead, alongside `flagella-04-complexation-
nfsim`'s existing (currently standalone, not yet WCM-coupled) rule-based
assembly work. See `workspace/investigations/flagella-cascade/studies/
flagella-02-transcription-regulation/study.yaml` and `flagella-06-reduced-
ode-model/study.yaml` for how this finding is tracked going forward, and
`investigation.yaml`'s open_decisions for the NFsim migration plan.

## Full citations

- Yamamoto S & Kutsukake K (2006). FliT acts as an anti-FlhD2C2 factor in
  the transcriptional control of the flagellar regulon in *Salmonella
  enterica* serovar Typhimurium. *J Bacteriol* 188(14):5124-31.
  https://pubmed.ncbi.nlm.nih.gov/16952964/
- Albanna A, Sim M, Hoskisson PA, Gillespie C, Rao CV, Aldridge PD (2018).
  Driving the expression of the *Salmonella enterica* sv Typhimurium
  flagellum using *flhDC* from *Escherichia coli* results in key regulatory
  and cellular differences. *Sci Rep* 8:16705.
  https://doi.org/10.1038/s41598-018-35005-2
- Utsey B & Keener JP (2020). PLOS Comput Biol 16:e1007689 (fast-equilibrium
  reduction used for the checkpoint's math).
- Tomoyasu T et al. (2003). Mol Microbiol 48:443 (ClpXP degrades assembled
  FlhD2C2, not free subunits).
- Claret L & Hughes KT (2000). J Bacteriol 182:833 (FlhD/FlhC half-life
  estimate in Proteus mirabilis, used for the basal degradation rate).

## Contents of this folder

- `code/flagella_flit_flhdc_checkpoint.py` -- the checkpoint Step, as it was.
- `code/flagella_flhdc_degradation.py` -- the basal degradation Step, as it was.
- `reaction_network_snapshot.md` -- exact removed TSV entries and sim_data.py
  config methods, for reference/restoration if ever needed.
