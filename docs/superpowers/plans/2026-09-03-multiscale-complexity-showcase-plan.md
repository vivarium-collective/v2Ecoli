# Investigation Plan — Multiscale Complexity in a Whole-Cell E. coli

**Repo:** v2ecoli · **Branch:** `investigation/multiscale-complexity-showcase`
**Worktree:** `~/code/v2ecoli--multiscale-showcase` (off fresh `origin/main`)
**Mode:** full-autonomous execution; finished published report handed back for review.

## Research question
What emergent biological phenomena does v2ecoli produce that are invisible at any
single scale — and only appear when molecular, cellular, and population scales are
coupled? Along the way, where the model diverges from known biology, apply the
smallest principled correction and demonstrate the improvement + the insight it exposes.

## Unifying thesis
Three *flavors* of multiscale complexity, each an arc, converging on one capstone:
cross-scale **control** (ppGpp), nonlinear **flux reallocation** (metabolism), and
**order-from-noise** (heterogeneity).

## Per-study discovery loop
1. **Observe** — run the native composite, measure the observable.
2. **Identify gap** — compare against a *cited* biological expectation / acceptance band.
3. **Fill it** — smallest principled correction in the real composite/step (never inline fudging).
   If no real, cited gap exists, record "model correct here" and skip the fix — that is a valid result.
4. **Validate** — re-run; the fix must improve the target observable AND not break a *held-out* observable.
5. **Insight** — biology-forward finding (statement / mechanism / evidence) + before/after viz.

## The four studies

### Study A — Arc 1 (ppGpp): replication timing is slaved to growth
- **Observables:** `origins_per_cell`, `ppgpp_conc`, `growth_rate`, `dnaA` across a rich vs minimal condition.
- **Cited expectation:** origins/cell scales with growth rate at ~constant initiation mass
  (Cooper–Helmstetter / Donachie); ppGpp restrains DnaA-driven initiation under stringency.
- **Candidate gap:** ppGpp→`chromosome_initiation` coupling may not reproduce origin scaling.
- **Candidate fix locus:** `steps/ppgpp_initiation.py`, `processes/chromosome_initiation.py`, dnaA steps
  (`dnaa_box_binding`, `rida`, `dars`, `ddah`).
- **Held-out check:** growth_rate & mass unchanged.

### Study B — Arc 2 (metabolism): acetate overflow as an emergent strategy
- **Observables:** acetate exchange/secretion flux vs `growth_rate`; `ecoli_baseline` tFBA vs `ecoli_millard` kinetic.
- **Cited expectation:** overflow metabolism — acetate secretion switches on above a growth-rate
  threshold (Basan et al. 2015; Crabtree-like proteome tradeoff).
- **Candidate gap:** overflow threshold / exchange scaling off between the two metabolism engines.
- **Candidate fix locus:** `processes/metabolism.py` constraints / exchange bounds; Millard bridge coupling.
- **Held-out check:** biomass yield preserved.

### Study C — Arc 3 (heterogeneity): single-cell variability from molecular noise
- **Observables:** `division_time` (and cell-size) distribution across a seed ensemble; CV; size-homeostasis slope.
- **Cited expectation:** interdivision-time CV ~10–30%; adder principle (constant added size).
- **Candidate gap:** ensemble too deterministic (CV too low) or homeostasis not adder-like.
- **Candidate fix locus:** stochastic initiation variance propagation; `steps/division.py`; seeding.
- **Held-out check:** mean division time & mean mass preserved.

### Capstone — one nutrient downshift, three complexities
- A single rich→minimal downshift run showing, simultaneously: the ppGpp spike (signaling),
  central-carbon flux reallocation + acetate change (metabolism), and a shift in the
  heterogeneity distribution (ensemble). No new fix — it synthesizes the three corrected mechanisms
  into one integrative visualization and narrative.

## Deliverables
- 4 filled-out studies (design → build → simulate → evaluate → decide) with real runs.
- Each: a gap finding, any correction (or a justified "no gap"), a validated before/after, a
  biology-forward insight, and a headline interactive before/after visualization.
- A published investigation report knitting the narrative into one story.

## Execution pipeline (agentic)
`/viva-investigation new` → per study `/viva-study` (baseline → variant/fix → runs → outcomes)
→ `/viva-viz` (before/after) → `/viva-biology-forward` (insight) → `/viva-cite-bands` (band provenance)
→ `/viva-harden-investigation` (rigor pass) → `/viva-report` (publish).

## Compute
Design + smoke-test on laptop (single/few seeds, full ParCa ~2.5 min). Escalate the seed
ensembles (Study C, capstone) to more seeds via Ray; use the mini if laptop wall-clock is limiting.

## Rigor guardrails
- Real native composites only; no inline numpy stand-ins for model behavior.
- Every correction cited to a biological band via `/viva-cite-bands`; validated on a held-out observable.
- "No gap found" is a publishable, honest outcome — no fabricated fixes.
- Isolated worktree; corrections committed on the investigation branch; never touch the shared checkout.
