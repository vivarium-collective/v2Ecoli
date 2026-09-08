# Flagella-Cascade Investigation — Master Document

*Living document. Started 2026-08-28, last updated 2026-09-04. Current
State, Parameters, History Appendix, and References sections are all up
to date as of this date — this consolidates and supersedes the older
per-date (`CHANGES_*.md`) and per-study (`study.yaml`) notes. Those
files, along with `NFSIM_WCM_WIRING_PLAN.md` and the `feedback/` notes,
have been removed from the working tree (2026-08-28) now that their
content lives here — all still fully recoverable from git history if
ever needed. `investigation.yaml` was trimmed the same day to a short
pointer back to this file; it is no longer the working record.*

---

## 1. Current State

### 1.1 Regulatory cascade (Class I → II → III)

- **Class I**: FlhDC (master regulator, transcription factor complex).
  Drives Class II gene expression. Not a structural component — never
  consumed by any assembly reaction.
- **Class II**: structural genes (C-ring, export apparatus, rod, rings,
  hook, and their associated ATPase/chaperone machinery). Expression is
  FlhDC-dependent.
- **Class III**: late genes (`fliC`, `fliD`, `flgK`/`flgL`, `motAB`,
  `cheAW`, `fliS`, `flgM`). Expression requires free FliA (σ28), which is
  normally sequestered by FlgM until the hook-basal body (HBB) is complete.

**FlgM:FliA sequestration** — modeled as an equilibrium reaction
(`FLGM-FLIA-CPLX_RXN`), still solved by the shared, general-purpose
equilibrium Step (numerical ODE, not yet moved to a dedicated exact-solve
Step). Current model Kd = 2×10⁻⁷ M — a **known, deliberate relaxation**
from the real, measured value (see §2). Not yet corrected — two earlier
attempts to tighten it toward the real Kd both failed in testing, and a
third attempt (2026-09-01, a dedicated exact-solve Step, same pattern as
FliS:FliC) technically ran without crashing but was reverted after
population-level dynamics were judged not correct on review (see History
§3.1 — the real Kd itself is not in question, but something about the
resulting dynamics is not yet understood).

**FlgM secretion** — its own dedicated Step
(`flagella_flgm_secretion.py`). As of 2026-08-27, the trigger is
`count(nascent_flagellum)` alone (in-progress HBBs), **not**
`count(CPLX0-7452)` (fully complete flagella including filament). This
matches the real, direct literature finding that the substrate-specificity
switch enabling FlgM export happens at HBB completion, before the filament
is built (see §2 citations). As of 2026-09-02, the export rate is
**first-order in the current FlgM pool** (`turnover_rate_per_s ≈
0.00158/s = ln(2)/7.3min`, Karlinsey et al. 1998, see §2), gated on
`hbb_count > 0` — replacing an earlier zero-order, uncited placeholder
(`secretion_rate=0.1/HBB/s`) that scaled export with completed-HBB count,
an assumption the real half-life data never supported.

### 1.2 Assembly cascade (NFsim / BNGL rule-based network)

Fixed 2026-08-27/28 to the real, literature-confirmed dependency order.
Each stage requires the previous stage's product — no stage can be skipped
or built out of order:

**C-ring → export apparatus (2 steps) → rod → P-ring → L-ring → hook →
complete HBB (→ `nascent_flagellum`)**

- C-ring (FliF/FliG/FliM/FliN) nucleates from free monomer, deliberately
  slow (real, literature-derived rate, not a placeholder).
- Export apparatus assembles onto the C-ring (2 sub-reactions, generator
  bookkeeping only, not 2 real biological stages).
- Rod (FliE, FlgB, FlgC, FlgF, FlgG) requires the export apparatus.
- P-ring (FlgI, 26 copies) requires the rod. **Newly added 2026-08-28** —
  previously skipped entirely.
- L-ring (FlgH, 26 copies) requires the rod+P-ring. This is the real,
  confirmed trigger for the rod-to-hook transition (see §2).
- Hook (FlgE, 120 copies) now requires the finished rod+P-ring+L-ring base
  (`FLAGELLAR-MOTOR-COMPLEX[j]`). **This dependency did not exist before
  2026-08-28** — hook used to form independently from free FlgE alone.
- **Stator (MotA/MotB) removed from this entire chain.** Real biology
  confirms structural completion does not require the stator — only
  rotation does. Free MotA/MotB are currently untouched by this pathway.
- Completion produces the `flagella` species, which becomes a real
  `nascent_flagellum` unique molecule.

**Filament elongation** is a separate, non-NFsim incremental Step
(combinatorial explosion in the rule-based engine at real subunit counts
made a single-shot reaction impossible — see History Appendix). Growth
follows an injection-diffusion law, `dL/dt = a/(b+L)`. Target length is
5,000 subunits (short end of the real 20,000–40,000 range — a deliberate
modeling simplification for practical simulation windows, not a claimed
biological value).

**FliS:FliC chaperone binding** — as of 2026-08-28, this reaction has its
**own dedicated Step** (`flagella_flis_flic_equilibrium.py`), solved by an
exact closed-form quadratic (not the shared numerical ODE solver). This
was a direct fix for a repeated division-triggered crash (see §1.3, §2).
Confirmed by direct testing: free FliS/FliC never overshoot negative,
by construction, at any scale.

**Stoichiometry corrected 2026-08-31**: FliS binds FliC as a homodimer,
not as a single monomer — confirmed directly against the primary
literature (Auvray et al. 2001: "FliS homodimers bind to FliC monomers";
consistent with Sajó et al. 2016 characterizing FliS's own structural
plasticity without ever describing a loose monomer/dimer equilibrium, the
usual signature of tight, fast self-association). Modeled via the
standard fast-pre-equilibrium approximation (no published FliS
self-dimerization Kd exists — searched directly, not found): dimer count
read directly off the raw monomer bulk count (`free_fliS // 2`). Each
unit of complex formed or dissociated now moves 2 raw FliS monomers, not
1 — fixed in both the equilibrium Step and in `flagella_filament_
elongation.py`'s chaperone-release-on-consumption logic, kept
consistent. Kd itself (5.26e-8 M, Muskotal et al. 2006) is unchanged —
now understood to be on a dimer:FliC basis, not raw-monomer:FliC.
Confirmed via population-scale test: halves the sustainable protected
(FLIS-FLIC-CPLX) pool for the same FliS synthesis rate, exactly as
predicted, since forming the same amount of complex now costs twice as
much raw FliS (see History §3.2).

### 1.3 Division handling

- Division splits all bulk species, including complete flagella and
  in-progress `nascent_flagellum` entries, binomially between daughters —
  consistent with direct Salmonella live-imaging evidence (see §2) that
  division "forcefully splits the number of existing filaments into
  half."
- Division does **not** pause for flagella assembly, and flagella assembly
  does not pause division. This was checked directly against the
  literature: the real mechanism connecting motility and division in
  E. coli is resource competition (shared ribosome/RNAP/ATP pools, already
  correctly modeled), not a hard on/off checkpoint. Direct checkpoint
  mechanisms of that kind exist in the literature (Caulobacter, FlhA-based
  sensor; Campylobacter, FlhG-based pole exclusion) but not in E. coli.
- **Known, still-live risk**: the equilibrium solver is most likely to
  crash right at division, because binomial splitting can suddenly cut a
  reaction's already-small absolute molecule counts in half. Confirmed
  directly (diagnostic printout) that FliS/FliC was the specific reaction
  crashing, twice, at increasing severity — this drove the exact-solve
  Step in §1.2. **FlgM:FliA is on the same shared, general solver, with a
  Kd relaxed for the same kind of reason, and has not yet been checked
  for the same vulnerability.**

### 1.4 Known open problems

1. ✅ FliS:FliC (§3.2) and FlgM secretion turnover-rate (§3.1) fixes both
   validated across a real 6-seed batch, 2026-09-03.
2. **FlgM:FliA remains on the shared, general ODE solver with a relaxed
   Kd.** Same class of fix (dedicated exact-solve Step) exists on disk but
   is unwired — see §3.1.
3. ✅ `secretion_rate` (FlgM secretion) recalibrated to a first-order,
   literature-derived rate, 2026-09-02 — see §3.1.
4. ✅ `complexation_reactions_modified.tsv` motor-complex stoichiometry
   audited 2026-09-04 — stale but confirmed inert, see §3.2.
5. ✅ NFsim's own internal random seed pinned to this codebase's outer
   `--seed`, 2026-09-01 — see §3.4.
6. The generated `.bngl` model file is not automatically rebuilt when its
   generator script changes — a real, already-encountered footgun. No
   permanent safeguard yet.
7. ✅ Temporary diagnostic print statements in `equilibrium.py` and
   `function_registry.py` removed 2026-09-04 (the division-crash
   investigation they supported is now fully closed out per item 1).
8. ✅ FliD double-consumption fix (2026-08-21) confirmed intact 2026-09-04
   via a direct unit test — see §3.2.

### 1.5 Execution-layer order (flagella-related Steps, real order)

1. `ecoli-flagella-nfsim-complexation` — C-ring → export apparatus → rod →
   P-ring → L-ring → hook → complete HBB
2. `ecoli-flagella-flis-flic-equilibrium` — exact closed-form solve
3. `ecoli-flagella-filament-elongation` — incremental FliC growth, draws
   protected pool first
4. `ecoli-flagella-flgm-secretion` — triggered by `nascent_flagellum` count
5. `ecoli-flagella-transcription-regulation` — Class II / III promoter
   activity

**Why this order does not need to match the conceptual regulation → transcription →
translation → assembly loop**: the real biological lag between a regulatory
change and its effect on the free-monomer pool is hundreds of ticks
(transcription, then translation). So assembly in any one tick always draws
from monomers made by *past* regulatory decisions, never the current tick's.
Confirmed directly: `diagnostic_transcription_to_protein_lag.py` showed free
FliC jump 0 → 10,807 → 17,576 molecules within the first ~600s of a run,
entirely from pre-existing mRNA and ribosomes, before that run's own
regulatory changes could have had any effect. Step ordering within one tick
is therefore a bookkeeping choice, not a biology-accuracy question.

---

## 2. Parameters and Sources

| Parameter | Value | Source |
|---|---|---|
| Kd, FlgM:FliA (real) | ~1.8–2.0×10⁻¹⁰ M | Chadsey, Karlinsey & Hughes 1998, *Genes Dev* 12:3123 (*Salmonella*, SPR: ka=8.9×10⁵ M⁻¹s⁻¹, kd=1.6×10⁻⁴ s⁻¹) |
| Kd, FlgM:FliA (current model) | 2.0×10⁻⁷ M | Deliberately relaxed from the real value above, for solver stability. Attempted switch to the real Kd via a dedicated exact-solve Step (2026-09-01) reverted after review — see History §3.1. |
| Kd, FliS:FliC | 5.26×10⁻⁸ M | Muskotal et al. 2006, *FEBS Lett* 580:3916 (ITC, Ka=1.9×10⁷ M⁻¹, 1:1 stoichiometry) — real value, unchanged by the 2026-08-28 Step move |
| k_bind (generic proxy rate) | 8.5×10⁴ M⁻¹s⁻¹ (→ ~1.412×10⁻⁴ /molecule/s) | McMurry et al. 2015 (real, measured FlhA:FlhB rate — used as a borrowed proxy for ~30 other reactions lacking their own measured kinetics) |
| Nucleation rate (C-ring, FlhDC) | 1.67×10⁻³ /s | Sim et al. 2017, *Sci Rep* 7:41189 (*E. coli* RP437, chemostat growth-rate study) |
| Filament elongation rate_a | 15,556 subunit²/s | Renault et al. 2017, *eLife* 6:e23136. Corrected 2026-09-01 by re-deriving the paper's own equations (kon=a/b) and matching directly to their Figure 3 dataset specifically (kon≈27.09/s, 291 filaments/1276 points — their stronger, six-color-labeling dataset), rather than an earlier, less-anchored value (26,450, kept as a comment in code). rate_b (575) already matched Fig 3 and is unchanged. |
| Filament elongation rate_b | 575 subunits | Same source as rate_a |
| Filament target_length | 5,000 subunits | Short end of real 20,000–40,000 range, PMC7696725 — a modeling simplification, not a claimed real value |
| P-ring (FlgI) copy number | 26 | Real, confirmed structural literature |
| L-ring (FlgH) copy number | 26 | Real, confirmed structural literature |
| Hook (FlgE) copy number | 120 | Real, structural (~55 nm hook, 11 protofilaments) |
| FliC degradation half-life (unprotected) | 7.4 min (k=0.00156/s) | Nagar et al. 2022, pulsed-SILAC |
| FlgM half-life, HBB-complete (Fla+) | 7.3 min | Karlinsey, Tsui, Winkler & Hughes 1998, *J Bacteriol* 180:5384-5397 (*Salmonella*, strain TH2592) — see citation-correction note below |
| FlgM half-life, HBB-incomplete | no detectable turnover | Same source, ring-mutant strains (ΔflgHI, flgB) |
| FlgM `turnover_rate_per_s` (current model) | 0.00158/s = ln(2)/7.3min | Derived directly from the half-life row above (2026-09-02 fix, see History §3.1). First-order, gated on `hbb_count > 0`; no per-HBB multiplier — replaces the old, uncited `secretion_rate=0.1/HBB/s` zero-order placeholder. |
| Hook-basal-body dependency on L-ring | rod-to-hook transition requires L-ring | Cohen & Hughes 2014, *J Bacteriol* 196:2387 (*Salmonella*) |
| Stator not required for structural completion | confirmed | General literature on *motA*/*motB* mutants — paralyzed but structurally complete flagella |
| Flagella split binomially at division | confirmed, direct imaging | Aizawa & Kubori 1998, *Genes to Cells* 3:625 (*Salmonella*, live dark-field imaging through real division events) |

*Note: several of these citations are from Salmonella, not E. coli directly
— used because the mechanism is treated as conserved across these closely
related species, and no direct E. coli data was found (checked directly,
not assumed) for several of them. This is flagged per-row, not hidden.*

*Citation correction (2026-09-02): earlier versions of this table and §4
merged two distinct Karlinsey et al. papers into one incorrect composite
citation (right year/journal, wrong title and author list). Verified
against CrossRef directly: the FlgM half-life data (TH2592, 7.3 min) is
from Karlinsey, Tsui, Winkler & Hughes (1998) *J Bacteriol* 180:5384-5397,
"Flk Couples flgM Translation to Flagellar Ring Assembly in Salmonella
typhimurium." The HBB-completion-coupled-to-secretion mechanism (§1.1) is a
separate paper: Karlinsey, Tanaka, Bettenworth, Yamaguchi, Boos, Aizawa &
Hughes (2000) *Mol Microbiol* 37:1220-1231. Both are now cited correctly
throughout this document and in `flagella_flgm_secretion.py`.*

---

## 3. History Appendix

Organized by topic (not chronologically by outcome) — each topic's full arc,
including anything tried and dropped along the way, lives in one place.
Drawn from `investigation.yaml`, the `CHANGES_2026-08-0{6,7,8,10}.md` series,
each study's `study.yaml`, the two `archive/*/README.md` files, the
`feedback/` review-response docs, and direct session history from
~2026-08-11 onward.

### 3.1 Regulatory cascade

- **Kalir & Alon SUM-gate transcriptional cascade** — promoter activity
  `p_i = (β·X + β'·Y)/(β+β')`, X=FlhDC activity, Y=free-FliA activity,
  driving Class II vs Class III `init_prob_override`. Ported from Maya's
  vEcoli `biofilm` branch (feature name `flagella_regulation` itself taken
  from her vEcoli config script name). Confirmed byte-neutral when off
  (`init_prob_override`==0 everywhere, cell grows/divides identically to
  pre-port main). Class II > Class III ordering holds stably across
  generations and the division boundary (Kalir & Alon 2004). **Current.**

- **FlgM secretion: gate mechanism and rate calibration.** Ported gate:
  `exported = min(FlgM, round(n_flagella·0.1·dt))`, the anti-sigma-factor
  negative feedback that releases free FliA as FlgM is exported — flgM is
  itself Class III, so rising free FliA drives more FlgM, which
  re-sequesters FliA, bounding free sigma28 (Stefan et al. 2015). Confirmed
  to keep free FliA bounded within a generation and across division. Trigger
  switched 2026-08-27 from `count(CPLX0-7452)` (complete flagellum) to
  `count(nascent_flagellum)` (HBB completion alone) — the real,
  literature-confirmed trigger point (Hughes et al. 1993; Karlinsey et al.
  2000), fixed after confirming the old trigger let free FliA run
  essentially unchecked (250→17,000 over 90 min while complete flagella
  barely grew 4→6). Rate: `secretion_rate=0.1/HBB/s` was an uncited
  zero-order placeholder, scaling export with completed-HBB count — an
  assumption never supported by real data. Fixed 2026-09-02: Karlinsey et
  al. 1998's pulse-chase (7.3-min FlgM half-life when HBB-complete, no
  detectable turnover when incomplete) is a first-order turnover rate
  constant, not a per-channel capacity — converted to `turnover_rate_per_s
  = ln(2)/438s ≈ 0.00158/s`, gated on `hbb_count > 0`, no further per-HBB
  multiplier. Caught two real bugs along the way: the FlgM-secretion unit
  tests had been silently broken since the trigger fix (`KeyError` on all
  5, stale fixtures — rewrote against the real trigger), and regenerating
  the cache surfaced a real infra gap where FliS:FliC's exact-equilibrium
  Step never had a real `get_config_by_name` entry in `LoadSimData` (only
  ever existed via a one-off cache hand-patch) — fixed with a real getter.
  Batch-validated across 6 seeds, 2026-09-03, no crashes. **Current
  mechanism — see §1.1.**

- **FlgM:FliA equilibrium (Kd) — three attempts, still unresolved.** Current:
  relaxed Kd=2×10⁻⁷ M on the shared, general numerical ODE solver, vs. the
  real 1.8×10⁻¹⁰ M (Chadsey et al. 1998) — a known, deliberate relaxation,
  not yet safely replaced.
  - *Attempt 1 (2026-08, division-crash investigation):* scaled the
    reaction's rate constants down 1000x, then another 100x (Kd unchanged
    both times), to stop division-triggered equilibrium-solver crashes.
    Both failed (crashed at ~1.8×10⁻¹⁰ M and again at ~2×10⁻⁸ M) — wrong
    knob. The real problem, later root-caused via FliS:FliC (§3.2), was the
    shared solver's absolute tolerance being far larger than one real
    molecule's concentration at this scale; rate-constant scaling never
    touches that.
  - *Attempt 2 (2026-09-01):* built a dedicated exact closed-form
    equilibrium Step (`flagella_flgm_flia_equilibrium.py`), mirroring
    FliS:FliC, using the real Kd. Hit a real infra gap: the shared solver
    reads pre-baked `rates_fwd`/`rates_rev` from `sim_data_cache.dill`'s
    cached config, not `simData.cPickle`, at runtime — zeroing only the
    cPickle left the baked copy live; fixed by patching both. Once fixed,
    the Step ran a full 92-min/2-generation test cleanly, no crash — but
    was **reverted the same day**: the resulting population dynamics were
    judged not correct on review, specifics not pinned down. Step kept on
    disk, unwired, for future reference.
  - *Re-examination (2026-09-03, analytical only, not implemented):*
    checked against one real data point (relaxed-Kd control run, t=2400s:
    free FliA=6551, FlgM=87, FLGM-FLIA-CPLX=2281) — confirms the relaxed
    Kd IS under-sequestering FliA as expected (implied effective Kd~250
    molecules, matching the relaxed value, not the real ~0.16-molecule
    value). But applying the real Kd to this same snapshot's totals
    (FliA_tot=8832, FlgM_tot=2368) predicts free FliA would only drop to
    ~6464 — barely below 6551 — because by t=2400s FliA already far
    outnumbers FlgM, leaving too little FlgM to sequester much more even
    at perfect binding. Suggests the real Kd's practical effect
    concentrates early in a generation (t=0 reference: FliA=2487 vs
    FlgM=1496, a much closer ratio), not permanent suppression — but this
    is inferred from one late snapshot, not a real early-timepoint trace.
    The standalone reduced ODE model (flagella-06) that would have let
    this be checked cheaply, without touching the live WCM, was deleted by
    Maya (didn't like it) — no longer available. A same-day (2026-09-04)
    attempt to re-wire the exact-Kd Step for further testing was started
    and explicitly reverted per Maya's direction (discuss the approach
    first, not unilaterally) — confirmed clean revert.
  - **Status: parked.** Root cause of the "wrong-looking" dynamics never
    found — candidates not yet checked: same-tick ordering (right after
    FlgM secretion vs. the shared solver's earlier position), interaction
    with the downstream Class II/III regulation Step, or a real biological
    effect that just looks surprising against the relaxed-Kd baseline
    everyone's used to reading charts against. If revisited: a
    properly-instrumented single-cell diagnostic with a full per-tick
    trace of free FliA/FlgM/Y/Class III override from t=0, so a
    wrong-looking result has data behind it instead of reverting blind a
    third time.

- **FlhDC shutdown-mechanism attempts — four tried, none currently active.**
  - *Artificial nucleation cap:* a hard ceiling on flagella count, applied
    as a stopgap once the FlgM/FliA positive-feedback runaway was found.
    Worked (held count at exactly 10) but was an artificial patch, not a
    real mechanism — **fully removed** per Maya's explicit instruction
    ("i dont want the artificial cap at all").
  - *`flagella_nucleation_cap.py` (NFsim-side equivalent):* worked (held
    `max_flagella=10` for 36 of 42 min) but explicitly logged as less
    "textbook correct" than a real consumed-nucleation-site alternative;
    superseded by the FliT:FlhDC checkpoint attempt below, then by the
    NFsim migration.
  - *FliT:FlhDC checkpoint (2026-08-05/10):* FliT-dimer (released once FliD
    is exported) binds FlhD4C2 and enhances its degradation via a
    fast-equilibrium reduction (Utsey & Keener 2020). Real, biochemically
    confirmed mechanism *in Salmonella* (Yamamoto & Kutsukake 2006,
    pull-down/far-Western). Basal ClpXP turnover (Tomoyasu et al. 2003;
    Claret & Hughes 2000 rate) was bundled in since FlhD4C2 otherwise has
    no degradation pathway at all (`protein_degradation.py` only handles
    monomers, never assembled complexes). Confirmed necessary-but-not-
    sufficient before removal: degradation alone brought FlhDC from an
    unbounded 625 to a stable ~173-174, but flagella count kept climbing
    4→19 regardless, since the SUM-gate's `X=FlhDC/(K+FlhDC)` term stays
    deep in its saturated regime even at the lower level — the real
    runaway engine is the separate FlgM-secretion feedback loop above.
    Removed 2026-08-10: Albanna et al. 2018 directly tested a Δ*fliT*
    mutant in **E. coli MG1655 — this WCM's exact K-12 reference strain**
    — and found no significant phenotype there, vs. a clear effect in
    *Salmonella*. `FLIT-DIMER_RXN` (real, checkpoint-independent FliT
    homodimer biology) kept as a building block for a planned NFsim
    replacement. Archived at `archive/flit-flhdc-regulation-2026-08/`.
  - *YdiV (E. coli-native candidate, not yet implemented):* since FliT was
    found Salmonella-specific, Maya's revised direction is to check YdiV
    (an EAL-domain pseudo-phosphodiesterase that binds FlhD4C2 and targets
    it for ClpXP degradation — real and E. coli-native, unlike FliT)
    against the literature before building any NFsim rule around it. Not
    yet checked or implemented.
  - **Status: no FlhDC shutdown mechanism currently exists at all** — the
    single largest open gap in the cascade's regulatory story (§1.4 item 2).

### 3.2 Assembly cascade

- **Real flagellar assembly stoichiometry, cryo-EM/structural-literature
  grounded** — corrected from mostly-unsourced -1 placeholders across many
  sessions: FliC×20,000 (later reduced for practical reasons, see Filament
  elongation below), FlgK/FlgL×11, FliF×34, FliG/FliM/FliN (PMC10128058),
  FlgH/FlgI×26, MotA×55/MotB×22, FliP:FliQ:FliR 5:4:1 (Kuhlen et al. 2018,
  resolving a 1:1:1 placeholder both of Maya's spreadsheets and the
  vendored default had wrong), FlhA×9 (homononameric export-gate ring),
  FliH₁₂FliI₆FliJ₁. Checked independently against Maya's two spreadsheets
  AND primary structural literature (not spreadsheet-only) after finding
  the spreadsheets themselves had at least one unsourced error. FliO left
  at -1 (not changed) — see §3.5, its consumed-vs-scaffold role is a real
  open question. `complexation_reactions_modified.tsv`'s own
  `FLAGELLAR-MOTOR-COMPLEX_RXN` still lists the stator (MotA/MotB), stale
  relative to the hierarchy fix below — audited 2026-09-04, confirmed
  inert: skipped entirely by `RUNTIME_EXCLUDED_REACTIONS`, feeds neither
  ParCa's mass-balance nor the runtime simulation, which runs entirely on
  NFsim's own already-correct `.bngl` model. Left unfixed (dead weight, not
  a live bug).

- **Assembly hierarchy/ordering — three successive fixes.** Deterministic-
  Step era (2026-08-11): MS-ring (FliF) → C-ring (FliG/FliM/FliN) merged
  into one stage; C-ring → export apparatus fixed from two independent
  branches only merging at the very last step (a real ordering bug, not
  matching Minamino & Namba 2008 / Chevance & Hughes 2008) into the correct
  dependency; export apparatus + rod/L-ring/stator → motor complex → hook
  → nucleation. Same-tick SSA-vs-Step race fix (2026-08-11): after the
  hierarchy fix made export-apparatus assembly depend on a C-ring molecule
  that only existed transiently inside another Step's own tick, moved
  `CPLX0-7451_RXN` out of ordinary Gillespie SSA into its own deterministic
  Step. Fixed a cross-mechanism timing gap, not the real FlhA-driven
  scarcity underneath it — motor-complex pool went from monotonically
  draining to a stable 5-6 oscillation. NFsim-era fix (2026-08-27/28): the
  full C-ring→export apparatus→rod→P-ring→L-ring→hook chain, replacing an
  earlier version where hook formed independently from free FlgE alone and
  P-ring was skipped entirely. L-ring is the confirmed real trigger for the
  rod-to-hook transition (Cohen & Hughes 2014). Stator (MotA/MotB) removed
  from the structural-completion chain entirely — real biology confirms
  rotation, not structural completion, requires the stator; motAB mutants
  assemble structurally complete, paralyzed flagella. **Current — see
  §1.2.**

- **Filament elongation — law, rate, and target length.** Law:
  `dL/dt = a/(b+L)` (Renault et al. 2017), pulled out of Gillespie SSA into
  its own incremental Step specifically because FliC's real ×20,000
  stoichiometry caused a combinatorial propensity blowup in the generic
  complexation framework (root-caused via macOS `sample`+`strings` on the
  compiled solver: a coefficient of 60 was safe, one of 20,000 was not).
  NFsim's own rule-based network deliberately excludes filament growth for
  the same combinatorial reason (fliC=-5000 alone ballooned 237 rules to
  5,588) and defers to this Step. `target_length`, three successive
  reductions — 20,000 (real, PMC7696725-cited) → 10,000 (2026-08-10) →
  5,000 (2026-08-11, current) — each a real literature-range value, not an
  arbitrary override, chosen because minimum completion time scales ~L²:
  20,000 needs ~133 min (structurally impossible within one ~42 min
  generation); 10,000 needs ~35 min but still nearly fully drained free
  FliC (51,967→14) within one generation once the hierarchy fix increased
  throughput; 5,000 was Maya's explicit call ("10,000 is a lot right now"),
  planned to move back toward 10,000 once the network is fully validated
  post-NFsim migration — not a permanent value. `rate_a` re-derivation
  (2026-09-01): the prior 26,450 subunit²/s figure did not cleanly
  reproduce Renault et al.'s own dynamics when re-checked directly.
  Re-derived their equations from first principles (`a=βD/l`, `b=D/(kon·l)`,
  giving the identity `kon=a/b`) and anchored to their Fig 3 dataset
  specifically (kon≈27.09/s, six-color labeling, 291 filaments/1,276
  points — the stronger of their two datasets, vs. Fig 2's 33.35/s,
  triple-color) — giving `rate_a=15,556` (`rate_b=575` already matched Fig
  3, unchanged). Old value kept as a comment in both `flagella_filament_
  elongation.py` and `sim_data.py`.

- **Nucleation rate/timing — literature grounding plus two real bugs
  found.** Literature grounding: Sim et al. 2017 (*Sci Rep* 7:41189,
  *E. coli* RP437 chemostat data), ≈1.67–1.81×10⁻³/s depending on
  derivation, used consistently for both the deterministic nucleation Step
  and (from 2026-08-17) NFsim's own BNGL calibration — two independently-
  derived numbers from the same paper agreeing within 8%. Fixed-interval
  rate-limiting pattern (`next_update_time`-based, not a per-tick
  probability): built after discovering `round(nucleation_rate*timestep)`
  silently rounds to 0 forever at small per-tick probabilities; reused
  directly for NFsim's own firing cadence and every other rate-limited
  flagella Step since. First-tick "free event" bug (2026-08-11):
  `next_update_time` defaults to 0.0, and `update_condition` fires whenever
  `next_update_time <= global_time` — true at t=0 — so the deliberately-
  rare nucleation Step fired immediately on its first call, skipping the
  intended ~600s wait entirely. Fixed by special-casing the first call;
  verified pre/post (first flagellum pre-fix already 2,016 subunits grown
  at t≈120s; post-fix, `n_nascent` correctly stayed 0 until t=600s).
  NFsim-side calibration: first suppressed via a fixed ratio to `k_bind`
  (fixed a first-real-run failure where new scaffolds nucleated faster
  than any existing one could finish — first-ever real completions
  confirmed at realistic multi-hour timescales). Replaced 2026-08-17 with
  a rate scaled per-reaction by each nucleating species' own real WCM
  ambient count (C-ring/FliF 657, hook/FlgE 3508, flhDC/FlhC 649) — caught
  and generalized a real bug in the same pass where only C-ring had been
  special-cased, leaving hook vulnerable to the identical "many parallel
  scaffolds, none finish" failure (confirmed: 1,226 stuck scaffolds in one
  chunk).

- **FliD consumption — double-counted, fixed, now unit-tested.** Found
  once NFsim was enabled: two separate consumption points fired for the
  same real flagellum, NFsim's own flagellum reaction (5× FliD, at
  hook-basal-body completion) and `flagella_filament_elongation.py`'s
  completion event (5× FliD, at filament completion) — 10 FliD consumed
  per completed flagellum instead of the real 5. Real biology: FliD caps
  the growing tip once, before elongation begins, and cannot accept
  another monomer once capped (Song et al. 2017; corroborated by Postel et
  al. 2020 cryo-EM). Fixed 2026-08-21: kept the consumption on
  `flagella_filament_elongation.py`'s side (the old deterministic
  pipeline's only FliD accounting point, still the default feature) and
  removed it from NFsim's own flagellum reaction (`generate_flagella_
  bngl.py`, old line kept as a comment; `.bngl` regenerated). Verified
  2026-09-04 via a direct unit test
  (`tests/test_flagella_filament_elongation.py`): one completion consumes
  exactly -5 FliD, two simultaneous completions -10 (not -20, the actual
  shape of the original bug), no completion leaves FliD untouched.
  **Confirmed intact — §1.4 item 8.**

- **FliS:FliC equilibrium (Kd) — exact-solve fix, then a real elongation-
  side bug found validating it.** Built 2026-08-28
  (`flagella_flis_flic_equilibrium.py`) as a direct fix for a
  division-triggered equilibrium-solver crash that hit twice, confirmed
  via diagnostic printout to be this exact reaction both times.
  Root-caused: the shared numerical ODE solver's default `atol` (1e-6 M)
  is >600x the concentration of a single real molecule in this cell's real
  volume at the post-division scale involved — this is what the two failed
  FlgM:FliA rate-slowdown attempts above had also been fighting, without
  knowing it. A simple 1:1:1 binding reaction has an exact quadratic
  solution; this Step computes it directly, no tolerance to get wrong, no
  possible overshoot. Stoichiometry corrected 2026-08-31: FliS binds FliC
  as a homodimer, not a single monomer (Auvray et al. 2001; consistent
  with Sajó et al. 2016's FliS structural-plasticity work never describing
  a loose monomer/dimer equilibrium) — dimer count read directly off the
  raw monomer bulk count (`free_fliS // 2`), each unit of complex now
  moving 2 raw FliS monomers, fixed consistently in both this Step and
  `flagella_filament_elongation.py`'s chaperone-release logic. Kd itself
  (5.26×10⁻⁸ M, Muskotal et al. 2006) unchanged, now understood as
  dimer:FliC not monomer:FliC. Population-scale validation (2026-08-30)
  first caught running against the wrong (pre-fix) cache, reproducing the
  old crash-prone behavior — not a new bug, just the wrong cache. Re-run
  against the correct cache confirmed the fix holds across real division
  events, but surfaced sharp crash-and-recover swings in FLIS-FLIC-CPLX
  right after division — traced to `flagella_filament_elongation.py`'s
  `MAX_COMPLEX_DRAW_FRACTION = 0.3` cap (added 2026-08-21 to protect the
  shared solver's legacy ODE path, which no longer handles this reaction
  at all now that the exact-solve Step exists). Removed the cap — produced
  a single clean step-response per division instead of a multi-cycle
  fight. Separately confirmed (finer `--sample 10` reporting) that
  individual filaments do reach the real 5,000-subunit target — the
  earlier appearance of "never completing" was a 120s report-sampling
  artifact, not a real stall. **Batch-validated across 6 seeds,
  2026-09-03 — §1.4 item 1.**

- **NFsim migration — porting, infra fixes, and scoping.** Ported
  2026-08-12 from `pbg-nfsim`'s own generic bundled example (unsourced
  placeholders from an unrelated codebase) into this study's own `models/`
  directory, restoichiometried to exactly match v2ecoli's own reaction
  network, per Maya's explicit direction ("I want it to be coupled to the
  whole cell model"). `pbg-nfsim` itself reverted to supplying only the
  runtime engine. Scaffold persistence fix: root-caused why
  `flagellar_hook` (120 sequential binds) never completed across any
  chunked run while shorter-chain stages did — `NFSimProcess` discards all
  in-progress scaffold state between chunk invocations by design. Patched
  to persist via BNG's own `.species` file; verified end-to-end (isolated
  hook test: 0→4 scaffolds persisting, completed at chunk 32), PR'd
  upstream (`pbg-nfsim`, viva-nfsim#2). Real-bulk-ID species renaming
  (2026-08-12): renamed directly to real v2ecoli EcoCyc bulk IDs so
  NFsim's observables ARE the bulk array's own names, mass conservation
  exact by construction — caught two real discrepancies in passing:
  FlhC's real bulk ID is `MONOMER0-2488[c]`, not `EG10319-MONOMER[c]`;
  and the canonical motor-complex spec includes FlgI but the deterministic
  Step never actually consumed it (flagged, not fixed). Wired into
  `ecoli_baseline.py` 2026-08-16 as a feature module mutually exclusive
  with the deterministic pipeline (A/B comparable by construction) —
  confirmed correct end-to-end but markedly slower to reach completions
  within the same window, an honest speed/calibration gap, not a
  correctness bug. Four deterministic assembly Steps
  (`flagella_motor_switch_assembly.py`, `flagella_export_apparatus_
  assembly.py`, `flagella_motor_complex_assembly.py`, `flagella_filament_
  nucleation.py`) archived 2026-08-21 — real, correctly cross-checked
  biology, removed only because Maya committed to NFsim as the one path
  forward, making two parallel pipelines pure maintenance surface.
  `flagella_filament_elongation.py`, `flagella_flgm_secretion.py`, and
  `flagella_transcription_regulation.py` were NOT archived — shared infra
  reused as-is by NFsim. Archived at `archive/deterministic-flagella-
  assembly-2026-08/`. Cumulative ("ever formed") tracker fix (2026-09-01):
  population charts showed a flat 0 for several intermediate stages
  despite real completions happening — `pbg-nfsim`'s net-delta computation
  (`final − initial` per chunk) is blind to any species produced then
  fully consumed within the same chunk. Fixed with a separate,
  non-destructive `gross_positive_deltas` computation for diagnostic-only
  `__cumulative` keys, leaving the real net-delta values (fed back into
  NFsim's next firing) untouched. Increasing `n_steps` (50→500) closed
  most of the remaining gap; Hook's own gap left unchased (Maya's call —
  display-only). **Open scoping question:** once fully WCM-coupled, should
  NFsim split into its own investigation with its own acceptance criteria
  (rule-based vs. Gillespie complexation as a distinct methods thread)
  rather than staying inside flagella-cascade as Aim 2B? Not yet decided.

- **Bottleneck/supply-chain analysis.** FlhA→FliN (2026-08-11): overturned
  an earlier same-session finding that FlhA was scarce (1–8 copies, drawn
  down by ungated SSA firing) — after the hierarchy fix above, FlhA held
  steady (105–190 copies) because `CPLX0-7451_RXN` now only fires when
  gated by `CPLX0-7450`, itself gated by FliN. The real bottleneck moved
  to FliN: co-transcribed with FliM on the same operon (`fliLMNOPQR`,
  TU0-1441, lockstep production), but FliN costs 111 copies/event vs.
  FliM's 34, so each event drains FliN's pool a proportionally bigger bite
  (observed 10–110 vs. FliM's 430–660). Recommendation: leave FliN alone —
  co-transcription rules out a synthesis-rate error, stoichiometry is
  already cited (111±13, PMC10128058), and the motor-complex pool it feeds
  stayed healthy (5–7) throughout. Broader question, still unresolved:
  across every diagnostic this investigation, something else (FliC supply,
  then motor-complex supply, then division/mass-homeostasis corruption,
  §3.3) always capped flagella count before the FliT/FlhDC checkpoint
  mechanism (§3.1) was ever directly observed to be binding. Whether the
  checkpoint (or its successor) would ever actually become the real
  limiting factor, or whether the system is bottlenecked layer-by-layer
  before that regardless, is unresolved. Parked at Maya's explicit call
  (2026-08-08).

### 3.3 Division & multi-generation testing

- **Division mechanics.** Binomial splitting (`divide_nascent_flagellum`,
  `v2ecoli/library/division.py`): in-progress and complete flagella split
  binomially between daughters, consistent with direct Salmonella
  live-imaging evidence (Aizawa & Kubori 1998) that division "forcefully
  splits the number of existing filaments into half." Verified 2026-08-06
  (fixed a real bug where `nascent_flagellum` was duplicated, not split),
  re-verified more rigorously 2026-08-11 against `unique_index` identity
  (not just `filament_length` values that could coincidentally collide) —
  exact partition confirmed, zero loss, zero duplication. Extended to 3
  real generations 2026-08-12, all passed. Opt-in `features=` threading
  through daughter rebuild: a general division-boundary bug where opt-in
  feature lists (e.g. `flagella_regulation`) were silently dropped when a
  daughter's composite was rebuilt post-division, raising unrelated-
  looking `ValueError`s. Fixed by threading `features=` through the
  rebuild path. **Current — see §1.3.**

- **Multi-generation testing methodology, and the division/mass-
  homeostasis bug it surfaced.** Manual multi-generation state-splicing
  (early approach): hand-splicing simulation state between runs, replaced
  after it produced its own real, separate bug (a dry-mass drift artifact
  of the splicing method itself, 706.7→262.0 fg over 7 generations)
  initially mistaken for a model bug. Replaced by driving real generations
  through the actual `Division` machinery instead — which is what surfaced
  the real (not diagnostic-artifact) bug below. A separate stdout-log-
  parsing timing feature was also removed after reporting an implausible
  ~5-8 min generation time, traced to a stale buffer bug in the parser
  itself, not the simulation. **Division/mass-homeostasis corruption** — a
  general (non-flagella-specific) WCM limitation: dry mass drifts down
  generation-over-generation past ~3-4 generations, with growing
  `GLP_NOFEAS` infeasibility flooding. Most likely cause (not confirmed as
  fixable): `division.py`'s D-period timer fires on a fixed duration
  regardless of mass threshold, producing underweight daughters FBA can't
  feed, and `allocator.py`'s negative-pool handling doesn't heal by design,
  so the error compounds across generations. Documented as a known
  limitation; Maya's explicit call was to document and not attempt a fix
  in-session, since it's outside flagella-specific code. **Blocks any
  multi-generation flagella-completion test past ~3 clean generations
  (~2hr)** — the original "does an inherited, in-progress flagellum
  complete across real generations" question this investigation has been
  chasing is therefore still not fully answered either way.

### 3.4 Infrastructure & reproducibility

- **General framework fixes (non-flagella-specific, found during flagella
  diagnostics).** `RUNTIME_EXCLUDED_REACTIONS` (`complexation.py`): the
  general mechanism for pulling numerically-dangerous reactions out of
  Gillespie SSA into dedicated deterministic/exact Steps. Used for every
  flagella structural-assembly reaction since 2026-08-06; the FliS:FliC
  (and originally-attempted FlgM:FliA) exact-solve Steps reuse the same
  pattern by zeroing the shared solver's rates for that reaction rather
  than deleting the row. Scale-aware mass-balance tolerance
  (`atol=1e-8 + rtol=1e-13·max_term_magnitude`, `complexation.py`): fixes
  floating-point mass-balance false-positives at large stoichiometric
  coefficients.

- **FliC synthesis-rate 10x override.** Measured real-time FliC synthesis
  (0.71→6.0 molecules/s) and applied a 10x transcription-rate override via
  ParCa's `adjust_rna_expression` mechanism, deliberately NOT closing the
  full ~65x gap to Renault-implied demand (≈46/s, using the pre-2026-09-01
  `rate_a`) since that would require ~13.5% of the total mRNA budget going
  to one gene; chose ~2.3% instead. (Demand figure corrected 2026-09-02
  alongside the `rate_a` re-derivation above — see the `rna_expression_
  adjustments.tsv` comment for the updated ~38.6x/~8.9%-budget math; the
  10x override itself is unchanged.)

- **Reproducibility & seeding — one fixed, one still open.** NFsim
  internal random-seed pinning (2026-09-01, resolved): NFsim's own BNGL
  `simulate` call previously drew its own internal seed independent of
  this codebase's outer `--seed`, breaking reproducibility of any run
  using the NFsim feature. Fixed across two repositories: this repo now
  derives and passes a `seed` config value (via the existing
  `_derive_process_seed` CRC32 convention already used by every other
  process); the sibling package `pbg-nfsim` now accepts it and increments
  it once per chunk firing. Verified byte-identical trajectories across a
  full 92-minute, multi-generation run at a fixed seed. **Same-seed
  reproducibility bug (2026-09-03, NOT resolved):**
  `run_nfsim_population_multigen.py --seed 0` should produce a
  byte-identical trajectory regardless of `--generations`/`--seconds-cap`/
  `--max-agents` (all three are pure loop-exit comparisons, never passed
  into `build_composite`, `comp.run`, or any seed derivation). Empirically
  false: bisected by holding two of the three fixed and varying one at a
  time against a common baseline. Result: `--seconds-cap` alone changes
  the trajectory; `--max-agents` alone changes it back; `--generations`
  alone has no effect — not a clean single-parameter culprit.
  `_derive_process_seed` checked and ruled out (only hashes
  `master_seed, process_name`, no CLI args). **Root cause not found** —
  parked rather than continuing via more black-box comparisons; would need
  instrumenting the composite build itself (logging a fingerprint of every
  derived seed/config at build time, diffing between two runs) to
  localize it. **Practical implication:** any same-seed "before/after"
  comparison in this document that didn't hold all three of those
  parameters fixed between the two runs being compared should be treated
  with reduced confidence — this threatens same-seed *reruns* specifically
  (e.g. if an earlier before/after test varied generation/time/agent-count
  targets between runs), not the confirmed multi-seed-*value* batch
  validations (different actual `--seed` integers) cited throughout this
  document, which don't depend on re-running one seed twice.

- **Repo synced with `origin/main`, 189 commits (2026-09-04).** 3 real
  merge conflicts: `tests/test_bundle_content_pins.py`'s content-hash
  pins needed recomputing against the actual merged TSVs, not either
  side's stale value; two additive, non-overlapping daughter-rebuild
  features in `_helpers.py`/`division.py` (kept both). Also fixed a stale
  `pbg_superpowers` import in this investigation's own `flagella_nfsim_
  assembly.py` (org-wide pbg→viva rename). Cache version-fingerprint went
  stale as expected (main touched `core.py`/`sim_data.py`) — fixed with
  the same fast `save_cache()` regeneration used throughout this
  investigation, no ParCa rebuild.

### 3.5 Still parked / out of scope

- **FliO's structural role in the export apparatus** — real literature
  describes FliO as a transient assembly scaffold, not part of the final
  mature complex (a Δ*fliO* mutant is rescued to wild-type motility by
  FliP overexpression alone), but it is currently modeled as an ordinary
  consumed reactant (-1) in the reaction forming the final complex.
  Maya's explicit call: leave as-is, since a copy-number fix doesn't
  address a role question; defer to being represented properly once
  NFsim rules can express a non-consumed scaffold relationship.
- **Report-study visual differentiation** — Maya's request for more
  detail / bigger font to differentiate studies in the generated report is
  a vivarium-workbench report-template change, out of scope for this
  investigation's own code; logged, not actioned.
- **Chart 04 (bursts-are-division-transients)** — the transient bursts at
  division boundaries in the multigen mean-override trace are hypothesized
  to be a re-initialization transient (pools halve, gate re-settles over a
  few ticks), but this has not been directly confirmed against logged
  division times — status remains `untested`.

---

## 4. References

- Aizawa SI & Kubori T (1998). Bacterial flagellation and cell division.
  *Genes to Cells* 3:625-634. (Direct Salmonella live dark-field imaging of
  flagellar filament number through real division events — binomial
  splitting.)
- Albanna A, Sim M, Hoskisson PA, Gillespie C, Rao CV, Aldridge PD (2018).
  Driving the expression of the *Salmonella enterica* sv Typhimurium
  flagellum using *flhDC* from *Escherichia coli* results in key regulatory
  and cellular differences. *Sci Rep* 8:16705.
  https://doi.org/10.1038/s41598-018-35005-2
- Chang YR, Sung YS & Hong DF (2025). Intrinsic clustering of flagellar
  basal body proteins in *Escherichia coli*. *Biochem Biophys Reports*
  42:102051. (Rationale for rate-limited nucleation: existing structures
  preferentially absorb material over nucleating new ones.)
- Chadsey MS, Karlinsey JE & Hughes KT (1998). The flagellar anti-sigma
  factor FlgM actively dissociates *Salmonella typhimurium* sigma28 RNA
  polymerase holoenzyme. *Genes Dev* 12:3123-3136.
- Chevance FFV & Hughes KT (2008). Coordinating assembly of a bacterial
  macromolecular machine. *Nat Rev Microbiol* 6:455-465.
- Claret L & Hughes C (2000). Rapid turnover of FlhD and FlhC, the
  flagellar regulon transcriptional activator proteins, during *Proteus*
  swarming. *J Bacteriol* 182:833-836. (FlhD/FlhC half-life estimate, *Proteus
  mirabilis*, used for the basal FlhD4C2 degradation rate. Corrected
  2026-09-04 — previously cited under this title with the wrong co-author
  (Hughes KT instead of Colin Hughes); verified against CrossRef.)
- Cohen EJ & Hughes KT (2014). Rod-to-hook transition for extracellular
  flagellum assembly is catalyzed by the L-ring-dependent rod scaffold
  removal. *J Bacteriol* 196:2387-2395.
- Fukumura T et al. (2017). Assembly and stoichiometry of the core
  structure of the bacterial flagellar type III export gate complex.
  *PLOS Biol* 15:e2002281.
- Kalir S & Alon U (2004). Using a quantitative blueprint to reprogram the
  dynamics of the flagella gene network. *Cell* 117:713-720. (Class I/II/III
  regulatory hierarchy and the SUM-gate promoter model.)
- Karlinsey JE, Tsui HC, Winkler ME & Hughes KT (1998). Flk Couples flgM
  Translation to Flagellar Ring Assembly in Salmonella typhimurium.
  *J Bacteriol* 180:5384-5397. (Strain TH2592, pulse-chase; FlgM
  half-life 7.3 min when HBB-complete, no detectable turnover when
  HBB-incomplete. Corrected 2026-09-02 — previously mis-cited under this
  entry's year/journal but with the title/authors of the 2000 paper below;
  see §2's citation-correction note.)
- Karlinsey JE, Tanaka S, Bettenworth V, Yamaguchi S, Boos W, Aizawa SI &
  Hughes KT (2000). Completion of the hook-basal body complex of the
  Salmonella typhimurium flagellum is coupled to FlgM secretion and fliC
  transcription. *Mol Microbiol* 37:1220-1231.
- Kuhlen L et al. (2018). Structure of the core of the type III secretion
  system export apparatus. *Nat Struct Mol Biol* 25:583-590. (Cryo-EM,
  building on Fukumura et al. 2017; FliP:FliQ:FliR 5:4:1.)
- McMurry JL, Minamino T, Furukawa Y, Francis JW, Hill SA, Helms KA &
  Namba K (2015). Weak interactions between Salmonella enterica FlhB and
  other flagellar export apparatus proteins govern type III secretion
  dynamics. *PLOS ONE* 10(8):e0134884. (FlhA:FlhB association rate
  constant, biosensor data, used both directly and as the literature-
  grounded proxy default for other flagellar binding reactions lacking
  their own measured kinetics. Corrected 2026-09-04 — the DOI was already
  right, but was previously attached to a different, unrelated 2004 J
  Bacteriol paper's title/author list; verified against CrossRef.)
- Minamino T & Namba K (2008). Distinct roles of the FliI ATPase and
  proton motive force in bacterial flagellar protein export. *Nature*
  451:485-488. (Real assembly order: MS-ring/C-ring before export
  apparatus before rod/hook.)
- Muskotal A, Kiraly R, Sebestyen A, Gugolya Z, Vegh BM & Vonderviszt F
  (2006). Interaction of FliS chaperone with a flagellin-specific export
  signal. *FEBS Lett* 580:3916-3920. (Isothermal titration calorimetry,
  Ka=1.9×10⁷ M⁻¹, 1:1 FliS:FliC stoichiometry.)
- Nagar N et al. (2022). Pulsed-SILAC proteomics reveals flagellin
  turnover kinetics. (FliC degradation half-life, unprotected: 7.4 min.)
  **UNVERIFIED (2026-09-04)** — this entry has no journal/volume/page even
  in its original form, and multiple CrossRef search strategies found no
  matching paper. Needs a direct check (e.g. PubMed/Google Scholar with
  tools not available here) before the 7.4-min figure is trusted further.
- Renault TT, Abraham AO, Bergmiller T, Paradis G, Rainville S, Charpentier
  E, Guet CC, Tu Y, Namba K, Keener JP, Minamino T & Erhardt M (2017).
  Bacterial flagella grow through an injection-diffusion mechanism. *eLife*
  6:e23136. (`dL/dt = a/(b+L)` filament growth law, a≈26,450, b≈575.)
- Postel S, Deng Z, Xu C, Sun S & Zhou J (2020). Cryo-EM structure of the
  bacterial flagellar filament tip and its molecular role in dual filament
  formation. *Nat Commun* 11:1965. (Cryo-EM mechanism for the FliD
  pentameric cap; corroborates single-cap-event behavior.) **UNVERIFIED
  (2026-09-04)** — no CrossRef match for this exact author list. The
  closest topical match (bacterial flagellum cap complex cryo-EM, same
  journal, same year) is a real paper by a completely different author
  list (Al-Otaibi, Taylor, Farrell, Tzokov, DiMaio, Kelly & Bergeron,
  *Nat Commun* 11:3210) — plausibly the actually-intended source with its
  citation badly garbled, or a genuinely separate paper this search
  couldn't surface. Needs direct verification before further use.
- Sim M et al. (2017). Growth rate control of flagellar assembly in
  *Escherichia coli* strain RP437. *Sci Rep* 7:41189. (Chemostat data;
  nucleation rate ≈1.67-1.81×10⁻³/s, and 7.8 flagella/cell at 1.2 hr
  doubling used as the direct back-calculation source for the same rate.)
- Song WS, Cho SY, Hong HJ, Park SC & Yoon SI (2017). Self-oligomerizing
  structure of the flagellar cap protein FliD and its implication in
  filament assembly. *J Mol Biol* 429:847-857. (FliD pentameric cap forms
  once, before elongation, and cannot accept a second binding event once
  complete. Corrected 2026-09-04 — previously cited with a paraphrased
  title, a fabricated co-author ("Jung HW"), and the wrong end-page (861);
  verified against CrossRef.)
- Stefan D et al. (2015). FlgM/FliA feedback loop and bounded free sigma28.
  *PLoS Comput Biol* 11:e1004028. **UNVERIFIED (2026-09-04)** — three
  CrossRef search strategies, including a direct DOI-pattern guess, found
  no matching paper; they did find a real, different Stefan D 2015 PLoS
  Comput Biol paper (bacterial promoter modeling from time-series data,
  unrelated topic). This citation actively supports a real claim in §3.1
  (free FliA bounded, not runaway) — needs direct verification before that
  claim leans on it further.
- Tomoyasu T, Takaya A, Isogai E & Yamamoto T (2003). Turnover of FlhD
  and FlhC, master regulator proteins for *Salmonella* flagellum
  biogenesis, by the ATP-dependent ClpXP protease. *Mol Microbiol*
  48:443-452. (ClpXP degrades assembled FlhD4C2, not free subunits.
  Corrected 2026-09-04 — previously cited with the title and 10-author
  list of a different, real 2002 J Bacteriol Tomoyasu paper attached to
  this one's correct journal/volume/pages; verified against CrossRef.)
- Utsey B & Keener JP (2020). A mathematical model for regulation of the
  MS ring switch complex assembly in bacterial flagellum biogenesis. *PLOS
  Comput Biol* 16:e1007689. (Fast-equilibrium reduction used for the
  FliT:FlhDC checkpoint's math.)
- Yamamoto S & Kutsukake K (2006). FliT acts as an anti-FlhD2C2 factor in
  the transcriptional control of the flagellar regulon in *Salmonella
  enterica* serovar Typhimurium. *J Bacteriol* 188:6703-6708.
  https://pubmed.ncbi.nlm.nih.gov/16952964/ (Pages corrected 2026-09-04 —
  previously cited as 5124-5131; PubMed record for this same PMID confirms
  6703-6708.)
- PMC7696725 — cryo-EM structural source for FliC subunit copy number
  (~20,000) and the 20,000-40,000 subunit / 5-20 μm real filament-length
  range.
- PMC10128058 — cryo-EM structural source for FliG/FliM/FliN switch-complex
  copy numbers.
