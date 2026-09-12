#!/usr/bin/env python
"""Route-B runner for genotype-06-trpr-regulon — multigeneration translation-level KO.

Three arms, each built from a cache so the perturbation survives cell division:

    wt          out/cache                        (unperturbed)
    trpR        out/genotype-06/cache-trpR       (EG11029)
    trpRtnaA    out/genotype-06/cache-trpRtnaA   (EG11029 + EG11005)

⚠ WHY CACHES AND NOT `knockouts=`. Measured 2026-08-21: a build-time
`knockouts=` argument is applied to the mother and LOST in both daughters
(`translation_efficiencies[3834]`: mother 0.0, daughters 2.4992e-04), because
`ecoli_baseline.py:1414-1415` compiles it into `config_overrides` and
`division.py:373-376` does not thread that to the daughter rebuild. The same
measurement against a cache-baked knockout gives 0.0 in BOTH daughters. v2ecoli#505
fixes the argument path; this runner does not depend on it.

⚠ WHY MULTIGENERATION. A translation-level knockout of a repressor is not
instantaneous: zeroing translation stops synthesis, but the existing TrpR pool
does not decay appreciably — measured flat at 135 copies across 300 steps while
the WT arm's grew. Clearance is by growth dilution and division partitioning, so
derepression can only appear over generations. At a 44 min doubling (2640 steps
at 1 s) a sub-generation run shows nothing, which is a property of the lever and
not a failed experiment.

Readouts are recorded as full traces; band windows are computed downstream so the
window rule can be changed without re-running. Nothing here grades anything.

Run from the workspace root:
    python workspace/studies/genotype-06-trpr-regulon/sims/run_route_b.py --arm wt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[4]

# ⚠ These index INTO the listener arrays, and a listener array carries no ids. They
# were resolved once against the old ParCa fixture; a new ecoli-sources rev adds rows
# and could shift either, which would silently repoint the readout at a different
# molecule while every number still looked plausible. So they are DECLARED here and
# CHECKED against the arm's own cache before the run (`resolve_indices`) rather than
# trusted. Measured 2026-08-25: both hold at ecoli-sources 840bc973 (3277 TUs, 4309
# monomers, unchanged by basal_with_trp) -- but the check, not that measurement, is
# what makes a future run safe.
TU_TRP = 2635          # TU00067 = trpLEDCBA (all six genes, one TU)
MONO_TRPR = 3834       # PD00423[c]
TU_TRP_ID = "TU00067[c]"
MONO_TRPR_ID = "PD00423[c]"
MONO_TNAA_HINT = "TRYPTOPHAN-MONOMER[c]"

# ⭑ INSTRUMENTATION, added 2026-08-25 because the first pass could not answer the
# question it raised. The original trace recorded `monomer_counts[MONO_TRPR]` --
# which is PD00423[c], the APO monomer -- and called it "TrpR". The species that
# actually binds DNA is the holorepressor CPLX-125[c] (TrpR·L-tryptophan), which was
# never recorded, so the run showed a duty cycle changing with the medium and carried
# no way to say why. Watching the wrong species is not a small error: it makes every
# mechanistic reading a conjecture.
#
# What is added, and what each one settles:
#   CPLX-125[c]  the ACTIVE holorepressor -- does the corepressor pool track the medium?
#   PD00423[c]   the APO monomer -- separates "no TrpR" from "TrpR not loaded"
#   TRP[c]       intracellular tryptophan -- the corepressor's own availability
#   dry mass     counts are not concentrations, and a 25-min cell is bigger than a
#                44-min one; without this the pool comparisons are uninterpretable
#   p_promoter_bound[trpR]  ⭑ the RUNTIME binding probability, recomputed each tick
#                by tf_binding.py:399 from the active/inactive counts. ParCa's FITTED
#                value is a per-condition constant (0.9443 for both with_aa and
#                basal_with_trp, identical to 16 digits), so this is the one readout
#                that can show the run and the parameterization disagreeing.
#   n_actual_bound / n_available_promoters  binding is capped at the number of active
#                TF molecules, and there are only tens of them -- these say whether
#                the switch is stochastic small-number behaviour or a saturated bound.
TF_TRPR_ID = "CPLX-125"
BULK_READOUTS = {
    "trpR_holo": "CPLX-125[c]",
    "trpR_apo": "PD00423[c]",
    "trp": "TRP[c]",
}

ARMS = {
    "wt":       ("out/cache",                      "unperturbed reference"),
    "trpR":     ("out/genotype-06/cache-trpR",     "EG11029 translation zeroed"),
    "trpRtnaA": ("out/genotype-06/cache-trpRtnaA", "EG11029 + EG11005 zeroed"),
    # AA-replete pair. `minimal` supplies TRP=0.0, `minimal_plus_amino_acids`
    # supplies TRP=0.1 — so these are the arms in which TrpR has its
    # corepressor available and can actually repress. The basal pair above
    # cannot show derepression if there was nothing to derepress, which is the
    # ambiguity these exist to resolve. ⚠ Blunt instrument: all 20 amino acids
    # are supplied (30 species differ from minimal) and the doubling time drops
    # 44 -> 25 min. The condition-level effects hit BOTH arms, so the
    # within-condition WT/KO ratio is still the comparison.
    "wt_withaa":   ("out/genotype-06/cache-wt-withaa",   "unperturbed, AA-replete"),
    "trpR_withaa": ("out/genotype-06/cache-trpR-withaa", "EG11029 zeroed, AA-replete"),
    # TRYPTOPHAN-ONLY pair -- what the AA-replete pair above was a stand-in for.
    # `basal_with_trp` (ecoli-sources 840bc973) is `minimal_plus_tryptophan` at
    # basal's OWN 44.0 min doubling with active TFs ["CPLX-125"], so TrpR has its
    # corepressor and nothing else moves: one nutrient differs from basal instead of
    # thirty, and the doubling time is unchanged instead of nearly halved. This is
    # the clean version of the with_aa result, and the reason the with_aa arms keep
    # their caveat rather than being deleted -- they are the same question asked with
    # a blunter instrument, and the two answers together say whether the blunt one
    # was measuring tryptophan or measuring growth rate.
    # ⚠ NOT reachable from the committed fixture: it carries 51 conditions and lacks
    # this one. These arms need a full-mode ParCa build at pin 840bc973 (~170 s at
    # --cpus 8), which is why their caches are not reproducible from the repo alone.
    "wt_withtrp":   ("out/genotype-06/cache-wt-withtrp",   "unperturbed, +tryptophan"),
    "trpR_withtrp": ("out/genotype-06/cache-trpR-withtrp", "EG11029 zeroed, +tryptophan"),
    # ⭑ TRYPTOPHAN DOSE-RESPONSE, added 2026-08-25. All four share ONE full-mode ParCa
    # state and the SAME condition (basal_with_trp: 44.0 min, CPLX-125 the only declared
    # active TF), so the ParCa cell spec -- expression, doubling time, and the fitted
    # pPromoterBound of 0.944322344818007 -- is byte-identical across them. The single
    # variable is the external tryptophan the RUN sees:
    #     minimal                 TRP 0.0
    #     minimal_plus_trp_low    TRP 0.1   <- deliberately the SAME as with_aa
    #     minimal_plus_tryptophan TRP inf
    # ⭑ Why 0.1 and not some round number: with_aa delivers exactly 0.1 (5X_supplement_EZ
    # TRP 0.5 into 0.2 of 1.0 L), and with_aa is the arm whose 74%-repressed wild type
    # this exists to explain. At the same tryptophan but basal's doubling time and none
    # of the other nineteen amino acids, a 74% result means the CONCENTRATION explains
    # with_aa; a 100% result means something else in with_aa does.
    # ⚠ `minimal_plus_trp_low` is supplied by a LOCAL bundle override
    # (out/genotype-06/trp-dose-override/), not by ecoli-sources -- a data-only override,
    # no new media file and so no LIST_OF_DICT_FILENAMES entry. Rebuild with:
    #   v2ecoli-parca --mode full --cpus 8 -o <dir> --bundle-overrides <that manifest>
    # ⚠⚠ And the override's units are a trap: the `ingredients weight` column is GRAMS.
    # A first attempt wrote [0.1] there and got 0.4897 mM (0.1 g / 204.23 g/mol) -- it
    # validated, hashed and built cleanly, and was caught only by reading the resulting
    # media dict. The row uses the `ingredients counts (units.mmol)` column instead.
    "wt_trpzero":  ("out/genotype-06/cache-wt-trpzero",  "unperturbed, TRP 0.0"),
    "wt_trplow":   ("out/genotype-06/cache-wt-trplow",   "unperturbed, TRP 0.1 (= with_aa)"),
    "wt_trpinf":   ("out/genotype-06/cache-wt-trpinf",   "unperturbed, TRP inf"),
    "trpR_trpinf": ("out/genotype-06/cache-trpR-trpinf", "EG11029 zeroed, TRP inf"),
    # ⭑⭑ THE ARM THAT FACES THE DATA. The reference cultivation this study is
    # ultimately compared against is grown in a minimal glucose medium with NO
    # tryptophan supplement, so the comparison lives in the MINIMAL regime, not the
    # +trp one.
    # ⚠ And in stock `basal` the model predicts a trpR knockout does essentially
    # nothing (measured 1.06x), because tf_condition.tsv declares trpR active only in
    # minimal_plus_amino_acids and INACTIVE in minimal: there is no repression to
    # relieve. That is a declaration, not a mechanism -- wt_trpzero shows the operon
    # represses fine on self-made tryptophan once trpR is allowed to be active.
    # This arm is wt_trpzero's knockout partner, so the pair gives the KO/WT ratio in
    # the condition the experiment actually occupies.
    "trpR_trpzero": ("out/genotype-06/cache-trpR-trpzero", "EG11029 zeroed, TRP 0.0"),
}

GEN_STEPS = 2640       # one doubling at 44 min (basal), 1 s/step
GEN_STEPS_WITHAA = 1500  # one doubling at 25 min (with_aa)
SAMPLE_EVERY = 120


#: Populated by ``resolve_indices`` from the arm's own simData, keyed by cache dir.
#: NOT module constants: unlike TU_TRP / MONO_TRPR (which are declared and checked
#: because they were resolved once against an older fixture), these are resolved by
#: ID every run. A declared-and-checked constant is right for a value a reader needs
#: to see; a looked-up one is right for a value nobody has memorised.
_IDX: dict = {}


def resolve_indices(cache: str) -> None:
    """Verify the two declared indices, and resolve the instrumentation ones by ID.

    The listener and bulk arrays are positional and carry no ids, so a shifted index
    does not raise -- it reports a different molecule's number under the right label.
    TU_TRP / MONO_TRPR are DECLARED above and only checked here: this refuses to run
    rather than silently re-point them, because following a moved index would turn a
    re-run into a different measurement without saying so.

    Everything in ``_IDX`` is looked up by id instead, and a miss is fatal for the
    same reason -- a readout that quietly returns None reads as "the species was
    absent" when it means "I could not find it".
    """
    import pickle
    with open(Path(cache) / "simData.cPickle", "rb") as fh:
        sd = pickle.load(fh)
    rna_ids = [str(x) for x in sd.process.transcription.rna_data["id"]]
    mono_ids = [str(x) for x in sd.process.translation.monomer_data["id"]]
    problems = []
    if not (len(rna_ids) > TU_TRP and rna_ids[TU_TRP] == TU_TRP_ID):
        at = rna_ids.index(TU_TRP_ID) if TU_TRP_ID in rna_ids else None
        problems.append(f"TU_TRP={TU_TRP} names "
                        f"{rna_ids[TU_TRP] if len(rna_ids) > TU_TRP else 'OUT OF RANGE'}, "
                        f"not {TU_TRP_ID} (which is at {at})")
    if not (len(mono_ids) > MONO_TRPR and mono_ids[MONO_TRPR] == MONO_TRPR_ID):
        at = mono_ids.index(MONO_TRPR_ID) if MONO_TRPR_ID in mono_ids else None
        problems.append(f"MONO_TRPR={MONO_TRPR} names "
                        f"{mono_ids[MONO_TRPR] if len(mono_ids) > MONO_TRPR else 'OUT OF RANGE'}, "
                        f"not {MONO_TRPR_ID} (which is at {at})")

    tf_ids = [str(x) for x in sd.process.transcription_regulation.tf_ids]
    bulk_ids = [str(x) for x in sd.internal_state.bulk_molecules.bulk_data["id"]]
    idx = {}
    if TF_TRPR_ID in tf_ids:
        idx["tf_trpR"] = tf_ids.index(TF_TRPR_ID)
    else:
        problems.append(f"TF {TF_TRPR_ID} absent from tf_ids ({len(tf_ids)} entries)")
    for label, mol in BULK_READOUTS.items():
        if mol in bulk_ids:
            idx[label] = bulk_ids.index(mol)
        else:
            problems.append(f"bulk molecule {mol} absent ({len(bulk_ids)} entries)")

    if problems:
        raise SystemExit("index resolution failed for " + cache + ":\n  "
                         + "\n  ".join(problems))
    _IDX.clear()
    _IDX.update(idx)
    print(f"  indices verified against {cache}: {TU_TRP_ID}@{TU_TRP}, "
          f"{MONO_TRPR_ID}@{MONO_TRPR}, TF {TF_TRPR_ID}@{idx['tf_trpR']}", flush=True)
    print("  instrumentation resolved: "
          + ", ".join(f"{BULK_READOUTS[k]}@{idx[k]}" for k in BULK_READOUTS), flush=True)


def lineage_agent(agents: dict) -> str:
    """Follow one lineage deterministically: the all-zeros daughter each time."""
    return sorted(agents)[0]


def _mass_fg(value):
    """Femtograms as a plain float; None stays None (absent != zero mass)."""
    if value is None:
        return None
    from v2ecoli.library.quantity_helpers import fg_magnitude
    return fg_magnitude(value)


def _at(arr, i):
    """Positional read that returns None for an ABSENT array, never for a real 0.

    ⚠ The distinction is the whole point of the instrumentation: a zero holorepressor
    count is a result (TrpR present but unloaded), while a missing listener is a
    measurement failure. Collapsing them -- which ``arr[i] if arr.size > i else 0``
    would do -- is the same class of error as reading a zero exchange flux as
    "not secreted".
    """
    a = np.asarray(arr)
    return float(a[i]) if a.size > i else None


def sample(comp) -> dict | None:
    ags = comp.state["agents"]
    aid = lineage_agent(ags)
    agent = ags[aid]
    li = agent["listeners"].get("rna_synth_prob", {})
    tu = np.asarray(li.get("actual_rna_synth_prob", []))
    if tu.size <= TU_TRP:
        return None
    mc = np.asarray(agent["listeners"].get("monomer_counts", []))
    bulk = np.asarray(agent.get("bulk", []))
    mass = agent["listeners"].get("mass", {}) or {}
    tf = _IDX["tf_trpR"]

    rec = {
        "agent": aid,
        "n_agents": len(ags),
        "trp_tu_synth_prob": float(tu[TU_TRP]),
        "total_rna_init": float(np.asarray(li.get("total_rna_init", 0)).sum()),
        # Kept under its original key so the pre-2026-08-25 traces stay comparable.
        # ⚠ It is the APO monomer, not the repressing species -- see BULK_READOUTS.
        "trpR_monomer": _at(mc, MONO_TRPR),
        # --- occupancy -----------------------------------------------------
        # ⚠ MEASURED 2026-08-25, and it decides which readout is usable: of the five
        # TF-indexed arrays tf_binding.py:450-461 writes, only `n_actual_bound`
        # materialises in the live agent store. `p_promoter_bound`,
        # `n_promoter_bound`, `n_available_promoters`, `n_bound_TF_per_TU` (flag-gated)
        # and `promoter_copy_number` all read back size 0 -- they are declared in the
        # schema and empty in practice, so a reader who adds them gets None, not a
        # number. Deliberately NOT recorded as None fields: a null column invites the
        # next person to treat it as "no binding" rather than "not emitted".
        #
        # ⭑ And `n_actual_bound` ACCUMULATES rather than reporting an instantaneous
        # count: 32 @ step 4, 234 @ 30, 941 @ 120, 1895 @ 240 -- a flat ~7.8/step. So
        # the useful quantity is its DERIVATIVE between samples (binding events per
        # second), not its value. Recorded raw; differenced downstream, so the choice
        # of window can change without re-running.
        "n_actual_bound_trpR_cumulative": _at(li.get("n_actual_bound", []), tf),
        # --- the corepressor equilibrium ----------------------------------
        # ⚠ Masses arrive as pint Quantity[fg] on this port, not plain floats -- a
        # bare float() raises DimensionalityError. Reuse v2ecoli's own helper rather
        # than stripping units by hand, which is how a fg value silently becomes a
        # different unit's number.
        "dry_mass_fg": _mass_fg(mass.get("dry_mass")),
        "cell_mass_fg": _mass_fg(mass.get("cell_mass")),
    }
    # bulk is a structured array of (id, count, ...); take the count field.
    for label in BULK_READOUTS:
        i = _IDX[label]
        if bulk.size > i:
            row = bulk[i]
            rec[label] = float(row["count"] if bulk.dtype.names else row)
        else:
            rec[label] = None
    return rec


def run_arm(arm: str, generations: int, gen_steps: int | None = None) -> dict:
    import v2ecoli
    global GEN_STEPS
    # basal_with_trp doubles at 44.0 min like basal, so only the _withaa pair differs.
    GEN_STEPS = gen_steps or (GEN_STEPS_WITHAA if arm.endswith("_withaa") else 2640)
    cache, note = ARMS[arm]
    print(f"== route B: {arm} ({note}) cache={cache} ==", flush=True)
    resolve_indices(cache)
    t0 = time.time()
    # ⚠ experiment_id MUST be distinct per arm. The ParquetEmitter writes its
    # configuration under .../experiment_id=<id>/generation=N/agent_id=M and
    # calls makedirs WITHOUT exist_ok, so two arms sharing the default id race
    # at the first division and one of them dies with FileExistsError. Cost a
    # 23-minute run of two arms to find.
    # ⛔ experiment_id ALONE IS NOT ENOUGH, measured 2026-08-25. The comment this
    # replaces claimed a distinct experiment_id made parallel arms safe; it does not.
    # The parquet emitter writes under .pbg/parquet-runs/default/.../experiment_id=
    # default/... -- the id never reaches it -- so two concurrent arms race on the
    # same configuration directory and one dies with FileExistsError, or later with
    # FileNotFoundError when a sibling removes a .tmp underneath it. Two arms DID run
    # concurrently earlier the same day, which is how the trap got written down as
    # retired: they were staggered by ~45 s and missed the window. That was luck.
    # ⊕ The measurements are unaffected either way -- every readout comes from
    # comp.state, never from the emitter -- so it is a crash risk, not a data risk.
    # The fix is a per-arm emitter sink, which makes the arms genuinely independent.
    comp = v2ecoli.build_composite("ecoli_baseline", seed=0, cache_dir=cache,
                                   experiment_id=f"genotype06-{arm}",
                                   emitter_out_dir=str(WS_ROOT / "out" / "genotype-06"
                                                       / "emitter" / arm))

    trace = []
    total = generations * GEN_STEPS
    for step in range(SAMPLE_EVERY, total + 1, SAMPLE_EVERY):
        comp.run(SAMPLE_EVERY)
        rec = sample(comp)
        if rec is not None:
            rec["step"] = step
            trace.append(rec)
            if step % (GEN_STEPS // 2) == 0:
                print(f"  step {step:>6} agents={rec['n_agents']:>2} "
                      f"trpTU={rec['trp_tu_synth_prob']:.4e} "
                      f"TrpR={rec['trpR_monomer']}", flush=True)
    out = {"arm": arm, "cache": cache, "note": note,
           "generations": generations, "gen_steps": GEN_STEPS,
           "sample_every": SAMPLE_EVERY,
           "wall_seconds": round(time.time() - t0, 1), "trace": trace}
    dest = STUDY_DIR / "data" / f"route_b_{arm}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  wrote {dest.name} ({len(trace)} samples, {out['wall_seconds']}s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--gen-steps", type=int, default=None,
                    help="override steps per generation (default: 2640 basal / 1500 with_aa)")
    args = ap.parse_args()
    os.chdir(WS_ROOT)
    run_arm(args.arm, args.generations, args.gen_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
