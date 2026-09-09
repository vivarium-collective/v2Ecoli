"""Diagnostic: empirically trace the causal lag from a Class III regulatory
decision (init_prob_override rising on fliC's promoter) through to mRNA
appearing, through to free FliC protein appearing.

Added 2026-08-12, part of the flagella-cascade investigation. Directly
answers a question raised during the execution-order discussion: does
flagella-transcription-regulation's output actually reach transcription
(confirmed in code, transcript_initiation.py:592-598), and if so, how many
ticks does it take for that regulatory decision to show up as free FliC
protein in the bulk pool? The claim under test is that this delay is real,
multi-tick, and NOT something the execution-layer ordering can or should
collapse to zero -- it reflects real RNA-polymerase/ribosome kinetics, not
an ordering artifact.

Same initial-condition pattern as run_diagnostic_no_division.py and other
scripts in this study, for comparability: 4 flagella, 0 motor at t=0 so
FlgM secretion (proportional to complete-flagella count) starts immediately;
free FliA started LOW and FlgM started HIGH so there is a clear low-to-high
FliA transition to trace, rather than starting already saturated.

fliC's TU_index (3062 in cache out/cache_full_flit_v11) is resolved directly
from the LIVE FlagellaTranscriptionRegulation Step instance inside the built
composite (comp.step_paths[...]['instance'].flg_classIII_TU_ids[0]) rather
than independently re-deriving it from sim_data -- guarantees this script
always uses whatever index the ACTUAL running regulation Step is using,
even if cistron ordering ever changes.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/diagnostic_transcription_to_protein_lag.py \
        --seconds 3600 --sample 20 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx
from v2ecoli.steps.division import Division

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIG_NEXT_UPDATE = Division.next_update

INIT = {
    "CPLX0-7452[j]": 4,          # complete flagella -- drives FlgM secretion from t=0
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 20,    # free FliA -- start LOW so a rise is visible
    "G369-MONOMER[c]": 800,      # FlgM -- start HIGH so FliA starts mostly sequestered
    "EG10321-MONOMER[e]": 0,     # free FliC -- start at zero so any appearance is a clear signal
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir):
    # Division is orthogonal to this diagnostic (it's about the transcription
    # -> translation lag, not lineage dynamics) and would otherwise replace
    # agent "0" with two daughters partway through a long enough run (D-period
    # division observed at t~2524s elsewhere in this investigation) --
    # disabled here the same way run_diagnostic_no_division.py does, fully
    # reverted in a finally block.
    Division.next_update = lambda self, timestep, states: {}
    try:
        enable_features("flagella_regulation")
        comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
        enable_features()

        # Resolve fliC's TU_index from the LIVE regulation Step instance.
        reg_path = ("agents", "0", "ecoli-flagella-transcription-regulation")
        reg_instance = comp.step_paths[reg_path]["instance"]
        fliC_tu_index = int(reg_instance.flg_classIII_TU_ids[0])
        fliC_rna_id = reg_instance.parameters["flg_classIII_rnaids"][0]
        print(f"fliC TU_index resolved from live Step instance: {fliC_tu_index} ({fliC_rna_id})")

        bulk = _arr(comp.state["agents"]["0"]["bulk"])
        bids = bulk["id"]
        for name, val in INIT.items():
            try:
                bulk["count"][bulk_name_to_idx(name, bids)] = val
            except Exception as e:
                print("  (skip IC", name, "->", e, ")")

        read_ids = ["EG11355-MONOMER[c]", "G369-MONOMER[c]", "EG10321-MONOMER[e]"]
        idx = {k: bulk_name_to_idx(k, bids) for k in read_ids}

        rec = {"t": [], "free_fliA": [], "free_flgM": [], "fliC_override": [],
               "fliC_mRNA_full": [], "fliC_mRNA_partial": [], "free_fliC": []}

        def snap(t):
            cell = comp.state["agents"]["0"]
            b = _arr(cell["bulk"])

            promoters = _arr(cell["unique"]["promoter"])
            p_mask = promoters["_entryState"].view(bool)
            p_tu = promoters["TU_index"][p_mask]
            p_override = promoters["init_prob_override"][p_mask]
            fliC_rows = p_override[p_tu == fliC_tu_index]
            override_val = float(fliC_rows.mean()) if len(fliC_rows) else 0.0

            rnas = _arr(cell["unique"]["RNA"])
            r_mask = rnas["_entryState"].view(bool)
            r_tu = rnas["TU_index"][r_mask]
            r_full = rnas["is_full_transcript"][r_mask]
            fliC_rna_mask = r_tu == fliC_tu_index
            n_full = int(np.sum(fliC_rna_mask & r_full.astype(bool)))
            n_partial = int(np.sum(fliC_rna_mask & ~r_full.astype(bool)))

            rec["t"].append(t)
            rec["free_fliA"].append(int(b["count"][idx["EG11355-MONOMER[c]"]]))
            rec["free_flgM"].append(int(b["count"][idx["G369-MONOMER[c]"]]))
            rec["fliC_override"].append(override_val)
            rec["fliC_mRNA_full"].append(n_full)
            rec["fliC_mRNA_partial"].append(n_partial)
            rec["free_fliC"].append(int(b["count"][idx["EG10321-MONOMER[e]"]]))

        snap(0)
        total = 0.0
        while total < seconds:
            chunk = min(sample, seconds - total)
            comp.run(chunk)
            total += chunk
            snap(total)

        return {k: np.array(v) for k, v in rec.items()}
    finally:
        Division.next_update = _ORIG_NEXT_UPDATE


def _first_rise(arr, t, factor=3.0, min_abs=3):
    """First time the series exceeds max(factor * baseline, baseline + min_abs)
    -- i.e. a genuine RISE above its own t=0 starting level, not just '>0'
    (which triggers trivially if the series already had a nonzero baseline
    from ParCa's own steady-state initial condition)."""
    baseline = arr[0]
    threshold = max(factor * baseline, baseline + min_abs)
    idxs = np.nonzero(arr > threshold)[0]
    return (t[idxs[0]], threshold) if len(idxs) else (None, threshold)


def report(rec):
    t = rec["t"]
    mrna_full = rec["fliC_mRNA_full"]
    mrna_any = rec["fliC_mRNA_full"] + rec["fliC_mRNA_partial"]
    protein = rec["free_fliC"]
    override = rec["fliC_override"]

    override_t, override_thresh = _first_rise(override, t, factor=3.0, min_abs=1e-4)
    mrna_any_t, mrna_any_thresh = _first_rise(mrna_any, t, factor=1e9, min_abs=1)
    mrna_full_t, mrna_full_thresh = _first_rise(mrna_full, t, factor=1e9, min_abs=1)
    protein_t, protein_thresh = _first_rise(protein, t, factor=1e9, min_abs=5)

    print("\n=== Causal lag: regulation -> mRNA -> protein (rise-above-baseline) ===")
    print(f"free FliA:  t=0 -> {rec['free_fliA'][0]}, final -> {rec['free_fliA'][-1]}")
    print(f"free FlgM:  t=0 -> {rec['free_flgM'][0]}, final -> {rec['free_flgM'][-1]}")
    print(f"fliC init_prob_override: baseline={override[0]:.6g}, first RISE (> {override_thresh:.4g}) at t = {override_t}")
    print(f"fliC mRNA (any):         baseline={mrna_any[0]}, first RISE (> {mrna_any_thresh:.0f}) at t = {mrna_any_t}"
          + (f"  (+{mrna_any_t - override_t:.0f}s after override rise)" if override_t is not None and mrna_any_t is not None else ""))
    print(f"fliC mRNA (FULL):        baseline={mrna_full[0]}, first RISE (> {mrna_full_thresh:.0f}) at t = {mrna_full_t}"
          + (f"  (+{mrna_full_t - override_t:.0f}s after override rise)" if override_t is not None and mrna_full_t is not None else ""))
    print(f"free FliC protein:       baseline={protein[0]}, first RISE (> {protein_thresh:.0f}) at t = {protein_t}"
          + (f"  (+{protein_t - mrna_full_t:.0f}s after full mRNA rise)" if mrna_full_t is not None and protein_t is not None else ""))
    if override_t is not None and protein_t is not None:
        print(f"\nTOTAL lag, regulation rise -> protein rise: {protein_t - override_t:.0f}s "
              f"({(protein_t - override_t) / 60:.1f} min)")

    print("\n--- full trajectory (every 10th sample) ---")
    print(f"{'t':>6} {'fliA':>6} {'flgM':>6} {'override':>10} {'mRNA_full':>10} {'mRNA_part':>10} {'FliC':>8}")
    for i in range(0, len(t), 10):
        print(f"{t[i]:6.0f} {rec['free_fliA'][i]:6d} {rec['free_flgM'][i]:6d} "
              f"{override[i]:10.4g} {mrna_full[i]:10d} {rec['fliC_mRNA_partial'][i]:10d} {protein[i]:8d}")


def figure(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = rec["t"] / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(t, rec["free_fliA"], color="#1f77b4", label="free FliA")
    axes[0].plot(t, rec["free_flgM"], color="#d62728", label="free FlgM")
    axes[0].set_ylabel("count"); axes[0].legend(fontsize=8); axes[0].set_title("Regulatory inputs")

    axes[1].plot(t, rec["fliC_override"], color="#2ca02c")
    axes[1].set_ylabel("init_prob_override"); axes[1].set_title("fliC promoter override (this Step's output)")

    axes[2].plot(t, rec["fliC_mRNA_partial"], color="#ff7f0e", label="partial")
    axes[2].plot(t, rec["fliC_mRNA_full"], color="#9467bd", label="full transcript")
    axes[2].set_ylabel("count"); axes[2].legend(fontsize=8); axes[2].set_title("fliC mRNA (unique RNAs, TU-filtered)")

    axes[3].plot(t, rec["free_fliC"], color="#8c564b")
    axes[3].set_ylabel("count"); axes[3].set_xlabel("time (min)"); axes[3].set_title("free FliC protein monomer")

    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/15_transcription_to_protein_lag.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=str, default="out/cache_full_flit_v11")
    args = ap.parse_args()

    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    report(rec)
    figure(rec)


if __name__ == "__main__":
    main()
