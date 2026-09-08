"""
ParCa Pipeline as Process-Bigraph Steps
========================================

Runs the full v2ecoli ParCa pipeline — 9 Steps wired to a nested bigraph store
mirroring ``SimulationDataEcoli``'s structure — and pickles the final
store state.

Usage::

    # Fast run (reduced TF conditions via debug=True, ~30 min)
    v2ecoli-parca --mode fast

    # Full production run (several hours)
    v2ecoli-parca --mode full

    # Custom output directory and Km-fitting cache
    v2ecoli-parca --mode fast -o out/sim_data --cache-dir out/cache
"""

import argparse
import contextlib
import io
import json
import os
import pickle
import re
import sys
import time


def _default_cpus() -> int:
    return os.cpu_count() or 4


def _build_arg_parser() -> argparse.ArgumentParser:
    """The CLI's parser, extracted so its behaviour is testable.

    Built here rather than inline in ``main()`` so a test can assert what the
    flags DO — defaults, round-tripping, rejection of unknown values — instead
    of grepping this file for a substring, which passes just as happily when a
    flag is parsed and then dropped on the floor.
    """
    parser = argparse.ArgumentParser(
        prog="v2ecoli-parca",
        description="ParCa Pipeline as Process-Bigraph Steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["fast", "full"], default="fast",
        help="fast: debug=True (reduced TF conditions, ~30 min); "
             "full: all conditions (several hours).",
    )
    parser.add_argument("-c", "--cpus", type=int, default=_default_cpus(),
                        help=f"Parallelism for Steps 4 and 5 "
                             f"(default {_default_cpus()}, from os.cpu_count). "
                             "Step 5 runs ~9 conditions each ~6 min serially, "
                             "so cpus=8 keeps them mostly in parallel.")
    parser.add_argument("-o", "--outdir", type=str, default="out/sim_data",
                        help="Output directory for pickled state.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for Km optimization "
                             "(default: <outdir>/cache).")
    parser.add_argument("--no-operons", action="store_true",
                        help="Disable operons in the KB.")
    parser.add_argument(
        "--bundle-manifest-path", type=str, default=None,
        help="Path to an ecoli-sources bundle manifest (default: the installed "
             "ecoli-sources reference bundle + v2ecoli overrides).")
    parser.add_argument(
        "--bundle-overrides", type=str, action="append", default=None,
        metavar="MANIFEST",
        help="Additional overrides manifest layered on the bundle; repeatable, "
             "applied in the order given. Always applied AFTER v2ecoli's own "
             "overrides, so naming one cannot revert v2ecoli's diverged flat "
             "files. This is how a private payload contributes extra canonical "
             "keys (e.g. a strain's new-gene inputs).")
    parser.add_argument(
        "--new-genes", type=str, default="off", metavar="SUBDIR",
        help="Insert a new_gene_data subdirectory into the genome (e.g. a "
             "heterologous pathway); 'off' (default) inserts none. The named "
             "subdirectory's flat inputs must be resolvable through the "
             "bundle — typically supplied by --bundle-overrides.")
    parser.add_argument(
        "--rnaseq-source", type=str, default="reference",
        choices=["reference", "experimental"],
        help="Transcriptome tier basal cistron expression is fitted from. "
             "'reference' (default) reads the legacy wide rna_seq_data table "
             "at sim_data.basal_expression_condition; 'experimental' reads "
             "ecoli-sources' long-form rnaseq_experimental_tpms key — the key "
             "a knockdown() variant bundle rewrites.")
    parser.add_argument(
        "--no-rnaseq-cross-fill", dest="rnaseq_cross_fill",
        action="store_false", default=True,
        help="On --rnaseq-source experimental, do NOT fill model genes the "
             "experimental dataset omits from the rnaseq_basal_tpms tier; "
             "leave them at zero expression (and not RNA-seq-covered).")
    parser.add_argument("--resume-from-step", type=int, default=1,
                        help="Skip steps 1..N-1; load --resume-pickle as the "
                             "initial composite state.  Use to debug late steps "
                             "without re-running step 5 (~60 min).")
    parser.add_argument("--resume-pickle", type=str, default=None,
                        help="Pickle of composite.state from an earlier run.")
    parser.add_argument(
        "--allow-partial-fit", action="store_true",
        help="Step 9's mechanistic_supply/export/uptake fits can fail on "
             "numerically-marginal kinetics (e.g. debug mode's truncated TF "
             "set). By default such a failure aborts the run so no "
             "partially-fit parca_state.pkl is written (PARCA_REVIEW A3). "
             "Pass this flag to write it anyway, with a per-fit ok/error "
             "status recorded at state['mechanistic_fit_status'].")
    return parser


def main():
    from v2ecoli.processes.parca.composite import build_parca_composite
    from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
        KnowledgeBaseEcoli,
    )

    parser = _build_arg_parser()
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'=' * 60}\n  v2ecoli ParCa — {args.mode} mode (cpus={args.cpus})\n{'=' * 60}")

    resume_state = None
    if args.resume_from_step > 1:
        if not args.resume_pickle:
            raise SystemExit("--resume-from-step > 1 requires --resume-pickle")
        print(f"[{time.strftime('%H:%M:%S')}] Loading resume state from "
              f"{args.resume_pickle}")
        with open(args.resume_pickle, 'rb') as f:
            resume_state = pickle.load(f)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading raw_data "
          f"(operons={not args.no_operons}, new_genes={args.new_genes}, "
          f"overrides={args.bundle_overrides or 'defaults only'}, "
          f"rnaseq_source={args.rnaseq_source}"
          f"{'' if args.rnaseq_cross_fill else ', cross-fill OFF'})")
    from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
    bundle = SourceBundle(base_manifest=args.bundle_manifest_path,
                          overrides=args.bundle_overrides)
    raw = KnowledgeBaseEcoli(
        operons_on=not args.no_operons,
        remove_rrna_operons=False, remove_rrff=False, stable_rrna=False,
        new_genes_option=args.new_genes,
        bundle=bundle,
    )
    print(f"    raw_data loaded in {time.time() - t0:.1f}s")

    # ---- Per-step disk checkpointing -----------------------------------
    # Wrap each Step class's .update so its output ports are pickled to
    # disk after the step completes.  If a later step crashes, you can
    # resume from any successful step via:
    #   --resume-from-step N --resume-pickle <outdir>/checkpoint_step_<N-1>.pkl
    # No-op when --resume-from-step > 1 (we already have a checkpoint).
    if args.resume_from_step <= 1:
        from v2ecoli.processes.parca.steps import ALL_STEP_CLASSES
        STEP_NUM_BY_CLASS = {
            'InitializeStep': 1, 'InputAdjustmentsStep': 2, 'BasalSpecsStep': 3,
            'TfConditionSpecsStep': 4, 'FitConditionStep': 5,
            'PromoterBindingStep': 6, 'AdjustPromotersStep': 7,
            'SetConditionsStep': 8, 'FinalAdjustmentsStep': 9,
        }
        running_checkpoint = {}
        step_times_live = {}
        runtimes_path = os.path.join(outdir, 'runtimes.json')
        def _wrap(cls, step_n):
            orig = cls.update
            def update(self, state):
                t_step = time.time()
                out = orig(self, state)
                step_times_live[f'step_{step_n}'] = time.time() - t_step
                try:
                    with open(runtimes_path, 'w') as f:
                        json.dump(step_times_live, f, indent=2, sort_keys=True)
                except Exception as e:
                    print(f"    WARN: runtimes write after step {step_n} failed: {e}")
                running_checkpoint.update(out)
                ckpt_path = os.path.join(outdir, f'checkpoint_step_{step_n}.pkl')
                try:
                    with open(ckpt_path, 'wb') as f:
                        pickle.dump(dict(running_checkpoint), f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"    checkpoint -> {ckpt_path}")
                except Exception as e:
                    print(f"    WARN: checkpoint after step {step_n} failed: {e}")
                return out
            cls.update = update
        for name, cls in ALL_STEP_CLASSES.items():
            n = STEP_NUM_BY_CLASS.get(name)
            if n is not None:
                _wrap(cls, n)

    t1 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] Running ParCa pipeline ...")
    captured = io.StringIO()
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
            self.encoding = getattr(streams[0], 'encoding', 'utf-8')
            self.errors = getattr(streams[0], 'errors', 'strict')
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
        def isatty(self):
            return getattr(self.streams[0], 'isatty', lambda: False)()
        def fileno(self):
            return self.streams[0].fileno()
        def writable(self):
            return True
    with contextlib.redirect_stdout(_Tee(sys.stdout, captured)):
        composite = build_parca_composite(
            raw,
            debug=(args.mode == 'fast'),
            cpus=args.cpus,
            cache_dir=cache_dir,
            resume_from_step=args.resume_from_step,
            resume_state=resume_state,
            allow_partial_fit=args.allow_partial_fit,
            # Record WHICH genotype this state was fit from. Without these the
            # emitted parca_state.pkl reads '' for all three and a new-gene
            # build is indistinguishable from a wild-type one by its own
            # recorded config -- so a cache's provenance has to be reconstructed
            # from outside it, which is exactly the ambiguity these fields exist
            # to remove.
            bundle_manifest=args.bundle_manifest_path or "",
            bundle_overrides=";".join(args.bundle_overrides or ()),
            new_genes=args.new_genes,
            # Same reasoning, one layer up: WHICH transcriptome tier the fit was
            # built from is equally part of a state's identity.
            rnaseq_source=args.rnaseq_source,
            rnaseq_cross_fill=args.rnaseq_cross_fill,
        )
    print(f"\n[{time.strftime('%H:%M:%S')}] Pipeline completed in "
          f"{time.time() - t1:.1f}s")

    step_times = {}
    for m in re.finditer(r'Step (\d) .*? completed in ([0-9.]+)s',
                         captured.getvalue()):
        step_times[f'step_{m.group(1)}'] = float(m.group(2))

    runtimes_path = os.path.join(outdir, 'runtimes.json')
    with open(runtimes_path, 'w') as f:
        json.dump(step_times, f, indent=2, sort_keys=True)
    print(f"    runtimes: {runtimes_path}")

    state = {k: v for k, v in composite.state.items()
             if not k.startswith('tick_')
             and not k.startswith('_')
             and k not in {'global_time', 'initialize', 'input_adjustments',
                           'basal_specs', 'tf_condition_specs', 'fit_condition',
                           'promoter_binding', 'adjust_promoters',
                           'set_conditions', 'final_adjustments'}}

    out_path = os.path.join(outdir, "parca_state.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    # Chassis provenance sidecar (parca_state.provenance.json): pins this
    # pickle's exact bytes + the code/workspace commits + the build inputs that
    # produced it, so a downstream sim_data cache can embed it in its
    # derived_from chain and a swap of the chassis is detectable rather than
    # silent. Best-effort: a provenance-write failure must never discard a
    # completed ParCa build, but it IS logged loudly.
    try:
        from v2ecoli.library.run_provenance import write_chassis_provenance
        build_info = {
            "mode": args.mode,
            "new_genes": args.new_genes,
            "bundle_manifest": args.bundle_manifest_path or "",
            "bundle_overrides": list(args.bundle_overrides or ()),
            "rnaseq_source": args.rnaseq_source,
            "argv": sys.argv,
        }
        prov = write_chassis_provenance(out_path, build=build_info)
        print(f"    chassis provenance -> {out_path.rsplit('.', 1)[0]}"
              f".provenance.json (sha256 {prov['artifact']['sha256'][:12]})")
    except Exception as e:  # noqa: BLE001 — provenance must not fail the build
        print(f"    WARN: chassis provenance write failed: "
              f"{type(e).__name__}: {e}")

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Total time:  {total:.1f}s ({total / 60:.1f} min)")
    print(f"Output:      {out_path} ({size_mb:.1f} MB)")
    print(f"Store keys:  {sorted(state.keys())}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
