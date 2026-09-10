"""Standalone post-hoc analysis for a Ray-dispatched simulation.

Two dispatch shapes reach this script, and it routes each analysis to whichever
of v2ecoli's two analysis-class families it belongs to (same
v2ecoli.workflow.analysis.ANALYSIS_REGISTRY for both -- distinguished by
``issubclass(cls, Analysis)``):

  * scripts/run_phase0_xarray_ensemble.py's single-generation dispatch writes
    only seed_NN/summary.json + seed_NN/store.zarr (no parquet). Its analyses
    are ``AnalysisStep`` subclasses (e.g. multiseed's doubling_time_distribution)
    -- this script builds row records straight from summary.json and calls
    ``cls({}, core=core).analyze(rows)`` (unchanged from before).
  * The multi-generation batch dispatch (BatchBaselineRunner, run through
    sms_api.compose.run_pbg's generic runner -- see viva-api backlog items
    26/27) uses emitter="both", which ALSO writes hive-parquet under the same
    S3 prefix. This unlocks the
    DuckDB-based ``Analysis`` family -- the cd1/ptools omics suite
    (ptools_rna, cd1_metabolomics, etc.), the actual original backlog target
    ("reproduce CD1 datasets"), never reachable via standalone analysis before.
    These are routed straight to v2ecoli.workflow.analysis_runner.run_analyses(),
    the SAME function the local batch_baseline flush path already uses
    end-to-end (sweep_dir may be an s3:// URI -- DuckDB reads the parquet in
    place via httpfs) -- no new analysis logic, just a real caller for GovCloud.

sms-api's own standalone-analysis endpoint (POST /simulations/{id}/analysis) was
built exclusively for a different, legacy simulation pipeline (vEcoli-private's
Nextflow-based parca/variant_sim_data/TSV output) and cannot read either of
these dispatch shapes' output at all -- this script is what actually runs.

The single-generation dispatch has no cell division, so `divided`/
`division_time` are always False/0.0 for THAT path's rows -- an honest
reflection of what it actually simulates, not a bug. The multi-generation path
has real division data (the batch dispatch's own real gap-2 fix).

Usage:
    python scripts/run_standalone_analysis.py \
        --out-uri s3://bucket/vecoli-output/<experiment_id> \
        --n-seeds 2 \
        --modules '{"multiseed": {"doubling_time_distribution": {}}}' \
        --analysis-name my-analysis

    # Or let v2ecoli pick every applicable analysis for this run's shape
    # (delegates to batch_baseline_runner.build_analysis_options -- the same
    # resolution the composite's own inline flush uses); --n-generations only
    # matters in this form, to decide whether multigeneration analyses apply:
    python scripts/run_standalone_analysis.py \
        --out-uri s3://bucket/vecoli-output/<experiment_id> \
        --n-seeds 2 --n-generations 2 \
        --modules applicable \
        --analysis-name my-analysis
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ``python scripts/run_standalone_analysis.py`` puts this file's own directory
# (scripts/) on sys.path[0], not the repo root -- `scripts` (no __init__.py,
# an implicit namespace package) is then unresolvable as a top-level import
# unless the repo root is ALSO on the path. v2ecoli.* already resolves
# regardless of invocation (installed into the container's venv); `scripts.*`
# never was, it only ever worked when a caller happened to also set
# PYTHONPATH/cwd to the repo root. Real repro: v2ecoli.workflow.analyses'
# comparison_summary module (imported transitively by
# batch_baseline_runner.build_analysis_options, reachable via "applicable" --
# item 60) does `from scripts._compare... import ...` and raised
# ModuleNotFoundError: No module named 'scripts' from inside the real Ray
# Batch container. Self-contained fix: ensure the repo root is importable
# regardless of how this script gets invoked, rather than pushing a
# PYTHONPATH requirement onto every caller.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_aws(args: list[str], tries: int = 3, backoff_s: float = 5.0) -> None:
    """Run an ``aws`` CLI subprocess, retrying transient failures.

    S3 network blips (e.g. a momentary DNS/connectivity hiccup to the
    bucket's virtual-hosted endpoint) have hit this exact call three times
    in one real investigation session -- always transient (a bare retry
    moments later always succeeded), never a credentials/permissions issue.
    Real ``aws sts get-caller-identity``/``aws s3 ls`` checks confirmed this
    each time before retrying by hand; this bakes that same retry in.
    """
    last_err: subprocess.CalledProcessError | None = None
    for attempt in range(1, tries + 1):
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            if attempt < tries:
                time.sleep(backoff_s * attempt)
    assert last_err is not None
    raise last_err


def _aws_cp(src: str, dst: str) -> None:
    _run_aws(["aws", "s3", "cp", src, dst])


def _aws_sync(src: str, dst: str) -> None:
    _run_aws(["aws", "s3", "sync", src, dst])


def build_multiseed_rows(out_uri: str, n_seeds: int, tmp: Path) -> list[dict[str, Any]]:
    """One row per seed, read from that seed's own summary.json."""
    rows = []
    for seed in range(n_seeds):
        local = tmp / f"seed_{seed:02d}_summary.json"
        try:
            _aws_cp(f"{out_uri.rstrip('/')}/seed_{seed:02d}/summary.json", str(local))
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()[:200] if e.stderr else str(e)
            raise RuntimeError(f"seed {seed}: no summary.json under {out_uri} -- {stderr}") from e
        summary = json.loads(local.read_text())
        rows.append({
            "variant": 0,
            "lineage_seed": summary.get("seed", seed),
            "generation": 1,
            "agent_id": "0",
            "divided": False,
            "division_time": 0.0,
            "final_dry_mass": float(summary.get("dry_mass_fg", 0.0)),
        })
    return rows


ROW_BUILDERS: dict[str, Callable[[str, int, Path], list[dict[str, Any]]]] = {
    "multiseed": build_multiseed_rows,
}


def run_duckdb_analyses(
    out_uri: str, scale: str, entries: dict[str, Any], tmp: Path, outdir: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Route DuckDB-based ``Analysis`` names (one scale's worth) straight to
    v2ecoli.workflow.analysis_runner.run_analyses() -- the same function the
    local batch_baseline flush path already runs end-to-end. ``out_uri`` is the
    sweep's S3 prefix; run_analyses reads its hive-parquet via DuckDB httpfs in
    place and writes analysis.json/viz/ptools locally, which we then sync up.
    """
    from v2ecoli.workflow import analysis_runner

    written: list[str] = []
    errors: list[dict[str, str]] = []
    local_out = tmp / "duckdb_out" / scale
    try:
        analysis_runner.run_analyses(
            sweep_dir=out_uri, analysis_options={scale: entries}, out_dir=str(local_out),
        )
    except Exception as e:  # noqa: BLE001 -- surface any analysis failure in the manifest
        errors.append({"scale": scale, "error": f"{type(e).__name__}: {e}"})
        return written, errors
    try:
        _aws_sync(str(local_out), outdir)
    except subprocess.CalledProcessError as e:
        # A sync failure here must NOT crash the whole multi-scale run --
        # real incident: an earlier version let this propagate uncaught,
        # which killed a multi-hour run at the FIRST scale's sync and lost
        # every downstream scale's real results (incl. every cd1_* module,
        # all of which live in a later scale) to what was, both times, a
        # transient network blip. Record it like every other per-scale
        # failure in this function and let the remaining scales still run.
        stderr = e.stderr.decode()[:500] if e.stderr else str(e)
        errors.append({"scale": scale, "error": f"aws s3 sync failed: {stderr}"})
        return written, errors
    # run_analyses() writes ONE analysis.json covering every name in `entries`
    # (plus viz/ptools per-group files) -- one manifest entry, not one per name.
    written.append(f"{outdir}/analysis.json")
    return written, errors


def _register_extra_analysis_modules() -> list[dict[str, str]]:
    """Import any modules named in ``V2ECOLI_EXTRA_ANALYSIS_MODULES`` for their
    ``ANALYSIS_REGISTRY`` registration side effect, and return one error record
    per module that failed to import.

    ``import v2ecoli.workflow.analyses`` only registers v2ecoli's own ports.
    Downstream packages register Analysis ports into the SAME registry purely as
    an import side effect, but they live outside v2ecoli, so this script never
    imported them and every such name silently resolved to "unknown analysis"
    in ``run()`` -- e.g. sms-ecoli's whole ``sms_modules`` suite
    (ptools_metabolites_multiseed, the fss_* family) was invisible to every
    standalone flush. A caller names those modules in the env var (comma- or
    whitespace-separated, e.g. ``sms_modules``); each is imported here.

    Best-effort by design: v2ecoli must stay standalone when the var is unset,
    and a typo'd or unimportable module name is surfaced in the manifest rather
    than crashing the run (it would otherwise take down analyses that ARE
    registered). The repo root is on sys.path (see top of file), so a
    downstream's top-level package resolves regardless of invocation."""
    errors: list[dict[str, str]] = []
    spec = os.environ.get("V2ECOLI_EXTRA_ANALYSIS_MODULES", "")
    for mod in spec.replace(",", " ").split():
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 -- a bad extra module must not crash the run
            errors.append({"module": mod,
                           "error": f"import failed: {type(e).__name__}: {e}"})
    return errors


def run(out_uri: str, n_seeds: int, modules: dict[str, dict[str, Any]],
        analysis_name: str, tmp: Path) -> dict[str, Any]:
    from bigraph_schema import allocate_core

    import v2ecoli.workflow.analyses  # noqa: F401 -- populates ANALYSIS_REGISTRY
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis

    core = allocate_core()
    outdir = f"{out_uri.rstrip('/')}/analyses/{analysis_name}"
    written: list[str] = []
    errors: list[dict[str, str]] = []
    # Register downstream Analysis ports (e.g. sms_modules) BEFORE any name is
    # looked up below; import failures are surfaced in the manifest, not fatal.
    errors.extend(_register_extra_analysis_modules())

    for scale, entries in modules.items():
        # Split this scale's requested names by which analysis-class family they
        # belong to -- the two families need entirely different inputs (row
        # records vs. a DuckDB connection over the sweep's parquet).
        duckdb_entries: dict[str, Any] = {}
        step_names: list[str] = []
        for name in entries:
            cls = ANALYSIS_REGISTRY.get(name)
            if cls is None:
                errors.append({"name": name, "error": f"unknown analysis {name!r}"})
                continue
            if cls.scale != scale:
                errors.append({"name": name, "error": f"{name} is scale={cls.scale!r}, not {scale!r}"})
                continue
            if issubclass(cls, Analysis):
                duckdb_entries[name] = entries[name]
            else:
                step_names.append(name)

        if duckdb_entries:
            w, e = run_duckdb_analyses(out_uri, scale, duckdb_entries, tmp, outdir)
            written.extend(w)
            errors.extend(e)

        if not step_names:
            continue
        builder = ROW_BUILDERS.get(scale)
        if builder is None:
            errors.append({"scale": scale, "error": f"no row builder for scale {scale!r}"})
            continue
        try:
            rows = builder(out_uri, n_seeds, tmp)
        except RuntimeError as e:
            errors.append({"scale": scale, "error": str(e)})
            continue
        for name in step_names:
            cls = ANALYSIS_REGISTRY[name]
            try:
                result = cls({}, core=core).analyze(rows)
            except Exception as e:  # noqa: BLE001 -- surface any analysis failure in the manifest
                errors.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                continue
            local_out = tmp / f"{name}.json"
            local_out.write_text(json.dumps(result, indent=2))
            dest = f"{outdir}/{name}.json"
            _aws_cp(str(local_out), dest)
            written.append(dest)

    status = "done" if written and not errors else ("failed" if errors and not written else "partial")
    manifest = {
        "analysis_name": analysis_name, "modules": modules,
        "written": written, "errors": errors, "status": status,
    }
    manifest_local = tmp / "_manifest.json"
    manifest_local.write_text(json.dumps(manifest, indent=2))
    _aws_cp(str(manifest_local), f"{outdir}/_manifest.json")
    return manifest


def resolve_modules(modules: "str | dict[str, dict[str, Any]]", *, n_seeds: int,
                     n_generations: int) -> dict[str, dict[str, Any]]:
    """Resolve the ``modules`` argument into a concrete ``{scale: {name: params}}`` mapping.

    Accepts an explicit mapping (used verbatim), the ``"applicable"`` keyword (every
    registered analysis at the scales this run covers), or a JSON string encoding
    either. ``"applicable"`` delegates to v2ecoli's own
    ``batch_baseline_runner.build_analysis_options`` -- the SAME resolution the
    composite's own inline flush already uses -- rather than re-deriving scale
    selection here.
    """
    if isinstance(modules, dict):
        return modules
    if modules.strip().lower() == "applicable":
        from v2ecoli.steps.batch_baseline_runner import build_analysis_options
        return build_analysis_options("applicable", n_seeds=n_seeds, n_generations=n_generations)
    return json.loads(modules)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", help="JSON file with out_uri/n_seeds/modules/analysis_name "
                                          "-- avoids shell-quoting the modules JSON on the CLI")
    p.add_argument("--out-uri", help="s3:// experiment output prefix")
    p.add_argument("--n-seeds", type=int)
    p.add_argument("--n-generations", type=int, default=1,
                    help="lineage depth; only consulted when --modules is 'applicable'")
    p.add_argument("--modules", help='JSON: {"scale": {"name": {}}}, or the keyword "applicable"')
    p.add_argument("--analysis-name")
    args = p.parse_args()

    if args.config_file:
        cfg = json.loads(Path(args.config_file).read_text())
        out_uri, n_seeds = cfg["out_uri"], int(cfg["n_seeds"])
        n_generations = int(cfg.get("n_generations", 1))
        modules = resolve_modules(cfg["modules"], n_seeds=n_seeds, n_generations=n_generations)
        analysis_name = cfg["analysis_name"]
    else:
        missing = [f for f in ("out_uri", "n_seeds", "modules", "analysis_name")
                   if getattr(args, f) is None]
        if missing:
            p.error(f"missing required arguments (or use --config-file): {missing}")
        out_uri, n_seeds = args.out_uri, args.n_seeds
        modules = resolve_modules(args.modules, n_seeds=n_seeds, n_generations=args.n_generations)
        analysis_name = args.analysis_name

    with tempfile.TemporaryDirectory() as td:
        manifest = run(out_uri, n_seeds, modules, analysis_name, Path(td))
    print(json.dumps(manifest, indent=2))
    if manifest["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
