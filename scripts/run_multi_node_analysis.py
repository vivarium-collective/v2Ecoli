"""Generic "Analysis flush" entrypoint for a completed multi-node
process-bigraph composite dispatch (backlog item 88 -- e.g. a colony
composite spread across N Ray-cluster nodes, or a pbg-native lineage batch,
item 109).

Two independent read paths, tried in order:

1. **Hive-parquet sweep (item 109's own fix)** -- when the caller supplies
   ``--n-seeds``/``--modules``, this is a ``lineage_ray_batch``-shaped
   dispatch: each ``LineageProcess`` actor wrote its own hive-partitioned
   parquet directly to ``history_uri`` (no flat file ever exists for this
   shape). Reuses ``run_standalone_analysis.py``'s own already-proven
   ``run_duckdb_analyses``/``resolve_modules`` -- the SAME DuckDB-httpfs
   in-place read + the SAME ``"applicable"``-resolution v2ecoli's local batch
   flush path already uses -- rather than a new read mechanism. This is what
   was missing: empirically confirmed twice (real GovCloud dispatches 282 and
   283) that without it, this script always reports "no history captured"
   for a pbg-native dispatch, because it only ever looked for the flat files
   below, which this dispatch shape never produces.
2. **Flat-file fallback (original, unchanged)** -- downloads whatever
   ``viva_api.compose.run_pbg``'s generic runner persisted (``emitter_history
   .json``/``final_state.json``) and hands it to ``v2ecoli.workflow.flush.
   run_flush`` -- the SAME mechanism the flush already dispatches for every
   other composite (cd1_*/ptools_* baseline analyses). This is colony's own
   path (item 88) and stays byte-for-byte as it was -- tried only when path 1
   is not applicable (no ``--n-seeds``/``--modules`` given) or produces
   nothing.

Writes ``_manifest.json`` matching the SAME S3-manifest contract every other
analysis kind already writes (``written``/``errors`` -- see
``viva_api.common.handlers.analyses.handle_get_ray_analysis_status``), so a
multi-node composite's analysis is exactly as discoverable through
``GET /analyses/{id}/status`` as a hand-triggered one.

Usage:
    python scripts/run_multi_node_analysis.py \
        --composite-id v2ecoli.composites.ecoli_colony.ecoli_colony \
        --history-uri s3://bucket/vecoli-output/<experiment_id> \
        --out-uri s3://bucket/vecoli-output/<experiment_id>/analyses/<name> \
        --experiment-id <experiment_id>

    # pbg-native (lineage_ray_batch) shape -- also tries the hive-parquet path:
    python scripts/run_multi_node_analysis.py \
        --composite-id v2ecoli.composites.lineage_ray_batch \
        --history-uri s3://bucket/vecoli-output/<experiment_id> \
        --out-uri s3://bucket/vecoli-output/<experiment_id>/analyses/<name> \
        --experiment-id <experiment_id> \
        --n-seeds 10 --n-generations 10 --modules applicable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Same real gap run_standalone_analysis.py's own comment documents: `scripts`
# (no __init__.py, an implicit namespace package) only resolves when the repo
# root is also on sys.path -- not guaranteed by every invocation shape.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_standalone_analysis import resolve_modules, run_duckdb_analyses  # noqa: E402


def _run_aws(args: list[str], tries: int = 3, backoff_s: float = 5.0) -> None:
    """Run an ``aws`` CLI subprocess, retrying transient failures -- mirrors
    run_standalone_analysis.py's own identical helper (real, previously-hit
    S3 network blips, always transient)."""
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


def _try_download(src: str, dst: Path) -> bool:
    """Best-effort ``aws s3 cp``: returns False (not an error) when the
    object simply doesn't exist -- e.g. a composite whose emitter couldn't be
    gathered generically (see ``run_pbg._persist_emitter_history``'s own
    best-effort contract), or a document with no file-backed emitter at all."""
    try:
        _aws_cp(src, str(dst))
        return dst.exists()
    except subprocess.CalledProcessError:
        return False


def run(
    *,
    composite_id: str,
    history_uri: str,
    out_uri: str,
    experiment_id: str,
    tmp: Path,
    n_seeds: int | None = None,
    n_generations: int = 1,
    modules: "str | dict[str, dict[str, Any]] | None" = None,
) -> dict[str, Any]:
    written: list[str] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    # Path 1: hive-parquet sweep, via the same DuckDB-httpfs mechanism
    # run_standalone_analysis.py already uses for the local batch_baseline
    # flush path. Only attempted when the caller told us there's something to
    # resolve -- n_seeds is the caller's own signal that history_uri actually
    # holds a hive-partitioned sweep, not just an unrelated flat-file dispatch.
    if n_seeds:
        # `modules is None` (not given at all) means "default to applicable";
        # an EXPLICITLY-passed empty dict/mapping means "no analyses
        # requested" and must stay empty, not silently become "applicable" --
        # `modules or "applicable"` would wrongly conflate the two (an empty
        # dict is falsy) and was a real bug caught by this file's own tests.
        resolved_modules = resolve_modules(
            "applicable" if modules is None else modules, n_seeds=n_seeds, n_generations=n_generations
        )
        for scale, entries in resolved_modules.items():
            w, e = run_duckdb_analyses(history_uri, scale, entries, tmp, out_uri.rstrip("/"))
            written.extend(w)
            errors.extend(e)

    if written:
        status = "done" if not errors else "partial"
        manifest = {
            "analysis_kind": "multi-node-composite-hive-parquet",
            "composite_id": composite_id,
            "experiment_id": experiment_id,
            "written": written,
            "errors": errors,
            "skipped": skipped,
            "status": status,
        }
        manifest_local = tmp / "_manifest.json"
        manifest_local.write_text(json.dumps(manifest, indent=2))
        _aws_cp(str(manifest_local), f"{out_uri.rstrip('/')}/_manifest.json")
        return manifest

    # Path 2 (original, unchanged): flat-file fallback -- colony's own shape
    # (item 88), and the path any Path-1 attempt that resolved to nothing
    # (no n_seeds given, or the DuckDB reads themselves found no data) falls
    # through to. `errors` from a Path-1 attempt that never wrote anything are
    # folded in below rather than dropped, so a real DuckDB-side failure is
    # still visible in the final manifest even though Path 2 also ran.
    run_dir = tmp / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    has_history = _try_download(f"{history_uri.rstrip('/')}/emitter_history.json", run_dir / "emitter_history.json")
    has_final_state = _try_download(f"{history_uri.rstrip('/')}/final_state.json", run_dir / "final_state.json")

    if not has_history and not has_final_state:
        errors.append(
            {"error": f"neither emitter_history.json nor final_state.json found under {history_uri}"}
        )
    else:
        try:
            import v2ecoli.workflow.post_sim_visualizations  # noqa: F401 -- populates VISUALIZATION_REGISTRY
            from v2ecoli.workflow.flush import run_flush

            result = run_flush(str(run_dir), config={}, ws_root=os.getcwd())
        except Exception as e:  # noqa: BLE001 -- surface any flush failure in the manifest, don't crash silently
            errors.append({"error": f"run_flush failed: {type(e).__name__}: {e}"})
        else:
            # run_flush's own per-step skips (e.g. a parquet-only analysis that
            # doesn't apply to this composite's non-parquet output shape) are
            # expected, benign no-ops for THIS script's purposes -- recorded for
            # debuggability, but deliberately not folded into `errors` (which
            # drives this manifest's own status below).
            skipped.extend(result.get("skipped") or [])
            for entry in result.get("placed") or []:
                local_path = Path(entry["path"])
                dest = f"{out_uri.rstrip('/')}/{local_path.name}"
                try:
                    _aws_cp(str(local_path), dest)
                    written.append(dest)
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr.decode()[:500] if e.stderr else str(e)
                    errors.append({"name": entry.get("name", "?"), "error": f"upload failed: {stderr}"})

    status = "done" if written and not errors else ("failed" if errors and not written else "partial")
    manifest = {
        "analysis_kind": "multi-node-composite",
        "composite_id": composite_id,
        "experiment_id": experiment_id,
        "written": written,
        "errors": errors,
        "skipped": skipped,
        "status": status,
    }
    manifest_local = tmp / "_manifest.json"
    manifest_local.write_text(json.dumps(manifest, indent=2))
    _aws_cp(str(manifest_local), f"{out_uri.rstrip('/')}/_manifest.json")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--composite-id", required=True)
    p.add_argument("--history-uri", required=True, help="s3:// prefix where run_pbg.py staged this dispatch's output")
    p.add_argument("--out-uri", required=True, help="s3:// prefix to write analysis output + _manifest.json")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--n-seeds", type=int, default=None,
                    help="if set, also try reading history_uri as a hive-parquet sweep "
                         "(lineage_ray_batch shape) via the same mechanism run_standalone_analysis.py uses")
    p.add_argument("--n-generations", type=int, default=1, help="only consulted when --modules is 'applicable'")
    p.add_argument("--modules", default="applicable",
                    help='JSON: {"scale": {"name": {}}}, or the keyword "applicable" (default)')
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        manifest = run(
            composite_id=args.composite_id,
            history_uri=args.history_uri,
            out_uri=args.out_uri,
            experiment_id=args.experiment_id,
            tmp=Path(td),
            n_seeds=args.n_seeds,
            n_generations=args.n_generations,
            modules=args.modules,
        )
    print(json.dumps(manifest, indent=2))
    if manifest["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
