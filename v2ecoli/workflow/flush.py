# v2ecoli/workflow/flush.py
"""The post-simulation analysis flush: extract a finished run once, then dispatch
to the unified POST_SIM_REGISTRY and place each output where the report renders
it. Plan 1 wires the report_card + visualization kinds; analyses keep their
existing run_analyses path (folded in by Plan 2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from viva_workspace import study_dir as _study_dir

from v2ecoli.workflow.report_cards import StudyContext, narrows_by_name

_STUDIES_RE = re.compile(r"(?:^|/)studies/([A-Za-z0-9_.\-]+)(?:/|$)")


def resolve_owning_study(out_dir: str, config: dict, ws_root) -> "str | None":
    """The study slug this run belongs to: config['study'] if set, else a
    studies/<slug>/ segment in out_dir, else None. Only returns a slug whose
    studies/<slug>/study.yaml exists under ws_root."""
    ws_root = Path(ws_root)
    slug = (config or {}).get("study")
    if not slug:
        m = _STUDIES_RE.search(str(out_dir).replace("\\", "/"))
        slug = m.group(1) if m else None
    if not slug:
        return None
    if (_study_dir(ws_root, slug) / "study.yaml").is_file():
        return slug
    return None


class RunExtract:
    """Lazy extraction context for a finished run. Heavy bits (DuckDB conn +
    sim_data) are provisioned only when conn_ctx()/records() are called."""

    def __init__(self, out_dir: str, config: dict, ws_root):
        self.out_dir = str(out_dir)
        self.config = config or {}
        self.ws_root = Path(ws_root)
        self.study_slug = resolve_owning_study(out_dir, config, ws_root)
        self._ctx: dict[str, Any] = {}
        self._records = None

    def study_ctx(self) -> "StudyContext | None":
        if not self.study_slug:
            return None
        return StudyContext.load(self.ws_root, self.study_slug)

    def study_viz_dir(self) -> "Path | None":
        if not self.study_slug:
            return None
        return _study_dir(self.ws_root, self.study_slug) / "viz"

    def records(self) -> list:
        if self._records is None:
            from v2ecoli.workflow.analysis_runner import build_cell_records
            self._records = list(build_cell_records(self.out_dir).values())
        return self._records

    def conn_ctx(self) -> tuple:
        if not self._ctx:
            import tempfile

            from viva_emitters import create_duckdb_conn

            from v2ecoli.library.sweep_io import apply_analysis_duckdb_config
            from v2ecoli.workflow.analysis_runner import (
                _history_from_clause, resolve_sim_data, resolve_validation_data)
            # Match the analysis runner's connection: temp_directory spill +
            # preserve_insertion_order (create_duckdb_conn) plus the explicit
            # memory budget, so a local flush reading a large hive spills instead
            # of OOMing the same way the dispatch path did.
            self._ctx["conn"] = create_duckdb_conn(temp_dir=tempfile.gettempdir())
            apply_analysis_duckdb_config(self._ctx["conn"])
            self._ctx["from_clause"] = _history_from_clause(self.out_dir)
            self._ctx["sim_data"] = resolve_sim_data(self.out_dir)
            self._ctx["validation_data"] = resolve_validation_data(self._ctx["sim_data"])
        return (self._ctx["conn"], self._ctx["from_clause"],
                self._ctx["sim_data"], self._ctx["validation_data"])

    def context_bag(self) -> dict:
        """The full provisioning bag the flush filters by each step's inputs().
        `study` is eager (cheap); `conn`/`sim_data`/`history_sql`/`records` are
        lazy CALLABLES so a step that doesn't declare them never triggers the
        heavy extraction."""
        return {
            "study": self.study_ctx(),
            "out_dir": self.out_dir,
            "conn": None,
            "history_sql": "",
            "config_sql": "",
            "success_sql": "",
            "sim_data": None,
            "validation_data": None,
            "_conn_ctx": self.conn_ctx,   # callable: () -> (conn, from_clause, sim_data, validation_data)
            "_records": self.records,     # callable: () -> records
        }

    def close(self) -> None:
        conn = self._ctx.get("conn")
        if conn is not None:
            conn.close()
        self._ctx = {}


def _write_html(path, html: str):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def place_output(kind: str, name: str, view: str, data: dict,
                 extract: "RunExtract") -> "str | None":
    """Route one step's output to the owning study's report location by kind.
    Returns the written html path (str) or None if nothing was written."""
    if not view:
        return None
    if kind == "report_card":
        from v2ecoli.workflow.report_cards import write_card
        ctx = extract.study_ctx()
        if ctx is not None:
            return str(write_card(ctx, name, data or {}, view))
        # no study: drop the card next to the run so it is not lost
        return str(_write_html(Path(extract.out_dir) / "viz" / "report_card" / f"{name}.html", view))
    # visualization / analysis view
    viz = extract.study_viz_dir() or (Path(extract.out_dir) / "viz")
    return str(_write_html(viz / f"{name}.html", view))


def _run_one_step(cls, kind, extract, core):
    """Instantiate + run one post-sim step; return (view, data). For report
    cards we call build()/applies() directly (their native API); visualizations
    and analyses go through update() with an inputs()-filtered bag."""
    step = cls({}, core=core)
    bag = extract.context_bag()
    inputs = {}
    try:
        inputs = step.inputs() or {}
    except Exception:  # noqa: BLE001
        inputs = {}
    # report cards: skip when applies() is False; build() returns (verdict, html)
    if kind == "report_card":
        ctx = bag.get("study")
        if ctx is None:
            return "", {}
        # honor a study's optional report_cards: allowlist (parity with the
        # standalone runner's report_cards.applicable) — a declared list of card
        # NAMES narrows which cards run; an absent key, or a machine-generated
        # list of embed paths, leaves every applicable card eligible.
        declared = (getattr(ctx, "spec", None) or {}).get("report_cards")
        if declared is not None and narrows_by_name(declared) and step.name not in declared:
            return "", {}
        if not step.applies(ctx):
            return "", {}
        res = step.build(ctx)
        if not res:
            return "", {}
        verdict, html = res
        return html, verdict
    # analyses/visualizations declaring DuckDB inputs get the lazy conn ctx
    state = {}
    for key in inputs:
        if key in ("conn", "history_sql", "sim_data", "validation_data") and bag.get("conn") is None:
            conn, from_clause, sim_data, validation_data = bag["_conn_ctx"]()
            state.update({"conn": conn, "history_sql": from_clause,
                          "sim_data": sim_data, "validation_data": validation_data})
        elif key in bag:
            state[key] = bag[key]
    out = step.update(state) or {}
    return out.get("view", ""), out.get("data", {}) or {}


def _flush_analyses(extract: "RunExtract", config: dict) -> tuple:
    """Run the configured analyses over the run, then copy their outputs into the
    owning study's report dir. Returns (placed, skipped). Empty analysis_options
    -> ([], []). A crash outside any single analysis (e.g. no sim_data
    resolvable) -> ([], [{"name":"analyses","error":...}]).

    P1-10 (CD2 audit §3.7): run_analyses() itself no longer raises just
    because one named analysis errored or a KPI column was missing from the
    emitted parquet -- it returns a structured summary instead (see its
    docstring). That summary's per-failure detail is folded into `skipped`
    here so a PARTIAL analysis run surfaces the same way a raised exception
    always has, instead of vanishing once run_analyses() stopped raising for
    it: `run_flush`'s caller (`_maybe_flush`) marks the whole run PARTIAL
    whenever `skipped` is non-empty."""
    analysis_options = (config or {}).get("analysis_options") or {}
    if not any(analysis_options.values()):
        return [], []
    try:
        from v2ecoli.workflow.analysis_runner import run_analyses
        summary = run_analyses(extract.out_dir, analysis_options) or {}
        skipped = []
        if summary.get("status") == "PARTIAL":
            for err in summary.get("errors") or []:
                skipped.append({"name": "analyses", **err})
        return place_analysis_outputs(extract), skipped
    except Exception as e:  # noqa: BLE001 — never abort the flush
        return [], [{"name": "analyses", "error": f"{type(e).__name__}: {e}"}]


def run_flush(out_dir, config, ws_root, *, core=None,
              kinds=("analysis", "report_card", "visualization")) -> dict:
    """Dispatch the registered post-sim steps of the given kinds over a finished
    run and place each output where the study report renders it. Graceful
    per-step skip. The 'analysis' kind calls _flush_analyses (not the per-step
    iter_post_sim loop, which is for report_card/visualization)."""
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.post_sim import iter_post_sim
    if core is None:
        core = allocate_core()
    extract = RunExtract(out_dir, config, ws_root)
    placed, skipped = [], []
    try:
        for kind in kinds:
            if kind == "analysis":
                a_placed, a_skipped = _flush_analyses(extract, config)
                placed.extend(a_placed)
                skipped.extend(a_skipped)
                continue
            for name, cls in iter_post_sim(kind):
                try:
                    view, data = _run_one_step(cls, kind, extract, core)
                    path = place_output(kind, name, view, data, extract)
                except Exception as e:  # noqa: BLE001 — one step never aborts the flush
                    skipped.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                    continue
                if path:
                    placed.append({"kind": kind, "name": name, "path": path})
    finally:
        extract.close()
    return {"placed": placed, "skipped": skipped, "study": extract.study_slug}


def place_analysis_outputs(extract: "RunExtract") -> list:
    """Copy a finished run's analysis artifacts into the owning study's report
    dir so the study report surfaces them: <out_dir>/viz/*.html ->
    <study viz>/<stem>.html and <out_dir>/ptools/*.tsv -> <study>/ptools/<stem>.tsv.
    Returns [{"kind":"analysis","name":<stem>,"path":<dest>}] per copied html.
    No owning study -> returns [] (the run's out_dir stays the home)."""
    import shutil

    study_viz = extract.study_viz_dir()
    if study_viz is None:
        return []
    placed = []
    src_viz = Path(extract.out_dir) / "viz"
    if src_viz.is_dir():
        study_viz.mkdir(parents=True, exist_ok=True)
        for html in sorted(src_viz.glob("*.html")):
            dest = study_viz / html.name
            shutil.copyfile(html, dest)
            placed.append({"kind": "analysis", "name": html.stem, "path": str(dest)})
    src_ptools = Path(extract.out_dir) / "ptools"
    if src_ptools.is_dir():
        study_ptools = study_viz.parent / "ptools"
        study_ptools.mkdir(parents=True, exist_ok=True)
        for tsv in sorted(src_ptools.glob("*.tsv")):
            shutil.copyfile(tsv, study_ptools / tsv.name)
    return placed


def main(argv=None) -> int:
    """Re-run the post-sim flush over an EXISTING run dir (parity with
    analysis_runner.main). Useful to regenerate a study's report cards /
    visualizations / analyses without re-simulating.

    Usage: python -m v2ecoli.workflow.flush <out_dir> [--study SLUG]
           [--ws-root DIR] [--kinds report_card,visualization,analysis]
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("out_dir", help="the finished run's output dir")
    ap.add_argument("--study", default=None, help="owning study slug (else inferred from out_dir)")
    ap.add_argument("--ws-root", default=None, help="workspace root (default: cwd)")
    ap.add_argument("--kinds", default="report_card,visualization,analysis",
                    help="comma-separated post-sim kinds to run")
    args = ap.parse_args(argv)

    ws_root = args.ws_root or os.getcwd()
    config = {"study": args.study} if args.study else {}
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    res = run_flush(args.out_dir, config, ws_root, kinds=kinds)
    print(f"placed {len(res['placed'])}, skipped {len(res['skipped'])}, "
          f"study={res.get('study')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
