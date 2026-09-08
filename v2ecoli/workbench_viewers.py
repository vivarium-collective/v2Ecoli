"""Vivarium-workbench analysis viewers contributed by v2ecoli.

The workbench (vivarium-workbench) discovers a ``<pkg>.workbench_viewers`` module
on the workspace package and every installed ``pbg-*`` distribution, and calls its
``get_viewers(ws_root)`` to render extra cards on the Analyses page. This is the
generic contribution seam; v2ecoli uses it to ship the **Pathway Tools Omics
Viewer** launcher, which used to be hardcoded inside the workbench itself.

The viewer overlays a study's exported PTools TSVs (written by
``v2ecoli.workflow.analyses.ptools_*``) onto the E. coli metabolic map in a
Pathway Tools / EcoCyc "Cellular Overview" Omics Viewer. It only appears when the
workspace configures ``ui.ptools_server_url`` in ``workspace.yaml``.

Nothing here imports at workbench startup unless PTools is configured — the module
is loaded lazily by the workbench's discovery pass and is otherwise inert.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import quote

import yaml

# Default targets the sms-ptools 0.8.2 Omics Viewer auto-load endpoint
# (omics=t&url=…&class=…&column1=…). Override via ui.ptools_omics_url_template.
# Placeholders: {server},{orgid},{tsv_url},{cls},{columns}.
_PTOOLS_DEFAULT_OMICS_URL_TEMPLATE = (
    "{server}/overviewsWeb/celOv.shtml"
    "?omics=t&url={tsv_url}&orgid={orgid}&class={cls}&column1={columns}"
)


def _ui_config(ws_root: Path) -> dict:
    """The ``ui:`` block from the workspace's ``workspace.yaml`` (or {})."""
    try:
        ws = yaml.safe_load(
            (Path(ws_root) / "workspace.yaml").read_text(encoding="utf-8")
        ) or {}
    except Exception:
        return {}
    return ws.get("ui") or {}


def _ptools_configured(ws_root: Path) -> bool:
    return bool((_ui_config(ws_root).get("ptools_server_url") or "").strip())


def _study_dir(ws_root: Path, slug: str) -> Path:
    """Resolve a study's directory via the shared ``viva_workspace`` resolver,
    which honours the ``layout:`` remap in workspace.yaml and searches nested
    ``investigations/<inv>/studies/`` in addition to the flat ``studies/`` root.

    ``must_exist`` is left at its default (False): the caller (``ptools_launch``)
    guards on ``sd.is_dir()`` and returns its own 404, so a not-yet-existing slug
    should resolve to the canonical path rather than raise here."""
    from viva_workspace import study_dir as _sd
    return Path(_sd(ws_root, slug))


def ptools_object_class(name: str) -> str:
    """Infer the Pathway Tools object class from an analysis/TSV name.

    The Omics Viewer needs to know whether the row IDs are genes, reactions,
    proteins, or compounds. v2ecoli's ptools analyses are named accordingly
    (ptools_rna → genes, ptools_rxns → reactions, ptools_proteins → proteins).
    """
    n = name.lower()
    if "rxn" in n or "reaction" in n:
        return "reaction"
    if "protein" in n:
        return "protein"
    if "metabolite" in n or "compound" in n:
        return "compound"
    return "gene"  # rna / default


def build_ptools_launch_url(
    study_dir: "Path | str",
    ws_root: "Path | str",
    ptools_server_url: str,
    ptools_omics_url_template: str,
    public_base: str,
    analysis: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> dict:
    """Discover ptools TSVs under *study_dir* and build an Omics-Viewer URL.

    Returns ``{url, tsv_url, available}`` on success, or ``{error, available}``.
    Two data-delivery modes: HTTP (``tsv_url`` on the dashboard's reachable host)
    or filesystem (``data_dir`` set → server-local ``<data_dir>/<rel>``).
    """
    study_dir = Path(study_dir)
    ws_root = Path(ws_root)

    all_tsvs = sorted(study_dir.glob("**/ptools/*.tsv"))
    if analysis:
        prefix = f"{analysis}__"
        all_tsvs = [p for p in all_tsvs if p.name.startswith(prefix)]

    def _relpath(p: Path) -> str:
        try:
            return p.relative_to(ws_root).as_posix()
        except ValueError:
            return p.as_posix()

    available = [_relpath(p) for p in all_tsvs]
    if not available:
        return {"error": "no ptools TSVs found for this run", "available": []}

    # Default overlay: prefer the combined "overview" export (genes + reactions +
    # proteins in one EcoCyc "a mixture" upload) so the launch is deterministic
    # and biologically complete. Fall back to any ``ptools_*`` export, then to
    # the first file — so a study whose ``ptools/`` dir also holds unrelated TSVs
    # (e.g. ``cd1_*`` omics tables that sort earlier alphabetically) still
    # launches the PTools overview rather than the alphabetically-first file.
    def _overlay_rank(p: Path) -> tuple:
        n = p.name.lower()
        if "overview" in n:
            return (0, n)
        if n.startswith("ptools_"):
            return (1, n)
        return (2, n)

    chosen = min(all_tsvs, key=_overlay_rank)
    rel = _relpath(chosen)
    if data_dir:
        tsv_url = f"{data_dir.rstrip('/')}/{rel}"
    else:
        tsv_url = f"{public_base.rstrip('/')}/{rel}"

    cls = ptools_object_class(analysis or chosen.name)

    # Animate across every data column.
    columns = "1"
    try:
        for line in chosen.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", ";")):
                continue
            ncol = len(line.split("\t")) - 1  # minus the name/ID column
            columns = f"1-{ncol}" if ncol > 1 else "1"
            break
    except Exception:
        pass

    # Percent-encode the nested TSV URL so strict URL parsers (WebKit/Safari
    # window.open) accept the Omics-Viewer query string. PTools decodes it.
    launch_url = ptools_omics_url_template.format(
        server=ptools_server_url.rstrip("/"),
        tsv_url=quote(tsv_url, safe=""),
        orgid="ECOLI",
        cls=cls,
        columns=columns,
    )
    return {"url": launch_url, "tsv_url": tsv_url, "available": available}


def _launch(ws_root, study, run, ctx) -> dict:
    """Workbench launcher callback: build the PTools Omics-Viewer URL for a study.

    Returns ``{"url": ...}`` on success, or ``{"error": ..., "status": ...}``.
    (``run`` is accepted for signature compatibility; TSV discovery globs the
    whole study directory.)
    """
    ws_root = Path(ws_root)
    ui = _ui_config(ws_root)

    ptools_server_url = (ui.get("ptools_server_url") or "").strip()
    if not ptools_server_url:
        return {"error": "ptools_server_url not configured", "status": 400}

    if not (study or run):
        return {"error": "study or run is required", "status": 400}

    template = ui.get("ptools_omics_url_template", _PTOOLS_DEFAULT_OMICS_URL_TEMPLATE)

    # workspace.yaml config wins over the request-derived public_base.
    public_base = (ctx or {}).get("public_base") or "http://localhost"
    cfg_base = (ui.get("dashboard_public_base_url") or "").strip()
    if cfg_base:
        public_base = cfg_base

    data_dir = (ui.get("ptools_data_dir") or "").strip() or None

    # A landed RUN (e.g. a GovCloud compose analysis, whose ptools TSVs the
    # workbench land path copies into .pbg/runs/<id>/ptools/) is launched by
    # ``run``; a local study by ``study``. Either resolves to a base directory
    # whose ptools/*.tsv exports build_ptools_launch_url discovers.
    if run:
        sd = _run_dir(ws_root, run)
        if not sd.is_dir():
            return {"error": f"run not found: {run}", "status": 404}
    else:
        sd = _study_dir(ws_root, study)
        if not sd.is_dir():
            return {"error": f"study not found: {study}", "status": 404}

    result = build_ptools_launch_url(
        study_dir=sd,
        ws_root=ws_root,
        ptools_server_url=ptools_server_url,
        ptools_omics_url_template=template,
        public_base=public_base,
        analysis=None,
        data_dir=data_dir,
    )
    if "error" in result:
        result.setdefault("status", 404)
    return result


def _studies_root(ws_root: Path) -> Path:
    """The directory holding study subdirs. Uses the shared ``viva_workspace``
    ``WorkspacePaths.studies`` resolver (honours a ``layout:`` remap /
    ``workspace/`` subdir)."""
    from viva_workspace import WorkspacePaths
    return WorkspacePaths.load(ws_root).studies


def _runs_root(ws_root: Path) -> Path:
    """The directory holding landed run subdirs (``<ws>/.pbg/runs``). Remote runs —
    including GovCloud compose analyses — land here; the workbench copies their
    ptools TSVs into ``.pbg/runs/<id>/ptools/``. Uses the shared ``viva_workspace``
    resolver (honours a ``layout:`` remap) with a plain fallback."""
    try:
        from viva_workspace import WorkspacePaths
        return WorkspacePaths.load(ws_root).pbg / "runs"
    except Exception:
        return Path(ws_root) / ".pbg" / "runs"


def _run_dir(ws_root: Path, run: str) -> Path:
    return _runs_root(Path(ws_root)) / run


def _ptools_targets(ws_root) -> list:
    """Studies and landed runs that have exported ``ptools/*.tsv`` files — the
    launchable rows. A local study is keyed by ``study``; a landed run (e.g. a
    GovCloud compose analysis) by ``run``, so the whole menu spans both."""
    ws_root = Path(ws_root)
    out = []
    root = _studies_root(ws_root)
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            n = len(list(p.glob("**/ptools/*.tsv")))
            if n:
                out.append({
                    "study": p.name,
                    "label": p.name,
                    "detail": f"{n} TSV{'' if n == 1 else 's'}",
                })
    runs_root = _runs_root(ws_root)
    if runs_root.is_dir():
        for p in sorted(runs_root.iterdir()):
            if not p.is_dir():
                continue
            n = len(list(p.glob("**/ptools/*.tsv")))
            if n:
                out.append({
                    "run": p.name,
                    "label": p.name,
                    "detail": f"{n} TSV{'' if n == 1 else 's'} · run",
                })
    return out


def get_viewers(ws_root) -> list:
    """Contribute the analysis viewers v2ecoli ships:

    * **Pathway Tools — Omics Viewer** (launcher; needs a configured PTools server).

    The Data Explorer (timeseries / scatter / flux maps for any run) is the
    workbench's own built-in and is shown alongside these automatically.
    """
    return [
        {
            "id": "pathway-tools",
            "title": "Pathway Tools — Omics Viewer",
            "description": (
                "Overlay a study's PTools TSV exports (genes / reactions / "
                "proteins / compounds) on the E. coli metabolic map in the "
                "EcoCyc Cellular Overview Omics Viewer."
            ),
            "kind": "launcher",
            "applies": _ptools_configured,
            "launch": _launch,
            "targets": _ptools_targets,
        },
    ]
