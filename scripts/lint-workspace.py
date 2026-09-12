#!/usr/bin/env python3
"""Validate workspace.yaml + cross-references. Exit non-zero on failure."""
from __future__ import annotations
import graphlib
import hashlib
import json
import locale
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter

import yaml
from jsonschema import Draft7Validator, FormatChecker

# Shared package-naming (viva_<slug> canonical, pbg_<slug> compat on disk).
from viva_workspace import resolve_package_dir


def _try_get_registry(ws_root: Path, ws_data: dict) -> set | None:
    """Try to introspect build_core() and return a set of registered process names.

    Returns None if introspection fails (caller should warn, not fail).
    """
    # Explicit package_path wins; otherwise resolve via the shared naming shim,
    # which returns the canonical ``viva_<slug>`` or an existing legacy
    # ``pbg_<slug>`` directory during the rebrand migration window.
    package_name = ws_data.get("package_path") or resolve_package_dir(ws_root, ws_data.get("name")).name
    try:
        py = sys.executable
        script = f"""
import json, sys
try:
    from {package_name}.core import build_core
    core = build_core()
    try:
        names = sorted(core.process_registry.list())
    except Exception:
        try:
            names = sorted(core.process_registry.registry.keys())
        except Exception:
            names = []
    print(json.dumps(names))
except Exception as e:
    print(json.dumps([]))
"""
        result = subprocess.run(
            [py, "-c", script],
            cwd=ws_root, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        names = json.loads(result.stdout.strip().split("\n")[-1])
        if isinstance(names, list):
            return set(names)
        return None
    except Exception:
        return None


def _validate_simulation_processes(ws_data: dict, registry: set | None) -> list[str]:
    """Return error strings for simulation process refs not in registry.

    If registry is None (introspection failed), returns empty list (warning only).
    """
    if registry is None:
        return []
    errors = []
    for sim in ws_data.get("simulations", []) or []:
        if not isinstance(sim, dict):
            continue
        for proc in sim.get("processes", []) or []:
            if proc not in registry:
                errors.append(
                    f"simulation '{sim.get('name', '?')}' references missing process '{proc}'"
                )
    return errors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


WS_ROOT = Path(__file__).resolve().parents[1]
WS_FILE = WS_ROOT / "workspace.yaml"

# Resolve workspace directories via the optional `layout:` map in workspace.yaml
# (unset keys default to their flat top-level name). Keep in sync with the
# WorkspacePaths resolver (vivarium_workbench.lib.workspace_paths).
_LAYOUT_DEFAULTS = {
    "studies": "studies", "investigations": "investigations",
    "composites": "composites", "references": "references",
    "datasets": "datasets", "notes": "notes", "experiments": "experiments",
    "reports": "reports", "pbg": ".pbg", "scripts": "scripts",
    "tests": "tests", "docs": "docs",
}


def _load_layout() -> dict:
    cfg = {}
    if WS_FILE.exists():
        try:
            cfg = yaml.safe_load(WS_FILE.read_text()) or {}
        except Exception:
            cfg = {}
    layout = dict(_LAYOUT_DEFAULTS)
    for k, v in (cfg.get("layout") or {}).items():
        if k in _LAYOUT_DEFAULTS and isinstance(v, str) and v:
            layout[k] = v
    return layout


_LAYOUT = _load_layout()


def _dir(name: str) -> Path:
    """Absolute path to a workspace directory, honoring workspace.yaml `layout:`."""
    return WS_ROOT / _LAYOUT[name]


def _schema(name: str) -> dict:
    p = _dir("pbg") / "schemas" / name
    if not p.exists():
        sys.exit(f"missing schema at {p}; was workspace scaffolded?")
    return json.loads(p.read_text())


def _fail(msg: str) -> None:
    print(f"LINT FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Phase-1 canonical data model: documented exceptions to the canonical
# conditions-form rule (mirrors tests/test_workspace_conformance.py).
_NO_MODEL = {"parca"}                                          # upstream artifact producer, no model
_MULTI_BASELINE_PENDING = {"mbp-07-millard-kinetic-metabolism"}  # multi_baseline_needs_human (user decision)

# Study-config <-> generator contract (mirrors tests/test_workspace_conformance.py --
# keep both in sync):
_RUN_CONTROL_KEYS = {"n_steps"}                    # engine-stripped run-control key (workbench #611)
_PARAM_CONTRACT_EXCEPTIONS = {
    "metabolism_redux",              # swap: not an ecoli_baseline param; pending owner mapping
    "showcase-1-parca",              # mode: full: not a parca generator param (CLI-only flag)
    "mbp-04-multigeneration-runs",   # features: not a reactor_bird_coupled param; pending owner fix
}
_COMPOSITE_UNRESOLVED_PREFIXES = ("v2ecoli_pdmp.", "pbg_copasi.")  # optional/external packages, not in _REGISTRY
_COMPOSITE_UNRESOLVED_SUFFIXES = ("millard2017_metabolism",)        # file-discovered YAML composite
_COMPOSITE_UNRESOLVED_EXACT = {"v2ecoli.composites.diagnostic.diagnostic"}  # known-nonexistent, aspirational ref


def check_canonical_layout() -> None:
    """Enforce the Phase-1 canonical data model as human-readable lint errors.

    Same four structural checks as tests/test_workspace_conformance.py:
    no nested study.yaml; investigations members-only; canonical conditions
    form (no stray top-level baseline/parent_studies/pipeline_gate.prerequisites,
    except the two documented exceptions); inputs.from DAG acyclic + resolvable.
    """
    errors: list[str] = []

    nested = list((_dir("investigations")).glob("*/studies/*/study.yaml"))
    if nested:
        errors.append(f"nested study.yaml must not exist: {nested}")

    for inv in (_dir("investigations")).glob("*/investigation.yaml"):
        spec = yaml.safe_load(inv.read_text()) or {}
        if "studies" in spec:
            errors.append(f"{inv.parent.name} still uses studies: (must be members:)")

    studies: dict[str, dict] = {
        p.parent.name: (yaml.safe_load(p.read_text()) or {})
        for p in (_dir("studies")).glob("*/study.yaml")
    }

    for slug, spec in studies.items():
        if slug in _MULTI_BASELINE_PENDING:
            bl = spec.get("baseline")
            if not (isinstance(bl, list) and bl and all(b.get("composite") for b in bl)):
                errors.append(f"{slug}: multi-baseline exception must be a valid top-level baseline list")
            continue
        if "baseline" in spec:
            errors.append(f"{slug} has a stray top-level baseline: (should be conditions.baseline)")
        if "parent_studies" in spec:
            errors.append(f"{slug} retains parent_studies (ordering must be inputs.from)")
        if "depends_on" in spec:
            errors.append(f"{slug} retains legacy depends_on (ordering must be inputs.from)")
        pg = spec.get("pipeline_gate") or {}
        if "prerequisites" in pg:
            errors.append(f"{slug} retains pipeline_gate.prerequisites (must be inputs.from)")
        if slug not in _NO_MODEL:
            comp = ((spec.get("conditions") or {}).get("baseline") or {}).get("composite")
            if not comp:
                errors.append(f"{slug} missing conditions.baseline.composite")
        if isinstance(spec.get("conditions"), dict) and isinstance(spec.get("tests"), dict):
            errors.append(f"{slug} conditions-form study has dict-shaped tests (must be a list)")

    slugs = set(studies) | {"parca"}
    ts = graphlib.TopologicalSorter()
    for slug, spec in studies.items():
        deps = []
        for e in (spec.get("inputs") or []):
            frm = e.get("from")
            if frm not in slugs:
                errors.append(f"{slug} inputs.from '{frm}' is not a real study")
            else:
                deps.append(frm)
        ts.add(slug, *deps)
    try:
        ts.prepare()
    except graphlib.CycleError as e:
        errors.append(f"inputs.from DAG has a cycle: {e}")

    if errors:
        _fail("canonical layout violations:\n  " + "\n  ".join(errors))


def check_generator_contract() -> None:
    """Enforce the study-config<->generator contract as human-readable lint errors.

    Same two checks as tests/test_workspace_conformance.py's
    test_all_composite_refs_resolve + test_baseline_params_are_generator_accepted:
    every baseline/variant composite ref must resolve in the generator registry
    (modulo the documented skip-list), and every baseline params key must be a
    declared generator parameter, an engine-stripped run-control key, or a
    documented pending-fix exception. Best-effort: if viva_superpowers isn't
    importable, warns and skips rather than failing the whole lint.
    """
    try:
        from viva_superpowers.composite_generator import discover_generators, _REGISTRY
    except Exception as e:
        print(
            f"LINT WARNING: could not import viva_superpowers.composite_generator ({e}); "
            "generator-contract checks skipped.",
            file=sys.stderr,
        )
        return

    # discover_generators() (via a transitive import, e.g. polars) resets
    # LC_CTYPE to "C", which silently flips Path.read_text()'s default
    # encoding to ASCII for every later unguarded read in this process --
    # breaking the claims.yaml / expert_docs / study.yaml reads further down
    # in main() on machines whose locale isn't already ASCII. Save + restore
    # it around the call so this check has no side effect on the rest of the
    # script.
    _saved_lc_ctype = locale.setlocale(locale.LC_CTYPE)
    try:
        discover_generators()
    finally:
        locale.setlocale(locale.LC_CTYPE, _saved_lc_ctype)
    errors: list[str] = []

    for study_yaml_path in sorted((_dir("studies")).glob("*/study.yaml")):
        slug = study_yaml_path.parent.name
        spec = yaml.safe_load(study_yaml_path.read_text(encoding="utf-8")) or {}
        cond = spec.get("conditions") or {}
        base = cond.get("baseline") or {}

        refs = []
        b = base.get("composite")
        if b:
            refs.append(b)
        for v in (cond.get("variants") or []):
            if v.get("composite"):
                refs.append(v["composite"])
        for r in refs:
            if r in _REGISTRY or r in _COMPOSITE_UNRESOLVED_EXACT:
                continue
            if r.startswith(_COMPOSITE_UNRESOLVED_PREFIXES) or r.endswith(_COMPOSITE_UNRESOLVED_SUFFIXES):
                continue
            errors.append(f"{slug}: unresolved composite ref '{r}'")

        if slug in _PARAM_CONTRACT_EXCEPTIONS:
            continue
        comp, params = base.get("composite"), (base.get("params") or {})
        if not comp:
            continue
        entry = _REGISTRY.get(comp)
        if entry is None:
            continue  # unresolved composite already reported above
        gen_params = set((getattr(entry, "parameters", {}) or {}).keys())
        unknown = set(params) - gen_params - _RUN_CONTROL_KEYS
        if unknown:
            errors.append(f"{slug}: params {sorted(unknown)} not accepted by {comp}")

    if errors:
        _fail("generator contract violations:\n  " + "\n  ".join(errors))


def main() -> None:
    if not WS_FILE.exists():
        _fail(f"{WS_FILE} not found — run inside a workspace")
    ws = yaml.safe_load(WS_FILE.read_text())

    # Schema version guard: fail clearly for v1 workspaces.
    schema_ver = ws.get("schema_version")
    if schema_ver != 2:
        _fail(
            f"workspace.yaml is schema v{schema_ver}; "
            "this linter requires schema_version: 2 (see .pbg/schemas/workspace.schema.json)."
        )

    Draft7Validator(_schema("workspace.schema.json"), format_checker=FormatChecker()).validate(ws)

    # Phase-1 canonical data model (no nested study.yaml; investigations
    # members-only; canonical conditions form; inputs.from DAG acyclic).
    check_canonical_layout()

    # Study-config <-> generator contract (Tasks 1-3): composite refs resolve;
    # baseline params are a subset of the generator's declared parameters.
    check_generator_contract()

    # Datasets
    for d in ws.get("datasets", []):
        path, url, sha = d.get("path"), d.get("url"), d.get("sha256")
        if path is not None:
            full = WS_ROOT / path
            if not full.exists():
                _fail(f"dataset '{d['name']}' path missing: {path}")
            elif sha:
                actual = _sha256(full)
                if actual != sha:
                    _fail(f"dataset '{d['name']}' sha256 mismatch (recorded={sha[:16]}…, actual={actual[:16]}…)")
        elif url is not None:
            if not sha:
                _fail(f"dataset '{d['name']}' has url but no sha256")
        else:
            _fail(f"dataset '{d['name']}' has neither path nor url")

    # References cross-ref
    refs_yaml = _dir("references") / "claims.yaml"
    bib = _dir("references") / "papers.bib"
    if refs_yaml.exists() and bib.exists():
        claims = (yaml.safe_load(refs_yaml.read_text()) or {}).get("claims", {}) or {}
        bib_text = bib.read_text()
        bib_keys = set(re.findall(r"@\w+\{([A-Za-z0-9_:-]+),", bib_text))
        for claim, key in claims.items():
            if isinstance(key, list):
                for k in key:
                    if k not in bib_keys:
                        _fail(f"claim '{claim}' references missing bib key '{k}'")
            elif isinstance(key, str):
                if key not in bib_keys:
                    _fail(f"claim '{claim}' references missing bib key '{key}'")

    # Expert docs path existence + optional sha256 verification
    for doc in ws.get("expert_docs", []) or []:
        doc_path = doc.get("path", "")
        if doc_path:
            full = WS_ROOT / doc_path
            if not full.exists():
                _fail(
                    f"expert_docs entry '{doc.get('name', '?')}' path missing: {doc_path} "
                    f"(expected at {full})"
                )
            sha = doc.get("sha256")
            if sha:
                actual = _sha256(full)
                if actual != sha:
                    _fail(f"expert_docs '{doc.get('name', '?')}' sha256 mismatch (recorded={sha[:16]}…, actual={actual[:16]}…)")

    # References PDFs: always verify sha256 + bib_key cross-reference
    bib = _dir("references") / "papers.bib"
    bib_keys: set = set()
    if bib.exists():
        bib_text = bib.read_text()
        bib_keys = set(re.findall(r"@\w+\{([A-Za-z0-9_:-]+),", bib_text))
    for pdf_entry in ws.get("references_pdfs", []) or []:
        bib_key = pdf_entry.get("bib_key", "")
        pdf_path = pdf_entry.get("path", "")
        sha = pdf_entry.get("sha256", "")
        if bib_key and bib_keys and bib_key not in bib_keys:
            _fail(f"references_pdfs entry '{bib_key}' has no matching entry in papers.bib")
        if pdf_path:
            full = WS_ROOT / pdf_path
            if not full.exists():
                _fail(f"references_pdfs '{bib_key}' path missing: {pdf_path}")
            elif sha:
                actual = _sha256(full)
                if actual != sha:
                    _fail(f"references_pdfs '{bib_key}' sha256 mismatch (recorded={sha[:16]}…, actual={actual[:16]}…)")

    # Collect registered simulation names.
    registered_sim_names: set = set()
    for sim in ws.get("simulations", []) or []:
        if not isinstance(sim, dict):
            continue
        sim_name = sim.get("name", "?")
        registered_sim_names.add(sim_name)
        # t_end > t_start
        t_start = sim.get("t_start")
        t_end = sim.get("t_end")
        if t_start is not None and t_end is not None:
            if float(t_end) <= float(t_start):
                _fail(f"simulation '{sim_name}' has t_end ({t_end}) <= t_start ({t_start})")

    # Visualization simulation references.
    for viz in ws.get("visualizations", []) or []:
        if not isinstance(viz, dict):
            continue
        viz_name = viz.get("name", "?")
        # simulation reference
        viz_sim = viz.get("simulation")
        if viz_sim and registered_sim_names and viz_sim not in registered_sim_names:
            _fail(f"visualization '{viz_name}' references missing simulation '{viz_sim}'")

    # Simulation process references (best-effort: warn if registry unavailable).
    has_process_refs = any(
        isinstance(s, dict) and s.get("processes")
        for s in ws.get("simulations", []) or []
    )
    if has_process_refs:
        registry = _try_get_registry(WS_ROOT, ws)
        if registry is None:
            print(
                "LINT WARNING: Could not introspect build_core() — "
                "simulation process references not validated.",
                file=sys.stderr,
            )
        else:
            for err in _validate_simulation_processes(ws, registry):
                _fail(err)

    # ---- Summary -------------------------------------------------------
    # Count expert_docs
    expert_docs = ws.get("expert_docs", []) or []
    n_expert = len(expert_docs)

    # Count bib keys
    bib_file = _dir("references") / "papers.bib"
    bib_keys_found: set = set()
    if bib_file.exists():
        bib_keys_found = set(re.findall(r"@\w+\{([A-Za-z0-9_:-]+),", bib_file.read_text()))
    n_bib = len(bib_keys_found)

    # Count claims
    claims_file = _dir("references") / "claims.yaml"
    n_claims = 0
    if claims_file.exists():
        try:
            claims_data = yaml.safe_load(claims_file.read_text()) or {}
            if isinstance(claims_data, dict):
                n_claims = len(claims_data.get("claims", claims_data) or {})
            elif isinstance(claims_data, list):
                n_claims = len(claims_data)
        except Exception:
            # Fall back to line count
            n_claims = sum(1 for ln in claims_file.read_text().splitlines() if ln.strip() and not ln.startswith("#"))

    # Enumerate studies
    study_schema_path = _dir("pbg") / "schemas" / "study.schema.json"
    study_schema: dict | None = None
    if study_schema_path.exists():
        try:
            study_schema = json.loads(study_schema_path.read_text())
        except Exception:
            pass

    study_names: list[str] = []
    study_statuses: list[str] = []
    study_warnings: list[str] = []
    studies_missing_viz: list[str] = []

    # A study "has a visualization" if it declares one in study.yaml
    # (visualizations[] or embed_visualizations[]) OR ships a chart/viz artifact
    # on disk (charts/, viz/, or the canonical reports/figures/<slug>/ — the
    # same WS_ROOT/reports/figures dir the dashboard auto-discovers embeds from).
    figures_root = WS_ROOT / "reports" / "figures"

    def _has_visualization(study_dir_path, study_data) -> bool:
        if study_data.get("visualizations") or study_data.get("embed_visualizations"):
            return True
        if any(study_dir_path.glob("charts/*")):
            return True
        if any(study_dir_path.glob("viz/*")):
            return True
        slug_figs = figures_root / study_dir_path.name
        if slug_figs.is_dir() and any(slug_figs.iterdir()):
            return True
        return False

    for study_yaml_path in sorted((_dir("studies")).glob("*/study.yaml")):
        study_dir = study_yaml_path.parent.name
        try:
            study_data = yaml.safe_load(study_yaml_path.read_text()) or {}
            s_name = study_data.get("name", study_dir)
            s_status = study_data.get("status", "unknown")
            study_names.append(s_name)
            study_statuses.append(s_status)

            # Every study must demonstrate its claims with at least one
            # visualization (declared in study.yaml or a chart/viz artifact).
            if not _has_visualization(study_yaml_path.parent, study_data):
                studies_missing_viz.append(s_name)

            # Validate against schema if available
            if study_schema is not None:
                try:
                    from jsonschema import Draft202012Validator
                    validator = Draft202012Validator(study_schema)
                    errs = sorted(validator.iter_errors(study_data), key=lambda e: list(e.path))
                    for err in errs:
                        path_str = ".".join(str(p) for p in err.path) if err.path else "(root)"
                        study_warnings.append(f"  WARN study {s_name}: [{path_str}] {err.message}")
                except ImportError:
                    # jsonschema draft 2020-12 validator not available; fall back silently
                    try:
                        Draft7Validator(study_schema, format_checker=FormatChecker()).validate(study_data)
                    except Exception as ve:
                        study_warnings.append(f"  WARN study {s_name}: {ve.message}")
                except Exception as ve:
                    study_warnings.append(f"  WARN study {s_name}: {ve}")
        except Exception as exc:
            study_warnings.append(f"  WARN could not parse {study_yaml_path}: {exc}")

    if studies_missing_viz:
        _fail(
            "every study must have at least one visualization "
            "(study.yaml visualizations[], or a charts/ / viz/ / "
            "reports/figures/<slug>/ artifact); missing in: "
            + ", ".join(sorted(studies_missing_viz))
        )

    n_studies = len(study_names)
    status_counts = Counter(study_statuses)
    status_summary = ", ".join(f"{v} {k}" for k, v in sorted(status_counts.items()))

    # Count runs from both emitter paths: runs.db (sqlite history) +
    # parquet-runs/ (per-study hive tree).
    n_active_runs = 0
    n_completed_runs = 0
    for runs_db in (_dir("studies")).glob("*/runs.db"):
        try:
            import sqlite3
            conn = sqlite3.connect(str(runs_db))
            try:
                cur = conn.execute("SELECT status FROM runs")
                for (row_status,) in cur.fetchall():
                    if row_status in ("running", "pending"):
                        n_active_runs += 1
                    elif row_status == "completed":
                        n_completed_runs += 1
            except Exception:
                pass
            finally:
                conn.close()
        except Exception:
            pass
    # Each studies/<slug>/parquet-runs/<run_id>/ with a success/ sentinel
    # counts as one completed run.
    for parquet_root in (_dir("studies")).glob("*/parquet-runs"):
        for run_dir in parquet_root.iterdir():
            if not run_dir.is_dir():
                continue
            if (run_dir / "success").exists():
                n_completed_runs += 1
            else:
                n_active_runs += 1

    # Print summary
    ws_name = ws.get("name", "?")
    ws_pkg = ws.get("package_path", "")
    study_names_preview = ", ".join(study_names[:3])
    if n_studies > 3:
        study_names_preview += f", ... (+{n_studies - 3} more)"

    print("workspace lint: OK")
    print(f"  workspace: {ws_name}  (package: {ws_pkg})")
    print(f"  {n_expert} expert_docs, {n_bib} bib keys, {n_claims} claims")
    if n_studies == 0:
        print("  0 studies")
    else:
        print(f"  {n_studies} studies: {study_names_preview}  (status: {status_summary})")
    print(f"  {n_active_runs} active runs, {n_completed_runs} completed runs")

    for w in study_warnings:
        print(w, file=sys.stderr)

    # Cross-study biology consistency (P2b-7): validates enum-typed fields
    # against the registered bigraph-schema enums in v2ecoli/types/biology.py
    # AND checks that the same literature/model observable maps to the same
    # pool across studies. Fails the lint on enum violations or pool conflicts.
    biology_validator = WS_ROOT / "scripts" / "validate_study_biology.py"
    if biology_validator.exists():
        result = subprocess.run(
            [sys.executable, str(biology_validator)],
            cwd=WS_ROOT, capture_output=True, text=True, timeout=30,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode != 0:
            _fail("cross-study biology consistency check failed")


if __name__ == "__main__":
    main()
