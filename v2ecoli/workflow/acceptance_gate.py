"""Acceptance gate: turn silent-success into loud failure for a run's output.

A run can exit 0, divide, and write a valid-looking store while producing NONE of
the observables the study exists to measure. A declared emit path that yields no
column is silent (sms-ecoli#210: "the check is the column list, not the exit
code"); a swapped process can be dropped between API and runner and the run still
completes green on the wild-type; a KPI column can be present and read exactly
zero. Presence/effect checks (a non-empty store, global_time advanced) pass all of
these.

This gate asserts, FROM THE ARTIFACT, what a study's output must satisfy:

  1. Required columns exist in the hive-partitioned history parquet, each with at
     least one non-null row. The required list is the study's OUTPUT CONTRACT and
     belongs in the versioned run config, not in code. A column can also be held
     to a fixed value (``must_equal``, e.g. the study's media_id -- a run against
     the wrong condition carries a present, single-valued column a bare presence
     check waves through) or required to vary (``must_vary``, a dead channel).
  2. (optional) The composite that actually ran contained the processes the
     submission declared, and none it declared forbidden -- closes the wild-type
     / dropped-injected_processes class, and the wrong-strain-still-carrying-the-
     stock-process case, without first winning the root-cause hunt for where they
     drop.
  3. (optional) The bulk species count matches the expected strain fingerprint
     (16,321 stock vs 16,323+ genuine) -- catches a candidate run staged from the
     stock cache whose columns and processes all look right, the failure class one
     leg downstream of the viva-api#437 staging fix.

It is route-independent: importable from run_pbg, a raw compute job, or a
post-run check, and it reads local paths or s3:// the same way (via sweep_io).
It writes a verdict.json so status can be derived from the verdict rather than a
container exit code (which has been wrong in both directions, sms-ecoli#210).

Exit codes: 0 = PASS, 1 = FAIL, 2 = the gate could not run (bad inputs). Kept
distinct from check_run_complete.py's exit-2-for-expected-partial convention so a
wrapper never confuses "gate could not run" with "run incomplete".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from v2ecoli.library.sweep_io import configure_duckdb_s3, history_files, is_s3_uri


def check_columns(
    sweep_dir: str,
    required_columns: Iterable[str],
    *,
    must_vary: Iterable[str] = (),
    must_equal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assert each required column exists in the sweep's history parquet with at
    least one non-null row, and (for ``must_vary`` columns) that it actually
    varies.

    Column names are the double-underscore-flattened emit paths the parquet
    carries (e.g. ``listeners__mass__dry_mass``), NOT slash paths. Returns a dict
    with ``passed`` and a per-column ``{present, nonnull_rows, distinct}`` map.

    A missing column and a present-but-all-null column both fail. For any column
    named in ``must_vary``, ``distinct <= 1`` also fails: a column pinned at its
    seed value is a dead channel wearing a column name (cplong90, sms-ecoli#210),
    and it passes a bare presence check. ``distinct`` is reported for every column
    so a reviewer can spot a dead channel even when it is not enforced.

    ``must_equal`` maps a column to the value EVERY non-null row must hold (e.g.
    ``{"environment__media_id": "basal_with_trp"}``). A run against the wrong
    media or condition carries a present, non-null, single-valued column that a
    presence check waves through; must_equal is how a study pins the condition it
    meant to run. A column named in must_equal need not be in ``required_columns``
    (it is checked when present); any non-null row that differs, or the column
    being absent, fails. The per-column entry gains ``{expected, mismatch_rows}``.
    """
    import duckdb

    required = list(required_columns)
    must_vary_set = set(must_vary)
    must_equal = dict(must_equal or {})
    # must_equal columns are checked even if not in the required list.
    all_cols = list(dict.fromkeys([*required, *must_equal]))
    files = history_files(sweep_dir)
    if not files:
        return {
            "passed": False,
            "error": f"no hive history parquet under {sweep_dir!r}",
            "columns": {c: {"present": False, "nonnull_rows": 0, "distinct": 0}
                        for c in all_cols},
            "n_files": 0,
        }
    con = duckdb.connect()
    if is_s3_uri(sweep_dir):
        configure_duckdb_s3(con)
    flist = "[" + ",".join(_sql_str(f) for f in files) + "]"
    src = f"read_parquet({flist}, hive_partitioning=true)"
    available = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}

    columns: dict[str, Any] = {}
    for col in all_cols:
        if col not in available:
            columns[col] = {"present": False, "nonnull_rows": 0, "distinct": 0}
            continue
        ident = _sql_ident(col)
        n, distinct = con.execute(
            f"SELECT count({ident}), count(DISTINCT {ident}) FROM {src}"
        ).fetchone()
        entry = {"present": True, "nonnull_rows": int(n), "distinct": int(distinct)}
        if col in must_equal:
            expected = must_equal[col]
            # A non-null row that differs from the expected value is a mismatch;
            # nulls are caught by the nonnull_rows check, not counted here.
            (mismatch,) = con.execute(
                f"SELECT count(*) FROM {src} "
                f"WHERE {ident} IS NOT NULL AND {ident} IS DISTINCT FROM ?",
                [expected],
            ).fetchone()
            entry["expected"] = expected
            entry["mismatch_rows"] = int(mismatch)
        columns[col] = entry

    def _ok(name: str, c: dict) -> bool:
        if not (c["present"] and c["nonnull_rows"] > 0):
            return False
        if name in must_vary_set and c["distinct"] <= 1:
            return False
        if name in must_equal and c.get("mismatch_rows", 0) > 0:
            return False
        return True

    # required columns must all pass; must_equal-only columns must also hold.
    checked = set(required) | set(must_equal)
    passed = bool(checked) and all(_ok(name, columns[name]) for name in checked)
    return {"passed": passed, "columns": columns, "n_files": len(files),
            "must_vary": sorted(must_vary_set),
            "must_equal": {k: must_equal[k] for k in sorted(must_equal)}}


def process_names_from_state(state: dict) -> set[str]:
    """The process ADDRESSES mounted in a built composite state (agents included).

    Walks the state tree for nodes that look like a process/step instance
    (``_type`` in {process, step} with an ``address``), and returns the set of
    addresses. Feed this to :func:`check_composition` as the "what actually ran"
    side. Address, not store key, because the store key is a wiring name
    (``ecoli-metabolism``) while the address is what was mounted
    (``local:MetabolismReduxClassic``) -- a swap changes the address.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            t = node.get("_type")
            addr = node.get("address")
            if isinstance(addr, str) and t in ("process", "step"):
                found.add(addr)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(state)
    return found


def check_composition(
    ran: Iterable[str],
    declared: Iterable[str],
    *,
    forbidden: Iterable[str] = (),
) -> dict[str, Any]:
    """Assert every declared process is present in what actually ran, and that no
    ``forbidden`` process is.

    ``ran``: the process addresses mounted in the composite (e.g. from
    :func:`process_names_from_state`). ``declared``: the EXPECTED mounted
    class/address for each process the run must contain -- e.g. for a
    ``swap_processes={ecoli-metabolism: ecoli-metabolism-redux}`` submission the
    study declares ``MetabolismReduxClassic`` (the class the swap mounts), because
    the registry name ``ecoli-metabolism-redux`` does not textually appear in the
    mounted address ``local:MetabolismReduxClassic``. The study's config owns that
    mapping, the same way it owns the required-column list.

    Match is a case-insensitive one-directional substring: a declared class is
    present iff it appears WITHIN some mounted address (``MetabolismReduxClassic``
    within ``local:MetabolismReduxClassic``). The match is NOT run the other way:
    an earlier version also passed when a mounted address was a substring of the
    declared name, so a short mounted address (``local:Metabolism``) satisfied a
    longer declared class (``MetabolismReduxClassic``) it is not -- a false pass
    on exactly the wild-type-not-swapped run this check exists to catch.

    ``forbidden``: classes that must be ABSENT after a swap -- e.g. declare the
    stock ``MetabolismFBA`` forbidden alongside the redux class required, so a run
    that mounted BOTH (or only the stock one) fails even though the required class
    is present. Catches the wrong-strain-still-carrying-the-stock-process case.
    """
    ran_low = [r.lower() for r in ran]
    declared_list = list(declared)
    forbidden_list = list(forbidden)
    missing = [d for d in declared_list if not any(d.lower() in r for r in ran_low)]
    present_forbidden = [f for f in forbidden_list if any(f.lower() in r for r in ran_low)]
    return {
        "passed": not missing and not present_forbidden,
        "missing": missing,
        "forbidden_present": present_forbidden,
        "declared": declared_list,
        "forbidden": forbidden_list,
    }


def bulk_species_count_from_state(state: dict) -> int | None:
    """Best-effort count of bulk molecular species in a composite state.

    The species-set size is the cheapest strain fingerprint we have: a stock
    build carries 16,321 bulk species, a genuine new-gene composition 16,323 /
    16,339 / 16,341 (cplong90's control, sms-ecoli#210). A candidate run whose
    columns and processes all look right but whose bulk count equals the stock's
    was staged from the wrong cache -- the failure class one leg downstream of the
    viva-api#437 staging fix.

    Walks the state for the FIRST node under a ``bulk`` key that is a dict or a
    list and returns its length (bulk is emitted per-agent as a mapping of
    molecule id -> count, or a list of records). Returns None if no bulk store is
    found, so a caller can tell "no bulk in this artifact" from a real count.
    """
    found: list[int] = []

    def walk(node: Any, key: str | None = None) -> None:
        if found:
            return
        if key == "bulk" and isinstance(node, (dict, list)) and node:
            # a bulk store, not a scalar leaf named "bulk"
            if isinstance(node, dict) and all(
                isinstance(v, (int, float)) for v in list(node.values())[:4]
            ):
                found.append(len(node))
                return
            if isinstance(node, list):
                found.append(len(node))
                return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk(state)
    return found[0] if found else None


def check_species_count(count: int | None, expected: int) -> dict[str, Any]:
    """Assert the bulk species count equals the expected strain fingerprint.

    ``count`` is a resolved integer (e.g. from :func:`bulk_species_count_from_state`
    over a ``final_state.json``); None means the count could not be read, which
    fails -- an unreadable fingerprint is not a pass.
    """
    passed = count is not None and count == expected
    return {"passed": passed, "count": count, "expected": expected}


def run_gate(
    sweep_dir: str,
    required_columns: Iterable[str],
    *,
    must_vary: Iterable[str] = (),
    must_equal: dict[str, Any] | None = None,
    ran_processes: Iterable[str] | None = None,
    declared_processes: Iterable[str] | None = None,
    forbidden_processes: Iterable[str] = (),
    species_count: int | None = None,
    expected_species_count: int | None = None,
) -> dict[str, Any]:
    """Run the column check and, when the inputs are given, the composition and
    species-count checks, and return a single verdict dict. ``passed`` is the AND
    of every check that ran."""
    verdict: dict[str, Any] = {
        "passed": False,
        "sweep_dir": sweep_dir,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "columns_check": check_columns(sweep_dir, required_columns,
                                       must_vary=must_vary, must_equal=must_equal),
    }
    passed = verdict["columns_check"]["passed"]
    if declared_processes is not None or forbidden_processes:
        verdict["composition_check"] = check_composition(
            ran_processes or [], declared_processes or [],
            forbidden=forbidden_processes,
        )
        passed = passed and verdict["composition_check"]["passed"]
    if expected_species_count is not None:
        verdict["species_check"] = check_species_count(
            species_count, expected_species_count
        )
        passed = passed and verdict["species_check"]["passed"]
    verdict["passed"] = passed
    return verdict


def write_verdict(verdict: dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(verdict, indent=2))


# The report-card verdict vocabulary (viva_superpowers.card_grade): within_tol
# (pass) / drift (warn) / mismatch (fail) / ungraded. Kept here as plain strings
# so the engine has no import dependency on the report-card layer -- the
# translation below lets a card express a gate result in that one shared schema.
_PASS, _WARN, _FAIL, _NA = "within_tol", "drift", "mismatch", "ungraded"
_RANK = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def report_from_gate_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    """Translate a :func:`run_gate` verdict into a grade_card-shaped report
    (``{overall, axes: {path: {group, label, verdict, value, meter, detail}}}``),
    the input :func:`viva_superpowers.card_grade.verdict_json` expects.

    This is the seam that unifies the gate with the report cards: the gate stays a
    route-independent check engine, and a report card feeds this report through the
    SAME verdict_json / render_verdict_html the science cards use -- one schema,
    one report surface, no parallel verdict format. Each check becomes one axis;
    ``overall`` is the worst axis verdict, so a card built from this dominates a
    study's worst-verdict rollup on any failure.
    """
    axes: dict[str, dict] = {}

    def add(path: str, group: str, label: str, verdict_: str,
            value: Any = None, meter: str = "", detail: dict | None = None) -> None:
        axes[path] = {"group": group, "label": label, "verdict": verdict_,
                      "value": value, "meter": meter, "detail": detail or {},
                      "path": path}

    cc = verdict.get("columns_check", {})
    if "error" in cc:
        add("output/history", "Output", "hive history parquet", _FAIL,
            meter=cc["error"])
    must_vary = set(cc.get("must_vary", []) or [])
    must_equal = cc.get("must_equal", {}) or {}
    for col, info in (cc.get("columns") or {}).items():
        present = info.get("present")
        nn = info.get("nonnull_rows", 0)
        dist = info.get("distinct", 0)
        if not (present and nn > 0):
            v, meter = _FAIL, ("absent" if not present else "present but all-null")
        elif col in must_vary and dist <= 1:
            v, meter = _FAIL, f"dead channel (distinct={dist})"
        elif col in must_equal and info.get("mismatch_rows", 0) > 0:
            v, meter = _FAIL, f"{info['mismatch_rows']} rows != {must_equal[col]!r}"
        else:
            meter = f"{nn} non-null" + (f", distinct={dist}" if col in must_vary else "")
            v = _PASS
        add(f"columns/{col}", "Output columns", col, v, meter=meter, detail=dict(info))

    comp = verdict.get("composition_check")
    if comp is not None:
        missing = set(comp.get("missing", []) or [])
        for d in comp.get("declared", []) or []:
            gone = d in missing
            add(f"composition/{d}", "Composition", f"{d} ran",
                _FAIL if gone else _PASS, meter="absent" if gone else "present")
        for f in comp.get("forbidden_present", []) or []:
            add(f"composition/forbidden/{f}", "Composition", f"{f} absent",
                _FAIL, meter="present (should be absent after the swap)")

    sp = verdict.get("species_check")
    if sp is not None:
        ok = sp.get("passed")
        add("strain/species_count", "Strain", "bulk species count",
            _PASS if ok else _FAIL, value=sp.get("count"),
            meter=f"{sp.get('count')} vs expected {sp.get('expected')}", detail=dict(sp))

    worst = _NA
    for ax in axes.values():
        if _RANK[ax["verdict"]] > _RANK[worst]:
            worst = ax["verdict"]
    if not axes:
        add("acceptance/status", "Acceptance", "no checks declared", _NA,
            meter="the acceptance block declared nothing to check")
        worst = _NA
    return {"overall": worst, "axes": axes}


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _self_test() -> int:
    """Validate the verifier itself against a known-bad artifact (sms-ecoli#210:
    "a positive control only protects the checks you actually run against it").
    Builds a tiny hive parquet, then asserts the gate PASSES a present column and
    FAILS a missing one and an all-null one. Returns 0 iff the gate behaves."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    ok = True
    with tempfile.TemporaryDirectory() as d:
        # Minimal hive tree history_files() will find:
        # <d>/history/experiment_id=t/variant=0/lineage_seed=0/generation=1/agent_id=1/0.pq
        leaf = (
            Path(d) / "history" / "experiment_id=t" / "variant=0"
            / "lineage_seed=0" / "generation=1" / "agent_id=1"
        )
        leaf.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "global_time": [0.0, 1.0],
                "listeners__mass__dry_mass": [430.0, 431.0],
                "all_null_col": pa.array([None, None], type=pa.float64()),
                "const_col": [7.0, 7.0],
            }),
            str(leaf / "0.pq"),
        )

        present = check_columns(d, ["global_time", "listeners__mass__dry_mass"])
        if not present["passed"]:
            print(f"SELF-TEST FAIL: present columns should pass: {present}")
            ok = False

        missing = check_columns(d, ["listeners__mass__dry_mass", "not_a_column"])
        if missing["passed"] or missing["columns"]["not_a_column"]["present"]:
            print(f"SELF-TEST FAIL: a missing column must fail: {missing}")
            ok = False

        allnull = check_columns(d, ["all_null_col"])
        if allnull["passed"]:
            print(f"SELF-TEST FAIL: an all-null column must fail: {allnull}")
            ok = False

        # a present, non-null, but CONSTANT column is a dead channel when it must vary
        dead = check_columns(d, ["const_col"], must_vary=["const_col"])
        if dead["passed"] or dead["columns"]["const_col"]["distinct"] != 1:
            print(f"SELF-TEST FAIL: a constant must_vary column must fail: {dead}")
            ok = False

    print("SELF-TEST PASS: gate flags missing, all-null, and dead-constant columns"
          if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", help="Run output dir (local or s3://), hive history under it.")
    p.add_argument("--required-columns", help="JSON list of required column names, or @file.")
    p.add_argument("--must-vary", default=None,
                   help="JSON list of required columns that must also VARY (distinct>1); "
                        "a column pinned at its seed value is a dead channel that passes "
                        "a bare presence check.")
    p.add_argument("--must-equal", default=None,
                   help="JSON object {column: expected_value}; every non-null row of "
                        "the column must equal the value (e.g. the study's media_id). "
                        "Catches a run against the wrong condition.")
    p.add_argument("--declared-processes", default=None,
                   help="JSON list of process names/addresses the submission declared (optional).")
    p.add_argument("--forbidden-processes", default=None,
                   help="JSON list of process classes that must be ABSENT from what ran "
                        "(e.g. the stock class after a swap). Needs --process-table.")
    p.add_argument("--process-table", default=None,
                   help="Path to a JSON composite state / final_state.json; the mounted "
                        "process addresses are extracted for the composition check, and "
                        "the bulk species count for --expected-species-count.")
    p.add_argument("--expected-species-count", type=int, default=None,
                   help="Assert the bulk species count in --process-table equals this "
                        "strain fingerprint (e.g. 16321 stock vs 16323+ genuine). An "
                        "unreadable count fails.")
    p.add_argument("--out", default=None, help="Write verdict.json here.")
    p.add_argument("--self-test", action="store_true",
                   help="Validate the gate against a known-bad artifact and exit.")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.sweep_dir or not args.required_columns:
        p.error("--sweep-dir and --required-columns are required (unless --self-test)")

    spec = args.required_columns
    required = json.loads(Path(spec[1:]).read_text() if spec.startswith("@") else spec)
    must_vary = json.loads(args.must_vary) if args.must_vary else []
    must_equal = json.loads(args.must_equal) if args.must_equal else None
    forbidden = json.loads(args.forbidden_processes) if args.forbidden_processes else []

    # The state file feeds both the composition check and the species count, so
    # read it once if either check needs it.
    state = None
    if args.process_table and (args.declared_processes is not None
                               or forbidden or args.expected_species_count is not None):
        state = json.loads(Path(args.process_table).read_text())

    ran = None
    declared = None
    if args.declared_processes is not None or forbidden:
        declared = json.loads(args.declared_processes) if args.declared_processes else []
        ran = sorted(process_names_from_state(state)) if state is not None else []

    species_count = bulk_species_count_from_state(state) if (
        state is not None and args.expected_species_count is not None) else None

    verdict = run_gate(args.sweep_dir, required, must_vary=must_vary,
                       must_equal=must_equal, ran_processes=ran,
                       declared_processes=declared, forbidden_processes=forbidden,
                       species_count=species_count,
                       expected_species_count=args.expected_species_count)
    print(json.dumps(verdict, indent=2))
    if args.out:
        write_verdict(verdict, args.out)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
