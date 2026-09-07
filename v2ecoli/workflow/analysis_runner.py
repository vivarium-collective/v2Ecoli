"""Post-sweep analysis runner.

Reads a finished sweep's emitted parquet (per-cell timeseries) + summary.json
(division metadata written by run_workflow), builds per-cell records, groups
them per scale, runs the AnalysisSteps named in analysis_options, and writes
analysis.json. Also runnable standalone:

    v2ecoli-analyze <sweep_dir> [--config cfg.json]

``sweep_dir`` may be a local path OR an ``s3://`` URI. The S3 form lets a run's
analyses be (re-)computed against output that stays in object storage — a
1000-seed lineage sweep is terabytes of hive parquet, far past what any single
node holds, but DuckDB's column projection means an analysis reads only the
columns it names.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import warnings
from typing import Any

# Sweep location/access lives in the library layer so the report-card vector
# extraction can share it (library must not import from workflow). Re-exported
# here because these names are part of this module's existing surface.
from v2ecoli.library.sweep_io import (
    _S3_PREFIX,
    analysis_memory_budget_bytes,
    apply_analysis_duckdb_config,
    configure_duckdb_s3,
    history_files,
    is_s3_uri,
)


def _default_analysis_workers() -> int:
    """Default concurrency ceiling for the DuckDB-backed (Analysis) family's
    per-module fan-out in run_analyses (see _run_duckdb_names below). Each named
    analysis runs a fully independent cursor+query and DuckDB releases the GIL
    during execution, so real threads give real speedup -- ~3x with 4 threads on
    a genuine aggregation query (2026-08-20).

    But each concurrent analysis is a full-hive query that can hold gigabytes, so
    the CPU count is the wrong ceiling on a memory-bounded container: N cores
    would stack N full ORDER-BYs into one buffer pool and OOM ("failed to pin
    block", observed running the 5 ptools views together). Precedence:
    ``V2E_ANALYSIS_MAX_WORKERS`` env; else, when a memory budget is known
    (:func:`analysis_memory_budget_bytes`), roughly one worker per 4 GiB (min 1);
    else the CPU count. A caller can still force strict serial via max_workers=1.
    """
    env = os.environ.get("V2E_ANALYSIS_MAX_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    budget = analysis_memory_budget_bytes()
    if budget:
        return max(1, min(cpu, budget // (4 * 1024 ** 3)))
    return cpu


DEFAULT_ANALYSIS_MAX_WORKERS = _default_analysis_workers()


def localize(uri: str, cache_dir: str | None = None) -> str:
    """Fetch an ``s3://`` object to a local file and return its path (cached).

    For the handful of sweep artifacts that must exist ON DISK — the sim_data
    pickle is unpickled by path, not streamed — as opposed to the parquet, which
    DuckDB reads in place.
    """
    if not is_s3_uri(uri):
        return uri
    import tempfile

    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "v2ecoli-sweep-cache")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(uri.rstrip("/")))
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    import boto3

    bucket, _, key = uri[len(_S3_PREFIX):].partition("/")
    boto3.client("s3").download_file(bucket, key, dest)
    return dest


_CELL_KEY_RE = re.compile(
    r"variant=(\d+)/lineage_seed=(\d+)/generation=(\d+)/agent_id=([^/]+)/")


def cell_keys(sweep_dir: str) -> list[dict]:
    """Minimal per-cell records — the four hive partition keys, no parquet read.

    ``group_for_scale`` only reads ``variant``/``lineage_seed``/``generation``/
    ``agent_id``, and every one of those is in the partition path. Enumerating
    groups therefore costs a LISTING, not a scan, which is what makes a
    DuckDB-backed analysis viable over a sweep with millions of rows: the heavy
    :func:`build_cell_records` is only needed by the record-based
    ``AnalysisStep`` family, which consumes per-cell timeseries.
    """
    seen: dict[tuple, dict] = {}
    for path in history_files(sweep_dir):
        m = _CELL_KEY_RE.search(path.replace("\\", "/"))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
        seen.setdefault(key, {"variant": key[0], "lineage_seed": key[1],
                              "generation": key[2], "agent_id": key[3]})
    return list(seen.values())


def group_for_scale(scale: str, records: list[dict]) -> dict[tuple, list[dict]]:
    """Group per-cell records by the key the scale aggregates over."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        v, s = int(r["variant"]), int(r["lineage_seed"])
        g, a = int(r["generation"]), str(r["agent_id"])
        if scale == "single":
            key = (v, s, g, a)
        elif scale == "multidaughter":
            # sisters share the parent id (phylogeny "00"/"01" -> parent "0");
            # the root cell "0" has no parent char to strip, so it keys to itself.
            parent = a[:-1] if len(a) > 1 else a
            key = (v, s, g, parent)
        elif scale == "multigeneration":
            key = (v, s)
        elif scale == "multiseed":
            key = (v,)
        elif scale == "multivariant":
            key = ()
        else:
            continue
        groups.setdefault(key, []).append(r)
    return groups


_MASS_COLS = ("listeners__mass__dry_mass", "listeners__mass__protein_mass",
              "listeners__mass__rRna_mass", "listeners__mass__dna_mass",
              "listeners__mass__rna_mass")  # rna_mass = total RNA (rRna+tRna+mRna)

# Per-timestep scalar physiology columns (time-averaged per cell). fork count is
# an array length expression appended separately.
_PHYSIO_COLS = ("listeners__mass__cell_mass", "listeners__mass__volume",
                "listeners__replication_data__number_of_oric")
_FORK_LEN = "len(listeners__replication_data__fork_coordinates)"

# Ribosome columns. active count + elongation rate + rRNA-initiation (a
# ribosome-biogenesis-rate proxy; ribosome_data__did_initialize isn't emitted
# in this build) are clean listeners; the inactive 30S/50S subunit counts have
# NO listener, so they're pulled from the bulk arrays by molecule id
# (sim_data.molecule_ids.s30_full_complex / s50_full_complex = the 30S/50S full
# complexes). Total ribosomes = active + min(30S, 50S), mirroring vEcoli's
# ecoli/analysis/multigeneration/ribosome_usage.py.
_RIBO_COLS = ("listeners__growth_limits__active_ribosome_allocated",
              "listeners__ribosome_data__effective_elongation_rate",
              "listeners__ribosome_data__total_rRNA_initiated")
_S30_COUNT = "list_extract(bulk__count, list_position(bulk__id, 'CPLX0-3953[c]'))"
_S50_COUNT = "list_extract(bulk__count, list_position(bulk__id, 'CPLX0-3962[c]'))"


def _replication_events(times, oric, nforks):
    """Per-cell replication event times (since birth — global_time resets each
    generation). Initiation = first oriC step-up (a new round fires, origin
    duplicates). Completion = forks first clear after being active (a round
    terminates). Either is None if the event doesn't occur within the cell's
    observed cycle (e.g. a cell born between rounds never re-initiates)."""
    init = next((times[i] for i in range(1, len(oric)) if oric[i] > oric[i - 1]),
                None)
    completion = next((times[i] for i in range(1, len(nforks))
                       if nforks[i] == 0 and nforks[i - 1] > 0), None)
    return init, completion


def _history_from_clause(sweep_dir: str) -> str:
    """A DuckDB FROM-clause selecting all of the sweep's history parquet.

    Accepts a local dir or an ``s3://`` URI; DuckDB reads ``s3://`` paths
    directly through httpfs (see :func:`configure_duckdb_s3`, which the caller
    must have applied to the connection).
    """
    files = history_files(sweep_dir)
    if not files:
        raise FileNotFoundError(f"no history parquet under {sweep_dir!r}")
    flist = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    return f"read_parquet({flist}, hive_partitioning=true)"


# scale -> the partition columns that scale's history_sql filters on.
_SCALE_FILTER_COLS = {
    "single": ("variant", "lineage_seed", "generation", "agent_id"),
    "multidaughter": ("variant", "lineage_seed", "generation"),  # parent handled below
    "multigeneration": ("variant", "lineage_seed"),
    "multiseed": ("variant",),
    "multivariant": (),
}


def scale_history_sql(scale: str, from_clause: str, key: tuple) -> str:
    """SELECT * scoped to the partition a scale aggregates over.

    ``key`` is the group key from ``group_for_scale`` for that scale.
    """
    cols = _SCALE_FILTER_COLS[scale]
    conds = []
    for col, val in zip(cols, key):
        if isinstance(val, str):
            conds.append(f"agent_id = '{val}'" if col == "agent_id"
                         else f"{col} = '{val}'")
        else:
            conds.append(f"{col} = {int(val)}")
    if scale == "multidaughter" and len(key) >= 4:
        # sisters share parent = agent_id without its last phylogeny char
        conds.append(f"agent_id LIKE '{key[3]}_' ESCAPE '\\'")
    elif scale in ("multigeneration", "multiseed", "multivariant"):
        # Restrict to the canonical single-daughter lineage — the all-zeros
        # agent_id chain (gen N = "0"*N). These scales collapse one lineage
        # across generations, so the transient d?1 birth-stub partitions
        # (agent_id containing a '1') are never wanted here (unlike multidaughter
        # above, which deliberately keeps sisters). Without this the stubs get
        # folded into the cross-generation aggregation and contaminate it — a
        # ~1% flux shift, and a stub whose estimated_exchange_dmdt column set
        # mismatches the lineage can block the read entirely (schema mismatch).
        # CAST because hive can read agent_id as BIGINT (0/1/10/…) rather than
        # VARCHAR ("00"); NOT LIKE needs VARCHAR. All-zeros → "0"/"00" (no '1');
        # a stub daughter → contains '1' either way.
        conds.append("CAST(agent_id AS VARCHAR) NOT LIKE '%1%'")
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    return f"SELECT * FROM {from_clause}{where} ORDER BY global_time"


def resolve_sim_data(sweep_dir: str):
    """Locate + load the sweep's ParCa sim_data via v2ecoli's loader.

    Resolution order:
      1. A sweep-local ``sim_data*.cPickle/.pkl`` — the exact pairing, preferred.
      2. ``$V2ECOLI_SIM_DATA`` — explicit override (use when you know the
         matching sim_data path).
      3. The ParCa knowledge-base build (``out/kb/simData.cPickle`` /
         ``out/workflow/simData.cPickle``) — the fallback that makes analyses
         (e.g. the ptools_* exports) runnable on a sweep that ran from the
         cached sim-input bundle, which does NOT itself contain a sim_data
         pickle (it stores process configs only). Emits a warning because the
         kb build can in principle be a different sim_data version than the
         sweep's cache; ensure they're from the same ParCa run.
    """
    from v2ecoli.library.sim_data import LoadSimData
    if not is_s3_uri(sweep_dir):
        for pat in ("sim_data*.cPickle", "sim_data*.pkl", "simData*.cPickle",
                    "**/sim_data*.cPickle", "**/simData*.cPickle"):
            hits = glob.glob(os.path.join(sweep_dir, pat), recursive=True)
            if hits:
                return LoadSimData(sim_data_path=hits[0]).sim_data
    env = os.environ.get("V2ECOLI_SIM_DATA")
    if env and is_s3_uri(env):
        # An S3 sweep has no co-located pickle to glob, so the pairing is named
        # explicitly. Fetch it once — LoadSimData needs a real file on disk.
        return LoadSimData(sim_data_path=localize(env)).sim_data
    if env and os.path.isfile(env):
        return LoadSimData(sim_data_path=env).sim_data
    for fallback in (os.path.join("out", "kb", "simData.cPickle"),
                     os.path.join("out", "workflow", "simData.cPickle")):
        if os.path.isfile(fallback):
            print(f"  sim_data: no sweep-local pickle; falling back to the ParCa "
                  f"build {fallback!r} — ensure it matches the sweep's cache "
                  f"(set $V2ECOLI_SIM_DATA to override).")
            return LoadSimData(sim_data_path=fallback).sim_data
    raise FileNotFoundError(
        f"no sim_data pickle under {sweep_dir!r}, no $V2ECOLI_SIM_DATA, and no "
        f"out/kb/simData.cPickle (needed by Analysis steps)")


def resolve_validation_data(sim_data):
    """Build minimal validation data from the copied flat files + sim_data.

    Returns a ``_ValidationData`` object exposing
    ``.protein.schmidt2015Data`` and ``.protein.wisniewski2014Data``, or
    ``None`` if the loader or flat files are unavailable (so unrelated
    analyses are never broken by a missing validation dataset).
    """
    try:
        from v2ecoli.library.validation_data import build_validation_data
        return build_validation_data(sim_data)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"validation_data unavailable ({type(exc).__name__}: {exc}); "
            "analyses that require it will receive None.",
            stacklevel=2,
        )
        return None


def build_cell_records(sweep_dir: str) -> dict[tuple, dict]:
    """Build per-cell summary records from the sweep's parquet + summary.json."""
    import tempfile

    from viva_emitters import create_duckdb_conn

    div_by_cell: dict[tuple, dict] = {}
    spath = os.path.join(sweep_dir, "summary.json")
    if not is_s3_uri(sweep_dir) and os.path.isfile(spath):
        with open(spath) as f:
            summary = json.load(f)
        for bkey, bs in summary.items():
            m = re.search(r"variant=(\d+)/seed=(\d+)", bkey)
            if not m:
                continue
            v, s = int(m.group(1)), int(m.group(2))
            for gen in bs.get("generations", []):
                ck = (v, s, int(gen["generation"]), str(gen["agent_id"]))
                div_by_cell[ck] = {"divided": bool(gen.get("divided", False)),
                                   "division_time": float(gen.get("duration", 0.0))}

    files = history_files(sweep_dir)
    if not files:
        return {}
    flist = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    sel = ("variant, lineage_seed, generation, agent_id, global_time, "
           + ", ".join(_MASS_COLS) + ", " + ", ".join(_PHYSIO_COLS)
           + ", " + _FORK_LEN + ", " + ", ".join(_RIBO_COLS)
           + ", " + _S30_COUNT + ", " + _S50_COUNT)
    conn = create_duckdb_conn(temp_dir=tempfile.gettempdir())
    apply_analysis_duckdb_config(conn)
    if is_s3_uri(sweep_dir):
        configure_duckdb_s3(conn)
    rows = conn.sql(
        f"SELECT {sel} FROM read_parquet({flist}, hive_partitioning=true) "
        f"ORDER BY variant, lineage_seed, generation, agent_id, global_time"
    ).fetchall()

    by_cell: dict[tuple, list] = {}
    for row in rows:
        (v, ls, g, a, t, dry, prot, rrna, dna, rna, cmass, vol, oric, nfork,
         active, elong, rrna_init, s30, s50) = row
        ck = (int(v), int(ls), int(g), str(a))
        by_cell.setdefault(ck, []).append(
            (float(t), float(dry), float(prot), float(rrna), float(dna), float(rna),
             float(cmass), float(vol), float(oric), int(nfork),
             float(active), float(elong), float(rrna_init),
             float(s30 or 0.0), float(s50 or 0.0)))

    records: dict[tuple, dict] = {}
    for ck, rs in by_cell.items():
        fr = {"protein": [], "rRna": [], "rna": [], "dna": []}
        ts = []
        cmasses, vols, orics, nforks, times = [], [], [], [], []
        ribo_total, ribo_active_frac, elongs, productions = [], [], [], []
        for (t, dry, prot, rrna, dna, rna, cmass, vol, oric, nfork,
             active, elong, rrna_init, s30, s50) in rs:
            ts.append({"listeners": {"mass": {"dry_mass": dry, "protein_mass": prot,
                                              "rRna_mass": rrna, "dna_mass": dna,
                                              "rna_mass": rna}}})
            times.append(t); cmasses.append(cmass); vols.append(vol)
            orics.append(oric); nforks.append(nfork)
            # Ribosomes: total = active + min(free 30S, free 50S) assemblable;
            # active fraction = translating / total (vEcoli ribosome_usage.py).
            total = active + min(s30, s50)
            ribo_total.append(total)
            if total > 0:
                ribo_active_frac.append(active / total)
            if elong > 0:
                elongs.append(elong)
            productions.append(rrna_init)
            if dry > 0:
                fr["protein"].append(prot / dry)
                fr["rRna"].append(rrna / dry)
                fr["rna"].append(rna / dry)      # total RNA / dry weight
                fr["dna"].append(dna / dry)
        div = div_by_cell.get(ck, {})
        repl_init, repl_complete = _replication_events(times, orics, nforks)

        def _mean(xs):
            return (sum(xs) / len(xs)) if xs else 0.0

        # Per-cell means are the CELL-level statistic (time-average within the
        # cell -> one value per cell). Population stats live across cells.
        records[ck] = {
            "variant": ck[0], "lineage_seed": ck[1], "generation": ck[2], "agent_id": ck[3],
            "divided": div.get("divided"),
            # division_time from summary.json (per-generation elapsed duration);
            # fallback to last global_time, which is ~the generation duration
            # because each generation runs a fresh composite (global_time resets).
            "division_time": div.get("division_time", float(rs[-1][0])),
            "newborn_dry_mass": rs[0][1], "final_dry_mass": rs[-1][1],
            "protein_fraction_mean": _mean(fr["protein"]),
            "rRna_fraction_mean": _mean(fr["rRna"]),
            "rna_fraction_mean": _mean(fr["rna"]),
            "dna_fraction_mean": _mean(fr["dna"]),
            # Physiology (cell cycle): time-mean level + per-cell event times.
            "cell_mass_mean": _mean(cmasses),
            "volume_mean": _mean(vols),
            "oric_mean": _mean(orics),
            "replication_initiation_time": repl_init,
            "replication_completion_time": repl_complete,
            # Ribosomes: total (components), active fraction + elongation rate
            # (usage), rRNA-initiation (production proxy).
            "ribosome_total_mean": _mean(ribo_total),
            "ribosome_active_fraction_mean": _mean(ribo_active_frac),
            "ribosome_elongation_mean": _mean(elongs),
            "ribosome_production_mean": _mean(productions),
            "timeseries": ts,
        }
    return records


_MISSING_COLUMN_RE = re.compile(
    r'[Cc]olumn(?: named)?\s+"([^"]+)"\s+(?:not found|does not exist)')


def _extract_missing_column(exc: Exception) -> str | None:
    """Best-effort extraction of a missing-column name from a DuckDB binder
    error, e.g. ``duckdb.BinderException: Binder Error: Referenced column
    "listeners__mass__dry_mass" not found in FROM clause!`` (confirmed live
    against a real DuckDB missing-column query 2026-09-01). Returns ``None``
    when the message doesn't name a column explicitly -- the caller still
    records the raw exception text either way, so nothing is lost.
    """
    m = _MISSING_COLUMN_RE.search(str(exc))
    return m.group(1) if m else None


def _name_status(per_group: dict) -> str:
    """Roll one named analysis's per-group results up into a single status --
    ``ok`` / ``partial`` / ``error`` / ``missing_column`` -- so ``run_analyses``
    can report a structured pass/fail summary (P1-10) instead of leaving a
    per-group ``{"error": ...}`` entry as the only signal a caller could act
    on. ``missing_column`` is distinct from ``error`` precisely so a zero/empty
    panel caused by an absent KPI column is never indistinguishable from a
    genuine null result (CD2 audit §3.7)."""
    if not per_group:
        return "ok"
    statuses = set()
    for v in per_group.values():
        if isinstance(v, dict) and "error" in v:
            statuses.add(v.get("status") or "error")
        else:
            statuses.add("ok")
    if statuses <= {"ok"}:
        return "ok"
    if statuses == {"missing_column"}:
        return "missing_column"
    if "ok" in statuses:
        return "partial"
    return "error"


def _group_key_str(scale: str, key: tuple) -> str:
    if scale == "single":
        return f"variant={key[0]}/seed={key[1]}/gen={key[2]}/agent={key[3]}"
    if scale == "multidaughter":
        return f"variant={key[0]}/seed={key[1]}/gen={key[2]}/parent={key[3]}"
    if scale == "multigeneration":
        return f"variant={key[0]}/seed={key[1]}"
    if scale == "multiseed":
        return f"variant={key[0]}"
    return "all"


def _register_builtin_analyses() -> None:
    """Import the built-in analyses package so its ``Analysis`` subclasses
    populate ``ANALYSIS_REGISTRY``.

    ``run_analyses`` resolves every requested name against ``ANALYSIS_REGISTRY``
    and skips (with an ``"unknown analysis"`` warning) any name that isn't
    there. Registration happens as a side effect of importing each analysis
    module, and ``v2ecoli.workflow.analyses.__init__`` imports the whole suite —
    but nothing in the workflow run path imports that package. So a bare
    ``python -m v2ecoli.workflow.run`` (as opposed to a session that already
    imported the analyses, e.g. via a downstream workspace) finds the registry
    empty and silently drops EVERY declared built-in analysis. Importing the
    package here guarantees the built-ins are resolvable regardless of what the
    caller imported. Idempotent — the import is cached after the first call.

    Also loads any analysis packages other installed distributions advertise
    under the ``v2ecoli.analyses`` entry-point group (see
    :func:`_register_plugin_analyses`), so a package that ships its own analyses
    -- e.g. sms_modules' ptools_metabolites -- registers them in the dispatch
    path without v2ecoli importing it by name (the vendored analysis entrypoints
    stay byte-identical to these, so they cannot import sms_modules themselves).
    """
    import v2ecoli.workflow.analyses  # noqa: F401 — import registers the suite

    _register_plugin_analyses()


def _register_plugin_analyses() -> None:
    """Import every module advertised under the ``v2ecoli.analyses`` entry-point
    group so its ``Analysis`` subclasses register in ``ANALYSIS_REGISTRY``.

    An installed distribution declares, in its own ``pyproject.toml``::

        [project.entry-points."v2ecoli.analyses"]
        sms_modules = "sms_modules.analyses"

    and importing that module fires its registration side effects. A plugin that
    fails to import is warned about, never fatal: a broken third-party analysis
    package must not take down the built-in suite. Idempotent -- the imports are
    cached after the first call.
    """
    import importlib
    import importlib.metadata as importlib_metadata

    try:
        eps = importlib_metadata.entry_points(group="v2ecoli.analyses")
    except TypeError:  # Python < 3.10 non-selectable entry_points() API
        eps = importlib_metadata.entry_points().get("v2ecoli.analyses", [])
    for ep in eps:
        # An entry-point value is "pkg.module" or "pkg.module:attr"; the module
        # is what carries the registration side effect, so import that.
        module_name = ep.value.split(":", 1)[0].strip()
        try:
            importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001 -- a bad plugin must not be fatal
            warnings.warn(
                f"v2ecoli.analyses plugin {ep.name!r} ({ep.value!r}) failed to "
                f"import; its analyses stay unregistered: {type(e).__name__}: {e}"
            )


def run_analyses(sweep_dir: str, analysis_options: dict,
                 sim_data_path: str | None = None,
                 out_dir: str | None = None,
                 max_workers: int | None = None) -> dict:
    """Run the analyses named in ``analysis_options`` over the sweep's cells,
    write ``analysis.json``, and return the nested results.

    The returned dict is ``{scale: {name: {group: data}}}`` (unchanged shape)
    PLUS three structured-summary keys a caller can check without walking
    every group of every named analysis (P1-10 / CD2 audit §3.7 -- an
    analysis failure, or a KPI column the emitter dropped, must never look
    like a clean `{"n": 0, "mean": 0.0}` result):

      ``status``   -- ``"OK"`` if every requested analysis/group succeeded,
                       else ``"PARTIAL"``.
      ``summary``  -- ``{scale: {name: "ok"|"partial"|"error"|"missing_column"}}``.
                       ``missing_column`` means the underlying per-cell record
                       lacked a column this analysis needed (see
                       ``build_cell_records``) -- distinct from ``error``
                       (the analysis itself raised) and never silently
                       collapsed into a zero-valued result.
      ``errors``   -- flat list of ``{"scale", "name", "group", "error",
                       "missing_column"}`` entries for every failure, plus one
                       ``scale=None`` entry if the per-cell record build
                       itself failed (see ``records_error`` below).

    This function never raises over an analysis-level failure or a missing
    KPI column -- both degrade to ``status: "PARTIAL"`` so the rest of the
    sweep's analyses still run and land in ``analysis.json``. It still
    raises for setup failures outside any single analysis's control (an
    unresolvable sim_data pickle, an s3:// sweep with no ``out_dir``, ...).

    Parameters
    ----------
    sweep_dir:
        Directory containing the sweep's history parquet and (optionally) a
        co-located sim_data pickle.  May be an ``s3://`` URI, in which case
        DuckDB reads the parquet in place and ``out_dir`` must be given.
    analysis_options:
        ``{scale: {name: params}}`` mapping selecting which analyses to run.
    sim_data_path:
        Optional explicit path to a sim_data pickle (local path or ``s3://``
        URI).  When provided, the pickle is loaded directly (no glob search
        under ``sweep_dir``).  When ``None`` (default), ``resolve_sim_data``
        is called as before.
    out_dir:
        Where ``analysis.json`` / ``viz/`` / ``ptools/`` are written.  Defaults
        to ``sweep_dir``, which is only writable when the sweep is local.
    max_workers:
        How many DuckDB-backed (``Analysis``-family) named analyses to run
        concurrently within a scale (see ``_run_duckdb_name``). ``None``
        (default) uses ``DEFAULT_ANALYSIS_MAX_WORKERS``, capped to however
        many such analyses this scale actually names. Pass ``1`` to force
        strict serial execution (e.g. for deterministic debugging).
    """
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY, ANALYSIS_SCALES

    # Populate ANALYSIS_REGISTRY with the built-in suite before resolving any
    # requested name against it — otherwise a bare workflow run finds it empty
    # and drops every declared analysis. See _register_builtin_analyses.
    _register_builtin_analyses()

    # Records are built ONLY if a record-based AnalysisStep is actually
    # requested. That family consumes per-cell timeseries, so building them
    # materializes every emitted row of the sweep in Python — ~24M rows for a
    # 1000-seed x 10-generation lineage. The DuckDB-backed Analysis family needs
    # nothing from records but the GROUP KEYS, and those come from the partition
    # paths (cell_keys) at listing cost. Mirrors the lazy provisioning in
    # flush.RunExtract.context_bag.
    def _needs_timeseries() -> bool:
        for scale, analyses in (analysis_options or {}).items():
            if scale not in ANALYSIS_SCALES:
                continue
            for name in (analyses or {}):
                cls = ANALYSIS_REGISTRY.get(name)
                if cls is not None and not issubclass(cls, Analysis):
                    return True
        return False

    # An s3:// sweep is read-only, so outputs need a local home.
    if out_dir is None:
        if is_s3_uri(sweep_dir):
            raise ValueError(
                "out_dir is required for an s3:// sweep (the sweep is read-only)")
        out_dir = sweep_dir

    # P1-10 (CD2 audit §3.7): build_cell_records() hard-codes 15+ columns in
    # one query; if the emitter dropped one, the query raises OUTSIDE any
    # per-group guard -- previously that propagated straight out of
    # run_analyses and every analysis (including ones on scales that never
    # needed the timeseries) was lost, reduced to one error string by the
    # flush's broad catch. Instead: catch it here, extract the missing column
    # name when the error names one, and fall back to the cheap partition-key
    # listing so DuckDB-backed analyses (which read columns from DuckDB
    # directly, not from these Python records) are unaffected. Record-based
    # analyses are flagged explicitly below (never silently computing a
    # hollow {"n": 0, "mean": 0.0, ...} over key-only records).
    records_error: dict[str, Any] | None = None
    if _needs_timeseries():
        try:
            records = list(build_cell_records(sweep_dir).values())
        except Exception as e:  # noqa: BLE001 -- converted into an explicit,
            # per-analysis "missing_column" signal below, not swallowed.
            records_error = {
                "error": f"{type(e).__name__}: {e}",
                "missing_column": _extract_missing_column(e),
            }
            records = cell_keys(sweep_dir)
    else:
        records = cell_keys(sweep_dir)
    core = allocate_core()
    results: dict[str, dict] = {}
    # Provisioned once on first use and shared across every Analysis step, so the
    # large sim_data pickle is loaded only once per run (not once per analysis),
    # and a single DuckDB connection is reused.
    _ctx: dict[str, Any] = {}
    # Named analyses now fan out across threads (see _run_duckdb_name below),
    # and this lazy provisioning is only safe to run once — guard the
    # check-then-populate with a lock so two threads racing on the very first
    # call can't both see `not _ctx`, both start building it, and leave a
    # reader mid-construction with a KeyError (or a discarded, redundantly
    # loaded sim_data pickle).
    import threading

    _ctx_lock = threading.Lock()

    def _analysis_ctx() -> tuple:
        with _ctx_lock:
            if not _ctx:
                import tempfile

                from viva_emitters import create_duckdb_conn
                _ctx["conn"] = create_duckdb_conn(temp_dir=tempfile.gettempdir())
                apply_analysis_duckdb_config(_ctx["conn"])
                _ctx["from_clause"] = _history_from_clause(sweep_dir)
                if sim_data_path is not None:
                    from v2ecoli.library.sim_data import LoadSimData
                    _ctx["sim_data"] = LoadSimData(
                        sim_data_path=localize(sim_data_path)).sim_data
                else:
                    _ctx["sim_data"] = resolve_sim_data(sweep_dir)
                _ctx["validation_data"] = resolve_validation_data(_ctx["sim_data"])
            return (_ctx["conn"], _ctx["from_clause"],
                    _ctx["sim_data"], _ctx["validation_data"])

    # (S3-secret refresh moved out of here and onto each thread's own cursor —
    # see _run_duckdb_name's own docstring for why the refresh call itself
    # must still be serialized via _ctx_lock even though query execution
    # after it is not.)

    def _run_duckdb_name(name: str, step_cls: type, params: dict, scale: str,
                          groups: Any) -> dict:
        """Run one DuckDB-backed (``Analysis``-family) named analysis over
        every group in ``scale``, on its own cursor.

        A DuckDB connection is safe to use concurrently from multiple Python
        threads only for QUERY EXECUTION, via ``conn.cursor()`` (a lightweight,
        independent session against the same in-memory catalog) — never the
        base connection object itself from more than one thread at a time.
        Cursors share the base connection's catalog (tables, secrets), which
        is exactly why concurrent CATALOG WRITES (``INSTALL``/``LOAD`` of an
        extension, ``CREATE OR REPLACE SECRET``) are NOT safe across threads:
        DuckDB's transactional catalog correctly raises a write-write
        TransactionException when two threads try to alter the same object at
        once (confirmed live 2026-08-21 — see item 79). So the S3 credential
        refresh below still needs to happen once per module, on each call's
        own cursor (item 71 b4's ExpiredToken fix — a long multi-module sweep
        can outlive one STS session), but the refresh CALL ITSELF must be
        serialized against every other thread's refresh call; only the query
        execution after it is safe to run concurrently.
        """
        conn, from_clause, sim_data, validation_data = _analysis_ctx()
        cursor = conn.cursor()
        if is_s3_uri(sweep_dir):
            with _ctx_lock:
                configure_duckdb_s3(cursor)
        step = step_cls(params, core=core)
        viz_dir = os.path.join(out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        per_group: dict[str, Any] = {}
        for gkey in groups:
            gstr = _group_key_str(scale, gkey)
            try:
                history_sql = scale_history_sql(scale, from_clause, gkey)
                out = step.update({
                    "conn": cursor, "history_sql": history_sql,
                    "config_sql": "", "success_sql": "",
                    "sim_data": sim_data,
                    "validation_data": validation_data,
                    "variant_metadata": params,
                })
                if out.get("view"):
                    vp = os.path.join(viz_dir, f"{name}__{gstr.replace('/', '_')}.html")
                    with open(vp, "w", encoding="utf-8") as vf:
                        vf.write(out["view"])
                data = out.get("data")
                if isinstance(data, dict) and data.get("tsv"):
                    ptools_dir = os.path.join(out_dir, "ptools")
                    os.makedirs(ptools_dir, exist_ok=True)
                    tsv_path = os.path.join(
                        ptools_dir,
                        f"{name}__{gstr.replace('/', '_')}.tsv",
                    )
                    with open(tsv_path, "w", encoding="utf-8") as tf:
                        tf.write(data["tsv"])
                per_group[gstr] = out.get("data", {})
            except Exception as e:
                per_group[gstr] = {"error": f"{type(e).__name__}: {e}"}
        return per_group

    # A declared analysis that never runs is a silent deliverable hole: a study
    # names an analysis that isn't registered in this runner (e.g. an sms_modules
    # KPI whose registration import never ran in the analysis container), and the
    # run reports OK with the plot simply absent. Collect these and surface them
    # as errors so status goes PARTIAL instead of a clean pass over a missing KPI.
    unknown_analyses: list[dict] = []
    for scale, analyses in (analysis_options or {}).items():
        if scale not in ANALYSIS_SCALES:
            warnings.warn(f"unknown analysis scale {scale!r}; skipping")
            unknown_analyses.append({
                "scale": scale, "name": None, "group": None,
                "error": f"unknown analysis scale {scale!r} not in ANALYSIS_SCALES",
                "missing_column": None,
            })
            continue
        groups = group_for_scale(scale, records)
        scale_out: dict[str, dict] = {}

        # Resolve + validate every requested name up front (unchanged
        # semantics), splitting into the DuckDB family (run concurrently
        # below — each is a fully independent query, and DuckDB releases the
        # GIL during execution, see _run_duckdb_name) and the record-based
        # family (run serially, unchanged — cheap, pure-Python, not the
        # observed bottleneck). Results are reassembled into scale_out in the
        # original ``analyses`` order regardless of which family or thread
        # actually produced them, so output ordering matches strict serial
        # execution exactly.
        duckdb_names: list[str] = []
        record_names: list[str] = []
        for name in (analyses or {}):
            step_cls = ANALYSIS_REGISTRY.get(name)
            if step_cls is None:
                warnings.warn(f"unknown analysis {name!r} (scale {scale})")
                unknown_analyses.append({
                    "scale": scale, "name": name, "group": None,
                    "error": f"unknown analysis {name!r} not in ANALYSIS_REGISTRY "
                             f"(declared but never registered/imported)",
                    "missing_column": None,
                })
                continue
            if step_cls.scale != scale:
                warnings.warn(f"analysis {name!r} is scale {step_cls.scale}, "
                              f"not {scale}; skipping")
                continue
            (duckdb_names if issubclass(step_cls, Analysis) else record_names).append(name)

        results_by_name: dict[str, dict] = {}

        for name in record_names:
            step_cls = ANALYSIS_REGISTRY[name]
            if records_error is not None:
                # The per-cell records this family needs failed to build (see
                # above) -- flag every group explicitly rather than running
                # analyze() over key-only records and returning a zero panel
                # that looks like a real (if boring) result.
                col = records_error.get("missing_column")
                msg = (f"missing KPI column {col!r} ({records_error['error']})"
                       if col else f"cell records unavailable ({records_error['error']})")
                flagged = {"error": msg, "status": "missing_column",
                          "missing_column": col}
                per_group = {_group_key_str(scale, gkey): dict(flagged)
                            for gkey in groups} or {"_all": flagged}
                results_by_name[name] = per_group
                continue
            step = step_cls(analyses.get(name) or {}, core=core)
            per_group: dict[str, Any] = {}
            for gkey, grp in groups.items():
                try:
                    # single-scale Steps consume a cell's timeseries; cross-scale
                    # Steps consume the list of per-cell summary records. (A single
                    # group is exactly one cell by construction — group_for_scale
                    # keys single by the full cell id — so grp[0] is that cell.)
                    rows = grp[0].get("timeseries") if scale == "single" else grp
                    per_group[_group_key_str(scale, gkey)] = step.analyze(rows or [])
                except Exception as e:
                    per_group[_group_key_str(scale, gkey)] = {
                        "error": f"{type(e).__name__}: {e}"}
            results_by_name[name] = per_group

        if duckdb_names:
            workers = max_workers or min(len(duckdb_names), DEFAULT_ANALYSIS_MAX_WORKERS)
            if workers <= 1 or len(duckdb_names) == 1:
                for name in duckdb_names:
                    params = (analyses or {}).get(name) or {}
                    results_by_name[name] = _run_duckdb_name(
                        name, ANALYSIS_REGISTRY[name], params, scale, groups)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {
                        name: ex.submit(
                            _run_duckdb_name, name, ANALYSIS_REGISTRY[name],
                            (analyses or {}).get(name) or {}, scale, groups)
                        for name in duckdb_names
                    }
                    for name, fut in futures.items():
                        results_by_name[name] = fut.result()

        for name in (analyses or {}):
            if name in results_by_name:
                scale_out[name] = results_by_name[name]
        results[scale] = scale_out

    if _ctx.get("conn") is not None:
        _ctx["conn"].close()

    # P1-10: a structured pass/fail summary, not just the nested per-group
    # results -- a caller (the post-sim flush, a batch summary, ...) can check
    # results["status"] instead of having to walk every group of every named
    # analysis looking for an "error" key to know whether anything failed.
    summary: dict[str, dict] = {}
    for scale_key, scale_result in results.items():
        scale_summary = {name: _name_status(per_group)
                         for name, per_group in scale_result.items()}
        if scale_summary:
            summary[scale_key] = scale_summary

    errors: list[dict] = []
    if records_error is not None:
        errors.append({"scale": None, "name": None, "group": None, **records_error})
    for scale_key, scale_summary in summary.items():
        for name, status in scale_summary.items():
            if status == "ok":
                continue
            for gkey, gval in results[scale_key][name].items():
                if isinstance(gval, dict) and "error" in gval:
                    errors.append({
                        "scale": scale_key, "name": name, "group": gkey,
                        "error": gval["error"],
                        "missing_column": gval.get("missing_column"),
                    })

    errors.extend(unknown_analyses)

    overall_bad = records_error is not None or bool(unknown_analyses) or any(
        status != "ok" for scale_summary in summary.values()
        for status in scale_summary.values())
    results["status"] = "PARTIAL" if overall_bad else "OK"
    results["summary"] = summary
    results["errors"] = errors

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def build_analysis_arg_parser() -> argparse.ArgumentParser:
    """The ``v2ecoli-analyze`` CLI contract, in one place so ``main()`` and a
    cross-repo contract test assert against the SAME parser.

    The command is ``v2ecoli-analyze <sweep_dir> [--config CONFIG]``. Everything
    an analysis needs beyond the sweep dir lives in the ``--config`` JSON
    (``analysis_options``, and an optional ``out_dir``). An emitter that passes
    any other flag -- workflow_nf's ``AnalysisTaskStep``, viva-api's
    ``_analysis_command`` -- is stale against this and its argv is rejected;
    parse the emitted argv against this parser to catch drift at the seam.
    """
    p = argparse.ArgumentParser(description="Run configured analyses over a sweep.")
    p.add_argument("sweep_dir", help="sweep output dir (parquet + summary.json)")
    p.add_argument("--config", default=None,
                   help="config JSON with analysis_options (+ optional out_dir; supports inherit_from)")
    return p


def main() -> None:
    args = build_analysis_arg_parser().parse_args()
    if not os.path.isdir(args.sweep_dir):
        raise SystemExit(f"sweep_dir not found: {args.sweep_dir!r}")

    analysis_options: dict = {}
    out_dir: str | None = None
    if args.config:
        from v2ecoli.workflow.config import load_config_with_inheritance
        cfg = load_config_with_inheritance(args.config)
        analysis_options = cfg.get("analysis_options") or {}
        # out_dir lives in the config (not a CLI flag) so the argv surface stays
        # `<sweep_dir> [--config]`; a Nextflow task sets it to a task-local name
        # ("analysis") that matches its declared `path` output.
        out_dir = cfg.get("out_dir") or None
    if not analysis_options:
        print("no analysis_options found; nothing to run")
        return
    run_analyses(args.sweep_dir, analysis_options, out_dir=out_dir)
    print(f"Wrote {os.path.join(out_dir or args.sweep_dir, 'analysis.json')}")


if __name__ == "__main__":
    main()
