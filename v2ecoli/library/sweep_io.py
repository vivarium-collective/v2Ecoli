"""Locating and reading a sweep's parquet, wherever the sweep lives.

A ``sweep_dir`` may be a local path OR an ``s3://`` URI. Both forms resolve to
the same flat list of history parquet files, so callers do not branch on
storage: :func:`history_files` hands back paths DuckDB can read either way, and
:func:`configure_duckdb_s3` gives a connection the credentials to do it.

These were introduced in the analysis runner (#418) for the standalone
``v2ecoli-analyze <sweep>`` path. They live here because the report-card vector
extraction (:mod:`v2ecoli.library.card_vectors`) needs them too, and
``library`` does not import from ``workflow``. ``workflow.analysis_runner``
re-exports them, so its existing callers are unaffected.
"""
from __future__ import annotations

import glob
import os
import re

_S3_PREFIX = "s3://"


def is_s3_uri(path: str) -> bool:
    return str(path).startswith(_S3_PREFIX)


def history_files(sweep_dir: str) -> list[str]:
    """The sweep's history parquet, as local paths or ``s3://`` URIs.

    Mirrors the local glob for S3 so every caller (FROM-clause builder, cell-key
    enumeration, per-cell record builder) sees one file list regardless of where
    the sweep lives.

    ⚠ Scoped to the HIVE-partitioned tree (``history/experiment_id=*/…/*.pq``), NOT
    a bare ``history/**/*.pq``. A run's output can also contain a stray, non-hive
    ``<out>/default/history/1.pq`` — an emit that fell back to experiment_id
    ``"default"`` (parquet_vecoli's default when the emitter decl carries no
    experiment_id) and wrote a flat file with none of the hive partition keys. A
    bare glob picks it up alongside the real hive files, and the downstream
    ``read_parquet(files, hive_partitioning=true)`` then aborts the whole read with
    "Hive partition mismatch … key 'agent_id' not found". Requiring the
    ``experiment_id=`` partition segment selects only real hive history and makes
    the read robust to any such stray/flat tree.
    """
    if not is_s3_uri(sweep_dir):
        return sorted(glob.glob(
            os.path.join(sweep_dir, "**", "history", "experiment_id=*", "**", "*.pq"),
            recursive=True))
    # DuckDB's glob() lists object storage through the same httpfs extension the
    # read needs — so listing costs no extra dependency and no parquet read.
    import tempfile

    from viva_emitters import create_duckdb_conn

    conn = create_duckdb_conn(temp_dir=tempfile.gettempdir())
    configure_duckdb_s3(conn)
    pattern = sweep_dir.rstrip("/") + "/**/history/experiment_id=*/**/*.pq"
    try:
        rows = conn.sql(
            f"SELECT file FROM glob('{pattern}')").fetchall()
    finally:
        conn.close()
    return sorted(r[0] for r in rows)


def configure_duckdb_s3(conn, region: str | None = None) -> None:
    """Give a DuckDB connection S3 read access via httpfs.

    Credentials come from the standard AWS chain. SSO callers should export them
    first (``eval "$(aws configure export-credentials --format env)"``); on an
    EC2/Batch node the instance role resolves through boto3 the same way.
    """
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    region = region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get(
        "AWS_REGION") or "us-gov-west-1"
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    token = os.environ.get("AWS_SESSION_TOKEN")
    if not (key and secret):
        import boto3

        creds = boto3.Session().get_credentials()
        if creds is None:
            raise RuntimeError(
                "no AWS credentials for the S3 sweep — export them with "
                '`eval "$(aws configure export-credentials --format env)"`')
        frozen = creds.get_frozen_credentials()
        key, secret, token = frozen.access_key, frozen.secret_key, frozen.token
    parts = [f"KEY_ID '{key}'", f"SECRET '{secret}'", f"REGION '{region}'"]
    if token:
        parts.append(f"SESSION_TOKEN '{token}'")
    conn.execute(f"CREATE OR REPLACE SECRET v2e_sweep_s3 (TYPE s3, {', '.join(parts)});")


def connect_for(sweep_dir: str):
    """A DuckDB connection ready to read ``sweep_dir`` (S3-configured if needed).

    The counterpart to :func:`history_files` for callers that own their own
    connection: pair the two and neither has to know where the sweep lives.
    """
    import tempfile

    from viva_emitters import create_duckdb_conn

    conn = create_duckdb_conn(temp_dir=tempfile.gettempdir())
    if is_s3_uri(sweep_dir):
        configure_duckdb_s3(conn)
    return conn


# --- Analysis DuckDB memory budget --------------------------------------------
#
# ``create_duckdb_conn`` (viva_emitters / vEcoli) already sets a temp_directory
# so DuckDB spills to disk, ``preserve_insertion_order = false``, and an object
# cache. What it does NOT set is an explicit ``memory_limit`` -- DuckDB then
# defaults to ~80% of the DETECTED (host) RAM. Inside a container that reads the
# HOST's RAM, not the cgroup limit, so a heavy analysis (a ptools view's ORDER BY
# over the whole hive) budgets far past what the container may use and dies with
# "failed to pin block ... NGiB/NGiB" instead of spilling. Reading the cgroup and
# setting a real budget makes DuckDB spill to its temp_directory and finish.

def _container_memory_limit_bytes() -> int | None:
    """This process's cgroup memory ceiling in bytes, or None if unbounded / not
    containerized (e.g. macOS, where the cgroup files do not exist)."""
    for path in ("/sys/fs/cgroup/memory.max",                     # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            raw = open(path, encoding="ascii").read().strip()
        except OSError:
            continue
        if raw in ("max", ""):
            continue
        try:
            n = int(raw)
        except ValueError:
            continue
        # cgroup v1 uses a near-INT64_MAX sentinel for "unlimited".
        if 0 < n < (1 << 62):
            return n
    return None


def _parse_size(s: str) -> int | None:
    """Parse a DuckDB-style size string ('8GB', '512MB', '4GiB') to bytes, or
    None if it is not a plain size (so a budget stays 'unknown' rather than
    wrong)."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]*)\s*", s or "")
    if not m:
        return None
    units = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
             "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    unit = m.group(2).upper()
    if unit not in units:
        return None
    try:
        return int(float(m.group(1)) * units[unit])
    except ValueError:
        return None


def analysis_memory_limit() -> str | None:
    """The value for DuckDB ``SET memory_limit`` on an analysis connection, or
    None to keep DuckDB's default.

    Precedence: ``V2E_ANALYSIS_MEMORY_LIMIT`` verbatim (e.g. '8GB'), else ~70% of
    the cgroup limit (leaving headroom for the process + spill bookkeeping), else
    None."""
    env = os.environ.get("V2E_ANALYSIS_MEMORY_LIMIT")
    if env:
        return env
    n = _container_memory_limit_bytes()
    if n:
        return f"{int(n * 0.7) // (1000 * 1000)}MB"
    return None


def analysis_memory_budget_bytes() -> int | None:
    """The analysis memory budget in bytes (for sizing the worker pool), mirroring
    :func:`analysis_memory_limit`'s precedence. None when unknown."""
    env = os.environ.get("V2E_ANALYSIS_MEMORY_LIMIT")
    if env:
        return _parse_size(env)
    n = _container_memory_limit_bytes()
    return int(n * 0.7) if n else None


def apply_analysis_duckdb_config(conn) -> None:
    """Set the explicit memory_limit on an analysis DuckDB connection so it spills
    to its temp_directory at the container's real budget instead of overshooting
    host RAM. No-op when no budget is known (keeps DuckDB's default)."""
    limit = analysis_memory_limit()
    if limit:
        conn.execute(f"SET memory_limit = '{limit}'")
