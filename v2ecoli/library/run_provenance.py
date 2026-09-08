"""Run identity + code provenance, importable from library code.

These two helpers used to live in ``scripts/`` (``_run_provenance.py``,
``_provenance.py``), which made them unavailable to anything under
``v2ecoli/`` — ``scripts/`` is not a package. The cache in
``sim_vector_cache`` needs both to stamp what it writes, so they live here
now and the script modules re-export them for their existing callers.

Run identity follows ``docs/conventions/run-provenance.md``:
``experiment_id == run_id == the top parquet partition key``.

``build_run_identity`` / ``write_run_identity`` (v2ecoli#472/#473) are the
one-record-per-run combiner: code identity + cache content fingerprint +
design/grid metadata, written to a canonical ``run_identity.json`` sidecar
that every runner writes and ``sim_vector_cache._run_commit`` reads. See
``RUN_IDENTITY_FILENAME``'s docstring for why this is a dedicated file
rather than a new key stuffed into each runner's own summary shape.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

from v2ecoli.library.cache_version import read_cache_version

_S3_PREFIX = "s3://"

#: Schema tag of a chassis provenance sidecar record.
CHASSIS_PROVENANCE_SCHEMA = "chassis-provenance/1"

#: Canonical, runner-agnostic sidecar name for ``build_run_identity``'s
#: record, written at the top of a run's output directory (the same level
#: ``summary.json`` / ``*_run_config.json`` land at today).
#:
#: A dedicated file rather than a new key inside each runner's existing
#: artifact, on purpose: ``v2ecoli/workflow/run.py``'s ``summary.json`` is a
#: flat ``{branch_key: {"summary": ...}}`` dict that every existing reader
#: (incl. ``pbg_superpowers.provenance``) iterates assuming every top-level
#: key is a branch — adding a sibling key there would be a breaking shape
#: change for a small win. A new, small, universal file avoids that, gives
#: every current and future ``run_*`` entrypoint one write call and one
#: filename, and needs no per-runner-shape knowledge in the reader.
RUN_IDENTITY_FILENAME = "run_identity.json"


def run_id_from_run_dir(run_dir: str) -> str:
    """Return the canonical run_id (``experiment_id``) for ``run_dir``.

    1. Authoritative: an ``experiment_id=<id>`` partition segment anywhere
       under the run dir — literally the key the emitter wrote.
    2. Fallback: the run dir's basename (runners name
       ``out/<experiment_id>/<experiment_id>/``, so the basename is the id).

    For an ``s3://`` sweep only the fallback applies: globbing the remote
    layout would cost a LIST per call, and the basename is the id by the
    same runner convention.
    """
    run_dir = run_dir.rstrip("/")
    if run_dir.startswith(_S3_PREFIX):
        return os.path.basename(run_dir)
    for cfg in glob.glob(os.path.join(run_dir, "**", "experiment_id=*"),
                         recursive=True):
        seg = os.path.basename(cfg)
        if seg.startswith("experiment_id="):
            return seg.split("=", 1)[1]
    return os.path.basename(run_dir)


def _git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def code_provenance(repo: Path) -> dict:
    """Commit + dirty/diff-sha + untracked count for the working tree at ``repo``.

    Warns when dirty/untracked so a mid-dev bake is loud about its provenance
    being identify-only rather than a clean commit."""
    commit = _git(["rev-parse", "HEAD"], repo)
    diff = _git(["diff", "HEAD"], repo) or ""          # tracked staged+unstaged vs HEAD
    untracked = _git(["ls-files", "--others", "--exclude-standard"], repo) or ""
    dirty = bool(diff)
    n_untracked = len([u for u in untracked.splitlines() if u])
    if dirty or n_untracked:
        warnings.warn(
            f"baking from a DIRTY tree (commit {commit[:12] if commit else '?'}, "
            f"dirty={dirty}, untracked={n_untracked}) — provenance is identify-only, "
            f"not a clean-commit claim")
    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None,
        "untracked": n_untracked,
    }


def _utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ`` (iso8601, ``Z`` suffix)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _direct_url_provenance(dist_name: str) -> dict:
    """Installed-dependency fallback: the VCS commit an installed package was
    built from, read from its ``*.dist-info/direct_url.json`` (PEP 610).

    When v2ecoli (or a workspace package) is consumed as an INSTALLED git
    dependency there is no ``.git`` tree to interrogate, but pip records the
    resolved commit in ``direct_url.json``'s ``vcs_info.commit_id``. Returns
    the honest-null shape ``{"commit": None, "source": None, "reason": ...}``
    on any failure — mirroring ``code_provenance``'s convention — so a missing
    or non-VCS install is a plain recorded fact, not a crash. ``dirty`` is
    ``None`` (unknowable for an installed artifact), not ``False``.
    """
    try:
        import importlib.metadata as metadata
        dist = metadata.distribution(dist_name)
        text = dist.read_text("direct_url.json")
        if not text:
            return {"commit": None, "source": None,
                    "reason": f"{dist_name}: no direct_url.json in dist-info"}
        info = json.loads(text)
        commit = (info.get("vcs_info") or {}).get("commit_id")
        if not commit:
            return {"commit": None, "source": None,
                    "reason": f"{dist_name}: direct_url.json has no "
                              f"vcs_info.commit_id (not a VCS install)"}
        return {"commit": commit, "dirty": None, "source": "direct_url"}
    except Exception as e:  # noqa: BLE001 — any lookup failure is honest-null
        return {"commit": None, "source": None,
                "reason": f"{dist_name}: {type(e).__name__}: {e}"}


def _repo_code_provenance(repo: Path | str | None,
                          *, dist_name: str | None = None) -> dict:
    """Code identity for a repo: git if there's a ``.git`` tree, else the
    installed-dependency ``direct_url.json`` fallback, else honest-null.

    ``source`` records WHICH path produced the record: ``"git"``,
    ``"direct_url"``, or ``None`` on total failure.
    """
    repo_path = Path(repo) if repo is not None else None
    if repo_path is not None and (repo_path / ".git").exists():
        prov = code_provenance(repo_path)
        prov["source"] = "git"
        return prov
    if dist_name:
        return _direct_url_provenance(dist_name)
    return {"commit": None, "source": None,
            "reason": f"no .git at {str(repo_path)!r} and no dist fallback"}


def _default_workspace_root() -> str | None:
    """Best-effort workspace root (nearest ``workspace.yaml`` ancestor), or
    ``None`` when there is no workspace / ``viva_workspace`` isn't installed."""
    try:
        from viva_workspace import find_workspace_root
        return str(find_workspace_root())
    except Exception:
        return None


def _workspace_code_provenance(workspace_root: Path | str | None) -> dict:
    """Code identity of the consuming WORKSPACE, prefixed with its ``repo`` id.

    The workspace holds the DATA files (``models/parca/...``) and any private
    payload, so its own commit is part of a chassis's provenance distinct from
    the v2ecoli package commit. Honest-null when there is no workspace.
    """
    if not workspace_root:
        return {"repo": None, "commit": None, "dirty": None,
                "diff_sha256": None, "untracked": None, "source": None,
                "reason": "no workspace root resolved"}
    root = Path(workspace_root)
    return {"repo": root.name, **_repo_code_provenance(root)}


def chassis_provenance(pkl_path: Path | str, *, build: dict | None = None,
                       repo_root: Path | str | None = None,
                       workspace_root: Path | str | None = None) -> dict:
    """Build the ``chassis-provenance/1`` record for a ParCa chassis pickle.

    The chassis layer is the bottom of the derived-from chain: the
    ``parca_state.pkl`` every downstream sim_data cache is built on. This
    record pins (a) the artifact's exact bytes (sha256 + size), (b) the code
    that produced it — the v2ecoli package commit AND the consuming workspace's
    commit, each git-or-``direct_url``-or-null — and (c) the caller-supplied
    ``build`` dict (mode / new_genes / bundle overrides / rnaseq source /
    argv). It is written beside the pickle by :func:`write_chassis_provenance`
    and embedded verbatim into a cache's ``derived_from`` chain by
    ``cache_version.compute_cache_version``.

    ``repo_root`` defaults to the v2ecoli source root (this file's package
    root); ``workspace_root`` defaults to the nearest ``workspace.yaml``
    ancestor (or ``None``).
    """
    path = Path(pkl_path)
    data = path.read_bytes()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if workspace_root is None:
        workspace_root = _default_workspace_root()
    return {
        "schema": CHASSIS_PROVENANCE_SCHEMA,
        "layer": "chassis",
        "artifact": {
            "file": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        },
        "code": {
            "v2ecoli": _repo_code_provenance(repo_root, dist_name="v2ecoli"),
            "workspace": _workspace_code_provenance(workspace_root),
        },
        "build": dict(build) if build else {},
        "created_at": _utc_now_iso(),
    }


def chassis_provenance_path(pkl_path: Path | str) -> Path:
    """Sidecar path for ``pkl_path``: ``parca_state.pkl`` →
    ``parca_state.provenance.json`` (the final extension is swapped)."""
    return Path(pkl_path).with_suffix(".provenance.json")


def write_chassis_provenance(pkl_path: Path | str, *,
                             build: dict | None = None,
                             repo_root: Path | str | None = None,
                             workspace_root: Path | str | None = None) -> dict:
    """Compute :func:`chassis_provenance` and write it to
    ``<pkl stem>.provenance.json`` beside the pickle. Returns the record.

    Atomic (tmp-file + ``os.replace``) so a killed writer leaves no half-file,
    matching :func:`write_run_identity_record`'s local-write convention.
    """
    record = chassis_provenance(pkl_path, build=build, repo_root=repo_root,
                                workspace_root=workspace_root)
    out = chassis_provenance_path(pkl_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    os.replace(tmp, out)
    return record


def read_chassis_provenance(pkl_path: Path | str) -> dict | None:
    """The chassis provenance sidecar for ``pkl_path``, or ``None`` if absent/
    unreadable. Accepts either the stem-swap or the literal-append name."""
    candidates = [chassis_provenance_path(pkl_path),
                  Path(str(pkl_path) + ".provenance.json")]
    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None
    return None


def _cache_fingerprint(cache_dir: str | None) -> dict:
    """``cache_version.json``'s content fingerprint for ``cache_dir``, read
    FRESH (never referenced by path/symlink).

    ``cache_version.json`` is mutable and gets silently regenerated by later
    work — a run's own record must copy the fields it saw at write time, or
    the "identity" drifts out from under it with no signal (v2ecoli#472 §2).
    Honest-null shape mirrors ``code_provenance``: a reason string, not a
    guess, when there's nothing to read.
    """
    if not cache_dir:
        return {"available": False, "reason": "no cache_dir supplied"}
    version = read_cache_version(cache_dir)
    if version is None:
        return {"available": False,
                "reason": f"no {read_cache_version.__module__}"
                          f".CACHE_VERSION_FILENAME under {cache_dir!r}"}
    return {
        "available": True,
        "inputs_hash": version.inputs_hash,
        "schema_version": version.schema_version,
    }


def sim_data_ref(cache_dir: str | None = None,
                 sim_data_uri: str | None = None) -> dict:
    """A RESOLVABLE pointer to the sim_data this run used, so a later STANDALONE
    analysis of the sweep can find it without the study context.

    The molecular Analysis steps (ptools_*, mass fractions, ...) need the ParCa
    ``simData.cPickle`` to map ids/masses, but a sweep does not carry it. When
    you run the study all at once the cache is right there; when you run the sim
    first and analyse the sweep later, resolution has nothing to go on. Recording
    the pointer here (consumed by ``analysis_runner.resolve_sim_data``) closes
    that gap. Precedence, most explicit first:
      1. ``sim_data_uri`` — an explicit path/URI the caller knows (e.g. the
         dispatch's staged S3 cache). Recorded verbatim.
      2. ``$V2ECOLI_SIM_DATA`` — the same override the analysis side honors.
      3. ``$RAY_STAGE_S3`` / ``$CONTAINER_STAGE_S3`` + ``simData.cPickle`` — the
         S3 cache a remote dispatch staged from (resolvable off-node).
      4. ``cache_dir/simData.cPickle`` — the local build (a local sim's cache is
         a real path a later local analysis can read). ``exists`` is recorded so
         a stale/missing pointer is debuggable rather than silently wrong.
    """
    import os as _os
    if sim_data_uri:
        return {"uri": sim_data_uri, "source": "explicit"}
    env = _os.environ.get("V2ECOLI_SIM_DATA")
    if env:
        return {"uri": env, "source": "V2ECOLI_SIM_DATA"}
    stage = _os.environ.get("RAY_STAGE_S3") or _os.environ.get("CONTAINER_STAGE_S3")
    if stage:
        return {"uri": stage.rstrip("/") + "/simData.cPickle", "source": "stage_s3"}
    if cache_dir:
        p = _os.path.join(str(cache_dir), "simData.cPickle")
        return {"uri": p, "source": "cache_dir", "exists": _os.path.isfile(p)}
    return {"uri": None, "source": None}


def build_run_identity(*, repo_root: Path | str | None = None,
                       cache_dir: str | None = None,
                       design: dict | None = None,
                       sim_data_uri: str | None = None) -> dict:
    """One record combining code identity + cache content fingerprint +
    design/grid metadata + a sim_data pointer, for any ``run_*`` entrypoint to
    write alongside its own output (v2ecoli#472/#473).

    ``design`` is whatever grid/config metadata is already available at the
    call site (``experiment_id``, ``variant``, ``lineage_seed``,
    ``generation``, seed/generation counts, ...) — the cheap write-side half
    of #473; the statistical reduction that consumes it is a separate,
    later piece of work and is explicitly out of scope here.

    ``sim_data`` records a resolvable pointer to the run's ParCa sim_data (see
    :func:`sim_data_ref`) so a standalone analysis of the sweep resolves it the
    same way the all-at-once study does. Pass ``sim_data_uri`` when the caller
    knows the resolvable location (e.g. a dispatch's staged S3 cache).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return {
        "code": code_provenance(Path(repo_root)),
        "cache_version": _cache_fingerprint(cache_dir),
        "design": dict(design) if design else {},
        "sim_data": sim_data_ref(cache_dir, sim_data_uri),
    }


def write_run_identity_record(out_dir: str, record: dict) -> None:
    """Write an already-built ``build_run_identity(...)`` record to
    ``<out_dir>/run_identity.json``.

    Split from :func:`write_run_identity` so a caller that already computed
    the record (e.g. to embed it in its own ``run_config``/``summary.json``
    too) doesn't pay for a second ``git``/cache-fingerprint round trip just
    to also write the canonical sidecar.

    ``out_dir`` may be an ``s3://`` URI, in which case the write is delegated
    to :func:`v2ecoli.cache.save_json`, exactly mirroring how the sibling
    ``summary.json`` in ``v2ecoli/workflow/run.py`` is written. That matters
    because the local path here is ``pathlib``-based and ``Path("s3://b/k")``
    silently collapses to ``s3:/b/k`` — a sweep dispatched to S3 would
    otherwise land its identity in a local directory named ``s3:`` and read
    back as having none, which is the exact silent-provenance-loss failure
    ``run_identity.json`` exists to prevent. ``save_json`` stages through a
    temp file and uploads, so the S3 branch keeps the same all-or-nothing
    property the local branch gets from ``os.replace``.
    """
    if str(out_dir).startswith(_S3_PREFIX):
        from v2ecoli.cache import save_json

        save_json(record, out_dir.rstrip("/") + "/" + RUN_IDENTITY_FILENAME)
        return
    path = Path(out_dir) / RUN_IDENTITY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    os.replace(tmp, path)          # atomic: a killed run leaves no half-file


def write_run_identity(out_dir: str, *, repo_root: Path | str | None = None,
                       cache_dir: str | None = None,
                       design: dict | None = None,
                       sim_data_uri: str | None = None) -> dict:
    """Compute ``build_run_identity(...)`` and write it to
    ``<out_dir>/run_identity.json``. Returns the record written.

    Best-effort on the write itself (mkdir + atomic replace) but not on the
    computation — a run identity that silently failed to compute would be
    exactly the "looks recorded but isn't" failure this brief exists to
    close, so errors from ``build_run_identity`` propagate.

    ``sim_data_uri`` is threaded into the recorded ``sim_data`` pointer so a
    standalone analysis of the sweep can resolve the run's sim_data.
    """
    record = build_run_identity(repo_root=repo_root, cache_dir=cache_dir,
                                design=design, sim_data_uri=sim_data_uri)
    write_run_identity_record(out_dir, record)
    return record


def read_run_identity(sweep_dir: str) -> dict | None:
    """``run_identity.json`` from ``sweep_dir``, or ``None`` if absent/unreadable."""
    path = Path(sweep_dir) / RUN_IDENTITY_FILENAME
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None
