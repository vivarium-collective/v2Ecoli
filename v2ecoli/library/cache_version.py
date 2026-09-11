"""Cache version fingerprinting.

The cache at ``out/cache/`` (``sim_data_cache.dill`` + ``initial_state.json``)
is derived from (a) the shipped ParCa fixture at ``models/parca/parca_state.pkl.gz``
and (b) the code in ``v2ecoli.library.sim_data`` and its pint-boundary
helpers.  When either side changes incompatibly — e.g. the unum→pint migration
in #18 — a cache built from the previous code drops through sim simulation
steps with obscure ``AttributeError: 'Unum' object has no attribute 'to'``
tracebacks several frames deep.

This module computes a content hash over the inputs that determine cache
compatibility, writes it into ``cache_version.json`` at build time, and
verifies it at load time.  On mismatch, ``verify_cache_version`` raises
``StaleCacheError`` with a one-line rebuild instruction — no detective work
required.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterable


SCHEMA_VERSION = "3"
CACHE_VERSION_FILENAME = "cache_version.json"

# Packages on the ParCa fit path whose version genuinely changes fit output
# (or how a previously-fit cache unpickles) — see the module docstring at
# ":238-266" / PARCA_REVIEW.md A9. Recorded into ``CacheVersion.context`` and
# folded into ``inputs_hash`` so a cache built under one scipy/numba/etc and
# loaded under another is detectable instead of silently mis-unpickling deep
# in a simulation step.
CONTEXT_PACKAGES: tuple[str, ...] = (
    "scipy", "numpy", "numba", "dill", "cvxpy", "ecos", "stochastic-arrow",
)

#: build_params keys that describe *which artifact* a bundle is (not the code
#: that produced it) — condition/media/seed/n_seeds/patch identity. Folding
#: these into inputs_hash is what makes ``out/cache`` (basal) and
#: ``out/cache-stage1-heuristic`` (dnaA-patched) produce distinguishable
#: ``cache_version.json`` files instead of byte-identical ones (PARCA_REVIEW
#: A7). ``None`` for every key is the basal/default build.
DEFAULT_BUILD_PARAMS: dict = {
    "condition": None,
    "fixed_media": None,
    "seed": None,
    "n_seeds": None,
    "condition_manifest_hash": None,
    # Strain-defining genotype content (P1-6). These identify WHICH STRAIN a
    # bundle is, not merely which nutrient condition. ``new_genes`` changes the
    # genome the fit is built from (a heterologous insertion / KO overlay);
    # ``bundle_overrides`` / ``bundle_manifest`` name the ecoli-sources bundle
    # the raw_data was built from; ``perturbations`` fingerprints an in-memory
    # sim_data perturbation baked into the cache before it was written (e.g. a
    # new-gene expression / translation-efficiency override — see
    # v2ecoli/perturbations/new_gene_cache.py). Two strains that differed only
    # in these previously produced byte-identical ``cache_version.json`` and a
    # wrong-strain cache verified clean, so they are folded into ``inputs_hash``
    # here and compared requested-vs-stored in ``verify_cache_version``. ``None``
    # for every key is the wild-type / unperturbed build.
    "new_genes": None,
    "bundle_overrides": None,
    "bundle_manifest": None,
    "perturbations": None,
}

#: Config names whose absence from a built bundle is fatal (PARCA_REVIEW A6).
#: The online sim divides by zero on ``listeners.mass.cell_mass`` /
#: crashes in Equilibrium when either is missing (see the comment at
#: ``v2ecoli/core.py``'s ``_write_sim_input_bundle``), so a bundle missing
#: them must never be stamped valid by ``verify_cache_version``. This is a
#: deliberately small subset of ``v2ecoli.core._CACHE_CONFIG_NAMES`` — the
#: other config-getters can legitimately fail against legacy vEcoli sim_data
#: (e.g. redux-specific attrs) without making the bundle unusable for the
#: baseline sim, so only the two configs the review calls out are required.
REQUIRED_CACHE_CONFIG_NAMES: tuple[str, ...] = (
    "ecoli-mass-listener",
    "ecoli-metabolism",
)


def _module_version(name: str) -> str:
    """Installed version of ``name``, or the sentinel ``"absent"``.

    Never raises: an uninstalled/unimportable optional dep (or a package
    metadata lookup failing for any other reason) must not crash cache-version
    computation — it should just show up plainly in the recorded context.
    """
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        return "absent"


def probe_context() -> dict:
    """Snapshot the runtime-environment versions that can silently change a fit.

    Mirrors the pattern in ``v2ecoli/comparison/vecoli_parca.py``'s
    ``VEcoliParcaBuild._ref`` (see its module docstring at ``:12-19`` for the
    failure this guards against — a cache built under one scipy silently
    mis-unpickling under another). Called fresh on every
    ``compute_cache_version()`` so build-time and verify-time context reflect
    whatever environment is *actually running*, not an echoed value — that is
    what lets a scipy/numpy/etc upgrade between build and load move
    ``inputs_hash``.
    """
    ctx = {
        "python": "%d.%d.%d" % sys.version_info[:3],
    }
    for pkg in CONTEXT_PACKAGES:
        ctx[pkg] = _module_version(pkg)
    return ctx

# Files whose content determines whether an existing cache is compatible with
# the current code.  Hash is computed over the *sorted* concatenation of
# ``path\n<sha256 of file>`` lines so reordering or renaming any file is
# detected.
#
# The ParCa fixture is hashed because it *is* the cache's biological content.
# The sim_data / unit-bridge modules are hashed because they shape how that
# content is projected into configs (the unum→pint migration boundary).
INPUT_FILES: tuple[str, ...] = (
    # Biological content.
    "models/parca/parca_state.pkl.gz",
    # LoadSimData: turns sim_data into process configs.
    "v2ecoli/library/sim_data.py",
    # Unum↔pint migration boundary — regressions here are the whole reason
    # this module exists.
    "v2ecoli/library/unit_bridge.py",
    # Custom pint UnitRegistry with nucleotide/amino_acid/count; also
    # defines the Quantity schema type. A registry change can silently
    # change how Quantity fields round-trip through dill.
    "v2ecoli/types/quantity.py",
    # Seeds bulk/unique molecules into initial_state.json.
    "v2ecoli/library/initial_conditions.py",
    # This module itself. Self-referential on purpose: editing INPUT_FILES
    # (or the hashing/raise logic below) changes this file's own bytes, so
    # future additions here automatically bust every existing cache without
    # requiring a separate SCHEMA_VERSION bump.
    "v2ecoli/library/cache_version.py",
    # save_cache + shared composite infrastructure.
    "v2ecoli/core.py",
    # Per-architecture document builders. A change here can shift the
    # document shape and silently invalidate a cache built against the
    # old architecture. Renamed to the ecoli_* scheme in 645fe178; keep
    # this list in sync per AGENTS.md's "Adding a new composite
    # architecture" step 3.
    "v2ecoli/composites/ecoli_baseline.py",
    "v2ecoli/composites/ecoli_population.py",
    "v2ecoli/composites/ecoli_time_varying_env.py",
    "v2ecoli/composites/ecoli_colony.py",
    "v2ecoli/composites/ecoli_millard.py",
    # This PR adds two leaves to the reactor-coupled document
    # (``reactor.kla_co2``, ``reactor.ammonium_medium_mM``), and the
    # composite was not listed here — so the change that shifts the
    # document shape would have busted no cache. A cache built against
    # the old reactor architecture would have verified clean against the
    # new one, which is the failure this module exists to prevent.
    # Only the three files this change touches are added; the wider gap
    # (the remaining unlisted composites, and the fact that
    # ``v2ecoli/steps/`` is not represented here at all) is tracked in
    # #650 and is deliberately NOT closed piecemeal from here.
    "v2ecoli/composites/reactor_bird_coupled.py",
    # The two steps THIS CHANGE EDITS. They are here because this diff
    # touches them, not because a structural rule now covers steps --
    # environment_driver.py is instantiated by the same lines of
    # add_reactor_coupling and shifts the document identically, and is
    # deliberately NOT added here. The general rule is #650's.
    "v2ecoli/steps/reactor_cell_coupler.py",
    "v2ecoli/steps/environment_mirror.py",
    # Shared builders imported by the composites above (make_edge,
    # _make_instance, _get_special_step, per-step config dispatch, ...).
    # A change here shifts document shape for every composite that imports
    # it, exactly like a change to the composite file itself.
    "v2ecoli/composites/_helpers.py",
    "v2ecoli/composites/_millard_helpers.py",
)


class StaleCacheError(RuntimeError):
    """Raised when cache_version.json does not match the current code/fixture.

    The message includes the rebuild command so humans and CI logs both get
    an actionable next step without reading this module.
    """


@dataclass(frozen=True)
class CacheVersion:
    schema_version: str
    inputs_hash: str
    per_file_hashes: dict[str, str]
    # Runtime-environment package versions (A9) and per-build parameters
    # (A7) — both folded into inputs_hash. default_factory=dict keeps old
    # callers (e.g. tests constructing CacheVersion directly) working without
    # passing these.
    context: dict = field(default_factory=dict)
    build_params: dict = field(default_factory=dict)
    # Config names actually built into this bundle's sim_data_cache.dill
    # (PARCA_REVIEW A6) — a completeness record, not a fingerprint input.
    # Deliberately NOT folded into inputs_hash: which configs happen to
    # build successfully is a property of a specific build attempt (can be
    # flaky/environment-dependent), not of the inputs that determine
    # whether a cache is *compatible* with the current code. Folding it in
    # would make inputs_hash move on a run-to-run basis for identical
    # inputs. default_factory=tuple keeps old callers (and pre-A6 cache_
    # version.json files, which lack this key) working without passing it.
    configs: tuple = field(default_factory=tuple)
    # The provenance chain (schema 3): the artifacts this cache was DERIVED
    # FROM — founder → sim_data → chassis (the ParCa ``parca_state.pkl``).
    # Each entry is ``{"layer", "path", "source_sha256", "provenance",
    # ["parent_cache_version"]}``: the exact bytes of the consumed artifact
    # (``source_sha256``), the ``chassis-provenance/1`` sidecar embedded
    # verbatim if one sits beside it (``provenance``), and, when the source
    # is itself a bundle dir carrying a ``cache_version.json``, that parent
    # record nested under ``parent_cache_version`` so the chain composes.
    # A stable projection of it (``[{layer,source_sha256,commit,dirty}]``) is
    # folded into ``inputs_hash`` — exactly like ``build_params`` — so swapping
    # the chassis a cache was built on changes the fingerprint instead of
    # verifying clean against a different founder/sim_data/chassis (the silent
    # failure schema 3 exists to close). default_factory=list keeps old callers
    # and pre-chain (schema 2) cache_version.json files working.
    derived_from: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "inputs_hash": self.inputs_hash,
            "per_file_hashes": dict(self.per_file_hashes),
            "context": dict(self.context),
            "build_params": dict(self.build_params),
            "configs": sorted(self.configs),
            "derived_from": list(self.derived_from),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheVersion":
        return cls(
            schema_version=d.get("schema_version", ""),
            inputs_hash=d.get("inputs_hash", ""),
            per_file_hashes=dict(d.get("per_file_hashes", {})),
            context=dict(d.get("context", {})),
            build_params=dict(d.get("build_params", {})),
            configs=tuple(d.get("configs", ())),
            derived_from=list(d.get("derived_from", [])),
        )


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


#: Suffix of a chassis/artifact provenance sidecar. Kept here (not imported
#: from ``run_provenance``) to avoid a circular import: ``run_provenance``
#: already imports ``read_cache_version`` from this module. The sidecar for
#: ``parca_state.pkl`` is ``parca_state.provenance.json`` (stem-swap), which is
#: what ``run_provenance.write_chassis_provenance`` writes; the append form
#: ``<path>.provenance.json`` is also accepted so a source declared with any
#: extension resolves.
PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"


def _sidecar_candidates(path: str) -> list[str]:
    """Both accepted sidecar names for ``path`` (stem-swap first, then append).

    ``parca_state.pkl`` → ``parca_state.provenance.json`` (what the writer
    emits) and ``parca_state.pkl.provenance.json`` (the literal append form),
    so a source resolves however the sidecar was named.
    """
    p = str(path)
    cands: list[str] = []
    root, ext = os.path.splitext(p)
    if ext:
        cands.append(root + PROVENANCE_SIDECAR_SUFFIX)
    cands.append(p + PROVENANCE_SIDECAR_SUFFIX)
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _read_sidecar_provenance(path: str | None) -> dict:
    """Embed a ``<path>.provenance.json`` sidecar verbatim, or an honest-null.

    Mirrors the honest-null convention in ``run_provenance``: a reason
    string, never a guess, when there is nothing to read.
    """
    if not path:
        return {"available": False, "reason": "no source path"}
    for cand in _sidecar_candidates(path):
        if os.path.isfile(cand):
            try:
                with open(cand, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:  # noqa: BLE001
                return {"available": False,
                        "reason": f"provenance sidecar {cand!r} unreadable: {e}"}
    return {"available": False,
            "reason": f"no provenance sidecar beside {path!r}"}


def _resolve_source(source: dict) -> dict:
    """Turn a ``{"layer","path"}`` source into a ``derived_from`` entry.

    A file source records the sha256 of its exact bytes. A bundle-dir source
    (one holding a ``cache_version.json``) records that parent record nested
    under ``parent_cache_version`` and uses the parent's ``inputs_hash`` as its
    ``source_sha256`` — a directory has no single byte-stream, and the parent's
    fingerprint is the identity that must move when the parent is rebuilt, so
    the chain NESTS (founder → sim_data → chassis).
    """
    layer = source.get("layer")
    path = source.get("path")
    entry: dict = {"layer": layer, "path": path}
    entry["source_sha256"] = None
    if path and os.path.isfile(path):
        entry["source_sha256"] = _hash_file(path)
    elif path and os.path.isdir(path):
        parent = read_cache_version(path)
        if parent is not None:
            entry["parent_cache_version"] = parent.to_dict()
            entry["source_sha256"] = parent.inputs_hash
    entry["provenance"] = _read_sidecar_provenance(path)
    return entry


def _resolve_derived_from(sources: Iterable[dict] | None,
                          derived_from: Iterable[dict] | None) -> list[dict]:
    """Echo ``derived_from`` verbatim if given (verify's non-recompute path),
    else resolve ``sources`` into fresh entries, else an empty chain."""
    if derived_from is not None:
        return [dict(e) for e in derived_from]
    if sources:
        return [_resolve_source(dict(s)) for s in sources]
    return []


def _chain_entry_identity(entry: dict) -> tuple[str | None, object]:
    """``(commit, dirty)`` for one chain entry, read from its embedded
    ``chassis-provenance`` sidecar (``code.v2ecoli``). Defensive: any shape
    that doesn't carry the block yields ``(None, None)``."""
    prov = entry.get("provenance") or {}
    code = prov.get("code") or {}
    v2 = code.get("v2ecoli") or {}
    return v2.get("commit"), v2.get("dirty")


def _chain_fold_projection(derived_from: Iterable[dict]) -> list[dict]:
    """The stable ``[{layer,source_sha256,commit,dirty}]`` projection folded
    into ``inputs_hash``. Deliberately narrow: unrelated sidecar fields
    (created_at, artifact.bytes, build.argv, ...) do NOT move the fingerprint,
    only the artifact identity (source_sha256) and its recorded code identity
    (commit/dirty) do."""
    proj: list[dict] = []
    for entry in derived_from:
        commit, dirty = _chain_entry_identity(entry)
        proj.append({
            "layer": entry.get("layer"),
            "source_sha256": entry.get("source_sha256"),
            "commit": commit,
            "dirty": dirty,
        })
    return proj


def _aggregate_inputs_hash(per_file: dict[str, str], context: dict,
                           build_params: dict,
                           derived_from: Iterable[dict]) -> str:
    """The single fingerprint aggregator, shared by build and verify so both
    fold ``derived_from`` identically."""
    agg = hashlib.sha256()
    for rel in sorted(per_file):
        agg.update(f"{rel}\n{per_file[rel]}\n".encode())
    agg.update(b"\ncontext\n")
    agg.update(json.dumps(context, sort_keys=True).encode())
    agg.update(b"\nbuild_params\n")
    agg.update(json.dumps(build_params, sort_keys=True).encode())
    agg.update(b"\nderived_from\n")
    agg.update(json.dumps(_chain_fold_projection(derived_from),
                          sort_keys=True).encode())
    return agg.hexdigest()


def _default_repo_root() -> str:
    """Repo root resolved from THIS file's location, not the cwd.

    Bundles are generated inside a chdir'd isolation dir (run_comparison_ensemble
    os.chdir's into ``.regen_*`` so the default emitter's relative side-writes
    don't collide across parallel seeds). With ``repo_root="."`` every INPUT_FILE
    then resolves under that throwaway dir, hashes as MISSING, and the fingerprint
    collapses to a constant that never changes when the source changes — so the
    whole staleness check silently no-ops. Anchor to the package instead:
    this file is ``<repo>/v2ecoli/library/cache_version.py``.

    Still used as the second/fallback candidate in :func:`candidate_repo_roots`,
    and directly by any external caller that just wants "the v2ecoli source
    root" without the installed-dependency two-root split.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def candidate_repo_roots() -> list[str]:
    """Ordered, de-duplicated roots to search for an INPUT_FILES-style entry.

    When v2ecoli is consumed as an INSTALLED dependency (e.g. sms-ecoli
    depending on it via git), its SOURCE files live under the package
    (``site-packages/v2ecoli/...``, anchored by :func:`_default_repo_root`)
    but its DATA files (``models/parca/parca_state.pkl.gz``) live in the
    consuming WORKSPACE — so no single root resolves both.

    Returns the workspace root first (via ``viva_workspace.find_workspace_root``,
    a chdir-safe upward walk to the nearest ``workspace.yaml`` — see that
    function's docstring; it still works from inside a chdir'd ``.regen_*``
    isolation dir as long as that dir nests under the workspace, same
    guarantee :func:`_default_repo_root`'s docstring describes), then the
    package/source root. Import is guarded: an environment with no
    ``viva_workspace`` installed, or no ``workspace.yaml`` in any ancestor
    (e.g. a bare `pip install v2ecoli` with no workspace at all), simply
    falls back to the package root alone.

    For a standalone v2ecoli checkout (this repo) both roots resolve to the
    same directory, so this collapses to a single-entry list — identical to
    the old single-root behavior.
    """
    roots: list[str] = []
    # An EXPLICITLY DECLARED root wins, because the usual discovery cannot work
    # everywhere: ``find_workspace_root`` walks up from CWD for a workspace.yaml,
    # and a task scheduler gives each task its own working directory with no
    # workspace above it. Nextflow is the case that forced this -- the walk
    # raised, the `except` below swallowed it, and the only remaining root was
    # site-packages, where `models/` is not shipped. The failure surfaced as
    # "INPUT_FILES entry does not exist ... tried roots: ['…/site-packages']",
    # which reads like a renamed data file rather than an unresolved workspace.
    declared = os.environ.get("V2E_ROOT", "").strip()
    if declared and os.path.isdir(declared):
        roots.append(declared)
    try:
        from viva_workspace import find_workspace_root
        roots.append(str(find_workspace_root()))
    except Exception:
        pass
    roots.append(_default_repo_root())
    seen: set[str] = set()
    ordered: list[str] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    return ordered


def compute_cache_version(repo_root: str | None = None,
                          files: Iterable[str] = INPUT_FILES,
                          build_params: dict | None = None,
                          context: dict | None = None,
                          configs: Iterable[str] | None = None,
                          sources: Iterable[dict] | None = None,
                          derived_from: Iterable[dict] | None = None) -> CacheVersion:
    """Compute the fingerprint over INPUT_FILES + context + build_params.

    ``context`` defaults to a fresh live probe (see ``probe_context``) so two
    calls in different environments naturally disagree — that is what makes a
    scipy/numpy/etc upgrade between build and load move ``inputs_hash``
    (A9). ``build_params`` defaults to :data:`DEFAULT_BUILD_PARAMS` (a plain
    basal build with no condition/seed/patch) and is otherwise supplied by
    the caller that actually knows what it built (``core.py``'s
    ``save_sim_input`` / ``save_cache``) — unlike ``context`` there is no
    environment to "probe" for build params, they are inherent to the
    artifact (A7).

    ``configs`` (PARCA_REVIEW A6): the config names actually built into this
    bundle's ``sim_data_cache.dill``, recorded on the returned
    ``CacheVersion`` for ``verify_cache_version`` to check completeness
    against. ``None`` (the default) records an empty set — callers that
    don't build a bundle (most, which just want ``inputs_hash``) shouldn't
    have to pass an empty list. Deliberately excluded from ``inputs_hash``
    — see the field docstring on ``CacheVersion.configs``.

    ``sources`` (schema 3): the artifacts this build CONSUMED, each a
    ``{"layer","path"}`` dict (e.g. ``{"layer":"chassis","path":".../
    parca_state.pkl"}``). Each is resolved into a ``derived_from`` entry
    (bytes hashed, ``*.provenance.json`` sidecar embedded, a bundle-dir's
    ``cache_version.json`` nested) and the chain's stable projection is folded
    into ``inputs_hash`` — so a cache built on a different chassis fingerprints
    differently. ``derived_from`` is the echo path used by
    ``verify_cache_version``: pass an already-resolved chain to fold it in
    verbatim WITHOUT re-resolving from disk (the parent artifact may be absent
    at verify time). Supply at most one of the two.
    """
    # An explicit repo_root (tests, or a caller that already knows exactly
    # where its files live) means "search only there" — the original,
    # single-root behavior. repo_root=None means "resolve per entry against
    # the workspace-then-package candidate roots" so a data file that only
    # exists in the workspace (installed-dependency case) still resolves.
    candidate_roots = [repo_root] if repo_root is not None else candidate_repo_roots()
    per_file: dict[str, str] = {}
    for rel in sorted(files):
        resolved_path = None
        for root in candidate_roots:
            path = os.path.join(root, rel)
            if os.path.exists(path):
                resolved_path = path
                break
        if resolved_path is None:
            # A vanished fingerprint input is a bug, not a state: hashing it
            # to a stable "MISSING" sentinel silently drops the file from
            # the fingerprint forever (its edits stop moving inputs_hash).
            # That is exactly how 5/11 INPUT_FILES went dead unnoticed after
            # the ecoli_* composite rename in 645fe178. Fail loudly instead.
            raise FileNotFoundError(
                f"cache_version INPUT_FILES entry does not exist: {rel!r} "
                f"(tried roots: {candidate_roots!r}). This file was "
                f"renamed or deleted without updating "
                f"v2ecoli/library/cache_version.py:INPUT_FILES — see "
                f"AGENTS.md 'Adding a new composite architecture' step 3."
            )
        per_file[rel] = _hash_file(resolved_path)

    if context is None:
        context = probe_context()
    resolved_build_params = dict(DEFAULT_BUILD_PARAMS)
    if build_params:
        resolved_build_params.update(
            {k: v for k, v in build_params.items() if k in resolved_build_params})

    resolved_derived_from = _resolve_derived_from(sources, derived_from)

    return CacheVersion(
        schema_version=SCHEMA_VERSION,
        inputs_hash=_aggregate_inputs_hash(
            per_file, context, resolved_build_params, resolved_derived_from),
        per_file_hashes=per_file,
        context=dict(context),
        build_params=resolved_build_params,
        configs=tuple(sorted(configs)) if configs is not None else (),
        derived_from=resolved_derived_from,
    )


def write_cache_version(cache_dir: str, version: CacheVersion | None = None,
                        repo_root: str | None = None,
                        build_params: dict | None = None,
                        configs: Iterable[str] | None = None,
                        sources: Iterable[dict] | None = None) -> CacheVersion:
    """Write cache_version.json inside ``cache_dir``.  Called by save_cache.

    ``sources`` (schema 3) declares the artifacts this bundle was derived from
    (founder / sim_data / chassis); it is threaded into ``compute_cache_version``
    so the chain is recorded and folded into ``inputs_hash``.
    """
    if version is None:
        version = compute_cache_version(repo_root=repo_root,
                                        build_params=build_params,
                                        configs=configs,
                                        sources=sources)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, CACHE_VERSION_FILENAME)
    with open(path, "w") as f:
        json.dump(version.to_dict(), f, indent=2, sort_keys=True)
    return version


def read_cache_version(cache_dir: str) -> CacheVersion | None:
    """Return the cached version, or ``None`` if not present."""
    path = os.path.join(cache_dir, CACHE_VERSION_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return CacheVersion.from_dict(json.load(f))


def _resolve_build_params(build_params: dict | None) -> dict:
    """Fill ``build_params`` against :data:`DEFAULT_BUILD_PARAMS`.

    Same normalization ``compute_cache_version`` applies before hashing:
    unknown keys are dropped, missing keys default to their ``None`` sentinel.
    Sharing it here lets ``verify_cache_version`` compare a *requested*
    build against a *stored* one on exactly the keys that shape the
    fingerprint.
    """
    resolved = dict(DEFAULT_BUILD_PARAMS)
    if build_params:
        resolved.update(
            {k: v for k, v in build_params.items() if k in resolved})
    return resolved


SKIP_CACHE_VERIFY_ENV = "V2ECOLI_SKIP_CACHE_VERIFY"
_skip_verify_warned = False


def verify_cache_version(cache_dir: str, repo_root: str | None = None,
                         expected_build_params: dict | None = None,
                         require_clean_chain: bool = False,
                         expected_chassis: dict | None = None) -> None:
    """Raise StaleCacheError if the cache on disk doesn't match current inputs.

    Called from the cache load path.  A missing ``cache_version.json`` is a
    hard error too — we can't prove a pre-versioning cache is safe, so treat
    it the same as a mismatch.

    ``expected_build_params`` (P1-6) is what makes a WRONG-STRAIN cache fail.
    A caller that knows which strain/condition it *requested* (e.g. new_genes,
    bundle_overrides, condition) passes those here; every supplied key is
    compared against the bundle's stored ``build_params`` and any divergence
    raises. Without it this function has no independent notion of the request
    and cannot tell a wild-type cache apart from a new-gene cache — the silent
    failure this parameter closes. Left ``None`` (the load paths that don't
    yet know the request) the comparison is skipped and behavior is unchanged.

    ``build_params`` (A7) describes *which artifact* the cache is (condition,
    seed, n_seeds, strain, ...) — it is a property of the bundle, not something
    "current code" can independently re-derive, so recomputing "current"
    echoes ``stored.build_params`` back rather than defaulting them away.
    That keeps a real non-basal bundle (e.g. built with a non-default seed)
    from failing verification against *itself* on the file/context inputs_hash;
    the requested-vs-stored strain check above is what catches a mismatched
    ``--cache-dir``, not the echoed recompute.
    ``context`` (A9) is the opposite: it is re-probed fresh here so an
    environment change between build and load is exactly what this catches.

    ``configs`` (A6): if the stored version recorded a non-empty config set
    (bundles written after this check existed), assert
    ``REQUIRED_CACHE_CONFIG_NAMES`` is a subset of it — a bundle missing
    ``ecoli-mass-listener``/``ecoli-metabolism`` must never verify clean,
    even though neither config participates in ``inputs_hash``. A bundle
    with an *empty* recorded config set (pre-A6, or written by a caller
    that doesn't build a ``configs`` dict at all — e.g. a hand-built test
    fixture) is not asserted against: we can't distinguish "nothing was
    recorded" from "nothing is required" from the marker alone, and the
    primary defense against an incomplete bundle is the build itself
    refusing to write one (``v2ecoli.core._write_sim_input_bundle``).

    ``derived_from`` (schema 3): the provenance chain is ECHOED here, never
    recomputed — the parent artifact (the ParCa ``parca_state.pkl`` a cache was
    built on) is routinely absent at load time, so ``current`` folds
    ``stored.derived_from`` back in verbatim, exactly as ``build_params`` is
    echoed, letting a real chained cache verify against itself. Three schema-3
    guards then run on the stored chain: (a) a schema-3 cache with no chain at
    all is stale (its chassis provenance was never recorded); (b) any chain
    layer that is ``dirty`` or has a null ``commit`` WARNs loudly by default and
    hard-fails when ``require_clean_chain`` (or ``$V2E_REQUIRE_CLEAN_CHAIN``) is
    set; (c) ``expected_chassis={"commit": ...}`` fails if the chain's chassis
    layer was built from a different commit — the chain-level mirror of
    ``expected_build_params``.
    """
    if os.environ.get(SKIP_CACHE_VERIFY_ENV):
        # The advertised escape hatch for deliberate cross-commit / cross-version
        # work (e.g. re-firing a run on a cache copied from another commit).
        # Honoured HERE, at the single verification chokepoint, so EVERY caller
        # respects it -- not only core.load_cache_bundle (which has its own
        # pre-check). Before this, direct callers on the staging path (the
        # ray-batch entrypoint's cache-verify, via run_pbg) ran verification
        # unconditionally, so the env var -- documented as the bypass -- had no
        # effect at that layer and a legitimately-copied cross-commit cache
        # failed at staging despite the flag being set.
        global _skip_verify_warned
        if not _skip_verify_warned:
            warnings.warn(
                f"{SKIP_CACHE_VERIFY_ENV} is set: skipping cache-version "
                f"verification for {cache_dir}. A stale or cross-commit cache "
                f"may load silently. Unset it for normal use.",
                stacklevel=2,
            )
            _skip_verify_warned = True
        return None

    stored = read_cache_version(cache_dir)
    current = compute_cache_version(
        repo_root=repo_root,
        build_params=(stored.build_params if stored is not None else None),
        derived_from=(stored.derived_from if stored is not None else None),
    )

    if stored is None:
        raise StaleCacheError(_rebuild_message(
            cache_dir,
            reason=f"{cache_dir}/{CACHE_VERSION_FILENAME} missing "
                   f"(cache was built before versioning was introduced, "
                   f"or was partially written)",
            expected=current,
            actual=None,
        ))

    # P1-6: compare the REQUESTED strain/condition against what the bundle was
    # actually built for. This is the real comparison — the echoed recompute of
    # ``current`` above deliberately folds in ``stored.build_params`` so a
    # bundle verifies against itself on file/context hashes, which means it can
    # never catch a wrong-strain --cache-dir on its own. Only an explicit
    # requested-vs-stored diff can, so do it here whenever the caller knows the
    # request.
    if expected_build_params is not None:
        requested = _resolve_build_params(expected_build_params)
        stored_bp = _resolve_build_params(stored.build_params)
        mismatched = {
            key: (requested[key], stored_bp[key])
            for key in requested
            if requested[key] != stored_bp[key]
        }
        if mismatched:
            detail = ", ".join(
                f"{key}: requested {req!r} != cached {cached!r}"
                for key, (req, cached) in sorted(mismatched.items()))
            raise StaleCacheError(_rebuild_message(
                cache_dir,
                reason=f"build_params mismatch (wrong strain/condition cache): "
                       f"{detail} — the cache at this path was built for a "
                       f"different strain/condition than requested (P1-6)",
                expected=current,
                actual=stored,
            ))

    if stored.schema_version != current.schema_version:
        raise StaleCacheError(_rebuild_message(
            cache_dir,
            reason=f"schema_version mismatch "
                   f"(stored={stored.schema_version!r}, "
                   f"current={current.schema_version!r})",
            expected=current,
            actual=stored,
        ))

    # --- schema-3 provenance-chain guards ------------------------------
    # Only for schema-3 caches; a schema-2 cache has already raised on the
    # schema check above (it predates the chain and its provenance is
    # unrecoverable — see the audit CLI's PRE-CHAIN message).
    if stored.schema_version == "3":
        if not stored.derived_from:
            # (a) A chained-schema cache that recorded no chain: its chassis
            # provenance was never captured, so nothing downstream can prove
            # which founder/sim_data/chassis it came from. WARN by default and
            # hard-fail only when a verified chain is demanded — the same opt-in
            # posture as the dirty-layer guard (b) below. This matters during the
            # rollout of ``sources=`` wiring: many cache-building callers
            # (new_gene_cache, variant_cache, build_condition_cache,
            # run_comparison_ensemble, …) do not declare their sources yet, and a
            # hard failure here would reject every one of their caches on load.
            # CD campaigns that require a clean chain set $V2E_REQUIRE_CLEAN_CHAIN
            # (or pass require_clean_chain=True) and DO hard-fail — which is what
            # catches the stale-chassis incident this design targets.
            require_clean = (require_clean_chain
                             or bool(os.environ.get("V2E_REQUIRE_CLEAN_CHAIN")))
            msg = _rebuild_message(
                cache_dir,
                reason="schema_version 3 cache has an empty/absent "
                       "'derived_from' chain — chassis provenance was NOT "
                       "recorded, so this cache cannot be traced to the ParCa "
                       "state it was built on. Rebuild declaring its sources.",
                expected=current,
                actual=stored,
            )
            if require_clean:
                raise StaleCacheError(msg)
            warnings.warn(msg)

        # (c) Chain-level analogue of expected_build_params: fail if the
        # chassis the cache was built on isn't the one the caller expects.
        if expected_chassis is not None:
            want_commit = expected_chassis.get("commit")
            chassis_commit = _chain_chassis_commit(stored.derived_from)
            if want_commit is not None and chassis_commit != want_commit:
                raise StaleCacheError(_rebuild_message(
                    cache_dir,
                    reason=f"chassis commit mismatch: requested "
                           f"{want_commit!r} != chain chassis "
                           f"{chassis_commit!r} — this cache was built on a "
                           f"different ParCa chassis than expected",
                    expected=current,
                    actual=stored,
                ))

        # (b) A dirty/uncommitted layer anywhere in the chain: WARN loudly by
        # default, hard-fail only when the caller (or the environment) demands
        # a clean chain.
        dirty_layers = _dirty_chain_layers(stored.derived_from)
        if dirty_layers:
            require_clean = (require_clean_chain
                             or bool(os.environ.get("V2E_REQUIRE_CLEAN_CHAIN")))
            message = _dirty_chain_message(cache_dir, dirty_layers, require_clean)
            if require_clean:
                raise StaleCacheError(message)
            warnings.warn(message)

    if stored.inputs_hash != current.inputs_hash:
        changed = [
            rel for rel in current.per_file_hashes
            if current.per_file_hashes.get(rel)
               != stored.per_file_hashes.get(rel)
        ]
        raise StaleCacheError(_rebuild_message(
            cache_dir,
            reason=f"inputs_hash mismatch; files differ: {changed}",
            expected=current,
            actual=stored,
        ))

    if stored.configs:
        missing_required = sorted(
            set(REQUIRED_CACHE_CONFIG_NAMES) - set(stored.configs))
        if missing_required:
            raise StaleCacheError(_rebuild_message(
                cache_dir,
                reason=f"required config(s) missing from bundle: "
                       f"{missing_required} (stored configs: "
                       f"{sorted(stored.configs)}) — PARCA_REVIEW A6",
                expected=current,
                actual=stored,
            ))


def _rebuild_message(cache_dir: str, reason: str,
                     expected: CacheVersion,
                     actual: CacheVersion | None) -> str:
    lines = [
        f"Cache at {cache_dir!r} is stale or unversioned: {reason}.",
        "",
        "Rebuild it:",
        "    python scripts/build_cache.py",
        "",
        f"Expected inputs_hash: {expected.inputs_hash[:16]}...",
    ]
    if actual is not None:
        lines.append(f"Actual   inputs_hash: {actual.inputs_hash[:16]}...")
    return "\n".join(lines)


def _chain_chassis_commit(derived_from: Iterable[dict]) -> str | None:
    """The recorded chassis-layer commit in the chain, or ``None`` if there is
    no chassis layer / it recorded no commit."""
    for entry in derived_from:
        if entry.get("layer") == "chassis":
            commit, _ = _chain_entry_identity(entry)
            return commit
    return None


def _dirty_chain_layers(derived_from: Iterable[dict]) -> list[dict]:
    """Chain entries whose recorded code identity is dirty or commit-less.

    Each returned item is ``{layer, commit, dirty, reason}`` so the caller can
    render a specific, multi-line warning naming which layer is untrustworthy.
    """
    flagged: list[dict] = []
    for entry in derived_from:
        commit, dirty = _chain_entry_identity(entry)
        reason = None
        if dirty is True:
            reason = "built from a DIRTY tree (provenance is identify-only)"
        elif commit is None:
            reason = "no commit recorded (provenance unavailable/unversioned)"
        if reason is not None:
            flagged.append({
                "layer": entry.get("layer"),
                "commit": commit,
                "dirty": dirty,
                "reason": reason,
            })
    return flagged


def _dirty_chain_message(cache_dir: str, dirty_layers: list[dict],
                         hard: bool) -> str:
    """Loud, multi-line message naming every untrustworthy chain layer."""
    verb = ("REFUSING cache" if hard else "WARNING")
    lines = [
        f"{verb}: provenance chain of {cache_dir!r} has "
        f"{len(dirty_layers)} untrustworthy layer(s):",
    ]
    for item in dirty_layers:
        commit = item["commit"]
        short = commit[:12] if isinstance(commit, str) else commit
        lines.append(
            f"    - layer {item['layer']!r}: commit={short}, "
            f"dirty={item['dirty']} — {item['reason']}")
    if hard:
        lines.append(
            "require_clean_chain / $V2E_REQUIRE_CLEAN_CHAIN is set: a cache "
            "derived from a dirty/unversioned chassis is not reproducible.")
    else:
        lines.append(
            "This cache's lineage cannot be pinned to clean commits. Set "
            "require_clean_chain=True or $V2E_REQUIRE_CLEAN_CHAIN to make this "
            "a hard error.")
    return "\n".join(lines)


def _format_chain(version: CacheVersion, indent: str = "") -> list[str]:
    """Human-readable lines describing a schema-3 chain (recursion-safe:
    nested ``parent_cache_version`` records are rendered one level deeper)."""
    lines: list[str] = []
    if not version.derived_from:
        lines.append(f"{indent}(no derived_from chain)")
        return lines
    for entry in version.derived_from:
        commit, dirty = _chain_entry_identity(entry)
        short = commit[:12] if isinstance(commit, str) else commit
        sha = entry.get("source_sha256")
        sha_short = sha[:12] if isinstance(sha, str) else sha
        lines.append(
            f"{indent}- layer={entry.get('layer')!r} "
            f"path={entry.get('path')!r}")
        lines.append(
            f"{indent}    source_sha256={sha_short} commit={short} "
            f"dirty={dirty}")
        parent = entry.get("parent_cache_version")
        if parent:
            lines.append(f"{indent}    parent_cache_version:")
            lines.extend(_format_chain(
                CacheVersion.from_dict(parent), indent + "        "))
    return lines


def _audit_main(argv: list[str]) -> int:
    """``python -m v2ecoli.library.cache_version <cache_dir_or_json>`` — print
    a cache's provenance chain human-readably.

    A schema-2 (pre-chain) cache is called out explicitly: its chassis
    provenance was never recorded and cannot be recovered, so any commit read
    off an S3 key is a claim, not a fact.
    """
    if not argv:
        print("usage: python -m v2ecoli.library.cache_version "
              "<cache_dir_or_json>", file=sys.stderr)
        return 2
    target = argv[0]
    if os.path.isdir(target):
        version = read_cache_version(target)
        source = os.path.join(target, CACHE_VERSION_FILENAME)
    else:
        source = target
        try:
            with open(target, encoding="utf-8") as f:
                version = CacheVersion.from_dict(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"cannot read {target!r}: {e}", file=sys.stderr)
            return 2
    if version is None:
        print(f"no {CACHE_VERSION_FILENAME} found under {target!r}",
              file=sys.stderr)
        return 2

    print(f"cache_version: {source}")
    print(f"  schema_version: {version.schema_version}")
    print(f"  inputs_hash:    {version.inputs_hash[:16]}...")
    print(f"  build_params:   {json.dumps(version.build_params, sort_keys=True)}")
    if version.schema_version != "3":
        print(f"PRE-CHAIN cache (schema {version.schema_version}): chassis "
              "provenance NOT RECORDED and unrecoverable; treat the S3 key's "
              "commit as a claim, not a fact.")
        return 0
    print("  derived_from chain:")
    for line in _format_chain(version, indent="    "):
        print(line)
    dirty = _dirty_chain_layers(version.derived_from)
    if dirty:
        print(_dirty_chain_message(source, dirty, hard=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_audit_main(sys.argv[1:]))
