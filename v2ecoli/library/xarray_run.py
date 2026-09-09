"""Multi-generation XArray run helper for v2ecoli composites.

Wraps the validated emitter-per-generation pattern (see
``docs/superpowers/notes/2026-05-22-xarray-multigen-emit-validated.py``) so the
three non-obvious gotchas of driving ``XArrayEmitter`` manually are handled
once, in one place:

  1. Buffer must flush before close, else ``flush`` asserts. We force a small
     buffer + ``subsample(1)`` so every update produces an emit-step and a
     flush happens within a few ticks per generation.

  2. Input must EXACTLY match the view subtree. We filter the agent's state
     down to the leaves declared in the view config before each ``update``.

  3. vEcoli daughters can re-use the parent's id at division (e.g. ``00`` ->
     ``00``, ``01``). We detect division by agent-count growth in the colony,
     not by ``followed in agents`` (which keeps being True across some
     divisions). When a division is seen, we close the current emitter and
     open a new one for the next generation, following a daughter.
"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

# NB: XArrayEmitter is imported lazily inside _build_emitter — it requires the
# [xarray] extra, but build_emitter_config (a pure dict builder) and the
# view/coord helpers must remain importable without it (e.g. in CI fast-tests).


#: v2ecoli observables known to be vectors/arrays.
#:
#: Vectors need a coord array in ``output_metadata`` so XArrayEmitter can
#: allocate them as (buf_size, N)-shape buffers instead of scalar slots.
#: :func:`extract_output_metadata_from_state` discovers coord arrays from
#: the composite's live state.
#:
#: ``view_from_emit_paths`` previously SKIPPED these leaves entirely
#: (so vector-only studies fell back to SQLite). Setting
#: ``include_vectors=True`` (the default since 2026-05-23) keeps them
#: in the view; the caller supplies output_metadata.
_KNOWN_VECTOR_LEAVES: set[str] = {
    "monomer_counts",
    "mRna_counts",
    "fork_coordinates",
    "RNAP_coordinates",
    "ribosome_coordinates",
}


# Env-gated unique-store emit (V2ECOLI_EMIT_UNIQUE=1): mirror of the
# parquet_run path. The chromosome-state renderer needs per-molecule genomic
# coordinates that live in the agent's structured ``unique`` numpy arrays --
# which the plain view filter cannot reach. When enabled we extract the named
# attribute arrays of the ACTIVE entries (``_entryState`` mask) and emit them
# as nested keys so the emitter flattens them to ``active_RNAP__coordinates``
# etc. ``chromosome_domain__child_domains`` (the (n_domain, 2) parent->child
# domain tree) is FLATTENED row-major to a 1-D list<int> aligned to
# ``chromosome_domain__domain_index`` so the renderer can place daughter-strand
# RNAPs on the replication bubbles, not just the rim.
_EMIT_UNIQUE = os.environ.get("V2ECOLI_EMIT_UNIQUE", "") not in ("", "0", "false", "False")
_UNIQUE_EMIT_SPEC = {
    "active_RNAP": ["coordinates", "domain_index"],
    "active_replisome": ["coordinates", "domain_index"],
    "full_chromosome": ["unique_index", "domain_index"],
    "chromosome_domain": ["domain_index", "child_domains"],
}


def _flatten_attr(col):
    """Flatten a unique-store attribute column to a 1-D python list.

    1-D attributes pass through; 2-D ones (child_domains is (n_active, 2)) are
    flattened row-major so they serialize as a plain list<int> column aligned
    to domain_index.
    """
    arr = np.asarray(col)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.tolist()


def _extract_unique_attrs(agent_state: dict) -> dict:
    """Pull active-entry attribute arrays out of an agent's ``unique`` store.

    Returns ``{mol: {attr: [values...]}}`` for the active entries of each
    configured unique molecule, ready to merge into the emit payload (flattens
    to ``<mol>__<attr>`` columns).
    """
    out: dict = {}
    unique = (agent_state or {}).get("unique") or {}
    for mol, attrs in _UNIQUE_EMIT_SPEC.items():
        arr = unique.get(mol)
        if arr is None or not hasattr(arr, "dtype") or arr.dtype.names is None:
            out[mol] = {a: [] for a in attrs}
            continue
        names = set(arr.dtype.names)
        if "_entryState" in names:
            active = arr[arr["_entryState"].view(np.bool_)]
        else:
            active = arr
        out[mol] = {
            a: (_flatten_attr(active[a]) if a in names else [])
            for a in attrs
        }
    return out


def view_from_emit_paths(
    emit_paths: list[str], *, default_dtype: str = "<f8",
    skip_leaves: set[str] | None = None,
    include_vectors: bool = True,
) -> list[dict]:
    """Build a v2ecoli-shaped XArray view config from a flat list of dotted emit paths.

    For dashboard-driven runs, each study declares its observables as dotted
    paths like ``"listeners.replication_data.number_of_oric"``. The XArray
    view config is nested by root + variable path. This helper builds it.

    Only handles ``listeners.<...>`` paths for v0 (the v2ecoli convention).
    Drops other roots (``bulk``, etc.) with a warning.

    Default dtype is ``<f8`` (float64); pass an explicit ``dtype_overrides``
    map at the call site if int dtypes are needed for specific leaves.
    """
    # Behaviour: by default INCLUDE known vector leaves (caller must supply
    # output_metadata coord arrays); old callers using include_vectors=False
    # keep the legacy scalar-only behaviour.
    if skip_leaves is None:
        skip = set() if include_vectors else _KNOWN_VECTOR_LEAVES
    else:
        skip = skip_leaves
    variables: dict = {}
    for p in emit_paths:
        # Accept dotted or slash-separated paths; strip optional agents/<id>/ prefix.
        parts = [x for x in p.replace(".", "/").split("/") if x]
        if parts and parts[0] == "agents" and len(parts) >= 3:
            parts = parts[2:]
        if len(parts) < 2 or parts[0] != "listeners":
            continue
        rest = parts[1:]
        leaf_name = rest[-1]
        if leaf_name in skip:
            continue
        cursor = variables
        for k in rest[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[leaf_name] = [{"path": leaf_name, "dtype": default_dtype}]
    if not variables:
        return []
    return [{"root": ("listeners",), "variables": variables}]


def _flatten_keys(d: dict, prefix: str = "") -> Any:
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten_keys(v, full)
        else:
            yield full


def extract_output_metadata_from_state(
    state: dict,
    view: list[dict],
    named_metadata: dict | None = None,
) -> dict:
    """Inspect a live composite state + a view config; build XArrayEmitter
    ``output_metadata`` with coord arrays for vector leaves.

    XArrayEmitter consumes ``config["output_metadata"]`` at construction time
    to learn the per-leaf coord arrays (which determine each variable's
    additional dimension beyond the time axis). Scalar leaves get no coord
    (and the storage allocates a scalar slot per emit step). Vector leaves
    need a coord array.

    When ``named_metadata`` is provided (typically the return value of
    ``v2ecoli.library.output_metadata.output_metadata(composite.state)``),
    element names from annotated listener ``outputs()`` schemas are preferred
    over the default integer-index ``range(N)`` fallback.  Un-annotated vector
    leaves still get ``range(N)`` so nothing breaks for them.

    The returned dict mirrors the view's path structure (so
    ``make_coords`` can resolve each leaf via ``get_in``).

    Reads from ``state['agents']['0']`` first; falls back to top-level state.
    Returns ``{}`` if no vector leaves found (XArrayEmitter then renders all
    leaves as scalars, the legacy behaviour).

    Output is shaped RELATIVE to the view root: e.g. for view
    ``root=('listeners',)`` with a vector leaf ``monomer_counts``, the result
    is ``{'monomer_counts': [0..N-1]}`` (integers, fallback) or
    ``{'monomer_counts': [name1, name2, ...]}`` (names, when annotated).
    TreeView.make_coords does ``get_in(coords, root.metadata_path)`` where
    metadata_path is ``()`` for plain listener roots, then walks the leaf
    paths from there. So the coords dict must be flat under the view root.

    Args:
        state: Composite state dict.
        view: XArrayEmitter view config (list of root/variables dicts).
        named_metadata: Optional store-relative metadata dict from
            ``output_metadata(state)``.  When a leaf's full store path
            (root_path + leaf_input_path) is present in this dict, its value
            is used as the coord array instead of ``range(N)``.  Pass ``None``
            (default) to keep the legacy ``range(N)`` behaviour for all leaves.
    """
    cell = (state.get("agents") or {}).get("0") or state
    if not isinstance(cell, dict):
        return {}

    out: dict = {}
    for entry in view:
        root_path = tuple(entry.get("root") or ())
        # Walk the state to the view root.
        cursor: Any = cell
        for k in root_path:
            cursor = (cursor or {}).get(k, {})
        if not isinstance(cursor, dict):
            continue
        # For each declared leaf, check if the value at that path is a vector.
        for leaf_input_path, leaf_output_name in _view_leaves([entry]):
            value: Any = cursor
            for k in leaf_input_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.size <= 1:
                continue  # scalar — no coord needed

            # Prefer element names from named_metadata when available.
            # Navigate named_metadata by root_path + leaf_input_path.
            named_coord: list | None = None
            if named_metadata is not None:
                nm: Any = named_metadata
                for k in root_path:
                    if not isinstance(nm, dict):
                        nm = None
                        break
                    nm = nm.get(k)
                for k in leaf_input_path:
                    if not isinstance(nm, dict):
                        nm = None
                        break
                    nm = nm.get(k)
                if isinstance(nm, (list, tuple)) and len(nm) == arr.shape[0]:
                    named_coord = list(nm)

            coord = named_coord if named_coord is not None else list(range(int(arr.shape[0])))

            # Nest coord under leaf path RELATIVE to root (see docstring).
            cursor_out = out
            for k in leaf_input_path[:-1]:
                cursor_out = cursor_out.setdefault(k, {})
            cursor_out[leaf_input_path[-1] if leaf_input_path else leaf_output_name] = coord
    return out


def _view_leaves(view: list[dict]) -> list[tuple[tuple[str, ...], str]]:
    """Walk a view config; return [(input_path_under_root, output_name), ...]."""
    leaves: list[tuple[tuple[str, ...], str]] = []

    def walk(node: Any, prefix: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for spec in node:
                leaves.append((prefix, spec["path"]))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, prefix + (k,))

    for entry in view:
        walk(entry.get("variables", {}), ())
    return leaves


def filter_view_to_existing_leaves(state: dict, view: list[dict]) -> list[dict]:
    """Drop view leaves that aren't produced by the composite's current state.

    XArrayEmitter raises ``KeyError: Missing emit paths`` when the view
    declares a leaf the composite doesn't emit. SQLite was lenient and
    silently captured only what was there. This mirrors that leniency
    for the xarray path: walks each leaf path in ``state['agents']['0']``,
    drops any leaf whose value is ``None`` (path doesn't exist), and
    returns a new view with the same shape minus the missing leaves.

    Empty variable subtrees collapse out — if a TreeView ends up with no
    leaves it is removed entirely (TreeView raises ``Missing arguments``
    on an empty variables dict).
    """
    cell = (state.get("agents") or {}).get("0") or state
    if not isinstance(cell, dict):
        return view

    out_view: list[dict] = []
    for entry in view:
        root_path = tuple(entry.get("root") or ())
        cursor: Any = cell
        for k in root_path:
            cursor = (cursor or {}).get(k, {})
        if not isinstance(cursor, dict):
            continue

        # Rebuild the variables dict, keeping only present leaves.
        # A leaf is "present" iff the composite emits a scalar/array at it.
        # Missing OR empty-container values (None, {}, []) are dropped — they
        # show up when ``inject_emitter_for_declared_paths`` registers a path
        # whose listener Step isn't in this particular composite variant
        # (the schema default for the slot is an empty container).
        # XArrayEmitter raises ``KeyError: Missing emit paths`` on those at
        # tick 1 because ``dict_to_paths`` yields nothing for an empty
        # container, so the path never gets discarded from ``emit_queue``.
        new_vars: dict = {}
        kept_any = False
        for leaf_input_path, leaf_output_name in _view_leaves([entry]):
            value: Any = cursor
            for k in leaf_input_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            if value is None:
                continue
            if isinstance(value, (dict, list, str)) and len(value) == 0:
                continue
            # Re-attach this leaf at its original path in new_vars.
            # Need to find the original list-spec from entry["variables"].
            orig_spec = entry.get("variables", {})
            for k in leaf_input_path[:-1]:
                orig_spec = orig_spec.get(k, {})
            if not isinstance(orig_spec, dict):
                continue
            spec_list = orig_spec.get(leaf_input_path[-1] if leaf_input_path else leaf_output_name)
            if not isinstance(spec_list, list):
                continue
            cursor_out = new_vars
            for k in leaf_input_path[:-1]:
                cursor_out = cursor_out.setdefault(k, {})
            cursor_out[leaf_input_path[-1] if leaf_input_path else leaf_output_name] = spec_list
            kept_any = True

        if kept_any:
            new_entry = dict(entry)
            new_entry["variables"] = new_vars
            out_view.append(new_entry)

    return out_view


def _filter_agent_state(agent_state: dict, view: list[dict]) -> dict:
    """Filter agent state to exactly the leaves declared by the view.

    Returns a dict shaped like ``{<root[0]>: {<root[1]>: ...{<leaf>: value}...}}``
    for each view entry. Unrelated keys (other listeners, bulk, etc.) are dropped.
    """
    out: dict[str, Any] = {}
    for entry in view:
        root: tuple[str, ...] = tuple(entry["root"])
        # Start at the agent and walk down to the root.
        cur: Any = agent_state
        for k in root:
            cur = (cur or {}).get(k, {})
        # Now `cur` is the subtree at the view root. Filter it to declared paths.
        for leaf_path, _out_name in _view_leaves([entry]):
            value: Any = cur
            for k in leaf_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            # Place value into out at full path: root/.../leaf_path
            cursor = out
            for k in root + leaf_path[:-1]:
                cursor = cursor.setdefault(k, {})
            cursor[leaf_path[-1]] = value
    return out


def build_emitter_config(
    *,
    store_path: Path,
    view: list[dict],
    metadata_base: dict,
    generation: int,
    agent_id: str,
    buffer_size: int = 600,
    output_metadata: dict | None = None,
    writer: dict | None = None,
    predicate: list | None = None,
    provenance: dict | None = None,
) -> dict:
    """Pure builder for an XArrayEmitter config dict (no IO).

    ``writer`` settings (buffers_per_chunk, backend_config, ...) merge over the
    zarr defaults — the store path is always forced. ``predicate`` overrides the
    default subsample(1). Leaf-spec ``unit`` / ``codecs`` keys in ``view`` pass
    through verbatim to LeafView.
    """
    md = dict(metadata_base)
    md["agent_id"] = agent_id
    md["generation"] = generation
    writer_cfg = {
        "backend": "zarr",
        "store": str(store_path),
        "buffers_per_chunk": 10,
        "backend_config": {"format": 3},
    }
    if writer:
        merged = {**writer_cfg, **writer}
        if isinstance(writer.get("backend_config"), dict):
            merged["backend_config"] = {
                **writer_cfg["backend_config"], **writer["backend_config"]}
        merged["store"] = str(store_path)
        writer_cfg = merged
    return {
        "emit": {"global_time": "node"},
        "out_uri": str(store_path),
        # Drive the (pbg-emitters >=0.2.0) generic XArrayEmitter Step with the
        # v2ecoli colony partition config so the multi-cell/lineage zarr layout
        # is preserved:
        #   * ``strategy="colony"`` makes ``extract_partition`` read ``agent_id``
        #     from metadata (``generation == len(agent_id)``, with a ``parent``
        #     cell) instead of the degenerate ``"flat"`` single partition.
        #   * ``emit_root=["agents", <agent_id>]`` tells the transducer to strip
        #     the ``{"agents": {<id>: ...}}`` envelope from each ``update()``
        #     payload (the Step default ``emit_root == ()`` would instead try to
        #     emit the whole wired state and raise ``Unexpected emit path:
        #     ('agents', <id>, ...)``). Keyed to ``agent_id`` so every
        #     generation's emitter strips its own lineage key.
        "strategy": "colony",
        "emit_root": ["agents", agent_id],
        "transducer": {
            "predicate": predicate or [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": buffer_size},
        },
        "view": view,
        "writer": writer_cfg,
        "metadata": md,
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": output_metadata or {},
        # Self-describing run provenance ({composite, config, run_id}) written to
        # the zarr store's root attrs by the XArrayEmitter, so a saved simulation
        # on disk knows what produced it. Empty -> emitter writes no attr.
        "provenance": provenance or {},
        "debug": False,
    }


def _build_emitter(*, core: Any, **kwargs):
    from v2ecoli.library.xarray_emitter import XArrayEmitter  # requires [xarray]
    return XArrayEmitter(config=build_emitter_config(**kwargs), core=core)


def _with_observable_bulk(agent: dict, ids: list) -> dict:
    """Return a shallow copy of ``agent`` whose ``listeners`` gains an
    ``observable_bulk`` group holding the declared bulk molecules' counts as
    scalars (``listeners.observable_bulk.<id>``).

    This is the two-arm comparison's bulk-KPI hook: bulk counts (e.g.
    ``VIOLACEIN[c]`` titer, ``mecillinam[p]-EG10606-MONOMER[i]`` drug-target
    complex) ride under the ``listeners`` root so the SAME listener view + emit
    machinery captures them, and BOTH engines expose an identical path to grade
    on. Selection is by molecule id from the bulk record (``agent["bulk"]`` carries
    both ``id`` and ``count``); a missing id yields 0.0 so the trace stays
    continuous. The live composite state is left untouched (a fresh dict is
    returned), so this never perturbs the simulation.
    """
    bulk = agent.get("bulk")
    if bulk is None or not ids:
        return agent
    bids = bulk["id"]
    counts = bulk["count"]
    grp = {}
    for mol in ids:
        hit = np.where(bids == mol)[0]
        grp[mol] = float(counts[hit[0]]) if len(hit) else 0.0
    return {**agent, "listeners": {**(agent.get("listeners") or {}),
                                   "observable_bulk": grp}}


def run_multigen_xarray(
    composite: Any,
    *,
    store_path: str | Path,
    view: list[dict],
    metadata_base: dict,
    max_steps: int,
    max_generations: int = 3,
    chunk: int = 60,
    initial_agent_id: str = "0",
    initial_generation: int = 1,
    initial_carry_state_path: str = "",
    daughter_state_out_path: str = "",
    overwrite: bool = True,
    buffer_size: int = 600,
    single_daughters: bool = False,
    division_detector: Callable[[set[str], set[str]], tuple[bool, str | None]] | None = None,
    provenance: dict | None = None,
    observable_bulk_ids: list | None = None,
    raise_on_zero_daughters: bool = False,
) -> dict:
    """Run a v2ecoli composite past divisions, swapping XArrayEmitters per generation.

    Args:
      composite: a process-bigraph ``Composite`` with v2ecoli agent state.
      store_path: zarr store path; partitions accumulate by generation.
      view: XArray view config (list of ``{"root": tuple, "variables": dict}``).
      metadata_base: base metadata for the emitter; ``agent_id`` and
        ``generation`` are filled per-generation. Must contain ``experiment_id``,
        ``variant``, ``lineage_seed``, ``time_step``, ``max_duration``.
      max_steps: stop after this many composite ticks.
      max_generations: cap on how many generations to follow.
      chunk: how many ticks between emitter updates.
      initial_agent_id: the INNER composite's agent key to start following
        (what the built composite names its cell — "0"), NOT the phylogeny
        label. The phylogeny/partition key is DERIVED from initial_generation.
      initial_generation: 1-based ABSOLUTE label for this invocation's first
        generation. 1 (the default) is a fresh lineage and changes nothing.
        >1 RESUMES a lineage an earlier invocation began — a chain — so the
        zarr generation labels continue instead of restarting at 1.
        ``max_generations`` stays a COUNT for this invocation, so
        ``initial_generation=4, max_generations=3`` runs generations 4, 5, 6.
        ⛔ Without this a resumed stage writes a SECOND ``generation=1``
        partition and a card windowing on generation grades the wrong cells.
        ⚠ The stages of a chain must share a STORE: the emitter's ``colony``
        strategy links each partition to its parent generation, so resuming
        into a fresh store fails on the missing parent.
      initial_carry_state_path: a carry state written by a previous stage's
        ``daughter_state_out_path``. Overlaid onto the freshly built composite's
        followed agent before the run, so this stage continues the previous
        stage's cell against whatever cache THIS composite was built from —
        which is the whole point of a chain.
      daughter_state_out_path: if set, write the daughter produced at this
        invocation's FINAL division here — the hand-off to the next stage of a
        chain. That daughter is otherwise discarded (the generation cap
        deliberately does not fold it into the parent partition), so this
        captures an existing moment rather than creating one.
      overwrite: if True, delete ``store_path`` before starting.
      buffer_size: XArrayEmitter transducer buffer size, in *emit steps*, held
        in memory before each flush to the zarr store. Default 600, matching the
        viva-emitters library default — sized to flush only a handful of times
        per generation (see :py:class:`viva_emitters.xarray_emitter.transducer.
        XarrayTransducer`). A tiny buffer (the old default of 3) forces a flush
        every few emit steps, degrading latency and compression by ~2 orders of
        magnitude; it was originally chosen to dodge a ``flush(final=True)``
        assertion in an early vendored emitter, which current viva-emitters has
        since fixed (the buffer-full/partial/aligned close paths are all
        regression-tested). The ``except AssertionError`` guards around the
        closes below are now belt-and-suspenders against that fixed quirk.
      division_detector: optional ``(prev_ids, curr_ids) -> (divided?, daughter_id|None)``.
        Default: detect division when ``len(curr) > len(prev)`` and pick the
        first new agent_id sorted.
      raise_on_zero_daughters: a division that yields zero survivors (e.g. a
        custom ``division_detector`` flags division but every daughter died or
        was pruned elsewhere) stops this generation early instead of running
        the requested ``max_generations`` — a silent lineage truncation (CD2
        audit §3.6). By default this only warns (``RuntimeWarning``) and
        returns the partial result; pass ``True`` to raise a ``RuntimeError``
        instead so a batch job fails loudly rather than reporting exit 0 with
        fewer generations than requested.

    Returns: ``{"steps": int, "generations": list[int], "store": str}``.
    """
    store_path = Path(store_path)
    # ⛔⛔ AND THE DEFAULT WOULD DELETE THE PREDECESSOR. ``overwrite=True`` (the
    # default) rmtree's the store, so a resumed stage run with default arguments
    # destroys the very generations its emitter must link back to — and then
    # fails with a KeyError about a MISSING PARENT, which reads as a linkage bug
    # rather than as "your first stage was deleted a moment ago".
    # ⇒ Refuse. A chain appends; it never re-creates.
    if int(initial_generation) > 1 and overwrite:
        raise ValueError(
            f"initial_generation={int(initial_generation)} resumes an existing "
            f"lineage, but overwrite=True would DELETE {store_path} first — "
            f"including the generation-{int(initial_generation) - 1} partition "
            f"this stage's emitter links to as its parent. Pass overwrite=False "
            f"for every stage after the first.")
    # ⛔ A RESUME WITH NOTHING TO RESUME FROM silently mislabels a FRESH cell as a
    # later generation — right partition, wrong biology, no error. The batch path
    # already refuses exactly this (``LineageProcess``: "initial_generation_index
    # must be 0 when initial_carry_state_path is empty"); keep the two drivers
    # answering the same way.
    if int(initial_generation) > 1 and not initial_carry_state_path:
        raise ValueError(
            f"initial_generation={int(initial_generation)} resumes a lineage, but "
            f"no initial_carry_state_path was given — the run would start a FRESH "
            f"cell and label it generation {int(initial_generation)}. Pass the "
            f"carry state the previous stage wrote with daughter_state_out_path.")
    if int(initial_generation) < 1:
        raise ValueError(
            f"initial_generation must be >= 1 (got {initial_generation}); "
            f"generation labels are 1-based on both engines. A 0 here is the "
            f"0-based convention from v2ecoli's OTHER emitting path "
            f"(workflow/lineage.py) leaking in.")

    if overwrite and store_path.exists():
        shutil.rmtree(store_path)
    core = composite.core

    if division_detector is None:
        def division_detector(prev, curr):
            new = sorted(curr - prev)
            if len(curr) > len(prev) and new:
                return True, new[0]
            return False, None

    # Discover output_metadata (coord arrays) for any vector leaves in the
    # view by inspecting the composite's live state. Without this, vector
    # observables (e.g. listeners.monomer_counts) crash the emitter with
    # "shape mismatch: value array of shape (N,) could not be broadcast to
    # indexing result with 0 dimensions."
    #
    # At t=0 v2ecoli's listeners are empty dicts — the listener Steps
    # populate them only on first run-tick. Warm the composite with 1 tick
    # so the listener vectors materialise and we can read their length.
    # Cost: we lose tick 0 from the capture, which is fine (the emit
    # predicate's subsample interval is usually > 1 anyway).
    # ⭐ SEED THIS STAGE FROM THE PREVIOUS ONE, before the warm-up tick — the
    # warm-up materialises listeners FROM the state, so overlaying afterwards
    # would leave the discovered coord metadata describing the fresh cell.
    # Reuses the batch path's own loader and overlay so a chain and a wave carry
    # a cell forward identically (``apply_carry_state`` preserves the fresh
    # agent's derived environment substores, which is not obvious and not ours
    # to re-derive).
    if initial_carry_state_path:
        from v2ecoli.cache import load_initial_state
        from v2ecoli.workflow.lineage import apply_carry_state
        _agents_in = (composite.state or {}).get("agents") or {}
        # ⚠ `followed`/`gen` are bound further down; use the parameters here.
        _key = (initial_agent_id if initial_agent_id in _agents_in
                else next(iter(_agents_in), None))
        if _key is None:
            raise ValueError(
                f"initial_carry_state_path was given but the composite has no "
                f"agent to seed (agents={sorted(_agents_in)}).")
        apply_carry_state(_agents_in[_key], load_initial_state(initial_carry_state_path))
        print(f"[multigen_xarray] seeded agent {_key!r} from carry state "
              f"{initial_carry_state_path} (generation {int(initial_generation)})")

    try:
        composite.run(1)
    except Exception as _e:
        print(f"[xarray_run] warm-up run failed: {_e!r}")
    # Filter out view leaves the composite doesn't actually emit. XArrayEmitter
    # is strict on missing-emit-paths (raises KeyError), so cross-variant views
    # that target listeners only some composites have would otherwise sink the
    # run. We mirror SQLite's lenient behaviour: keep what's there, drop what
    # isn't.
    state_after_warmup = composite.state or {}
    # Augment a COPY of the warmup state with the declared bulk observables so the
    # view-filter (below) and coord-metadata discovery keep the synthetic
    # listeners.observable_bulk.<id> leaves — the live state is never mutated.
    if observable_bulk_ids:
        _agents = state_after_warmup.get("agents") or {}
        _cell = _agents.get(initial_agent_id) or next(iter(_agents.values()), None)
        if isinstance(_cell, dict):
            _key = initial_agent_id if initial_agent_id in _agents else next(iter(_agents))
            state_after_warmup = {**state_after_warmup, "agents": {
                **_agents, _key: _with_observable_bulk(_cell, observable_bulk_ids)}}
    filtered_view = filter_view_to_existing_leaves(state_after_warmup, view)
    if not filtered_view:
        raise RuntimeError(
            "[xarray_run] no view leaves remain after filtering against "
            "composite state — the composite emits none of the declared "
            f"observables. Declared view: {view}"
        )
    dropped = sum(len(_view_leaves([e])) for e in view) - \
              sum(len(_view_leaves([e])) for e in filtered_view)
    if dropped:
        print(f"[xarray_run] dropped {dropped} view leaf(s) absent from composite state")
    view = filtered_view
    # Harvest element-name labels from listener outputs() schemas via the
    # registry walker, then feed them to extract_output_metadata_from_state
    # so XArrayEmitter uses real IDs (e.g. monomer names) as coord arrays
    # instead of the integer-index fallback.
    from v2ecoli.library.output_metadata import output_metadata as _get_named_metadata
    named_metadata = _get_named_metadata(state_after_warmup)
    output_metadata = extract_output_metadata_from_state(
        state_after_warmup, view, named_metadata=named_metadata)
    if output_metadata:
        print(f"[xarray_run] discovered vector coord arrays for: "
              f"{list(_flatten_keys(output_metadata))}")

    from v2ecoli.steps.division import daughter_phylogeny_id

    # ``followed`` = the inner-composite key for the tracked cell. The inner
    # Division step always names its mother "0", so EVERY division yields
    # daughters "00"/"01" regardless of lineage depth — the followed key can be
    # REUSED by a daughter ("00" -> "00"/"01"). Detecting division by
    # ``followed in agents`` (which stays True on reuse) folded the last
    # generation's post-division daughter into the parent partition. Detect
    # division structurally (a NEW agent id appeared = an ``_add`` this chunk)
    # and carry a SEPARATE ``partition_agent_id`` along the true phylogeny
    # ("0" -> "00" -> "000") for the zarr generation/agent_id metadata.
    # ⛔ A RESUME SEPARATES TWO THINGS A FRESH RUN CANNOT, AND THEY ARE NOT THE
    # SAME KEY: the INNER composite's agent key (what the freshly built
    # composite names its cell — "0", for a stage-2 build off a cache exactly as
    # for stage 1) and the PHYLOGENY key, which labels the zarr partition and
    # from which the emitter's ``colony`` strategy derives the generation
    # (``generation == len(agent_id)``, with a parent). On a fresh lineage both
    # are "0" and nothing can tell them apart.
    # ⇒ Taking ONE parameter for both is how the wrapped-fork driver came to run
    # every generation as the founder (fixed in ``vivarium_ecoli_engine``). Do
    # not repeat it: the phylogeny key is DERIVED, so the two cannot disagree
    # and no caller is left holding the invariant.
    followed = initial_agent_id
    gen = int(initial_generation)
    partition_agent_id = "0" * gen
    # ``max_generations`` is a COUNT for this invocation, not an absolute stop —
    # a resumed stage asks for "3 more generations", not "run until 6".
    last_gen = gen + int(max_generations) - 1
    em = _build_emitter(
        core=core, store_path=store_path, view=view,
        metadata_base=metadata_base, generation=gen, agent_id=partition_agent_id,
        output_metadata=output_metadata, buffer_size=buffer_size,
        provenance=provenance,
    )
    # ``done`` counts ticks already advanced past tick 0 (we used 1 for the
    # coord-discovery warm-up so first chunk completes at done = 1 + chunk).
    done = 1
    gens_seen = [gen]
    _closed_at_cap = False   # set when the cap-break closes this generation itself
    prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    def _emit_followed(emitter, agents_map, key):
        if key not in agents_map:
            return
        _agent = agents_map[key]
        if observable_bulk_ids:
            _agent = _with_observable_bulk(_agent, observable_bulk_ids)
        payload = _filter_agent_state(_agent, view)
        if _EMIT_UNIQUE:
            payload = {**payload, **_extract_unique_attrs(agents_map[key])}
        # CRITICAL: emit the payload under the emitter's OWN agent_id
        # (``partition_agent_id``), NOT the inner-composite key ``key``
        # (``followed``). The XArrayEmitter transducer strips the agent prefix
        # via ``get_in(data, ("agents", partition.agent_id))`` where
        # ``partition.agent_id`` is the metadata agent_id this generation's
        # emitter was built with (== ``partition_agent_id``). When the inner
        # composite REUSES a key so ``followed`` diverges from the true
        # phylogeny ``partition_agent_id`` (happens at deeper divisions, gen 4+),
        # emitting under ``followed`` makes ``get_in`` miss → ``dict_to_paths``
        # yields the empty-tuple path → ``KeyError: 'Unexpected emit path: ()'``.
        # Relabelling the payload to ``partition_agent_id`` here is exact: the
        # emitter only uses the key to locate the cell subtree, and the
        # generation/agent_id partition metadata is already keyed to
        # ``partition_agent_id``. (``partition_agent_id`` is read by late binding
        # so each call uses the current generation's value.)
        emitter.update({
            "time": float(done),
            "global_time": float(done),
            "agents": {partition_agent_id: payload},
        })

    while done < max_steps and gen <= last_gen:
        try:
            composite.run(chunk)
        except Exception as e:
            # FAIL LOUD when a crash TRUNCATES the requested run. A swallowed
            # ``break`` here returns normally → the process exits 0 → the batch
            # job reports SUCCESS while having written only partial, often
            # unreadable data (the 4x4x5 crash that masqueraded as success).
            #
            # The ONLY tolerable raise is one that fires AFTER all requested
            # generations have already completed (``gen >= max_generations`` and
            # we are past the warm-up tick) — post-run noise, e.g. a cell that
            # can no longer divide. Everything else — a crash mid-generation,
            # before the requested generation count — is a real failure that
            # must surface as a non-zero exit, never a silent truncation.
            import traceback as _tb
            tb = _tb.format_exc()
            reached_target = gen >= last_gen and done > 1
            print(f"[multigen_xarray] composite raised at {done}s "
                  f"(gen {gen}/{last_gen}, target_reached={reached_target}):\n{tb}",
                  flush=True)
            if reached_target:
                break
            # Finalize whatever data exists (for diagnosis) then re-raise so the
            # batch job FAILS visibly instead of exiting 0 with a partial zarr.
            try:
                em.close(success=False)
            except Exception:
                pass
            raise RuntimeError(
                f"[multigen_xarray] composite CRASHED at {done}s in generation "
                f"{gen} (requested generations {initial_generation}..{last_gen}) — "
                f"see traceback above") from e
        done += chunk
        agents = (composite.state or {}).get("agents") or {}
        curr_ids = set(agents.keys())

        divided, _detected_daughter = division_detector(prev_ids, curr_ids)
        new_ids = curr_ids - prev_ids
        if not divided and new_ids:
            divided = True

        if not divided:
            # Same pbg-emitters quirk guarded at the per-gen and final closes
            # below: a per-tick emit that triggers an internal buffer-full flush
            # (buffer_size overflow) can hit zarr_writer's flush assert MID-RUN.
            # Unguarded it propagates out of the Ray task and fails the whole run
            # BEFORE reaching division — an emitter-layer flakiness orthogonal to
            # the biology. The overflowed rows are already on disk; swallow the
            # trailing-buffer assert and keep going, exactly as the closes do.
            try:
                _emit_followed(em, agents, followed)
            except AssertionError:
                print(f"[multigen_xarray] tick {done}: pbg-emitters buffer-flush "
                      "assert (buffer full mid-run); flushed rows retained, "
                      "continuing.")
            prev_ids = curr_ids
            continue

        # --- DIVISION: end this generation here; the parent partition must NOT
        # receive the post-division daughter row. ---
        survivors = sorted(curr_ids)
        inner_next = next((i for i in survivors if i.endswith("0")), None)
        if inner_next is None:
            inner_next = survivors[0] if survivors else None

        if gen >= last_gen:
            # Generation cap reached: stop (don't fold the daughter in).
            # ⭐ But that daughter is exactly the hand-off a CHAIN needs — the
            # next stage's founder, born from this stage's last cell. It is
            # dropped here only because this invocation has no generation left to
            # put it in. The daughter is selected ABOVE the cap check so the
            # carry-on and carry-out paths name the same cell by construction.
            if daughter_state_out_path and inner_next is not None:
                _daughter = ((composite.state or {}).get("agents") or {}).get(inner_next)
                if _daughter is not None:
                    import os as _os
                    from v2ecoli.cache import save_initial_state
                    # ⛔ CARRY ONLY WHAT THE READ SIDE CONSUMES. `apply_carry_state`
                    # overlays exactly bulk/unique/environment/boundary, and the
                    # batch path's `select_carry_daughter` hands it exactly those.
                    # Saving the WHOLE agent subtree instead drags along
                    # `listeners`, `process_state`, `next_update_time`… — state
                    # that was never meant to serialize. Measured: it fails on a
                    # pint Quantity whose magnitude the JSON encoder cannot coerce
                    # (`TypeError: float() argument must be ... not 'Float'`), and
                    # it fails AFTER writing megabytes, so the corpse looks like a
                    # hand-off. Project to the contract instead.
                    _carry = {k: _daughter[k] for k in
                              ("bulk", "unique", "environment", "boundary")
                              if k in _daughter}
                    # ⛔ AND WRITE IT ATOMICALLY. A partial write leaves a file that
                    # EXISTS, so an `-f` check (and the next stage) accepts a
                    # truncated carry state as a valid hand-off.
                    _tmp = f"{daughter_state_out_path}.partial"
                    save_initial_state(_carry, _tmp)
                    _os.replace(_tmp, daughter_state_out_path)
                    print(f"[multigen_xarray] wrote generation-{gen + 1} carry "
                          f"state ({inner_next!r}, keys={sorted(_carry)}) -> "
                          f"{daughter_state_out_path}")
                else:
                    print(f"[multigen_xarray] carry state NOT written: daughter "
                          f"{inner_next!r} absent from the composite.")
            # ⛔⛔ CLOSE THIS GENERATION HERE, exactly as a non-final generation
            # does a few lines below. Falling through to the post-loop close
            # instead writes NOTHING for this generation.
            # `[m]` A single-generation run that ends at its DIVISION left a store
            # holding only the group hierarchy — 8 entries, no variables — while
            # reporting exit 0 and `generations: [1]`. The same run stopped by
            # `max_steps` (no division) wrote all 8 partitions. The difference is
            # exactly this close: the per-generation close writes, the post-loop
            # close does not once the division has torn down the followed agent.
            # ⇒ A chain's first stage is PRECISELY this shape — run generation 1,
            # stop at its division, hand the daughter on — so without this the
            # next stage fails on a parent generation that was never written.
            # Terminal finalize of this (generation-cap) generation via the
            # lineage-scoped emitter — flush + mark division + consolidate. NOT
            # swallowed: a real persistence failure must surface loudly rather
            # than silently leave a skeleton store.
            em.close(success=True)
            _closed_at_cap = True
            break

        if inner_next is None:
            # A division was detected but left ZERO survivors to follow — this
            # generation ends here even though `gen < last_gen` (the requested
            # generation count is not reached). Falling through to `break`
            # silently truncates the lineage: the process still exits 0 and
            # `gens_seen` just looks shorter, with no signal that this stopped
            # early rather than by design (CD2 audit §3.6). Surface it.
            _msg = (
                f"[multigen_xarray] zero-daughter division: generation {gen} "
                f"(followed={followed!r}, lineage_seed="
                f"{metadata_base.get('lineage_seed')!r}, experiment_id="
                f"{metadata_base.get('experiment_id')!r}, variant="
                f"{metadata_base.get('variant')!r}) produced no survivors at "
                f"tick {done} — stopping at generation {gen} instead of the "
                f"requested {last_gen} ({last_gen - gen} generation(s) not run).")
            if raise_on_zero_daughters:
                try:
                    em.close(success=False)
                except Exception:
                    pass
                raise RuntimeError(_msg)
            warnings.warn(_msg, RuntimeWarning, stacklevel=2)
            break

        followed = inner_next
        # Prune the non-followed daughters so ONLY the followed lineage survives
        # in the inner composite. Kept siblings keep growing and dividing, each
        # surfacing a NEW agent id that the division detector reads as a division
        # — and once gen == max_generations the very next such signal hits the
        # `gen >= max_generations` break below, TRUNCATING the last generation to
        # a few ticks (the "v2ecoli runs one generation fewer than vEcoli"
        # artifact). With pruning, only the followed cell's own division ends a
        # generation, so the last generation runs to completion like vEcoli.
        # Mirrors run_multigen_parquet / sqlite_run's single_daughters prune.
        if single_daughters:
            import gc
            from v2ecoli.library.sqlite_run import prune_to_followed_lineage
            dropped = prune_to_followed_lineage(composite, followed)
            gc.collect()
            print(f"[multigen_xarray] gen {gen + 1} → following {followed!r} at "
                  f"tick {done} (single_daughters: dropped {dropped} sibling(s))")
        partition_agent_id = daughter_phylogeny_id(partition_agent_id)[0]
        gen += 1
        gens_seen.append(gen)
        # Finalize this generation (flush trailing buffer + mark the division
        # event + consolidate) and open the daughter's partition in the SAME
        # store, all through the one lineage-scoped emitter. This replaces the
        # old close()+rebuild flow whose per-generation close() was wrapped in
        # `except AssertionError: pass`; when a (sub-buffer_size) generation
        # failed to persist, that swallow left the zarr store as bare group
        # skeletons and crashed the next generation's _check_group.
        # advance_generation() guarantees the finished generation is on disk
        # before the daughter's partition opens.
        em.advance_generation(agent_id=partition_agent_id, success=True)
        # Emit the daughter's birth row into the NEW generation's partition.
        _emit_followed(em, (composite.state or {}).get("agents") or {}, followed)
        prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    if not _closed_at_cap:
        # Terminal finalize of the final generation (flush trailing buffer +
        # mark division + consolidate) via the one lineage-scoped emitter. NOT
        # swallowed: a real persistence failure must surface as a loud failure
        # (this driver's fail-loud philosophy) rather than silently leave a
        # skeleton store — the exact silent-success mode this fix removes.
        em.close(success=True)

    return {"steps": done, "generations": gens_seen, "store": str(store_path)}
