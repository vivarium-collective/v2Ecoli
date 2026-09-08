"""Multi-generation Parquet run helper for v2ecoli composites.

Sibling to :mod:`v2ecoli.library.sqlite_run`. Same external-emitter driving
model (own the emitter, run the composite in chunks, push the followed
agent's state per-chunk), but the persistence target is a hive-partitioned
parquet directory instead of a sqlite db.

Storage: ~3-5× smaller than the equivalent sqlite run for v2ecoli-shaped
sims — Parquet is column-oriented and dictionary-encodes repeated listener
fields. Trade-off: the dashboard's Simulations-DB tab cannot read parquet
yet (vivarium-workbench follow-up); for now use this runner when downstream
analysis is DuckDB/Polars-based, ``sqlite_run`` when dashboard inspection
is required.

Lineage layout mirrors vEcoli's Parquet convention::

    <out_dir>/<experiment_id>/history/
      experiment_id=<quoted>/variant=<v>/lineage_seed=<s>/generation=<g>/agent_id=<id>/N.pq

Every division advances the partitioning's ``generation`` and ``agent_id``
hive levels — so each daughter lineage gets its own subtree and can be
queried independently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from v2ecoli.library.sqlite_run import (
    _normalize_emit_paths,
    _normalize_root_paths,
    _filter_agent_state,
    _filter_root_state,
    _build_emit_schema,
    _merge_into,
    prune_to_followed_lineage,
)
from v2ecoli.library.output_metadata import output_metadata as _get_output_metadata

# Env-gated unique-store emit (V2ECOLI_EMIT_UNIQUE=1): the chromosome-state
# renderer (scripts/render_chromosome_gif.py) needs per-molecule genomic
# coordinates that live in the agent's structured ``unique`` numpy arrays —
# which the plain-dict emit-path filter cannot reach. When enabled we extract
# the named attribute arrays of the ACTIVE entries (``_entryState`` mask) and
# emit them as TOP-LEVEL nested keys so ParquetEmitter flattens them to the
# columns the renderer reads: ``active_RNAP__coordinates`` etc. (no ``unique__``
# prefix). ``chromosome_domain__child_domains`` is the (n_domain, 2) parent->child
# domain tree; it is FLATTENED row-major to a 1-D ``list<int>`` of length
# ``2*n_domain`` (aligned to ``chromosome_domain__domain_index``) so the renderer
# can place daughter-strand RNAPs on the replication bubbles, not just the rim.
_EMIT_UNIQUE = os.environ.get("V2ECOLI_EMIT_UNIQUE", "") not in ("", "0", "false", "False")
_UNIQUE_EMIT_SPEC = {
    "active_RNAP": ["coordinates", "domain_index"],
    "active_replisome": ["coordinates", "domain_index"],
    "full_chromosome": ["unique_index", "domain_index"],
    "chromosome_domain": ["domain_index", "child_domains"],
}


def _unique_emit_schema() -> dict:
    """Emit-schema fragment declaring the unique coordinate columns."""
    return {
        mol: {attr: "any" for attr in attrs}
        for mol, attrs in _UNIQUE_EMIT_SPEC.items()
    }


def _flatten_attr(col):
    """Flatten a unique-store attribute column to a 1-D python list.

    1-D attributes (coordinates, domain_index) pass through. 2-D attributes
    (child_domains is (n_active, 2)) are flattened row-major to a flat
    list<int> of length 2*n_active so they serialize as a plain parquet list
    column, aligned to domain_index.
    """
    arr = np.asarray(col)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.tolist()


def _extract_unique_attrs(agent_state: dict) -> dict:
    """Pull active-entry attribute arrays out of an agent's ``unique`` store.

    Returns ``{mol: {attr: [values...]}}`` for the active entries of each
    configured unique molecule, ready to merge into the emit payload (flattens
    to ``<mol>__<attr>`` parquet columns).
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



def _declared_emit_roots(generator: Any = None) -> list[str]:
    """The store roots a generator's ``emitters=`` declaration names.

    ``generator`` is whatever ``viva_superpowers.composite_generator
    .emitter_defaults`` accepts (a decorated builder fn, a registry entry, a
    static-spec dict) or a registered spec id. ``None`` falls back to
    ``_helpers._declared_parquet_roots()`` -- the declaration of the generator
    currently being built if one is in scope, else ecoli_baseline's set
    (``global_time`` / ``bulk`` / ``listeners``) -- the same fallback #741 gave
    a lineage with no emit_paths.
    """
    from v2ecoli.composites._helpers import _declared_parquet_roots

    if generator is None:
        return list(_declared_parquet_roots())
    from viva_superpowers.composite_generator import _REGISTRY, emitter_defaults

    source = generator
    if isinstance(generator, str):
        # A registry id, else an architecture name (the form build_composite
        # takes; aliases share one function, so any match reads the same decl).
        source = _REGISTRY.get(generator) or next(
            (e for e in _REGISTRY.values() if e.name == generator), None)
    if source is None:
        raise ValueError(
            f"run_multigen_parquet: no registered composite {generator!r} to "
            "read an emitters= declaration from")
    roots: list[str] = []
    for decl in emitter_defaults(source):
        for p in decl.get("paths") or []:
            parts = [seg for seg in str(p).replace(".", "/").split("/") if seg]
            if parts and parts[0] not in roots:
                roots.append(parts[0])
    if not roots:
        raise ValueError(
            f"run_multigen_parquet: {generator!r} declares no emitter paths "
            "(no @composite_generator(emitters=[...]) entry), so an empty "
            "emit_paths cannot mean 'the declared set'. Pass explicit "
            "emit_paths, or declare the composite's sink on its generator.")
    return roots


def declared_emit_set(
    composite: Any, generator: Any = None,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """``(agent_leaves, root_leaves)`` for an EMPTY ``emit_paths``: the
    composite's DECLARED emit set instead of an explicit allow-list.

    The roots come from the generator's ``emitters=`` declaration
    (:func:`_declared_emit_roots`). Each is resolved against the BUILT state
    the same way the composite's own in-document sink resolves it
    (``_helpers.set_enclosing_emitter_decl`` / ``_parquet_emit_set``): a root
    that exists inside the followed agent is agent-relative (``bulk``,
    ``listeners``, ``boundary`` -> captured whole, so the hive carries
    ``bulk__id`` / ``bulk__count`` / ``listeners__fba_results__*`` / ...);
    a root that exists only at the document level (``reactor`` /
    ``population`` / ``lineage`` for reactor_bird_coupled) is a root path,
    merged into the same row. ``global_time`` is the runner's own column and
    is skipped. A declared root present in neither place at build time is
    kept agent-relative (the in-document sink's default frame) so a store a
    Step materialises on its first tick is still captured.

    This is the whole-cell emit volume (one cell's bulk + listeners per tick),
    which is tractable for a SINGLE-LINEAGE run (``single_daughters=True``) --
    the case the coupled dispatch opts into. A many-agent population run must
    keep passing its explicit reduced ``emit_paths``; that allow-list is what
    keeps bulk x agents x generations off the emit path (#754/#776).
    """
    agents = (getattr(composite, "state", None) or {}).get("agents") or {}
    agent_state = next(iter(agents.values()), {}) if agents else {}
    state = getattr(composite, "state", None) or {}
    agent_leaves: list[tuple[str, ...]] = []
    root_leaves: list[tuple[str, ...]] = []
    for root in _declared_emit_roots(generator):
        if root == "global_time":
            continue
        if isinstance(agent_state, dict) and root in agent_state:
            agent_leaves.append((root,))
        elif isinstance(state, dict) and root in state:
            root_leaves.append((root,))
        else:
            agent_leaves.append((root,))
    return agent_leaves, root_leaves


def run_multigen_parquet(
    composite: Any,
    *,
    experiment_id: str,
    out_dir: str | Path,
    emit_paths: list[str],
    max_steps: int,
    max_generations: int = 1,
    chunk: int = 100,
    initial_agent_id: str = "0",
    initial_variant: int = 0,
    initial_lineage_seed: int = 0,
    initial_generation: int = 1,
    division_detector: Callable[[set[str], set[str]], tuple[bool, str | None]] | None = None,
    core: Any = None,
    single_daughters: bool = False,
    batch_size: int = 400,
    threaded: bool = True,
    study_slug: str | None = None,
    investigation_slug: str | None = None,
    extra_root_paths: list[str] | None = None,
    declared_generator: Any = None,
) -> dict:
    """Run a v2ecoli composite across divisions, externally-driven ParquetEmitter.

    Args mostly mirror :func:`v2ecoli.library.sqlite_run.run_multigen_sqlite`.
    Differences:

      * ``emit_paths``: a NON-EMPTY list is an explicit allow-list of
        agent-relative store paths (unchanged). An EMPTY list means "the
        composite's DECLARED emit set" -- the roots its generator names in
        ``@composite_generator(emitters=[...])``, resolved by
        :func:`declared_emit_set` (agent stores captured whole, document-level
        stores merged into the same row). ``declared_generator`` names the
        generator to read (builder fn / registry entry / spec id); ``None``
        falls back to the generator in build scope, else ecoli_baseline's set.
        The declared set is whole-cell volume: opt in for single-lineage runs,
        keep the explicit reduced list for many-agent population runs.
      * ``out_dir`` (vs ``db_file``): root directory for the parquet hive.
      * ``experiment_id``: top-level partition key. Quoted via ``parse.quote_plus``
        internally so it survives any path-unsafe characters.
      * ``initial_variant`` / ``initial_lineage_seed`` / ``initial_generation``:
        hive partition values seeded on the first generation. Each subsequent
        generation increments ``generation`` and uses the daughter agent_id.
      * ``batch_size`` / ``threaded``: forwarded to ParquetEmitter.

    Returns: ``{"steps": int, "generations": list[int], "out_dir": str}``.

    Per-generation ParquetEmitter lifecycle: each new generation rotates the
    emitter (close + re-create with new partition keys). Without this each
    generation would overwrite the prior one's history dir on next config
    write — the per-generation rotation is what lets ``read_parquet(out_dir/
    experiment_id/history/**/*.pq)`` pick up all generations in one read.
    """
    # Imported directly from pbg-emitters (the upstream library);
    # ``v2ecoli.library.parquet_emitter`` is just a re-export shim.
    from viva_emitters import ParquetEmitter

    if division_detector is None:
        def division_detector(prev: set[str], curr: set[str]) -> tuple[bool, str | None]:
            new = sorted(curr - prev)
            if len(curr) > len(prev) and new:
                return True, new[0]
            return False, None

    if emit_paths:
        leaves = _normalize_emit_paths(emit_paths)
        root_leaves = _normalize_root_paths(extra_root_paths or [])
    else:
        # Empty allow-list = the generator's declared set (see the docstring).
        # An extra root path under a declared document root is subsumed by
        # that root (captured whole) -- keep only the ones outside it.
        leaves, root_leaves = declared_emit_set(composite, declared_generator)
        declared_root_names = {r[0] for r in root_leaves}
        root_leaves = root_leaves + [
            rl for rl in _normalize_root_paths(extra_root_paths or [])
            if rl[0] not in declared_root_names
        ]
        print("[multigen_parquet] emit_paths empty -> the composite's DECLARED "
              f"emit set: agent roots {['/'.join(l) for l in leaves]}, "
              f"document roots {['/'.join(r) for r in root_leaves]}")
    emit_schema = _build_emit_schema(leaves)
    if root_leaves:
        _merge_into(emit_schema, _build_emit_schema(root_leaves))
    if _EMIT_UNIQUE:
        _merge_into(emit_schema, _unique_emit_schema())
        print("[multigen_parquet] V2ECOLI_EMIT_UNIQUE=1 -> emitting unique "
              "coordinate columns: "
              + ", ".join(f"{m}__{a}" for m, aa in _UNIQUE_EMIT_SPEC.items() for a in aa))
    out_dir = str(Path(out_dir).resolve())

    # Harvest element-name labels from listener outputs() schemas. Labels are
    # registered in the type core during initialize(), so they are present in
    # composite.state immediately — no warmup tick required (unlike the xarray
    # path which also needs vector shapes from live state values).
    # Un-annotated composites return {} → backward-compat: no output_metadata
    # key is added and existing runs are unaffected.
    _named_metadata: dict = _get_output_metadata(composite.state or {})
    if _named_metadata:
        print(f"[multigen_parquet] discovered output_metadata labels for: "
              f"{list(_named_metadata.keys())}")

    def _make_emitter(agent_id: str, generation: int) -> ParquetEmitter:
        metadata: dict[str, Any] = {
            "experiment_id": experiment_id,
            "variant": initial_variant,
            "lineage_seed": initial_lineage_seed,
            "generation": generation,
            "agent_id": agent_id,
        }
        if study_slug:
            metadata["study_slug"] = study_slug
        if investigation_slug:
            metadata["investigation_slug"] = investigation_slug
        # Merge element-name labels into config metadata so ParquetEmitter
        # persists them via flatten_dict as output_metadata__<path> columns
        # in the configuration parquet. Recoverable via field_metadata().
        if _named_metadata:
            metadata["output_metadata"] = _named_metadata

        return ParquetEmitter(
            config={
                "emit": emit_schema,
                "out_dir": out_dir,
                "batch_size": batch_size,
                "threaded": threaded,
                "flatten_separator": "__",
                "partitioning_keys": [
                    "experiment_id", "variant", "lineage_seed",
                    "generation", "agent_id",
                ],
                "metadata": metadata,
            },
            core=core or composite.core,
        )

    import gc
    from v2ecoli.steps.division import daughter_phylogeny_id

    max_steps = int(max_steps)
    # ``followed`` = the key the inner composite uses for the cell we track.
    # The inner Division step always names its mother "0", so EVERY division
    # produces daughters "00"/"01" regardless of lineage depth — the followed
    # key can be REUSED by a daughter ("00" -> "00"/"01"). So we must NOT detect
    # division by ``followed in agents`` (it stays True when a daughter reuses
    # the id) — that bug folded the last generation's post-division daughter
    # into the parent generation's partition. Instead detect division
    # structurally (a new agent id appeared = an ``_add`` = a division this
    # chunk) and carry a SEPARATE ``partition_agent_id`` that increments along
    # the true phylogeny ("0" -> "00" -> "000") for the hive layout.
    followed = initial_agent_id
    partition_agent_id = initial_agent_id
    gen = int(initial_generation)
    done = 0
    gens_seen = [gen]
    prev_ids = set(((composite.state or {}).get("agents") or {}).keys())
    # Seed the represented-population doubling count (#225 item #1); see
    # run_multigen_sqlite.set_lineage_doublings. No-op without a lineage store.
    from v2ecoli.library.sqlite_run import set_lineage_doublings
    set_lineage_doublings(composite, gen - 1)

    def _emit(agents_map: dict, agent_key: str, emitter) -> None:
        if agent_key not in agents_map:
            return
        payload = _filter_agent_state(agents_map[agent_key], leaves)
        # ParquetEmitter takes the flat tick state directly — no
        # `agents/<id>/` wrapper (which is sqlite-only convention).
        update_state: dict = {"global_time": float(done), **payload}
        if _EMIT_UNIQUE:
            _merge_into(update_state, _extract_unique_attrs(agents_map[agent_key]))
        if root_leaves:
            _merge_into(
                update_state,
                _filter_root_state(composite.state or {}, root_leaves),
            )
        try:
            emitter.update(update_state)
        except Exception as e:
            print(f"[multigen_parquet] emit failed at tick {done}: "
                  f"{type(e).__name__}: {str(e)[:120]}")

    em = _make_emitter(partition_agent_id, gen)

    try:
        while done < max_steps:
            n = min(chunk, max_steps - done)
            try:
                composite.run(n)
            except Exception as e:
                print(f"[multigen_parquet] composite stopped at tick {done}: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                break
            done += n
            agents = (composite.state or {}).get("agents") or {}
            curr_ids = set(agents.keys())

            # A division this chunk = a new agent id surfaced (an ``_add``).
            # This is robust to the inner step reusing the followed id for a
            # daughter (the case that previously slipped past ``followed in
            # agents`` and ran the last generation past its own division).
            divided, _detected_daughter = division_detector(prev_ids, curr_ids)
            new_ids = curr_ids - prev_ids
            if not divided and new_ids:
                divided = True

            if not divided:
                # No division: emit the followed cell into this generation's
                # partition.
                _emit(agents, followed, em)
                prev_ids = curr_ids
                if single_daughters and n >= 50:
                    gc.collect()
                continue

            # --- DIVISION: end this generation here. The parent's partition
            # must NOT receive the post-division daughter row. ---
            if gen >= max_generations:
                # Generation cap reached: stop following. The daughter is
                # intentionally dropped (its generation wasn't requested) rather
                # than folded into the parent partition.
                break

            # Choose the inner survivor to follow: prefer a daughter whose id
            # ends in "0" (vEcoli single-daughter convention), else any agent.
            survivors = sorted(curr_ids)
            inner_next = next((i for i in survivors if i.endswith("0")), None)
            if inner_next is None:
                inner_next = survivors[0] if survivors else None
            if inner_next is None:
                break

            # Rotate the emitter: close the parent generation's hive, open a
            # fresh one keyed on the next generation + the true phylogeny id
            # ("0" -> "00" -> "000"), independent of the inner key's reuse.
            em.close(success=True)
            followed = inner_next
            partition_agent_id = daughter_phylogeny_id(partition_agent_id)[0]
            gen += 1
            gens_seen.append(gen)
            # Grow the represented population 2x for the new generation (#225
            # item #1); aggregator applies 2^(gen-1) in representative_doubling.
            set_lineage_doublings(composite, gen - 1)
            em = _make_emitter(partition_agent_id, gen)

            if single_daughters:
                dropped = prune_to_followed_lineage(composite, followed)
                gc.collect()
                print(f"[multigen_parquet] gen {gen} → following inner agent "
                      f"{followed!r} (partition agent_id={partition_agent_id!r}) "
                      f"at tick {done} (single_daughters: dropped {dropped} "
                      f"sibling agent(s) + ran gc)")
            else:
                print(f"[multigen_parquet] gen {gen} → following inner agent "
                      f"{followed!r} (partition agent_id={partition_agent_id!r}) "
                      f"at tick {done}")

            # Emit the daughter's birth row into the NEW generation's partition.
            _emit((composite.state or {}).get("agents") or {}, followed, em)
            prev_ids = set(((composite.state or {}).get("agents") or {}).keys())
            if single_daughters and n >= 50:
                gc.collect()
    finally:
        em.close(success=True)

    return {"steps": done, "generations": gens_seen, "out_dir": out_dir}
