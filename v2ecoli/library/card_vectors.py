"""Cell-level extraction of vector observables for the report card.

Transcriptome / proteome / exchange-flux axes are *vectors* (one value per
gene / protein / exchange reaction). Following the cell-level discipline used
everywhere in the card: time-average each vector **within** a cell (one vector
per cell), then take the ensemble mean across the N cells that pass burn-in.
The ensemble-mean vector is what the card grades (R^2 vs the pinned reference)
and plots (scatter).

This reads the sweep parquet directly (the array columns aren't carried in the
scalar per-cell records). It is heavier than the scalar analysis (~minute over
a 4x8 ensemble), so it runs at report-render time, not in the workflow step.

``sweep_dir`` may be a local path or an ``s3://`` URI (see
:mod:`v2ecoli.library.sweep_io`). The S3 form lets an equivalence reference be
pinned — and a measured card rendered — against a sweep that never lands on the
local disk, which is what makes a multi-condition card affordable: the parquet
stays in object storage and DuckDB's column projection reads only the three
array columns below.
"""
from __future__ import annotations

# Bump whenever a node's CONTENT changes — the aggregation semantics (the column
# set, the ragged-row rule, the cell-first order) *or* the set of keys a node
# carries. It is part of the sim_vector_cache key, so a bump invalidates every
# cached vector rather than silently serving one built by older code — the vector
# is a function of this code as much as of the run.
#
# v1 -> v2: every group now carries ``per_cell`` (was: fluxes only) and every node
# declares its ``units``. Neither changes a single number, which is exactly why
# the bump is needed rather than optional: a v1 envelope and a v2 envelope are
# numerically identical, so nothing else would tell a consumer asking for
# ``per_cell`` that this cache simply predates it. Without the bump, "the run
# recorded no per-cell samples" and "this file was written by older code" are
# indistinguishable — and the first is a fact about the run, the second a fact
# about the tooling.
#
# v2 -> v3: a node's CONTENT can change, because a group whose observable column
# is absent from the sweep is now OMITTED rather than the whole extraction
# failing. An envelope written by v2 for a sweep that v2 could not read does not
# exist -- but one written by v2 for a sweep whose column set has since changed
# would differ, and the cache key is the only thing that distinguishes them.
#
# v3 -> v4: cells that are not complete cell cycles are excluded from the
# ensemble, and a node carries `n_cells_excluded_partial` plus
# `partial_cell_detection`. Content AND keys change, so the bump is required
# twice over. A v3 envelope was written WITHOUT the exclusion and its numbers
# are plausible, so nothing but the key distinguishes them.
EXTRACTOR_VERSION = 4

#: A cell whose row count sits below the split is not a complete cell cycle.
#:
#: ⛔ **THE SPLIT IS FOUND, NOT ASSUMED — and this function REFUSES rather than
#: guessing.** A fixed fraction of the longest cell was tried first and was
#: withdrawn after independent review: on a coarse-emit sweep already in this
#: tree (per-cell rows ``25, 29, 26, 4``) a genuinely partial cell is 14% of the
#: longest, so a 10% floor admitted it **while the node affirmatively reported
#: that nothing had been dropped.** Whether a stub falls under any fixed fraction
#: is a property of the producing run's emit cadence — the very thing this module
#: must stop assuming.
#:
#: ⇒ Instead: sort the per-cell row counts and look for the largest RATIO gap. A
#: sweep whose cells separate into "ran a cycle" and "was born and abandoned"
#: has a gap of orders of magnitude; one that does not, does not. If no gap
#: reaches this ratio the split is **UNDECIDABLE**, and the extraction says so
#: (``partial_cell_detection: "ambiguous"``) and excludes NOTHING rather than
#: excluding arbitrarily.
#:
#: ⚠ **"Ambiguous" is not "clean" — a consumer must not read n_cells as complete
#: cycles in that case.** It is the honest state for a truncated run, where every
#: cell is partial to a different degree and no split exists to find.
_MIN_SEPARATION_RATIO = 10.0

#: Rows pulled from DuckDB per batch. Bounds peak memory during extraction: the
#: rows themselves are released after each batch, so what persists is the
#: per-cell running sums, not the run. Large enough that per-batch overhead is
#: irrelevant, small enough that a batch of wide omics rows is megabytes.
_FETCH_BATCH_ROWS = 2000

#: observable column -> ``(group, name, units)``.
#:
#: **Units are declared HERE, beside the parquet column they describe, and
#: nowhere else.** They are a property of what the listener emits — monomer
#: counts are copies/cell whatever anybody does with them downstream — so a
#: consumer that names its own units is stating a belief about this column
#: rather than reading a fact from it. ``scripts/bake_model_omics.py`` and the
#: committed fixtures already declare exactly these strings; this is the
#: declaration they should eventually read rather than a fourth copy.
_VECTOR_COLS = {
    "listeners__rna_counts__mRNA_cistron_counts": ("omics", "transcriptome", "counts/cell"),
    "listeners__monomer_counts": ("omics", "proteome", "copies/cell"),
    "listeners__fba_results__external_exchange_fluxes": ("fluxes", "exchange", "mmol/gDCW/h"),
}


#: ``(group, name) -> units``. The lookup a resolver uses when it holds a card
#: path rather than a parquet column — see ``operands.run_operand``, which stamps
#: units from HERE rather than from the cached node, so that a hand-built or
#: pre-v2 envelope cannot produce an operand that silently declares nothing.
VECTOR_UNITS = {(group, name): units for group, name, units in _VECTOR_COLS.values()}


def _complete_cells(cell_order: list, per_cell_rows: dict) -> tuple[list, int, str]:
    """Split cells into complete cycles and partials, or decline to.

    Returns ``(included, n_excluded, detection)`` where ``detection`` is
    ``"clean"`` when a real separation was found and ``"ambiguous"`` when none
    was — in which case NOTHING is excluded and ``included`` is every cell.

    ⛔ **Declining is a real outcome, not a fallback.** The alternative — pick a
    threshold anyway — is what the withdrawn version did, and it silently
    admitted a partial cell on a real sweep while reporting that it had not.

    ⚠ Deliberately makes no claim about WHY a cell is short. A birth stub and a
    run killed mid-cycle are both "not a complete cycle"; distinguishing them
    needs the per-cell ``divided`` flag, which is not an emitted column.
    """
    if len(cell_order) < 2:
        # One cell cannot separate into two populations. Not ambiguous in the
        # interesting sense, but nothing is excludable either.
        return list(cell_order), 0, "clean" if cell_order else "ambiguous"
    counts = sorted(per_cell_rows[c] for c in cell_order)
    best_ratio, split_at = 1.0, None
    for lo, hi in zip(counts, counts[1:]):
        ratio = (hi / lo) if lo else float("inf")
        if ratio > best_ratio:
            best_ratio, split_at = ratio, hi
    if split_at is None or best_ratio < _MIN_SEPARATION_RATIO:
        return list(cell_order), 0, "ambiguous"
    included = [c for c in cell_order if per_cell_rows[c] >= split_at]
    return included, len(cell_order) - len(included), "clean"

def extract_vectors(sweep_dir: str, generation_lower_bound: int = 0) -> dict:
    """Return ``{group: {name: {...}}}`` of cell-first aggregated vectors
    (time-mean within cell, then mean across cells).

    Each node carries the ensemble-mean ``vector``, ``n_cells``,
    ``n_cells_excluded_partial``, its ``units``, and ``per_cell`` — the
    n_cells x n_features matrix of per-cell time-mean vectors whose column mean
    IS ``vector``.

    ⭐ **Which observable columns exist is CHECKED here rather than assumed.**
    A sweep is extracted for the columns it has; a group whose column is absent
    is OMITTED, never zero-filled. Different metabolism processes write
    different exchange leaves, and that was previously fatal to the whole
    extraction rather than to the one group.

    ⛔ **KNOWN GAP, deliberately not addressed here — cells are NOT filtered.**
    Depending on how the producing runner emits daughters at division, grouping
    by ``(lineage_seed, generation, agent_id)`` can yield short birth stubs, and
    a run that stops mid-cycle leaves a truncated final generation. Both are
    averaged in as if they were complete cells, which biases every ensemble
    statistic toward the newborn state. A row-count heuristic was tried and
    withdrawn: whether a stub falls under any given fraction is a function of the
    producing run's emit cadence, so the heuristic silently did nothing on
    coarse-emit sweeps while asserting that it had. The real signal is a
    per-cell ``divided`` flag (``workflow/analysis_runner.py``), which is not
    currently an emitted parquet column. ⇒ **Callers must not treat ``n_cells``
    as a count of complete cell cycles.**

    ⚠ **A second, PRE-EXISTING inconsistency, stated because it is easy to
    assume this function prevents it and it does not:** the ragged-row rule
    below drops cells per COLUMN, so a cell emitting ``[]`` for one observable
    is absent from that group and present in the others. Different groups can
    therefore describe different cell sets, and their ``per_cell`` row indices
    do not align. Nothing in the output says so.

    Ragged/empty array rows are dropped per column: ``external_exchange_fluxes``
    emits a ``[]`` default on some timesteps, so only rows whose array matches
    the column's modal length are averaged."""
    import numpy as np

    from v2ecoli.library.sweep_io import connect_for, history_files

    files = history_files(sweep_dir)
    if not files:
        return {}
    con = connect_for(sweep_dir)
    # ⛔ ``union_by_name`` IS LOAD-BEARING, NOT TIDINESS. Without it DuckDB binds
    # the glob against the FIRST file's schema, and the two failure modes are
    # asymmetric: if the first file LACKS a column the later ones have, the group
    # is silently omitted with no error; if it HAS one the later ones lack, the
    # read raises a schema-mismatch. The silent branch is reachable whenever a
    # directory pools runs with different column sets, because ``history_files``
    # globs recursively.
    rel = ("read_parquet(" + repr(files)
           + ", hive_partitioning=true, union_by_name=true)")

    # ⛔ SELECT ONLY THE COLUMNS THIS SWEEP ACTUALLY HAS, and omit the rest from
    # the result rather than failing the whole extraction.
    #
    # **Not every metabolism writes every leaf.** ``metabolism.py`` writes
    # ``listeners.fba_results.external_exchange_fluxes``; ``metabolism_redux``
    # does not — it writes ``estimated_exchange_dmdt`` instead. The same
    # divergence is already documented, and already guarded with an explanatory
    # refusal, in ``library/vivarium_ecoli_engine.py`` for the ``gdcw``
    # exchange-flux basis. This function had no equivalent, so a sweep produced
    # by a metabolism that does not write the leaf died inside DuckDB with a
    # binder error naming a column — loud, but it reports the symptom rather
    # than the cause, and it takes the omics groups down with it even though
    # those columns are present and perfectly extractable.
    #
    # ⭐ **Absent means ABSENT — the group is omitted, never emitted as zeros.**
    # A zero-filled exchange vector is indistinguishable from a cell exchanging
    # nothing, which is the failure mode the engine's own guard exists to
    # refuse. A consumer that requires a group it cannot find must say so
    # itself; silently handing it zeros moves an error into a result.
    available = {c.lower() for c in con.sql(f"SELECT * FROM {rel} LIMIT 0").columns}
    present = [
        (col, meta) for col, meta in _VECTOR_COLS.items() if col.lower() in available
    ]
    if not present:
        raise ValueError(
            f"no observable columns found in {sweep_dir!r}. Looked for: "
            + ", ".join(_VECTOR_COLS)
            + ". A sweep with none of them is not gradeable — check that the "
            "run emitted its listeners, and note that the metabolism in use "
            "determines which exchange leaf (if any) is written."
        )
    cols = ", ".join(col for col, _ in present)
    result = con.sql(
        f"SELECT lineage_seed, generation, agent_id, {cols} FROM {rel} "
        f"WHERE generation >= {int(generation_lower_bound)}"
    )

    # ⛔ STREAMED IN BATCHES, AND THIS IS NOT A MICRO-OPTIMISATION.
    #
    # This read used to be a single ``.fetchall()``, which materialises every
    # timestep x every cell x every vector column into Python objects at once,
    # and was then grouped into a SECOND full copy keyed by cell. Peak footprint
    # was therefore ~2x the sweep's vector columns, in Python objects.
    #
    # ⚠ Measured 2026-08-21 on an 8x16 basal sweep (50 GB of history parquet) at
    # ``generation_lower_bound=0``: 52 GB resident, 63 GB peak, 42.5 GB of 44 GB
    # swap consumed, the process pinned in uninterruptible I/O wait, and killed
    # rather than failing on its own -- taking the machine with it.
    #
    # ⚠ NOT "it could never have completed": the surviving v1 cache envelope for
    # this same sweep records ``extract_seconds: 680.76`` at ``gen_lb=3`` over
    # the same 104 cells, extracted 2026-07-30 by the ``fetchall`` code. The old
    # path completed this sweep at gen_lb=3 and blew past the machine at
    # gen_lb=0, +21% rows. The defect is that peak scales with the RUN, so the
    # margin is a property of the hardware and not of the code.
    #
    # The fix is to never hold the rows. We accumulate a RUNNING SUM per
    # (cell, column, array-length) and divide at the end, so peak memory is a
    # function of the ENSEMBLE (cells x features), not of the RUN (timesteps x
    # cells x features). For that sweep that is ~900k floats instead of billions.
    #
    # Bucketing by array length rather than resolving the modal length first is
    # what keeps this a SINGLE pass: the modal length is not known until every
    # row has been seen, and a second pass over 50 GB to learn it would trade
    # the memory problem for an I/O one. Distinct lengths are few (a column is
    # ragged only where a listener emits its ``[]`` default), so the buckets
    # cost nothing.
    per_cell_sum: dict[tuple, "np.ndarray"] = {}
    per_cell_n: dict[tuple, int] = {}
    cell_order: list[tuple] = []          # first-appearance order, which IS the
    seen_cells: set[tuple] = set()        # row order of the per_cell matrix
    col_len = [0] * len(present)          # modal (max) feature length per column
    per_cell_rows: dict[tuple, int] = {}  # rows per cell, for the split below

    while True:
        batch = result.fetchmany(_FETCH_BATCH_ROWS)
        if not batch:
            break
        for r in batch:
            cell = (r[0], r[1], r[2])
            if cell not in seen_cells:
                seen_cells.add(cell)
                cell_order.append(cell)
            # Counted over ROWS, so membership is a property of the cell rather
            # than of whichever observable happens to be widest.
            per_cell_rows[cell] = per_cell_rows.get(cell, 0) + 1
            for i, val in enumerate(r[3:]):
                if val is None:
                    continue
                length = len(val)
                if length > col_len[i]:
                    col_len[i] = length
                key = (cell, i, length)
                acc = per_cell_sum.get(key)
                if acc is None:
                    per_cell_sum[key] = np.array(val, dtype=float)  # copy: acc is mutated in place
                    per_cell_n[key] = 1
                else:
                    acc += val
                    per_cell_n[key] += 1

    # ⭐ MEMBERSHIP IS DECIDED ONCE AND APPLIED TO EVERY COLUMN, so the groups
    # cannot describe different cell sets.
    # ⚠ The ragged-row rule below can still drop a cell from ONE column (a
    # listener emitting its `[]` default for a whole cell). That is pre-existing
    # and NOT fixed here — `n_cells` is therefore per-node, and two nodes may
    # still differ. Stated rather than implied.
    included, n_excluded, detection = _complete_cells(cell_order, per_cell_rows)

    out: dict[str, dict] = {}
    for i, (col, (group, name, units)) in enumerate(present):
        n = col_len[i]
        # per-cell time-mean vector over rows whose array is full-length (drops
        # the [] empties); skip cells with no full-length rows.
        cell_means = []
        for c in included:
            key = (c, i, n)
            count = per_cell_n.get(key)
            if count:
                cell_means.append(per_cell_sum[key] / count)
        per_cell_means = np.array(cell_means)
        ensemble_mean = per_cell_means.mean(axis=0)
        node = {
            "vector": [float(x) for x in ensemble_mean],
            "n_cells": len(cell_means),
            "n_cells_excluded_partial": n_excluded,
            # "clean" -> a real split was found. "ambiguous" -> none was, nothing
            # was excluded, and n_cells is NOT a count of complete cycles.
            "partial_cell_detection": detection,
            "units": units,
            "per_cell": [[float(x) for x in row] for row in per_cell_means],
        }
        out.setdefault(group, {})[name] = node
    return out
