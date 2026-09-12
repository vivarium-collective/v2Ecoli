"""The per-cell table — the contract two engines have to meet to be compared.

A design screen's grading layer consumes ONE table: a long-format row per cell,
with the screen's observables as columns. Everything above that table — KPIs,
per-variant summaries, contrasts against a named reference, the ranking, the
panel — is arithmetic on those rows and knows nothing about how they were
produced. Everything below it is an engine's own stored output and id
resolution, which differs per engine and cannot be shared.

⇒ That makes this table the seam. Write it down and a second engine has
something to conform to; leave it implicit and the second producer reverse-
engineers the first one's incidental output, which is how two screens stop
comparing the same thing without anyone noticing.

⚠ THIS CONTRACT IS ESTABLISHED HERE, NOT INHERITED. The upstream analysis that
first wrote such a table describes it as "substrate", and no upstream consumer
reads it back — each recomputes its own frame from the raw store. So this is a
contract we are adopting, and the obligation to keep producers conformant is
ours.

THE GRAIN — one row per cell, keyed by all five of ``KEY_COLUMNS``.
⚠ ``experiment_id`` is part of the key. The upstream docstring describes the
grain as four columns and its code groups by five; the code is right, and a
producer that drops ``experiment_id`` silently merges cells from different runs
that share a lineage seed.

THE VALUES — every non-key column is an observable that has ALREADY been
time-averaged WITHIN the cell. That ordering is the whole point: a screen's unit
of replication is the cell, so a statistic is computed over cells, never over
raw timepoints. Pooling timepoints would weight long-lived cells more heavily
and report a confidence the design never earned.

Derived columns (yields) are ``numerator / abs(denominator)`` — the absolute
value because an uptake flux is negative by sign convention, and a yield on a
consumed substrate is positive.
"""
from __future__ import annotations

#: Together these identify one cell. All five are required; see the module note
#: on ``experiment_id``.
KEY_COLUMNS = ("experiment_id", "variant", "lineage_seed", "generation", "agent_id")


class SeamViolation(ValueError):
    """The table does not meet the contract. Carries every violation found, not
    just the first — a producer being brought into conformance wants the whole
    list, and reporting one at a time turns that into a guessing game."""

    def __init__(self, violations: list):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


def observable_columns(rows: list) -> list:
    """The observable column names — every column that is not a key. Order is
    the first row's, so a table stays readable in the order its producer chose."""
    if not rows:
        return []
    return [c for c in rows[0] if c not in KEY_COLUMNS]


def cell_key(row: dict) -> tuple:
    return tuple(row.get(c) for c in KEY_COLUMNS)


def check(rows: list, *, required_observables=()) -> list:
    """Return the list of contract violations — empty when the table conforms.

    Returns rather than raises so a caller can report them all; see
    :func:`validate` for the raising form.
    """
    violations = []
    if not rows:
        return ["the table has no rows"]

    first = list(rows[0])
    missing_keys = [c for c in KEY_COLUMNS if c not in first]
    if missing_keys:
        violations.append(f"missing key column(s): {missing_keys}")

    obs = observable_columns(rows)
    if not obs:
        violations.append("no observable columns — a table of keys grades nothing")

    for name in sorted(set(required_observables) - set(obs)):
        violations.append(f"required observable absent: {name!r}")

    seen: dict = {}
    for i, row in enumerate(rows):
        if list(row) != first:
            violations.append(f"row {i}: columns differ from the first row")
            continue
        if missing_keys:
            continue
        key = cell_key(row)
        if any(v is None for v in key):
            violations.append(f"row {i}: incomplete cell key {key}")
            continue
        if key in seen:
            # ⛔ The one that silently corrupts a ranking: two rows for one cell
            # double-weight it in every statistic computed over the panel.
            violations.append(f"row {i}: duplicate cell key {key} (first seen at row {seen[key]})")
            continue
        seen[key] = i
        for c in obs:
            v = row[c]
            if v is None:
                continue          # an unobserved observable is honest; see below
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                violations.append(f"row {i}: observable {c!r} is not numeric ({v!r})")
    return violations


def validate(rows: list, *, required_observables=()) -> None:
    """Raise :class:`SeamViolation` if the table does not conform."""
    violations = check(rows, required_observables=required_observables)
    if violations:
        raise SeamViolation(violations)


def cells_per_variant(rows: list) -> dict:
    """``{variant: n_cells}`` — the replication actually achieved.

    ⚠ A screen's resolvability is a function of this, so it belongs beside the
    numbers rather than being assumed uniform: variants do not all survive to
    the same cell count, and a variant with fewer cells has a wider interval
    that a ranking has to respect.
    """
    counts: dict = {}
    for row in rows:
        counts[row.get("variant")] = counts.get(row.get("variant"), 0) + 1
    return counts
