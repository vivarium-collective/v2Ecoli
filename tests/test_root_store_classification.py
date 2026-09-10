"""Every agent-root store a step/process declares must have a CARRY CLASSIFICATION.

The lineage carry policy (``v2ecoli/library/division.py``) is a policy over
whatever happens to sit at the agent root: core stores are divided, registered
stores are split by their divider, deny-listed stores are dropped, and
everything else is COPIED. That last branch is where #765 went wrong -- the
per-tick partition roots ``request``/``allocate`` were neither listed nor
excluded, were silently copied into every daughter, and killed generation 1 of
every multi-generation run built from ``main`` for five hours (sims 943/944/945,
2026-09-10; fixed by #769). Nothing in the test-suite could have named them,
because no test ever saw the real root set.

This test does. It enumerates every single-element root port path declared as
a wiring value (``: ("name",)``) across the steps, processes, composites and
library, and fails -- naming the store -- if any of them is not in exactly one
of: ``CORE_DIVISIBLE_KEYS``, ``NON_CARRIED_ROOT_KEYS``, ``CARRIED_BY_COPY``,
the divider registry, or the explicit list of COMPOSITE-level roots below
(stores that live beside the ``agents`` map, never inside a cell, and are
therefore never subject to the policy). Adding a root store means deciding,
in one line, what division does to it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from v2ecoli.library import division as div

pytestmark = pytest.mark.fast

ROOT = pathlib.Path(__file__).resolve().parents[1] / "v2ecoli"
SCAN_DIRS = ("steps", "processes", "composites", "library")
PORT_VALUE = re.compile(r":\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*,\s*\)")

#: Roots that sit at the COMPOSITE level (siblings of ``agents``), not inside a
#: cell document, so the per-cell carry policy never sees them. Each with the
#: file that declares it and why it is not an agent root.
COMPOSITE_LEVEL_ROOTS = {
    "agents": "the agents map itself (composites/_helpers.py); also in NON_CARRIED",
    "population": "population_aggregator / reactor_cell_coupler: colony-level aggregate",
    "lineage": "lineage_bookkeeper / population_aggregator: colony-level bookkeeping",
    "batch": "batch_baseline_runner: per-seed results written by the meta-composite",
    "reactor": "reactor_cell_coupler: the bioreactor (ecoli_millard), one per composite",
}


def declared_root_ports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for sub in SCAN_DIRS:
        for path in (ROOT / sub).rglob("*.py"):
            text = path.read_text(errors="ignore")
            for match in PORT_VALUE.finditer(text):
                name = match.group(1)
                found.setdefault(name, set()).add(str(path.relative_to(ROOT.parent)))
    return found


def test_the_scan_sees_the_real_root_set():
    """Guard the guard: if the regex ever stops matching the wiring style, this
    test would pass vacuously. It must see the well-known roots."""
    names = set(declared_root_ports())
    for expected in ("bulk", "unique", "listeners", "global_time", "request", "allocate"):
        assert expected in names, f"scan no longer sees {expected!r}: {sorted(names)}"


def test_every_declared_root_store_is_classified():
    from v2ecoli.library.division import (
        CARRIED_BY_COPY,
        CORE_DIVISIBLE_KEYS,
        NON_CARRIED_ROOT_KEYS,
        STORE_DIVIDERS,
    )

    classified = (
        set(CORE_DIVISIBLE_KEYS)
        | set(NON_CARRIED_ROOT_KEYS)
        | set(CARRIED_BY_COPY)
        | set(STORE_DIVIDERS)
        | set(COMPOSITE_LEVEL_ROOTS)
    )
    unclassified = {}
    for name, files in declared_root_ports().items():
        if name.startswith("_"):
            continue  # schema/private keys are skipped by extra_store_keys itself
        if name not in classified:
            unclassified[name] = sorted(files)
    assert not unclassified, (
        "agent-root store(s) with NO carry classification -- decide what division "
        "does to each (CORE_DIVISIBLE_KEYS / NON_CARRIED_ROOT_KEYS / CARRIED_BY_COPY / "
        "register_store_divider, or COMPOSITE_LEVEL_ROOTS in this test if it is not "
        "an agent root):\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(unclassified.items()))
    )


def test_classifications_do_not_overlap():
    from v2ecoli.library.division import CARRIED_BY_COPY, CORE_DIVISIBLE_KEYS, NON_CARRIED_ROOT_KEYS

    core, non, copy_ = set(CORE_DIVISIBLE_KEYS), set(NON_CARRIED_ROOT_KEYS), set(CARRIED_BY_COPY)
    assert not (core & non), core & non
    assert not (core & copy_), core & copy_
    assert not (non & copy_), non & copy_


def test_partition_bookkeeping_is_never_carried():
    """The #765 class, pinned: the two per-tick partition roots are excluded."""
    assert {"request", "allocate"} <= set(div.NON_CARRIED_ROOT_KEYS)
    assert "request" not in div.extra_store_keys({"request": {}, "allocate": {}, "fields": {}})
    assert div.extra_store_keys({"request": {}, "allocate": {}, "fields": {}}) == ("fields",)
