"""Regression test for the per-generation RNG seed combiner.

The historical combiner ``gen_seed = (seed + generation) % 2**31`` collapsed a
multiseed x multigeneration sweep's stochastic heterogeneity two ways: it ignored
``lineage_seed`` (so every lineage drew the same per-generation seed) and it
aliased the ``(seed, generation)`` grid (e.g. ``(1, 4)`` and ``(0, 5)`` both -> 5).
Because ``gen_seed`` is also the ``master_seed`` every stochastic process derives
from, colliding cells became bit-identical.

``_derive_generation_seed`` replaces it with a ``SeedSequence`` combine of all
three axes. These tests pin the properties that fix depends on.
"""
import numpy as np

from v2ecoli.workflow.lineage import _derive_generation_seed


def test_historical_additive_alias_pair_now_differs():
    # The canonical collision the additive combiner produced.
    assert (1 + 4) % (2**31) == (0 + 5) % (2**31)  # the old bug, for the record
    a = _derive_generation_seed(seed=1, lineage_seed=0, generation=4)
    b = _derive_generation_seed(seed=0, lineage_seed=0, generation=5)
    assert a != b, "additive alias pair (1,4) vs (0,5) still collides"


def test_lineage_seed_actually_matters():
    # Under the old combiner, lineage_seed never entered gen_seed, so the N
    # lineages of a multiseed run were bit-identical. Every lineage must now draw
    # a distinct seed.
    seeds = {
        _derive_generation_seed(seed=0, lineage_seed=ls, generation=g)
        for ls in range(4)
        for g in range(5)
    }
    # 4 lineages x 5 generations = 20 cells, all distinct.
    assert len(seeds) == 20


def test_no_collisions_across_a_multiseed_multigen_grid():
    grid = [
        _derive_generation_seed(seed=s, lineage_seed=ls, generation=g)
        for s in range(4)
        for ls in range(4)
        for g in range(20)
    ]
    assert len(set(grid)) == len(grid), "grid cells alias (RNG heterogeneity lost)"


def test_reproducible_per_tuple():
    for s, ls, g in [(0, 0, 0), (3, 2, 19), (7, 1, 5)]:
        assert _derive_generation_seed(s, ls, g) == _derive_generation_seed(s, ls, g)


def test_seed_is_nonnegative_int32():
    for s, ls, g in [(0, 0, 0), (2**31 - 1, 5, 10), (123, 456, 789)]:
        v = _derive_generation_seed(s, ls, g)
        assert isinstance(v, int)
        assert 0 <= v <= 0x7FFFFFFF


def test_accepts_numpy_and_python_ints():
    # config values may arrive as numpy ints from an upstream store.
    a = _derive_generation_seed(np.int64(2), np.int64(1), np.int64(3))
    b = _derive_generation_seed(2, 1, 3)
    assert a == b
