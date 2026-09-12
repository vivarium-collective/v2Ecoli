"""Memory-class routing for analysis submission (v2ecoli#788).

``analysis_memory_class`` picks the instance class ("standard" 60 GB vs "large"
200 GB r7i) at submission time from the analyses named and the sweep's scale, so
a heavy multi-generation gather goes to the big box by declaration instead of
OOM-then-hand-rerun (v2ecoli#786). Generations drive the per-lineage peak
(~8 GB/generation, from #786's ~78 GB over 10 generations); the chunked readers
make seed count a non-factor.
"""

from __future__ import annotations

import pytest

from v2ecoli.workflow.analysis_runner import (
    analysis_memory_class,
    scale_memory_class,
)


# ---------------------------------------------------------------------------
# scale_memory_class — the per-scale sizing rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", ["single", "multidaughter", None])
def test_single_cell_scales_are_always_standard(scale):
    assert scale_memory_class(scale, n_generations=100) == "standard"


@pytest.mark.parametrize("scale", ["multigeneration", "multiseed"])
def test_multicell_scale_routes_large_past_the_standard_box(scale):
    # ~8 GB/generation: 4 gens (32 GB) fits the 60 GB box; 8 gens (64 GB) and
    # 10 gens (80 GB) — the CD2 failing cases — do not.
    assert scale_memory_class(scale, n_generations=4) == "standard"
    assert scale_memory_class(scale, n_generations=8) == "large"
    assert scale_memory_class(scale, n_generations=10) == "large"


def test_missing_generation_count_stays_standard():
    assert scale_memory_class("multigeneration", n_generations=None) == "standard"


# ---------------------------------------------------------------------------
# analysis_memory_class — reduction over an analysis_options mapping
# ---------------------------------------------------------------------------

def test_options_reduce_to_the_max_class():
    # A light single-scale analysis and a heavy 10-generation multigeneration
    # one in the same job: the whole job routes large.
    options = {
        "single": {"ptools_rxns": {}},
        "multigeneration": {"ptools_rxns_multigeneration": {}},
    }
    assert analysis_memory_class(options, n_generations=10) == "large"
    assert analysis_memory_class(options, n_generations=3) == "standard"


def test_single_scale_only_is_standard_at_any_size():
    options = {"single": {"ptools_rxns": {}, "cell_mass": {}}}
    assert analysis_memory_class(options, n_seeds=64, n_generations=64) == "standard"


def test_empty_options_is_standard():
    assert analysis_memory_class({}, n_generations=10) == "standard"
    assert analysis_memory_class(None, n_generations=10) == "standard"


def test_declared_large_forces_large_regardless_of_scale(monkeypatch):
    # A module that declares itself heavy routes large even at single scale on a
    # tiny sweep. Set the attribute on a registered analysis class.
    import v2ecoli.workflow.analyses  # noqa: F401  (populate the registry)
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    cls = ANALYSIS_REGISTRY["ptools_rxns"]
    monkeypatch.setattr(cls, "memory_class", "large", raising=False)

    options = {"single": {"ptools_rxns": {}}}
    assert analysis_memory_class(options, n_generations=1) == "large"
