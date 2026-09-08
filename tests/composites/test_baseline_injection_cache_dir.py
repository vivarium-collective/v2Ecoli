"""``baseline()`` must thread its own ``cache_dir`` into the injection spec.

An injected process whose config is built from the ParCa bundle needs a path to
it, and the native resolver looks for that path **on the injection spec** — not
in ``baseline()``'s arguments, which it cannot see. A caller that declares only
``swap_processes`` therefore gets a process built with NO config: every
``config_schema`` default fires. For a swapped metabolism-redux that is an empty
stoichiometry / 0 homeostatic targets, which collapses the generation to one tick
while reporting success (sms-ecoli#210 Gate 0).

Test structure (not behavior): the fix is a one-line seed, but it must be the
right shape — copied not mutated, override-wins, and BEFORE the resolve call —
and the resolver that consumes it lives in the image, not this repo's tree, so a
behavioral test here can't exercise it. Pins the source shape instead. (Design
ported from the closed v2ecoli#667, which had the diagnosis right.)
"""
import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "v2ecoli" / "composites" / "ecoli_baseline.py"


def _injection_block() -> str:
    """Source of the branch that assembles + applies injected_processes."""
    src = _SRC.read_text()
    i = src.index("assert_injection_sourcing(injected_processes)")
    return src[i:src.index("remove_processes(cell_state", i)]


def test_cache_dir_is_threaded_into_the_spec_before_resolution():
    block = _injection_block()
    assert '"cache_dir": injected_processes.get("cache_dir") or cache_dir' in block, (
        "baseline() must seed the spec's cache_dir from its own, or an injected "
        "process needing a bundle-built config is constructed config-less")
    assert block.index('"cache_dir"') < block.index("resolve_injections("), (
        "the seed must happen BEFORE resolve_injections, or it has no effect")


def test_an_explicit_cache_dir_on_the_spec_still_wins():
    """``or``, not an unconditional assignment: a caller deliberately pointing an
    injected process at a different bundle (e.g. a per-seed cache) keeps it."""
    block = _injection_block()
    assert 'injected_processes.get("cache_dir") or cache_dir' in block
    assert '"cache_dir": cache_dir,' not in block, (
        "unconditional assignment would silently override a deliberate override")


def test_the_spec_is_copied_not_mutated():
    """The caller's dict must not be mutated — callers reuse it across builds
    (e.g. one spec threaded to every per-seed build in a batch dispatch)."""
    block = _injection_block()
    assert "**injected_processes," in block, "must build a new dict, not mutate"
    assert "injected_processes[" not in block.split("resolve_injections(")[0], (
        "no item assignment on the caller's dict before resolution")


def test_module_still_parses():
    ast.parse(_SRC.read_text())
