"""v2ecoli — whole-cell E. coli model on process-bigraph."""

from __future__ import annotations

import warnings
from typing import Any

from process_bigraph import Composite
from viva_superpowers.composite_generator import _REGISTRY, build_generator

from v2ecoli import core
from v2ecoli.core import build_core
from v2ecoli import composites  # noqa: F401 — forces @composite_generator decorators to fire
from v2ecoli import visualizations  # noqa: F401 — forces Visualization Steps into link_registry

# Convention hook: a generic runner (e.g. process_bigraph.workflow.provision)
# discovers a package's core-registration function by this name. Points at
# the same function ecoli_baseline's core_extensions declares, so a bare
# core provisioned via either path ends up identically configured.
register_types = core.register_ecoli_core


def build_composite(
    name: str,
    *,
    core: Any = None,
    **kwargs: Any,
) -> Composite:
    """Build a Composite by architecture name.

    Parameters
    ----------
    name:
        One of ``"ecoli_baseline"``, ``"ecoli_colony"``, ``"ecoli_millard"``.
    core:
        Optional bigraph-schema core. If omitted, a fresh one is built via
        ``v2ecoli.core.build_core()``.
    **kwargs:
        Passed through to the generator's declared parameters (currently
        ``seed`` and ``cache_dir``).

    Raises
    ------
    ValueError
        If ``name`` does not match any registered architecture, or matches more
        than one, or if ``kwargs`` contains an unknown parameter name.
    """
    auto_core = core is None
    matches = [e for e in _REGISTRY.values() if e.name == name]
    if not matches:
        available = sorted({e.name for e in _REGISTRY.values()})
        raise ValueError(
            f"unknown composite architecture {name!r}; available: {available}"
        )
    if len(matches) > 1:
        # The same architecture may be registered under more than one id (e.g. a
        # clean-id alias added so the dashboard's ".composites.<slug>" resolver
        # finds it — see v2ecoli/composites/__init__.py). Those entries share the
        # same generator function, so dedupe by function identity and pick the
        # canonical (shortest) id. Only genuinely-distinct generators are
        # ambiguous.
        if len({e.func for e in matches}) == 1:
            matches = [min(matches, key=lambda e: len(e.id))]
        else:
            raise ValueError(
                f"ambiguous architecture name {name!r}; multiple generators registered: "
                f"{[e.id for e in matches]}"
            )
    gen = matches[0]
    if auto_core:
        core = build_core()
        # Apply the generator's core_extensions so a composite that needs extra type
        # or link registration (e.g. ecoli_colony's ``pymunk_agent`` type via
        # viva_munk) builds on a proper core through this convenience path too -- not
        # only inside the workbench env-worker (which already applies them). An
        # extension may RETURN a fresh core (the colony builds one on viva_munk's base
        # via ``core_import``); use the returned core when it does.
        for _ext in (getattr(gen, "core_extensions", None) or []):
            _returned = _ext(core)
            if _returned is not None:
                core = _returned
    doc = build_generator(gen, overrides=kwargs, core=core)
    composite = Composite(doc, core=core)
    _install_xarray_flush_hook(composite)
    return composite


def _find_lazy_xarray_emitters(state: Any) -> list:
    """Collect any in-document ``SingleCellXArrayEmitter`` instances in ``state``.

    The single-cell ``emitter="xarray"`` path wires one such step inside
    ``agents/<id>/emitter``; a run may hold several (one per agent). Returns the
    live step instances so their trailing buffers can be flushed after ``run()``.
    """
    from v2ecoli.composites.ecoli_baseline import SingleCellXArrayEmitter
    found: list = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            inst = node.get("instance")
            if isinstance(inst, SingleCellXArrayEmitter):
                found.append(inst)
            for v in node.values():
                _walk(v)

    _walk(state)
    return found


def _install_xarray_flush_hook(composite: Composite) -> None:
    """Flush in-document XArray emitters after each ``run()``.

    ``XArrayEmitter`` only auto-flushes a FULL buffer mid-run, and
    ``flush(final=False)`` asserts a full buffer — so the trailing partial buffer
    (and the zarr store finalization) can only be written via ``close()``. There
    is no process_bigraph lifecycle hook that fires at run completion, so we wrap
    ``run`` to close the emitters after the requested interval completes. Only
    installed when a ``SingleCellXArrayEmitter`` is present — the parquet/sqlite/
    default paths are left byte-identical.
    """
    emitters = _find_lazy_xarray_emitters(composite.state)
    if not emitters:
        return
    _orig_run = composite.run

    def _run_and_flush(interval, *args, **kwargs):
        # The flush must happen however run() ends. An exception out of
        # run() (a process crash, an OOM) used to skip close_emitter()
        # entirely, losing the up-to-600-row trailing buffer AND the store
        # finalization (no consolidate, no division_reached attr) — the
        # durability gap CD2 hit as "empty group skeletons crash the next
        # generation". On the failure path a flush error is reported, never
        # raised over the run's own exception; on the success path a flush
        # failure stays loud exactly as before.
        try:
            result = _orig_run(interval, *args, **kwargs)
        except BaseException:
            for em in _find_lazy_xarray_emitters(composite.state):
                try:
                    em.close_emitter()
                except Exception as flush_err:  # noqa: BLE001
                    warnings.warn(
                        f"xarray flush after failed run() also failed "
                        f"(the run's own exception follows): {flush_err}")
            raise
        for em in _find_lazy_xarray_emitters(composite.state):
            em.close_emitter()
        return result

    composite.run = _run_and_flush  # type: ignore[method-assign]


__all__ = ["build_composite", "build_core", "register_types"]
