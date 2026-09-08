"""Resolve declared port-schema units into axis-label strings.

Units live on a resolved bigraph-schema type node as ``_units`` (e.g.
``core.access('quantity[float,fg]')._units == 'fg'``). Wrappers such as
``overwrite[...]`` carry the unit one level down on ``_value``; ``array[...]``
and parameterized ``float[...]`` propagate it to the top node. This module
extracts that unit per type string, builds a ``dotted-path -> unit`` index by
walking the baseline composite's declared port schemas, and formats axis
labels. No sim_data is loaded.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _unit_from_node(node: Any, _depth: int = 0) -> Optional[str]:
    """Recover a unit string from a resolved type node, unwrapping wrappers.

    Reads ``node._units``; an empty string or ``None`` means "no unit here",
    so unwrap one level via ``_value`` (e.g. ``overwrite[...]``) up to a small
    depth. Returns the first non-empty unit found, else ``None``.
    """
    if node is None or _depth > 3:
        return None
    unit = getattr(node, "_units", None)
    if unit:  # non-empty string
        return unit
    return _unit_from_node(getattr(node, "_value", None), _depth + 1)


def unit_from_type(type_str: Any, core: Any) -> Optional[str]:
    """Return the unit string declared by a bigraph-schema type string, or None.

    ``core`` is a bigraph-schema Core (``v2ecoli.core.build_core()``). Any
    resolution failure yields ``None`` (units are best-effort decoration).
    """
    if not isinstance(type_str, str) or core is None:
        return None
    try:
        node = core.access(type_str)
    except Exception:
        return None
    return _unit_from_node(node)


def units_from_schema(schema: Any, core: Any, _prefix: str = "") -> dict[str, str]:
    """Walk a port-schema value into a flat ``dotted-path -> unit`` dict.

    Mirrors the traversal in ``output_metadata._extract_labels_recursive`` but
    records units instead of element labels. Leaves with no unit are omitted.

    Handles: bare string type names (``'quantity[float,fg]'``), typed-leaf
    dicts (``{'_type': '...', '_default': ...}``), and nested port dicts.
    """
    index: dict[str, str] = {}

    if isinstance(schema, str):
        unit = unit_from_type(schema, core)
        if unit and _prefix:
            index[_prefix] = unit
        return index

    if not isinstance(schema, dict):
        return index

    if "_type" in schema:
        unit = unit_from_type(schema.get("_type"), core)
        if unit and _prefix:
            index[_prefix] = unit
        return index

    for key, sub in schema.items():
        if key.startswith("_"):
            continue
        child_prefix = f"{_prefix}.{key}" if _prefix else key
        index.update(units_from_schema(sub, core, child_prefix))
    return index


def _iter_process_classes():
    """Yield (name, class) for baseline composite process/step classes.

    Best-effort enumeration for schema introspection only — uses the static
    ``PARTITIONED_PROCESSES`` registry plus the explicitly-imported listener
    classes. Import failures (optional deps) are skipped silently.
    """
    try:
        from v2ecoli.composites._helpers import PARTITIONED_PROCESSES
    except Exception:
        PARTITIONED_PROCESSES = {}
    for name, cls in (PARTITIONED_PROCESSES or {}).items():
        yield name, cls

    explicit = []
    for modpath, clsname in [
        ("v2ecoli.processes.equilibrium", "Equilibrium"),
        ("v2ecoli.processes.metabolism", "Metabolism"),
        ("v2ecoli.processes.two_component_system", "TwoComponentSystem"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            explicit.append((clsname, getattr(mod, clsname)))
        except Exception:
            continue
    for item in explicit:
        yield item


def _instantiate(cls, core):
    """Best-effort instance for schema introspection only.

    Tries a normal ``cls(config={}, core=core)`` first; if ``__init__`` is
    config-dependent and raises, falls back to a bare ``cls.__new__(cls)``
    instance. The ``inputs()``/``outputs()`` schema declarations in v2ecoli
    processes are pure (they do not read ``self``), so the bare instance still
    yields the declared port types. Returns ``None`` if both paths fail.
    """
    try:
        return cls(config={}, core=core)
    except Exception:
        pass
    try:
        return cls.__new__(cls)
    except Exception as exc:
        logger.debug(
            "units_resolver: skipping %r — could not instantiate for schema "
            "introspection: %r",
            getattr(cls, "__name__", cls),
            exc,
        )
        return None


def _index_from_classes(core) -> dict[str, str]:
    index: dict[str, str] = {}
    for name, cls in _iter_process_classes():
        inst = _instantiate(cls, core)
        if inst is None:
            continue
        for method in ("inputs", "outputs"):
            try:
                schema = getattr(inst, method)()
            except Exception as exc:
                logger.debug(
                    "units_resolver: skipping %s.%s() — schema introspection "
                    "raised, dropping its unit contribution: %r",
                    name,
                    method,
                    exc,
                )
                continue
            if isinstance(schema, dict):
                index.update(units_from_schema(schema, core))
    return index


@lru_cache(maxsize=1)
def build_units_index() -> dict[str, str]:
    """Composite-wide ``dotted-path -> unit`` index, built once and cached.

    Reads declared port types from the baseline process classes. No sim_data is
    loaded. Returns the same dict object on repeat calls (callers must treat it
    as read-only).
    """
    from v2ecoli.core import build_core
    core = build_core()
    return _index_from_classes(core)


def resolve_unit(units_index: dict, path: Optional[str]) -> Optional[str]:
    """Look up the unit for an observable path; tolerate array/sub-leaf paths.

    Exact match first; otherwise strip trailing ``.<segment>`` components (array
    indices, sub-leaves) and retry against the parent path. Returns ``None`` for
    unitless or unknown paths.
    """
    if not path or not units_index:
        return None
    if path in units_index:
        return units_index[path]
    parts = path.split(".")
    while len(parts) > 1:
        parts = parts[:-1]
        parent = ".".join(parts)
        if parent in units_index:
            return units_index[parent]
    return None


def format_axis_label(base_label: str, unit: Optional[str]) -> str:
    """Append ``(unit)`` to a label, idempotently. ``None`` unit -> unchanged."""
    if not unit:
        return base_label
    label = (base_label or "").rstrip()
    if label.endswith(f"({unit})"):
        return label
    return f"{label} ({unit})".strip()


class V2EcoliUnitsResolver:
    """Callable ``path -> unit`` resolver backed by the cached composite index.

    Registered onto the Visualization base class so every v2ecoli viz can label
    axes from the declared schema. Reads the live declarations (no persisted
    snapshot); the underlying index is memoized by ``build_units_index``.
    """

    def __call__(self, path: Optional[str]) -> Optional[str]:
        return resolve_unit(build_units_index(), path)


# ---------------------------------------------------------------------------
# Resilient figure helpers — prefer the shared base hook, fall back locally
# ---------------------------------------------------------------------------
#
# v2ecoli visualizations label axes via the pluggable finalize_figure hook on
# the shared ``viva_superpowers.visualization.Visualization`` base. That hook is
# present in the shared package now; the local fallback below exists only for a
# stale or non-editable shared venv that predates it. These wrappers delegate to
# the base hook when it is present and apply the identical labeling locally
# otherwise, so v2ecoli rendering never crashes on an old base.


def _base_visualization():
    """Return the shared Visualization base class, or None if unavailable."""
    try:
        from viva_superpowers.visualization import Visualization
        return Visualization
    except Exception:
        return None


def units_finalize_figure(fig, axis_units=()):
    """Append schema units to matplotlib axis labels in-place.

    ``axis_units`` is an iterable of ``(ax, which, path)`` with ``which`` in
    ``{'x', 'y'}``. Delegates to ``Visualization.finalize_figure`` when present;
    otherwise applies the same labeling via the v2ecoli resolver. Returns ``fig``.
    """
    base = _base_visualization()
    if base is not None and hasattr(base, "finalize_figure"):
        return base.finalize_figure(fig, axis_units)
    resolver = V2EcoliUnitsResolver()
    for ax, which, path in axis_units:
        unit = resolver(path)
        if not unit:
            continue
        if which == "y" and hasattr(ax, "set_ylabel"):
            ax.set_ylabel(format_axis_label(ax.get_ylabel(), unit))
        elif which == "x" and hasattr(ax, "set_xlabel"):
            ax.set_xlabel(format_axis_label(ax.get_xlabel(), unit))
    return fig


def units_figure_to_html(fig, axis_units=(), *, dpi=150, close=True):
    """Finalize axis units, then serialize a matplotlib figure to an ``<img>``.

    Delegates to ``Visualization.figure_to_html`` when present; otherwise
    finalizes locally via :func:`units_finalize_figure` and serializes the same
    way the base hook does. Returns a complete ``<img>`` tag.
    """
    base = _base_visualization()
    if base is not None and hasattr(base, "figure_to_html"):
        return base.figure_to_html(fig, axis_units, dpi=dpi, close=close)
    import base64
    import io
    units_finalize_figure(fig, axis_units)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    if close:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
