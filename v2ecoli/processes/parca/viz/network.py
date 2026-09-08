"""Interactive Cytoscape.js network visualization for the v2ecoli.processes.parca composite.

Public API:
    build_graph() -> dict
        Walks ``ALL_STEP_CLASSES`` + each step's ``INPUT_PORTS`` /
        ``OUTPUT_PORTS`` and v2ecoli.processes.parca's ``STORE_PATH`` registry, and emits
        ``{nodes, edges, layers, legend}`` for the Cytoscape viewer. No
        composite needs to be instantiated — the static port manifests
        are the source of truth.
    render_html(data, title, subtitle) -> str
    write_outputs(data, out_dir, name, title, subtitle) -> (json_path, html_path)

Same HTML viewer as v2ecoli (Cytoscape.js, bipartite/dagre/fcose layouts,
legend filtering, click-to-inspect details panel). The graph builder is
v2ecoli.processes.parca-specific: the composite topology here is a linear 9-Step chain
wired through a nested sim_data-shaped store, not a whole-cell composite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence


# ---------------------------------------------------------------------------
# Step classification — one category per logical phase of the ParCa.
# ---------------------------------------------------------------------------
# (key, display_label, hex_color, matcher(step_name: str) -> bool)
BIO_COLORS: list[tuple[str, str, str, Callable[[str], bool]]] = [
    ('init',     'Initialize (scatter)',        '#9CC8E3',
        lambda n: n == 'initialize'),
    ('basal',    'Basal fitting',               '#A4D4A4',
        lambda n: n in ('input_adjustments', 'basal_specs')),
    ('fit',      'Per-condition fitting',       '#F7D488',
        lambda n: n in ('tf_condition_specs', 'fit_condition')),
    ('promoter', 'Promoter binding',            '#D9B8E0',
        lambda n: n in ('promoter_binding', 'adjust_promoters')),
    ('condition','Set conditions',              '#FFD58C',
        lambda n: n == 'set_conditions'),
    ('final',    'Final adjustments',           '#F4A7A1',
        lambda n: n == 'final_adjustments'),
]


def classify(name: str) -> tuple[str, str]:
    """Return (phase_key, color) for a v2ecoli.processes.parca step name."""
    for key, _label, color, matcher in BIO_COLORS:
        if matcher(name):
            return key, color
    return 'other', '#CCCCCC'


# ---------------------------------------------------------------------------
# Composite → graph data
# ---------------------------------------------------------------------------

_STORE_NODES = (
    'bulk', 'unique', 'environment', 'boundary', 'listeners',
    'request', 'allocate', 'process_state', 'next_update_time',
    'global_time', 'timestep', 'ppgpp_state', 'attenuation_config',
)

# Parent stores whose first-level children should get their own node,
# connected to the parent by a place-graph "contains" edge. Processes
# that wire into ``('unique', 'promoter')`` will then connect to a
# dedicated ``unique.promoter`` node rather than the ``unique`` circle.
_NESTABLE_PARENTS = ('unique',)

# Config keys to omit from the inspection panel — callables, shared
# registries, and ParCa-sized arrays that aren't useful at a glance.
_CONFIG_HIDE = {
    'process', 'step', 'processes', 'listeners', 'sim_data', 'simData',
    'simdata', 'core', 'topology', 'flow', 'parent',
}


def _wire_to_store_id(wire: Any) -> tuple[str | None, str | None]:
    """Return (store_id, parent_id) for a wire path.

    - Generic wires resolve to their first segment: ``('bulk', ...)``
      -> ``('bulk', None)``.
    - Wires whose head is in ``_NESTABLE_PARENTS`` and that carry a
      second segment resolve to a compound id: ``('unique', 'promoter', ...)``
      -> ``('unique.promoter', 'unique')``. Callers register both the
      child and the parent and add a containment edge.
    """
    if not (isinstance(wire, (list, tuple)) and wire):
        return None, None
    # Skip wire-path navigation segments — ``..`` means "go up one store"
    # and is not itself a store. Walk past leading ``..`` segments so the
    # resolved id is the first real store name.
    segs = list(wire)
    while segs and segs[0] == '..':
        segs.pop(0)
    if not segs:
        return None, None
    head = segs[0]
    if not (isinstance(head, str) and head and not head.startswith('_')):
        return None, None
    if head in _NESTABLE_PARENTS and len(segs) > 1:
        child = segs[1]
        if isinstance(child, str) and child and not child.startswith('_'):
            return f'{head}.{child}', head
    return head, None


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Best-effort JSON-safe reduction of an arbitrary value for the panel."""
    if depth > 3:
        return f'<{type(value).__name__}>'
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace')
        except Exception:
            return f'<{len(value)} bytes>'
    if callable(value):
        return f'<callable {getattr(value, "__qualname__", type(value).__name__)}>'
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            shape = tuple(int(d) for d in value.shape)
            dtype = str(value.dtype)
            if value.size <= 8:
                return {'_type': 'ndarray', 'shape': shape, 'dtype': dtype,
                        'values': value.tolist()}
            return {'_type': 'ndarray', 'shape': shape, 'dtype': dtype}
        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:40]:
            if k in _CONFIG_HIDE:
                continue
            out[str(k)] = _jsonable(v, depth + 1)
        if len(value) > 40:
            out['…'] = f'{len(value) - 40} more entries'
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if len(seq) <= 12:
            return [_jsonable(x, depth + 1) for x in seq]
        return {
            '_type': 'sequence',
            'len': len(seq),
            'head': [_jsonable(x, depth + 1) for x in seq[:8]],
        }
    cls = type(value).__name__
    try:
        return f'<{cls} {value!r}>'
    except Exception:
        return f'<{cls}>'


_MATH_HEADINGS = (
    'mathematical model', 'mathematics', 'math', 'equations', 'model',
)


def _split_rst_sections(doc: str) -> list[tuple[str, str]]:
    """Split a reST-style docstring into (heading, body) sections.

    Recognises headings that are followed by an underline made of ``-`` or
    ``=`` characters. Lines before the first such heading form a synthetic
    ``''`` (empty-heading) section so the lead paragraph is preserved.
    """
    if not doc:
        return []
    # Dedent so inline headings keep their underline alignment.
    import textwrap
    lines = textwrap.dedent(doc).splitlines()
    sections: list[tuple[str, list[str]]] = [('', [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        if (line.strip()
                and nxt.strip()
                and set(nxt.strip()) <= set('-=~^"')
                and len(nxt.strip()) >= max(3, len(line.strip()) - 2)):
            sections.append((line.strip(), []))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1
    return [(h, '\n'.join(body).strip('\n')) for h, body in sections]


def _extract_math_and_doc(raw_doc: str) -> tuple[str, str]:
    """Return (math, rest) extracted from a reST-style docstring.

    The ``math`` string contains the joined bodies of any sections whose
    heading (case-insensitive) matches one of ``_MATH_HEADINGS`` or begins
    with ``math``. ``rest`` is the remaining prose with those sections
    removed, so the general Docstring panel no longer duplicates the math.
    """
    if not raw_doc:
        return '', ''
    sections = _split_rst_sections(raw_doc)
    math_parts: list[str] = []
    keep_parts: list[str] = []
    for heading, body in sections:
        h = heading.lower().strip()
        is_math = (
            h in _MATH_HEADINGS
            or h.startswith('math')
            or h.endswith('equations')
        )
        if is_math:
            math_parts.append(body.strip())
        else:
            if heading:
                keep_parts.append(heading)
                keep_parts.append('-' * len(heading))
            if body.strip():
                keep_parts.append(body.strip())
            keep_parts.append('')  # blank between sections
    math = '\n\n'.join(p for p in math_parts if p).strip()
    rest = '\n'.join(keep_parts).strip()
    return math, rest


def _underlying_partitioned_process(instance) -> tuple[object | None, str]:
    """If the edge wraps a PartitionedProcess (Requester/Evolver), return
    (underlying_process_instance, role) where role is 'request' or 'evolve'.
    Otherwise return (None, '')."""
    if instance is None:
        return None, ''
    cls_name = type(instance).__name__
    role = ''
    if cls_name == 'Requester':
        role = 'request'
    elif cls_name == 'Evolver':
        role = 'evolve'
    else:
        return None, ''
    params = getattr(instance, 'parameters', None)
    if not isinstance(params, dict):
        return None, role
    proc = params.get('process')
    if isinstance(proc, (list, tuple)) and proc:
        proc = proc[0]
    return proc, role


def _method_doc(obj, name: str) -> str:
    """Return the dedented docstring of ``obj.<name>`` or '' if none."""
    fn = getattr(obj, name, None)
    if fn is None:
        return ''
    doc = getattr(fn, '__doc__', None) or ''
    if not doc:
        return ''
    # Strip the first line's indentation, then dedent the rest together
    # (standard Python docstring convention). textwrap.dedent alone
    # can't handle the ``"""First line ...`` style.
    import textwrap
    lines = doc.expandtabs().splitlines()
    first = lines[0].lstrip() if lines else ''
    rest = textwrap.dedent('\n'.join(lines[1:])) if len(lines) > 1 else ''
    result = (first + ('\n' + rest if rest else '')).strip()
    return result


def _extract_metadata(edge: dict) -> dict:
    """Collect inputs/outputs schemas, config, doc + math for an edge.

    For plain Step/Process edges: math is pulled from the module docstring
    (our "Mathematical Model" reST convention) with the class docstring
    supplying implementation notes.

    For Requester/Evolver wrappers around a ``PartitionedProcess``: math is
    additionally pulled from the underlying process — both the module
    docstring and the specific method (``calculate_request`` for
    requesters, ``evolve_state`` for evolvers). This surfaces the role-
    specific math even though the viewer node is a wrapper step.
    """
    meta: dict[str, Any] = {
        '_inputs': _jsonable(edge.get('_inputs') or {}),
        '_outputs': _jsonable(edge.get('_outputs') or {}),
        'address': edge.get('address', ''),
    }

    raw_config = edge.get('config') or {}
    instance = edge.get('instance')
    if instance is not None and hasattr(instance, 'parameters'):
        params = getattr(instance, 'parameters', None)
        if isinstance(params, dict) and params:
            raw_config = params
    meta['config'] = _jsonable(raw_config)

    class_doc = ''
    module_doc = ''
    class_name = ''
    method_doc = ''
    method_label = ''

    def _module_doc_of(cls_obj):
        try:
            import sys as _sys
            mod = _sys.modules.get(cls_obj.__module__)
            return (getattr(mod, '__doc__', '') or '').strip() if mod else ''
        except Exception:
            return ''

    if instance is not None:
        cls = type(instance)
        class_name = f'{cls.__module__}.{cls.__qualname__}'
        class_doc = (cls.__doc__ or '').strip()
        module_doc = _module_doc_of(cls)

    underlying, role = _underlying_partitioned_process(instance)
    if underlying is not None:
        # Replace the effective source of math with the underlying process.
        u_cls = type(underlying)
        class_doc = (u_cls.__doc__ or '').strip() or class_doc
        module_doc = _module_doc_of(u_cls) or module_doc
        class_name = (
            f'{u_cls.__module__}.{u_cls.__qualname__} '
            f'({"requester" if role == "request" else "evolver"} wrapper)'
        )
        if role == 'request':
            method_label = 'calculate_request'
            method_doc = _method_doc(underlying, 'calculate_request')
        elif role == 'evolve':
            method_label = 'evolve_state'
            method_doc = _method_doc(underlying, 'evolve_state')

    # Prefer math from module doc (our convention); fall back to class doc.
    mod_math, mod_rest = _extract_math_and_doc(module_doc)
    cls_math, cls_rest = _extract_math_and_doc(class_doc)
    mth_math, mth_rest = _extract_math_and_doc(method_doc)

    # For requester/evolver, prepend the role-specific method math so it
    # reads as: "Inputs/Parameters/Calculation/Outputs of THIS half-step"
    # followed by the full process-level math for context.
    math_parts = []
    if mth_math:
        math_parts.append(mth_math)
    elif method_doc:
        # No tagged "Mathematical Model" in the method doc — still show
        # whatever the method docstring says, so authors get credit.
        math_parts.append(method_doc.strip())
    if mod_math:
        if math_parts:
            math_parts.append('--- Full process math ---')
        math_parts.append(mod_math)
    elif cls_math:
        math_parts.append(cls_math)
    math = '\n\n'.join(math_parts).strip()

    parts = [p for p in (cls_rest, mod_rest, mth_rest) if p]
    doc = '\n\n'.join(parts).strip()

    meta['doc'] = doc
    meta['math'] = math
    meta['class'] = class_name or edge.get('address', '')
    meta['role'] = role or ''
    meta['method'] = method_label or ''
    return meta


_STEP_SPECS: list[tuple[int, str, str, str]] = [
    # (tick, step_name, class_name, module_path)
    (1, 'initialize',         'InitializeStep',        'v2ecoli.processes.parca.steps.step_01_initialize'),
    (2, 'input_adjustments',  'InputAdjustmentsStep',  'v2ecoli.processes.parca.steps.step_02_input_adjustments'),
    (3, 'basal_specs',        'BasalSpecsStep',        'v2ecoli.processes.parca.steps.step_03_basal_specs'),
    (4, 'tf_condition_specs', 'TfConditionSpecsStep',  'v2ecoli.processes.parca.steps.step_04_tf_condition_specs'),
    (5, 'fit_condition',      'FitConditionStep',      'v2ecoli.processes.parca.steps.step_05_fit_condition'),
    (6, 'promoter_binding',   'PromoterBindingStep',   'v2ecoli.processes.parca.steps.step_06_promoter_binding'),
    (7, 'adjust_promoters',   'AdjustPromotersStep',   'v2ecoli.processes.parca.steps.step_07_adjust_promoters'),
    (8, 'set_conditions',     'SetConditionsStep',     'v2ecoli.processes.parca.steps.step_08_set_conditions'),
    (9, 'final_adjustments',  'FinalAdjustmentsStep',  'v2ecoli.processes.parca.steps.step_09_final_adjustments'),
]


def _store_label(path: Sequence[str]) -> str:
    """Short human label for a store at ``path``."""
    if not path:
        return '(root)'
    if len(path) == 1:
        return path[0]
    return '/'.join(path)


def build_graph() -> dict:
    """Build the v2ecoli.processes.parca network from static step port manifests.

    Walks each step module's ``INPUT_PORTS`` / ``OUTPUT_PORTS`` + the
    global ``STORE_PATH`` registry, producing nodes for steps and stores
    and edges for every port wire. No Composite instance is required —
    this is a pure static view of the declared data flow.
    """
    import importlib
    from v2ecoli.processes.parca.composite import STORE_PATH

    nodes: dict[str, dict] = {}
    edge_index: dict[tuple[str, str, str], dict] = {}
    layers: list[list[str]] = []

    # One "execution layer" per tick — v2ecoli.processes.parca's steps are strictly serial.
    for tick, step_name, cls_name, mod_path in _STEP_SPECS:
        layers.append([step_name])
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            nodes[step_name] = {'data': {
                'id': step_name, 'label': f'{step_name}\n(import error)',
                'kind': 'process', 'subsystem': 'other', 'color': '#CCCCCC',
                'layer': tick - 1, 'klass': cls_name,
                'meta': {'doc': f'Import failed: {e}'},
            }}
            continue
        cls = getattr(mod, cls_name, None)
        ins  = getattr(mod, 'INPUT_PORTS',  {}) or {}
        outs = getattr(mod, 'OUTPUT_PORTS', {}) or {}

        # Schemas (for the Port-schemas panel): role description per port.
        in_schema  = {p: v for p, v in ins.items()  if not p.startswith('tick_')}
        out_schema = {p: v for p, v in outs.items() if not p.startswith('tick_')}

        subsystem, color = classify(step_name)
        cls_doc = (cls.__doc__ or '').strip() if cls is not None else ''
        mod_doc = (mod.__doc__ or '').strip()
        # Split Mathematical Model section out of module doc so the viewer
        # renders it in the Mathematics panel (colored sub-blocks) rather
        # than lumping it into the general Docstring panel.
        mod_math, mod_rest = _extract_math_and_doc(mod_doc)
        cls_math, cls_rest = _extract_math_and_doc(cls_doc)
        math = '\n\n'.join(p for p in (cls_math, mod_math) if p)
        doc  = '\n\n'.join(p for p in (cls_rest, mod_rest) if p)
        meta = {
            '_inputs':  in_schema,
            '_outputs': out_schema,
            'address':  f'local:{cls_name}',
            'config':   {},
            'doc':      doc,
            'math':     math,
            'class':    f'{mod_path}.{cls_name}',
            'role':     '',
            'method':   '',
        }
        nodes[step_name] = {'data': {
            'id':        step_name,
            'label':     f'Step {tick}\n{step_name}',
            'kind':      'process',
            'subsystem': subsystem,
            'color':     color,
            'layer':     tick - 1,
            'klass':     cls_name,
            'meta':      meta,
        }}

        # Wires: every non-tick port resolved to its store path.
        for direction, port_dict in (('input', ins), ('output', outs)):
            for port, role in port_dict.items():
                if port.startswith('tick_'):
                    continue
                path = STORE_PATH.get(port)
                if not path:
                    continue
                store_id = '/'.join(path)
                parent_id = '/'.join(path[:-1]) if len(path) > 1 else ''
                # Register store node (and parent container if nested).
                if store_id not in nodes:
                    nodes[store_id] = {'data': {
                        'id':            store_id,
                        'label':         _store_label(path),
                        'kind':          'store',
                        'subsystem':     'store',
                        'color':         '#FFFFFF',
                        'parent_store':  parent_id,
                    }}
                if parent_id and parent_id not in nodes:
                    nodes[parent_id] = {'data': {
                        'id':            parent_id,
                        'label':         _store_label(path[:-1]),
                        'kind':          'store',
                        'subsystem':     'store',
                        'color':         '#FFFFFF',
                        'is_container':  True,
                    }}
                # Place-graph edge parent → child (once).
                if parent_id:
                    ckey = (parent_id, store_id, 'contains')
                    if ckey not in edge_index:
                        edge_index[ckey] = {'data': {
                            'id':        f'{parent_id}__{store_id}__contains',
                            'source':    parent_id,
                            'target':    store_id,
                            'direction': 'contains',
                            'ports':     [],
                        }}
                # Data-flow edge store ↔ step.
                src, dst = (store_id, step_name) if direction == 'input' else (step_name, store_id)
                key = (src, dst, direction)
                if key in edge_index:
                    edge_index[key]['data']['ports'].append(port)
                else:
                    edge_index[key] = {'data': {
                        'id':        f'{src}__{dst}__{direction}',
                        'source':    src,
                        'target':    dst,
                        'direction': direction,
                        'ports':     [port],
                    }}

    # Compact edge labels.
    for e in edge_index.values():
        ports = sorted(set(e['data']['ports']))
        e['data']['ports'] = ports
        e['data']['label'] = (
            ', '.join(ports) if len(ports) <= 3
            else f'{ports[0]} (+{len(ports) - 1} more)'
        )

    return {
        'nodes':  list(nodes.values()),
        'edges':  list(edge_index.values()),
        'layers': layers,
        'legend': [
            {'key': k, 'label': lbl, 'color': c}
            for (k, lbl, c, _m) in BIO_COLORS
        ],
    }


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #222; }}
    header {{ padding: 10px 16px; border-bottom: 1px solid #ddd; background: #fff; display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
    header h1 {{ font-size: 16px; margin: 0; }}
    header .subtitle {{ font-size: 12px; color: #666; }}
    .app {{ display: flex; height: calc(100vh - 44px); }}
    aside {{
      width: 300px; flex: 0 0 300px;
      overflow-y: auto; border-right: 1px solid #ddd; background: #fafafa;
      padding: 12px 14px; font-size: 12px;
    }}
    aside h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #555; margin: 16px 0 6px; }}
    aside h2:first-child {{ margin-top: 0; }}
    aside input[type=search] {{
      width: 100%; padding: 6px 8px; font-size: 13px;
      border: 1px solid #bbb; border-radius: 4px;
    }}
    .filter-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; cursor: pointer; user-select: none; }}
    .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #888; flex: 0 0 14px; }}
    .filter-row.disabled .legend-label {{ color: #aaa; text-decoration: line-through; }}
    .filter-row.disabled .legend-swatch {{ opacity: 0.3; }}
    .layer-row {{ padding: 3px 0; border-top: 1px solid #eee; }}
    .layer-row:first-child {{ border-top: none; }}
    .layer-num {{ font-family: 'SF Mono', Menlo, monospace; font-size: 10px; color: #888; }}
    .chip {{
      display: inline-block; padding: 1px 6px; margin: 2px 2px 0 0;
      border-radius: 9px; font-size: 11px; cursor: pointer;
      border: 1px solid rgba(0,0,0,0.15); user-select: none;
    }}
    .chip:hover {{ outline: 1px solid #333; }}
    main {{ flex: 1; position: relative; overflow: hidden; }}
    #cy {{ position: absolute; inset: 0; background: #fff; }}
    .toolbar {{
      position: absolute; top: 10px; right: 10px; z-index: 10;
      background: rgba(255,255,255,0.96); border: 1px solid #ccc;
      border-radius: 6px; padding: 6px; display: flex; gap: 6px; align-items: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 12px;
    }}
    .toolbar label {{ color: #666; }}
    .toolbar select, .toolbar button {{
      border: 1px solid #ccc; background: #fff; padding: 4px 8px;
      font-size: 12px; border-radius: 4px; cursor: pointer;
    }}
    .toolbar button:hover, .toolbar select:hover {{ background: #f0f0f0; }}
    #details {{
      position: absolute; bottom: 10px; left: 10px; z-index: 10;
      width: 420px; max-height: 45vh; overflow-y: auto;
      background: rgba(255,255,255,0.98); border: 1px solid #ccc;
      border-radius: 6px; padding: 10px 12px; font-size: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      display: none;
    }}
    #details.expanded {{
      width: calc(100% - 20px); max-height: calc(100vh - 120px);
      max-width: none;
    }}
    #details .toolbar-mini {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 6px;
    }}
    #details .toolbar-mini button {{
      background: #fff; border: 1px solid #ccc; border-radius: 4px;
      padding: 2px 8px; font-size: 11px; cursor: pointer;
    }}
    #details .toolbar-mini button:hover {{ background: #f0f0f0; }}
    #details h3 {{ margin: 0; font-size: 13px; }}
    #details .meta {{ color: #666; font-size: 11px; margin-bottom: 8px; }}
    #details table {{ width: 100%; border-collapse: collapse; }}
    #details td {{ padding: 2px 6px; border-top: 1px solid #eee; font-size: 11px; vertical-align: top; }}
    #details td:first-child {{ color: #666; white-space: nowrap; }}
    #details code {{ font-size: 11px; background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
    #details details {{ margin-top: 8px; border-top: 1px solid #ddd; padding-top: 6px; }}
    #details details > summary {{
      cursor: pointer; font-weight: 600; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.04em; color: #444;
      list-style: none; padding: 2px 0;
    }}
    #details details > summary::-webkit-details-marker {{ display: none; }}
    #details details > summary::before {{ content: '▸ '; color: #888; }}
    #details details[open] > summary::before {{ content: '▾ '; }}
    #details pre {{
      font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 11px;
      background: #f7f7f7; border: 1px solid #eee; border-radius: 4px;
      padding: 6px 8px; white-space: pre-wrap; word-break: break-word;
      max-height: 280px; overflow: auto; margin: 4px 0;
    }}
    #details.expanded pre {{ max-height: 50vh; }}
    #details .doc {{
      white-space: pre-wrap; font-size: 12px; line-height: 1.4;
      color: #333; background: #fffce8; border-left: 3px solid #e9c46a;
      padding: 6px 10px; margin: 4px 0; border-radius: 0 4px 4px 0;
    }}
    #details .math {{
      font-family: 'STIX Two Math', 'Latin Modern Math', 'Cambria Math', 'SF Mono', Menlo, monospace;
      font-size: 13px; line-height: 1.5;
      white-space: pre-wrap; word-break: break-word;
      color: #222; background: #f1f5ff; border-left: 3px solid #4a6cf7;
      padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0;
    }}
    #details.expanded .math {{ font-size: 14px; }}
    #details .math-block {{ margin: 0 0 8px 0; }}
    #details .math-block:last-child {{ margin-bottom: 0; }}
    #details .math-label {{
      display: inline-block; font-weight: 700; letter-spacing: 0.02em;
      color: #2a3b8f; font-size: 12px; margin-bottom: 2px;
    }}
    #details .math-label.inputs  {{ color: #1f6f9c; }}
    #details .math-label.outputs {{ color: #a6552f; }}
    #details .math-label.params  {{ color: #5b4796; }}
    #details .math-label.calc    {{ color: #2a3b8f; }}
    #details .port-row td:first-child {{ width: 40%; }}
    #details .schema-type {{
      font-family: 'SF Mono', Menlo, monospace; color: #0a5;
      font-size: 11px;
    }}
    .hotkey {{ font-family: 'SF Mono', Menlo, monospace; background: #eee; padding: 1px 4px; border-radius: 3px; color: #444; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
</head>
<body>
  {repro_banner}
  <header>
    <h1>{title}</h1>
    <span class="subtitle">{subtitle}</span>
    <span class="subtitle">· <span class="hotkey">click</span> to focus  <span class="hotkey">esc</span> to clear  <span class="hotkey">drag</span> to move nodes</span>
  </header>
  <div class="app">
    <aside>
      <h2>Search</h2>
      <input type="search" id="q" placeholder="Filter nodes by name…">

      <h2>Subsystems</h2>
      <div id="legend"></div>

      <h2>Execution layers</h2>
      <div id="layers"></div>
    </aside>
    <main>
      <div class="toolbar">
        <label>Layout:</label>
        <select id="layout-select">
          <option value="bipartite" selected>Bipartite (stores ← | → processes)</option>
          <option value="dagre">Dagre (hierarchy)</option>
          <option value="fcose">fCoSE (force)</option>
          <option value="grid">Grid</option>
          <option value="circle">Circle</option>
          <option value="concentric">Concentric</option>
          <option value="breadthfirst">Breadth-first</option>
        </select>
        <button id="btn-fit">Fit</button>
        <button id="btn-reset">Reset</button>
      </div>
      <div id="cy"></div>
      <div id="details"></div>
    </main>
  </div>
  <script>
    const DATA = {data_json};

    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements: {{ nodes: DATA.nodes, edges: DATA.edges }},
      minZoom: 0.1, maxZoom: 5, wheelSensitivity: 0.2,
      style: [
        {{ selector: 'node', style: {{
            'label': 'data(label)',
            'background-color': 'data(color)',
            'border-color': '#333', 'border-width': 1,
            'font-size': '16px',
            'text-valign': 'center', 'text-halign': 'center',
            'text-wrap': 'wrap', 'text-max-width': 120,
            'width': 'label', 'height': 'label',
            'padding': '8px', 'shape': 'rectangle',
        }} }},
        {{ selector: 'node[kind="store"]', style: {{
            'shape': 'ellipse', 'background-color': '#ffffff',
            'border-color': '#666', 'border-width': 2, 'border-style': 'solid',
            'font-weight': 600, 'color': '#333',
            'width': 'label', 'height': 'label',
            'padding': '14px',
        }} }},
        {{ selector: 'edge', style: {{
            'curve-style': 'bezier', 'width': 1.5,
            'line-color': '#bbb', 'target-arrow-shape': 'triangle',
            'target-arrow-color': '#bbb', 'arrow-scale': 0.9, 'opacity': 0.75,
            'line-style': 'dashed',
        }} }},
        {{ selector: 'edge[direction="input"]', style: {{
            'line-color': '#7BA7D9', 'target-arrow-color': '#7BA7D9',
            'line-style': 'dashed',
        }} }},
        {{ selector: 'edge[direction="output"]', style: {{
            'line-color': '#D88A7B', 'target-arrow-color': '#D88A7B',
            'line-style': 'dashed',
        }} }},
        {{ selector: 'edge[direction="contains"]', style: {{
            'line-color': '#666', 'line-style': 'solid',
            'target-arrow-shape': 'none', 'width': 1.5, 'opacity': 0.8,
            'curve-style': 'bezier',
        }} }},
        {{ selector: 'node[?is_container]', style: {{
            'background-color': '#fafafa', 'border-style': 'solid',
            'border-color': '#333', 'border-width': 2.5,
            'font-weight': 700, 'font-size': '18px',
        }} }},
        {{ selector: '.faded', style: {{ 'opacity': 0.1, 'text-opacity': 0.1 }} }},
        {{ selector: '.highlighted', style: {{ 'border-color': '#000', 'border-width': 3 }} }},
        {{ selector: 'edge.highlighted', style: {{ 'width': 3, 'opacity': 1 }} }},
      ],
    }});

    const layoutOpts = {{
      dagre:        {{ name: 'dagre', rankDir: 'LR', nodeSep: 40, rankSep: 120, edgeSep: 20, animate: true }},
      fcose:        {{ name: 'fcose', quality: 'proof', randomize: false, animate: true, idealEdgeLength: 90, nodeRepulsion: 4500 }},
      grid:         {{ name: 'grid', avoidOverlap: true, padding: 24, animate: true }},
      circle:       {{ name: 'circle', padding: 24, animate: true }},
      concentric:   {{ name: 'concentric', padding: 24, animate: true, concentric: n => (n.data('kind') === 'store' ? 1 : 10) }},
      breadthfirst: {{ name: 'breadthfirst', directed: true, padding: 24, animate: true }},
    }};
    function bipartitePositions() {{
      // Stores on left, processes on right. Stores are grouped by parent:
      // a container store (e.g. ``unique``) is followed by its children
      // (e.g. ``unique.promoter``, ``unique.active_replisome``), indented
      // slightly so the place-graph nesting reads as a tree. Processes
      // sort by execution layer then name for a stable tie-break.
      const allStores = cy.nodes().filter(n => n.data('kind') === 'store');
      const parents = allStores.filter(n =>
        !n.data('parent_store') &&
        allStores.some(c => c.data('parent_store') === n.id())
      );
      const orphans = allStores.filter(n => !n.data('parent_store')
        && !parents.some(p => p.id() === n.id()));
      const byParent = new Map();
      allStores.forEach(n => {{
        const p = n.data('parent_store');
        if (p) {{
          if (!byParent.has(p)) byParent.set(p, []);
          byParent.get(p).push(n);
        }}
      }});
      const alpha = (a, b) => a.data('label').localeCompare(b.data('label'));

      // Ordered list: orphans (alpha), then parent + its children (alpha),
      // keeping the place-graph cluster contiguous in the column.
      const ordered = [];
      orphans.sort(alpha).forEach(n => ordered.push({{node: n, depth: 0}}));
      parents.sort(alpha).forEach(p => {{
        ordered.push({{node: p, depth: 0}});
        const kids = (byParent.get(p.id()) || []).slice().sort(alpha);
        kids.forEach(k => ordered.push({{node: k, depth: 1}}));
      }});

      const procs = cy.nodes().filter(n => n.data('kind') !== 'store')
        .sort((a, b) => {{
          const la = a.data('layer'); const lb = b.data('layer');
          if (la !== lb) return (la ?? 99) - (lb ?? 99);
          return a.data('label').localeCompare(b.data('label'));
        }});
      const storeStep = 70, procStep = 60;
      const storeX = 0, procX = 760;
      const positions = new Map();
      // Sub-stores sit directly below their parent (same x). The
      // place-graph edge provides the visible link; no indent needed.
      ordered.forEach(({{node}}, i) => {{
        positions.set(node.id(), {{ x: storeX, y: i * storeStep }});
      }});
      procs.forEach((n, i) => positions.set(n.id(), {{ x: procX, y: i * procStep }}));
      return positions;
    }}
    function applyLayout(name) {{
      if (name === 'bipartite') {{
        const pos = bipartitePositions();
        cy.layout({{
          name: 'preset',
          positions: node => pos.get(node.id()) || {{ x: 0, y: 0 }},
          animate: true, fit: true, padding: 30,
        }}).run();
        return;
      }}
      cy.layout(layoutOpts[name] || layoutOpts.dagre).run();
    }}
    applyLayout('bipartite');

    document.getElementById('layout-select').addEventListener('change', e => applyLayout(e.target.value));
    document.getElementById('btn-fit').onclick = () => cy.fit(null, 24);
    document.getElementById('btn-reset').onclick = () => {{
      cy.elements().removeClass('faded highlighted'); hideDetails();
      applyLayout(document.getElementById('layout-select').value);
    }};

    function focusNode(node) {{
      cy.elements().addClass('faded').removeClass('highlighted');
      const nh = node.closedNeighborhood();
      nh.removeClass('faded').addClass('highlighted');
      showDetails(node);
    }}
    cy.on('tap', 'node', evt => focusNode(evt.target));
    cy.on('tap', evt => {{
      if (evt.target === cy) {{ cy.elements().removeClass('faded highlighted'); hideDetails(); }}
    }});
    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') {{ cy.elements().removeClass('faded highlighted'); hideDetails(); }}
    }});

    const detailsEl = document.getElementById('details');
    let detailsExpanded = false;
    function toggleExpanded() {{
      detailsExpanded = !detailsExpanded;
      detailsEl.classList.toggle('expanded', detailsExpanded);
      const btn = document.getElementById('btn-expand');
      if (btn) btn.textContent = detailsExpanded ? 'Collapse' : 'Expand';
    }}
    // Split a math section into labeled sub-blocks. Recognises headings
    // of the form "Inputs:", "Inputs (...)", "Parameters:", "Calculation:",
    // "Outputs:", "Notes:" at column 0 (left-trimmed). Falls back to a
    // single unlabeled block if no such heading is found.
    function splitMathBlocks(text) {{
      const lines = String(text || '').split('\\n');
      const headingRe = /^(Inputs|Parameters|Calculation|Outputs|Notes)\\b/i;
      const blocks = [];
      let current = {{ label: null, cls: 'calc', body: [] }};
      for (const line of lines) {{
        const m = headingRe.exec(line);
        if (m) {{
          if (current.body.length || current.label) blocks.push(current);
          const key = m[1].toLowerCase();
          const cls = key === 'inputs' ? 'inputs'
                    : key === 'outputs' ? 'outputs'
                    : key === 'parameters' ? 'params'
                    : key === 'notes' ? 'calc'
                    : 'calc';
          current = {{ label: line.trim(), cls, body: [] }};
        }} else {{
          current.body.push(line);
        }}
      }}
      if (current.body.length || current.label) blocks.push(current);
      return blocks;
    }}
    function renderMath(text) {{
      const blocks = splitMathBlocks(text);
      if (!blocks.length) return '';
      // If there's no labeled heading at all, just render one plain block.
      if (blocks.length === 1 && !blocks[0].label) {{
        return `<div class="math">${{escapeHtml(blocks[0].body.join('\\n').trim())}}</div>`;
      }}
      return blocks.map(b => {{
        const body = b.body.join('\\n').replace(/^\\n+|\\n+$/g, '');
        if (!body && !b.label) return '';
        const label = b.label
          ? `<div class="math-label ${{b.cls}}">${{escapeHtml(b.label)}}</div>`
          : '';
        return `<div class="math-block">${{label}}<div class="math">${{escapeHtml(body)}}</div></div>`;
      }}).join('');
    }}
    function fmtSchemaValue(v) {{
      // Compact one-line representation for a schema node.
      if (v === null || v === undefined) return '<code>any</code>';
      if (typeof v === 'string') return `<code class="schema-type">${{escapeHtml(v)}}</code>`;
      if (typeof v !== 'object') return `<code>${{escapeHtml(String(v))}}</code>`;
      if (Array.isArray(v)) {{
        return `<code>[${{v.map(fmtSchemaValue).join(', ')}}]</code>`;
      }}
      if ('_type' in v) {{
        const dflt = ('_default' in v) ? ` = ${{escapeHtml(JSON.stringify(v._default))}}` : '';
        return `<code class="schema-type">${{escapeHtml(String(v._type))}}</code>${{escapeHtml(dflt)}}`;
      }}
      return '<code>{{…}}</code>';
    }}
    function renderSchemaTable(schema, dirColor, label) {{
      if (!schema || typeof schema !== 'object' || !Object.keys(schema).length) return '';
      let rows = `<tr><td colspan=2 style="color:${{dirColor}};font-weight:600">${{label}}</td></tr>`;
      const walk = (obj, prefix) => {{
        for (const [k, v] of Object.entries(obj)) {{
          const key = prefix ? `${{prefix}}.${{k}}` : k;
          if (v && typeof v === 'object' && !('_type' in v) && !Array.isArray(v)) {{
            walk(v, key);
          }} else {{
            rows += `<tr class="port-row"><td><code>${{escapeHtml(key)}}</code></td><td>${{fmtSchemaValue(v)}}</td></tr>`;
          }}
        }}
      }};
      walk(schema, '');
      return rows;
    }}
    function showDetails(node) {{
      const d = node.data();
      const m = d.meta || {{}};
      const incoming = node.incomers('edge').map(e => ({{ peer: e.source().id(), ports: e.data('ports') }}));
      const outgoing = node.outgoers('edge').map(e => ({{ peer: e.target().id(), ports: e.data('ports') }}));

      const meta = [];
      if (d.kind) meta.push(d.kind);
      if (d.subsystem) meta.push(d.subsystem);
      if (d.layer !== undefined && d.layer !== -1) meta.push(`layer L${{String(d.layer).padStart(2,'0')}}`);
      if (d.klass) meta.push(`<code>${{escapeHtml(d.klass)}}</code>`);

      let html = '';
      html += `<div class="toolbar-mini"><h3>${{escapeHtml(d.label)}}</h3>`;
      html += `<div><button id="btn-expand">${{detailsExpanded ? 'Collapse' : 'Expand'}}</button> `;
      html += `<button id="btn-close">×</button></div></div>`;
      html += `<div class="meta">${{meta.join(' · ')}}</div>`;

      if (m.math) {{
        html += `<details open><summary>Mathematics</summary>${{renderMath(m.math)}}</details>`;
      }}
      if (m.doc) {{
        html += `<details ${{m.math ? '' : 'open'}}><summary>Docstring</summary><div class="doc">${{escapeHtml(m.doc)}}</div></details>`;
      }}
      if (m.class) {{
        html += `<details><summary>Class</summary><pre>${{escapeHtml(m.class)}}</pre></details>`;
      }}

      // Neighbor wires — same compact view as before
      html += `<details open><summary>Wires (${{incoming.length + outgoing.length}})</summary><table>`;
      if (incoming.length) {{
        html += `<tr><td colspan=2 style="color:#7BA7D9;font-weight:600">Inputs (${{incoming.length}})</td></tr>`;
        for (const e of incoming) html += `<tr><td>${{escapeHtml(e.peer)}}</td><td>${{escapeHtml(e.ports.join(', '))}}</td></tr>`;
      }}
      if (outgoing.length) {{
        html += `<tr><td colspan=2 style="color:#D88A7B;font-weight:600">Outputs (${{outgoing.length}})</td></tr>`;
        for (const e of outgoing) html += `<tr><td>${{escapeHtml(e.peer)}}</td><td>${{escapeHtml(e.ports.join(', '))}}</td></tr>`;
      }}
      html += `</table></details>`;

      // Port schemas from `_inputs` / `_outputs` collected by build_graph
      const portRows =
        renderSchemaTable(m._inputs, '#7BA7D9', 'Input ports (schema)') +
        renderSchemaTable(m._outputs, '#D88A7B', 'Output ports (schema)');
      if (portRows) {{
        html += `<details><summary>Port schemas</summary><table>${{portRows}}</table></details>`;
      }}

      if (m.config && Object.keys(m.config).length) {{
        html += `<details><summary>Config</summary><pre>${{escapeHtml(JSON.stringify(m.config, null, 2))}}</pre></details>`;
      }}

      detailsEl.innerHTML = html;
      detailsEl.style.display = 'block';
      const btnExp = document.getElementById('btn-expand');
      if (btnExp) btnExp.onclick = toggleExpanded;
      const btnCls = document.getElementById('btn-close');
      if (btnCls) btnCls.onclick = hideDetails;
    }}
    function hideDetails() {{ detailsEl.style.display = 'none'; }}
    function escapeHtml(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]); }}

    const legendEl = document.getElementById('legend');
    const hiddenSubs = new Set();
    DATA.legend.forEach(({{key, label, color}}) => {{
      const row = document.createElement('div');
      row.className = 'filter-row';
      row.dataset.sub = key;
      row.innerHTML = `<span class="legend-swatch" style="background:${{color}}"></span><span class="legend-label">${{escapeHtml(label)}}</span>`;
      row.onclick = () => {{
        if (hiddenSubs.has(key)) {{ hiddenSubs.delete(key); row.classList.remove('disabled'); }}
        else {{ hiddenSubs.add(key); row.classList.add('disabled'); }}
        applyFilter();
      }};
      legendEl.appendChild(row);
    }});
    function applyFilter() {{
      const q = (document.getElementById('q').value || '').trim().toLowerCase();
      cy.batch(() => {{
        cy.nodes().forEach(n => {{
          const d = n.data();
          const hideSub = d.kind === 'process' && hiddenSubs.has(d.subsystem);
          const hideSearch = q && !(d.label || '').toLowerCase().includes(q);
          n.style('display', hideSub || hideSearch ? 'none' : 'element');
        }});
      }});
    }}
    document.getElementById('q').addEventListener('input', applyFilter);

    const layersEl = document.getElementById('layers');
    DATA.layers.forEach((layer, idx) => {{
      const row = document.createElement('div');
      row.className = 'layer-row';
      const num = `L${{String(idx).padStart(2,'0')}}`;
      const chips = layer.map(step => {{
        const node = cy.getElementById(step);
        const color = node.nonempty() ? node.data('color') : '#eee';
        return `<span class="chip" data-step="${{escapeHtml(step)}}" style="background:${{color}}">${{escapeHtml(step)}}</span>`;
      }}).join('');
      row.innerHTML = `<div class="layer-num">${{num}}</div><div>${{chips}}</div>`;
      layersEl.appendChild(row);
    }});
    layersEl.addEventListener('click', e => {{
      const chip = e.target.closest('.chip');
      if (!chip) return;
      const node = cy.getElementById(chip.dataset.step);
      if (node.nonempty()) {{ focusNode(node); cy.animate({{ center: {{ eles: node }}, zoom: 1.5 }}, {{ duration: 400 }}); }}
    }});
  </script>
</body>
</html>
"""


def render_html(data: dict, title: str, subtitle: str) -> str:
    """Return the interactive HTML viewer as a string."""
    import html as _html
    return _HTML_TEMPLATE.format(
        title=_html.escape(title),
        subtitle=_html.escape(subtitle),
        data_json=json.dumps(data),
        repro_banner='',
    )


def write_outputs(
    data: dict,
    out_dir: str | Path,
    name: str,
    title: str,
    subtitle: str,
) -> tuple[Path, Path]:
    """Write ``<name>.json`` and ``<name>.html`` side-by-side, return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f'{name}.json'
    json_path.write_text(json.dumps(data, indent=2))

    html_path = out_dir / f'{name}.html'
    html_path.write_text(render_html(data, title, subtitle))

    return json_path, html_path
