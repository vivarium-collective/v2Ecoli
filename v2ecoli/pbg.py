"""Process-bigraph document serialization (.pbg files).

Saves composite documents as JSON with step addresses, wiring,
and the full simulation state. Configs are referenced by step name
(loaded from cache at runtime) rather than inlined, keeping file
sizes manageable.
"""

import os
import json
import numpy as np


class PbgEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)


def save_pbg(composite, path):
    """Serialize a Composite's document to a .pbg JSON file.

    Saves each step/process edge with its address and wiring,
    plus the full simulation state (bulk, unique, listeners, etc.).
    Configs are omitted (too large) — they are loaded from cache
    at runtime.
    """
    state = composite.state
    cell = state.get('agents', {}).get('0', {})

    doc = {
        'global_time': state.get('global_time', 0.0),
        'agents': {'0': {}},
    }
    cell_doc = doc['agents']['0']

    for key, val in cell.items():
        if isinstance(val, dict) and '_type' in val:
            # Step/process edge — save address and wiring
            cell_doc[key] = {
                '_type': val.get('_type'),
                'address': val.get('address'),
                'inputs': _clean_wiring(val.get('inputs', {})),
                'outputs': _clean_wiring(val.get('outputs', {})),
            }
        elif isinstance(val, dict):
            # Nested state data — serialize recursively
            cell_doc[key] = _serialize_state(val)
        elif hasattr(val, 'dtype'):
            # Numpy array (bulk, unique molecules)
            cell_doc[key] = _serialize_array(val)
        else:
            cell_doc[key] = val

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(doc, f, cls=PbgEncoder, indent=2)
    return path


def _clean_wiring(wiring):
    """Convert tuple wires to lists for JSON."""
    if isinstance(wiring, dict):
        return {str(k): _clean_wiring(v) for k, v in wiring.items()}
    if isinstance(wiring, tuple):
        return list(wiring)
    if isinstance(wiring, list):
        return [_clean_wiring(x) for x in wiring]
    return wiring


def _serialize_state(state):
    """Recursively serialize state values."""
    if isinstance(state, dict):
        return {str(k): _serialize_state(v) for k, v in state.items()}
    if hasattr(state, 'dtype'):
        return _serialize_array(state)
    if isinstance(state, (list, tuple)):
        return [_serialize_state(x) for x in state]
    return state


def _serialize_array(arr):
    """Serialize a numpy array, handling structured arrays."""
    if hasattr(arr, 'dtype') and arr.dtype.names:
        # Structured array — serialize as dict of columns
        return {
            '__structured_array__': True,
            'dtype': [(name, str(arr.dtype[name])) for name in arr.dtype.names],
            'columns': {name: arr[name].tolist() for name in arr.dtype.names},
            'length': len(arr),
        }
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return arr


def save_pbg_doc(state_dict, path):
    """Save a flat composite state dict (no agents/0 wrapping) as .pbg JSON.

    Use this for composites whose top-level state isn't wrapped in the
    multi-cell ``agents/0`` pattern — e.g. the ParCa pipeline, where each
    Step entry sits directly at the top level alongside data leaves.
    Output mirrors ``save_pbg``: ``_type`` entries become {address, inputs,
    outputs} stubs (configs stripped), nested data dicts are serialized
    recursively, numpy arrays are JSON-friendly.
    """
    doc = {'global_time': 0.0}
    for key, val in state_dict.items():
        if isinstance(val, dict) and '_type' in val:
            doc[key] = {
                '_type': val.get('_type'),
                'address': val.get('address'),
                'inputs': _clean_wiring(val.get('inputs', {})),
                'outputs': _clean_wiring(val.get('outputs', {})),
            }
        elif isinstance(val, dict):
            doc[key] = _serialize_state(val)
        elif hasattr(val, 'dtype'):
            doc[key] = _serialize_array(val)
        else:
            doc[key] = val
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(doc, f, cls=PbgEncoder, indent=2)
    return path
