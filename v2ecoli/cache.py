"""
JSON cache for v2ecoli simulation states.

Provides save/load for two checkpoint levels:
1. sim_data cache — ParCa output (process configs + initial state parameters)
2. ecoli_wcm cache — fully wired simulation document ready for Composite

Uses bigraph-schema's serialize/realize for numpy array round-tripping.
"""

import os
import gzip
import json
import tempfile

import numpy as np

from v2ecoli.library.schema import MetadataArray
from v2ecoli.library.unit_bridge import unum_to_pint
from v2ecoli.types.quantity import ureg

# Mirrors v2ecoli.workflow.analysis_runner's own local-path-or-s3-URI
# convention (is_s3_uri / localize) — duplicated rather than imported so this
# module (on LineageProcess's per-generation hot path) doesn't pull in
# analysis_runner's heavier deps (duckdb) at import time.
_S3_PREFIX = "s3://"


def is_s3_uri(path: str) -> bool:
    return str(path).startswith(_S3_PREFIX)


# Bounded client config for checkpoint/state S3 I/O. A bare ``boto3.client("s3")``
# applies no effective socket timeout to a stalled connection, so a hung S3 write
# can block indefinitely with no error — observed as a lineage checkpoint upload
# leaving a GovCloud run IDLE for 4+ hours right before the checkpoint was written
# (sms-ecoli#210 / dispatch 313). An explicit connect/read timeout plus bounded
# retries turns a stalled transfer into a loud failure in minutes instead of a
# silent hang. read_timeout is per-request (per multipart part), so a legitimately
# large upload is NOT killed — only a genuinely stuck part is.
_S3_TIMEOUT_CONFIG = None


def _s3_client():
    """A boto3 S3 client with bounded timeouts + retries (see _S3_TIMEOUT_CONFIG)."""
    import boto3
    global _S3_TIMEOUT_CONFIG
    if _S3_TIMEOUT_CONFIG is None:
        from botocore.config import Config
        _S3_TIMEOUT_CONFIG = Config(
            connect_timeout=15,
            read_timeout=60,
            retries={"max_attempts": 5, "mode": "standard"},
        )
    return boto3.client("s3", config=_S3_TIMEOUT_CONFIG)


def _s3_download(uri: str, dest: str) -> None:
    bucket, _, key = uri[len(_S3_PREFIX):].partition("/")
    _s3_client().download_file(bucket, key, dest)


def _s3_upload(local_path: str, uri: str) -> None:
    bucket, _, key = uri[len(_S3_PREFIX):].partition("/")
    _s3_client().upload_file(local_path, bucket, key)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Handle structured arrays specially
            if obj.dtype.names:
                # Serialize dtype preserving sub-array shapes
                dtype_list = []
                for name in obj.dtype.names:
                    field_dtype = obj.dtype[name]
                    if field_dtype.shape:
                        dtype_list.append((name, str(field_dtype.base), list(field_dtype.shape)))
                    else:
                        dtype_list.append((name, str(field_dtype)))
                return {
                    '__numpy_structured__': True,
                    'dtype': dtype_list,
                    'shape': list(obj.shape),
                    'data': [list(row) for row in obj.tolist()],
                }
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'shape': list(obj.shape),
                'data': obj.tolist(),
            }
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return {'__set__': True, 'data': sorted(list(obj))}
        if isinstance(obj, bytes):
            return {'__bytes__': True, 'data': obj.hex()}
        # Both Unum and pint round-trip through pint via the bridge.
        q = unum_to_pint(obj)
        if hasattr(q, 'magnitude') and hasattr(q, 'units'):
            mag = q.magnitude
            if isinstance(mag, np.ndarray):
                return {'__pint_array__': True, 'magnitude': mag.tolist(), 'units': str(q.units)}
            return {'__pint__': True, 'magnitude': float(mag), 'units': str(q.units)}
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'data': list(obj)}
        if isinstance(obj, type):
            return {'__type__': True, 'name': obj.__name__, 'module': obj.__module__}
        if callable(obj) and not isinstance(obj, type):
            name = getattr(obj, '__name__', str(obj))
            module = getattr(obj, '__module__', '')
            return {'__callable__': True, 'name': name, 'module': module}
        if hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
            return {'__object__': True, 'class': type(obj).__name__, 'data': str(obj)[:200]}
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def numpy_json_hook(obj):
    """JSON object hook that reconstructs numpy arrays and other types."""
    if isinstance(obj, dict):
        if obj.get('__numpy_structured__'):
            dtype = np.dtype([tuple(field) for field in obj['dtype']])
            return np.array([tuple(row) for row in obj['data']], dtype=dtype).reshape(obj['shape'])
        if obj.get('__numpy__'):
            return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        if obj.get('__set__'):
            return set(obj['data'])
        if obj.get('__bytes__'):
            return bytes.fromhex(obj['data'])
        if obj.get('__pint__') or obj.get('__pint_array__'):
            mag = obj.get('magnitude', obj.get('value'))
            if obj.get('__pint_array__'):
                mag = np.array(mag)
            return ureg.Quantity(mag, obj['units'])
        if obj.get('__unum__') or obj.get('__unum_array__'):
            # Legacy save format used a dict-repr for the unit, which is not
            # pint-parseable. Match the prior behavior of returning the bare
            # magnitude — these entries were never unit-checked downstream.
            mag = obj.get('value', obj.get('magnitude'))
            return np.array(mag) if obj.get('__unum_array__') else mag
    return obj


def _stringify_keys(obj):
    """Convert non-string dict keys to strings for JSON compatibility."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_stringify_keys(v) for v in obj]
    return obj


def _opener(path, mode):
    """Open `path` with gzip if it ends in .gz, else plain open.

    Text mode is used in both cases so json.dump/json.load see str, not bytes.
    """
    if path.endswith('.gz'):
        return gzip.open(path, mode + 't', encoding='utf-8')
    return open(path, mode)


def save_json(data, path):
    """Save data to JSON with numpy support.

    If `path` ends in .gz, the file is transparently gzipped. JSON state
    files compress ~15x with gzip, so using .json.gz makes large
    bigraph-schema save states cheap to commit and distribute.

    ``path`` may also be an ``s3://`` URI — staged through a local temp file
    then uploaded, mirroring how every other S3 output in this codebase is
    written (a local write followed by an explicit upload/sync, not a
    streaming write; see ``v2ecoli.workflow.analysis_runner.localize`` for
    the read-side precedent this follows).
    """
    data = _stringify_keys(data)
    if is_s3_uri(path):
        suffix = ".json.gz" if path.endswith(".gz") else ".json"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with _opener(tmp_path, 'w') as f:
                json.dump(data, f, cls=NumpyJSONEncoder, indent=1)
            _s3_upload(tmp_path, path)
            print(f"Saved {path} ({os.path.getsize(tmp_path) // 1024}KB)")
        finally:
            os.remove(tmp_path)
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with _opener(path, 'w') as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=1)
    print(f"Saved {path} ({os.path.getsize(path) // 1024}KB)")


def load_json(path):
    """Load data from JSON with numpy reconstruction.

    Transparently gunzips if `path` ends in .gz. ``path`` may also be an
    ``s3://`` URI, downloaded to a local temp file first (mirrors
    ``v2ecoli.workflow.analysis_runner.localize``'s download-then-read
    pattern for the sim_data pickle).
    """
    if is_s3_uri(path):
        suffix = ".json.gz" if path.endswith(".gz") else ".json"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            _s3_download(path, tmp_path)
            with _opener(tmp_path, 'r') as f:
                return json.load(f, object_hook=numpy_json_hook)
        finally:
            os.remove(tmp_path)
    with _opener(path, 'r') as f:
        return json.load(f, object_hook=numpy_json_hook)


def save_initial_state(initial_state, path='out/initial_state.json'):
    """Save the E. coli initial state (bulk, unique, environment, boundary) as JSON."""
    # Convert MetadataArray objects to regular arrays with metadata preserved
    state = {}
    for key, value in initial_state.items():
        if isinstance(value, np.ndarray):
            state[key] = value
        elif isinstance(value, dict):
            sub = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    entry = {'array': v}
                    if hasattr(v, 'metadata'):
                        entry['metadata'] = v.metadata
                    sub[k] = entry
                else:
                    sub[k] = v
            state[key] = sub
        else:
            state[key] = value
    save_json(state, path)


def load_initial_state(path='out/initial_state.json'):
    """Load E. coli initial state from JSON."""
    state = load_json(path)

    # Reconstruct MetadataArray objects for unique molecules
    if 'unique' in state and isinstance(state['unique'], dict):
        unique = {}
        for name, entry in state['unique'].items():
            if isinstance(entry, dict) and 'array' in entry:
                arr = entry['array']
                metadata = entry.get('metadata')
                if isinstance(arr, np.ndarray) and 'unique_index' in arr.dtype.names:
                    unique[name] = MetadataArray(arr, metadata=metadata)
                else:
                    unique[name] = arr
            elif isinstance(entry, np.ndarray):
                unique[name] = entry
            else:
                unique[name] = entry
        state['unique'] = unique

    return state
