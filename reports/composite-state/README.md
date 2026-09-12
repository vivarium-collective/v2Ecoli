# Pre-resolved composite states (dashboard explorer)

Each `<composite-id>.json` here is a **pre-resolved composite state** the
read-only dashboard publish step ships verbatim as
`api/composite-state/<id>.json` and marks navigable (`has_wiring=True`) — even
if the composite can't resolve at publish time.

`vivarium_workbench.publish` checks this directory first; a committed file wins
over live resolution. (See the publish loop in vivarium-workbench's
`publish.py`.)

## Why this exists

Some composites can't resolve in CI/publish because they need on-disk inputs.
The full **`baseline`** composite (the 55-process whole-cell model) calls
`load_cache_bundle("out/cache")`, which needs the ParCa cache — absent at
publish time. Without a committed state it would resolve to `has_wiring=False`
and the dashboard would hide its Explore button.

## Regenerating `v2ecoli.composites.ecoli_baseline.ecoli_baseline.json`

This is a **manual artifact** — refresh it when the baseline model topology
changes. On a machine with a valid ParCa cache (`out/cache/`, e.g. the lab
mini — full ParCa build is ~2.5 min):

```python
# resolve the baseline, prune heavy arrays, serialize via the server serializer
import os, sys, json, math
os.environ.setdefault("POLARS_MAX_THREADS", "1")
from pathlib import Path
WS = Path("/path/to/v2ecoli"); sys.path.insert(0, str(WS))
from vivarium_workbench import server
from vivarium_workbench.lib._root import set_workspace_root
from vivarium_workbench.server import _json_default
server.WORKSPACE = WS; set_workspace_root(WS)

CID = "v2ecoli.composites.ecoli_baseline.ecoli_baseline"
data = server._composite_resolve_data(CID)            # needs out/cache

def prune(o):                                          # truncate heavy arrays; keep topology
    tn = type(o).__name__
    if tn == "ndarray":
        n = int(getattr(o, "size", 0))
        return f"<ndarray shape={tuple(o.shape)} dtype={o.dtype}>" if n > 16 else prune(o.tolist())
    if tn.startswith("float"): v = float(o); return v if math.isfinite(v) else None
    if tn.startswith(("int", "uint")): return int(o)
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, (list, tuple)):
        return ([prune(x) for x in o[:16]] + [f"...(+{len(o)-16} more)"]) if len(o) > 16 else [prune(x) for x in o]
    if isinstance(o, dict): return {k: prune(v) for k, v in o.items()}
    if isinstance(o, str) and len(o) > 400: return o[:400] + "…"
    return o

data["state"] = prune(data.get("state", {}))
Path(f"reports/composite-state/{CID}.json").write_text(
    json.dumps(data, default=_json_default, allow_nan=False), encoding="utf-8")
```

The pruning truncates the multi-MB bulk store and other large arrays (the
explorer only needs the **topology** — process/store nodes + wiring — not the
data values), keeping the committed file a few MB.

## `v2ecoli.composites.biological.biological.json` (historical)

The **`biological`** composite was `baseline` with a pure path-remap of the
store hierarchy (same simulation, just relabeled). The composite module and its
`_remap` helper were removed from the package in #362; the committed state file
here remains as a static viewer artifact and can no longer be regenerated from
this repo. If the baseline state file is refreshed and this one is wanted too,
recover the remap recipe from the pre-#362 history of this README
(`git log -- reports/composite-state/README.md`).
