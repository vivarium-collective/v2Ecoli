"""Standalone network viewer for the v2parca composite.

Builds a Cytoscape.js interactive diagram of the 9-Step ParCa pipeline
from its static port manifests (no Composite instance needed), writes
the HTML side-by-side with ``network.json`` under ``docs/``, and opens
it in the default browser.

Usage:
    python scripts/parca_network.py
    python scripts/parca_network.py --output docs/network.html
    python scripts/parca_network.py --no-open
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2ecoli.processes.parca.viz import build_graph, render_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description='v2parca composition-diagram viewer')
    parser.add_argument('--output', default='docs/network.html',
                        help='output HTML path (default: docs/network.html)')
    parser.add_argument('--no-open', action='store_true',
                        help='skip opening the report in a browser')
    args = parser.parse_args()

    print('Extracting v2parca composition graph ...')
    data = build_graph()

    n_proc  = sum(1 for n in data['nodes'] if n['data']['kind'] == 'process')
    n_store = sum(1 for n in data['nodes'] if n['data']['kind'] == 'store')
    n_edges = len(data['edges'])
    title    = 'v2parca · 9-Step ParCa pipeline'
    subtitle = f'{n_proc} steps · {n_store} stores · {n_edges} wires'

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(data, title, subtitle))

    # Write JSON side-by-side for reuse.
    json_path = out_path.with_suffix('.json')
    import json
    json_path.write_text(json.dumps(data, indent=2))

    print(f'Wrote {out_path}')
    print(f'  {subtitle}')

    if not args.no_open:
        subprocess.run(['open', str(out_path)], capture_output=True)


if __name__ == '__main__':
    main()
