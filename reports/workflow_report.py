"""
v2ecoli Workflow Testing Framework

Step-based pipeline that caches intermediate states, replacing benchmark_report.py.
Each step checks for cached metadata before executing, enabling incremental
development and fast re-runs.

Pipeline Steps:
0. biocyc — Fetch raw data files from the EcoCyc API
1. raw_data — Catalog raw TSV files and knowledge base stats
2. parca — Run parameter calculator (ParCa) or load cached simData
3. load_model — Build composite from cache
4. single_cell — Run single cell to division
5. division — Cell division, conservation, daughter viability
6. daughters — Divide and run both daughters

Usage:
    python reports/workflow_report.py              # run full pipeline
    python reports/workflow_report.py --clean      # clear cache and re-run

Rendering has been migrated to WorkflowVisualization (v2ecoli/visualizations/workflow.py).
This wrapper keeps all pipeline orchestration and dispatches to the Step for HTML.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dill
import numpy as np

from contextlib import chdir

try:
    from wholecell.utils.filepath import ROOT_PATH as V1_ROOT_PATH
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.library.schema import not_a_process  # noqa: F401  # V1-availability probe
    V1_AVAILABLE = True
except ImportError:
    V1_ROOT_PATH = os.getcwd()
    V1_AVAILABLE = False

from v2ecoli import build_composite
from v2ecoli.core import build_core, save_cache, save_sim_input
from v2ecoli.composites.ecoli_baseline import (
    baseline as _baseline_doc,
    seed_mass_listener,
    build_execution_layers,
    DEFAULT_FEATURES,
    FLOW_ORDER,
)
from process_bigraph import Composite
from v2ecoli.visualizations._helpers import build_graph, write_outputs
from v2ecoli.cache import NumpyJSONEncoder, load_initial_state

try:
    from v2ecoli.library.division import divide_cell, divide_bulk
except ImportError:
    divide_cell = divide_bulk = None

try:
    from bigraph_viz import plot_bigraph
except ImportError:
    plot_bigraph = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mfloat(x, default=0.0):
    """Coerce a listener value to a plain float, stripping pint units.

    Under units-on-ports the mass listeners (dry_mass, cell_mass, …) are
    unit-bearing pint Quantities (e.g. femtogram), so a bare ``float(q)`` raises
    ``DimensionalityError``. Take the magnitude when present; otherwise float
    directly, falling back to ``default`` on non-numeric input.
    """
    if hasattr(x, 'magnitude'):
        try:
            return float(x.magnitude)
        except (TypeError, ValueError):
            return float(default)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKFLOW_DIR = 'out/workflow'
# Use existing cache if available, otherwise workflow-local cache
CACHE_DIR = 'out/cache' if os.path.isdir('out/cache') else 'out/workflow/cache'
LONG_DURATION = 1800.0  # Legacy label
MAX_LONG_DURATION = 3600  # Max seconds before giving up on division
SNAPSHOT_INTERVAL = 50  # Seconds between chromosome snapshots
PLASMID_SNAPSHOT_INTERVAL = 1  # Seconds between plasmid snapshots (when enabled)
DAUGHTER_DURATION = None  # Set to half the single-cell division time at runtime

# Runtime options (overridden by CLI args)
_OPTIONS = {
    'composite_factory': lambda **kw: build_composite("ecoli_baseline", **kw),
    'fetch_biocyc': False,
    'parca_rerun': False,
    'parca_cpus': 4,
    'max_duration': MAX_LONG_DURATION,
}

# Try to find simData.  Priority:
#   1. Previous in-workflow run (out/workflow/simData.cPickle).
#   2. Legacy vEcoli ParCa output (out/kb/simData.cPickle), if present.
# When neither is found, step_parca will materialize the shipped
# ``models/parca/parca_state.pkl.gz`` fixture into
# out/workflow/simData.cPickle, or run the 9-Step pipeline from scratch.
_sim_data_candidates = [
    os.path.join(WORKFLOW_DIR, 'simData.cPickle'),
    'out/kb/simData.cPickle',
]
SIM_DATA_PATH = next((p for p in _sim_data_candidates if os.path.exists(p)), None)


# ---------------------------------------------------------------------------
# Caching Infrastructure
# ---------------------------------------------------------------------------

def load_meta(step_name):
    """Load cached metadata for a step, or None if not cached."""
    path = os.path.join(WORKFLOW_DIR, f'{step_name}_meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_meta(step_name, meta):
    """Save step metadata with timestamp."""
    meta['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = os.path.join(WORKFLOW_DIR, f'{step_name}_meta.json')
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, cls=NumpyJSONEncoder)


def save_state_data(step_name, data):
    """Save step state data as dill pickle."""
    path = os.path.join(WORKFLOW_DIR, f'{step_name}.dill')
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        dill.dump(data, f)


def load_state_data(step_name):
    """Load step state data from dill pickle."""
    path = os.path.join(WORKFLOW_DIR, f'{step_name}.dill')
    with open(path, 'rb') as f:
        return dill.load(f)


# ---------------------------------------------------------------------------
# Step Diagnostics
# ---------------------------------------------------------------------------

def bench_step_diagnostics(composite):
    """Per-step analysis of composite structure."""
    cell = composite.state['agents']['0']

    diagnostics = []
    for step_name in FLOW_ORDER:
        edge = cell.get(step_name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue

        inst = edge['instance']
        proc = getattr(inst, 'process', inst)

        info = {
            'name': step_name,
            'class': type(inst).__name__,
            'inner_class': type(proc).__name__ if proc is not inst else None,
            'has_ports_schema': hasattr(proc, 'ports_schema') and callable(getattr(proc, 'ports_schema')),
            'has_inputs': hasattr(inst, 'inputs') and 'inputs' in type(inst).__dict__,
            'has_config_schema': bool(getattr(inst, 'config_schema', {})),
            'has_defaults': bool(getattr(proc, 'defaults', {})),
            'n_config_keys': len(getattr(proc, 'parameters', {})),
            'priority': edge.get('priority', 0),
        }

        if info['has_ports_schema']:
            try:
                ps = proc.ports_schema()
                info['input_ports'] = sorted(ps.keys())
            except Exception:
                info['input_ports'] = []
        else:
            info['input_ports'] = []

        wires = edge.get('inputs', {})
        info['wires'] = {k: v for k, v in wires.items() if not k.startswith('_flow')}

        diagnostics.append(info)

    return diagnostics


# ---------------------------------------------------------------------------
# v1 Lifecycle Data Collection
# ---------------------------------------------------------------------------

def _collect_v1_lifecycle(duration):
    """Run v1 for the full lifecycle, extracting data from emitted listeners."""
    try:
        if not hasattr(np, 'in1d'):
            np.in1d = np.isin

        import ecoli.experiments.ecoli_master_sim as _ems
        os.environ.setdefault('IMAGE_GIT_HASH', 'v2ecoli')
        if not hasattr(_ems, '_orig_get_git_diff'):
            _ems._orig_get_git_diff = _ems.get_git_diff
            _ems.get_git_diff = lambda: ''

        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]

        sim_data_candidates = [
            'out/kb/simData.cPickle',
            os.path.join(WORKFLOW_DIR, 'simData.cPickle'),
            os.path.join(V1_ROOT_PATH, 'out', 'kb', 'simData.cPickle'),
        ]
        sim_data_path = None
        for p in sim_data_candidates:
            if os.path.exists(p):
                sim_data_path = os.path.abspath(p)
                break
        if sim_data_path is None:
            raise FileNotFoundError(
                f"v1 simData not found. Tried: {sim_data_candidates}")

        with chdir(V1_ROOT_PATH):
            sim = EcoliSim.from_file()
            sim.sim_data_path = sim_data_path
            sim.max_duration = int(duration)
            sim.emitter = 'timeseries'
            sim.divide = False
            sim.build_ecoli()

            t0 = time.time()
            sim.run()
            wall_time = time.time() - t0

        sys.argv = saved_argv
        print(f"    v1 completed in {wall_time:.0f}s")

        v1_ts = sim.query()
        snapshots = []
        for t_key in sorted(v1_ts.keys()):
            if not isinstance(t_key, (int, float)):
                continue
            t = int(t_key)
            if t % SNAPSHOT_INTERVAL != 0 and t != 1:
                continue
            snap = v1_ts[t_key]
            if not isinstance(snap, dict):
                continue

            listeners = snap.get('listeners', {})
            mass = listeners.get('mass', {})
            dry_mass = _mfloat(mass.get('dry_mass', 0)) if isinstance(mass, dict) else 0
            dna_mass = _mfloat(mass.get('dna_mass', 0)) if isinstance(mass, dict) else 0

            umc = listeners.get('unique_molecule_counts', {})
            n_chrom = 0
            if isinstance(umc, dict):
                fc_count = umc.get('full_chromosome', 0)
                n_chrom = int(fc_count) if isinstance(fc_count, (int, float, np.integer)) else 0

            rd = listeners.get('replication_data', {})
            fork_coords = []
            if isinstance(rd, dict):
                fc = rd.get('fork_coordinates')
                if fc is not None:
                    if isinstance(fc, (list, np.ndarray)) and len(fc) > 0:
                        fork_coords = [int(c) for c in fc]

            rnap_data = listeners.get('rnap_data', {})
            n_rnap = 0
            if isinstance(rnap_data, dict):
                rnap_coords = rnap_data.get('active_rnap_coordinates')
                if rnap_coords is not None and hasattr(rnap_coords, '__len__'):
                    n_rnap = len(rnap_coords)

            snapshots.append({
                'time': t,
                'n_chromosomes': n_chrom,
                'fork_coords': fork_coords,
                'n_rnap': n_rnap,
                'dna_mass': dna_mass,
                'dry_mass': dry_mass,
            })

        print(f"    {len(snapshots)} snapshots extracted")
        return {'snapshots': snapshots, 'wall_time': wall_time, 'duration': int(duration)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    v1 lifecycle failed: {e}")
        return {'snapshots': [], 'wall_time': 0, 'duration': int(duration)}


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

BIOCYC_FILE_IDS = [
    "complexation_reactions", "dna_sites", "equilibrium_reactions",
    "genes", "metabolic_reactions", "metabolites", "proteins",
    "rnas", "transcription_units", "trna_charging_reactions",
]

# Flat-file knowledge base — now vendored inside the merged parca
# subpackage so the workflow runs without any vEcoli checkout.
FLAT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'v2ecoli', 'processes', 'parca',
    'reconstruction', 'ecoli', 'flat')


def step_biocyc():
    """Step 0: Fetch raw data files from the EcoCyc API."""
    step_name = 'biocyc'
    meta = load_meta(step_name)
    if meta is not None:
        print("  Step 0: EcoCyc API (cached)")
        return meta

    print("  Step 0: EcoCyc API")
    import requests
    base_url = "https://websvc.biocyc.org/wc-get?type="
    results = {}

    for file_id in BIOCYC_FILE_IDS:
        outpath = os.path.join(FLAT_DIR, file_id + ".tsv")
        print(f"    Fetching {file_id}...", end=" ", flush=True)
        try:
            response = requests.get(base_url + file_id, timeout=30)
            response.raise_for_status()
            with open(outpath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            n_bytes = len(response.text)
            n_lines = response.text.count('\n')
            results[file_id] = {'bytes': n_bytes, 'lines': n_lines, 'status': 'ok'}
            print(f"{n_bytes:,} bytes, {n_lines} lines")
        except Exception as e:
            results[file_id] = {'bytes': 0, 'lines': 0, 'status': str(e)}
            print(f"FAILED: {e}")
        time.sleep(1)

    meta = {
        'n_files': len(BIOCYC_FILE_IDS),
        'files': results,
        'n_fetched': sum(1 for v in results.values() if v['status'] == 'ok'),
    }
    save_meta(step_name, meta)
    print(f"    {meta['n_fetched']}/{meta['n_files']} files fetched")
    return meta


def step_raw_data():
    """Step 1: Catalog raw data files and knowledge base statistics."""
    step_name = 'raw_data'
    meta = load_meta(step_name)
    if meta is not None:
        print("  Step 1: Raw Data (cached)")
        return meta

    print("  Step 1: Raw Data")
    t0 = time.time()

    try:
        from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
            KnowledgeBaseEcoli,
        )
    except ImportError as e:
        print(f"    Skipped (merged ParCa package unavailable: {e})")
        meta = {'skipped': True, 'reason': f'import failed: {e}'}
        save_meta(step_name, meta)
        return meta
    raw_data = KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False,
        remove_rrff=False, stable_rrna=False)

    # Enumerate source files via the SourceBundle (flat/ was deleted in the
    # ecoli-sources migration; all data now comes from the installed package +
    # local flat_overrides/).
    from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
    bundle = SourceBundle()
    n_files = 0
    total_size = 0
    by_subdir: dict = {}
    file_list = []
    for key in sorted(bundle._index):
        path = bundle._index[key]
        sz = path.stat().st_size if path.exists() else 0
        n_files += 1
        total_size += sz
        # Group by top-level key component (e.g. "adjustments" vs "root")
        parts = key.split('__')
        rel = parts[0] if len(parts) > 1 else 'root'
        by_subdir.setdefault(rel, {'count': 0, 'size': 0})
        by_subdir[rel]['count'] += 1
        by_subdir[rel]['size'] += sz
        base = parts[-1]
        is_biocyc = base in BIOCYC_FILE_IDS
        is_modifier = any(base.endswith(s) for s in ('_added', '_removed', '_modified'))
        file_list.append({
            'name': key,
            'size': sz,
            'source': 'biocyc' if is_biocyc else ('modifier' if is_modifier else 'curated'),
        })

    n_genes = len(raw_data.genes) if hasattr(raw_data, 'genes') else 0
    n_rnas = len(raw_data.rnas) if hasattr(raw_data, 'rnas') else 0
    n_proteins = len(raw_data.proteins) if hasattr(raw_data, 'proteins') else 0
    n_metabolites = len(raw_data.metabolites) if hasattr(raw_data, 'metabolites') else 0
    genome_length = len(raw_data.genome_sequence) if hasattr(raw_data, 'genome_sequence') else 0

    elapsed = time.time() - t0

    meta = {
        'n_files': n_files,
        'total_size': total_size,
        'total_size_mb': round(total_size / 1e6, 1),
        'by_subdir': by_subdir,
        'file_list': file_list,
        'n_genes': n_genes,
        'n_rnas': n_rnas,
        'n_proteins': n_proteins,
        'n_metabolites': n_metabolites,
        'genome_length': genome_length,
        'elapsed': elapsed,
    }
    save_meta(step_name, meta)
    print(f"    {n_files} files, {meta['total_size_mb']}MB, "
          f"{n_genes} genes, {n_rnas} RNAs, {n_proteins} proteins, "
          f"{n_metabolites} metabolites, genome={genome_length}bp")
    return meta


def step_parca():
    """Step 2: Run ParCa (parameter calculator) or load cached simData."""
    step_name = 'parca'
    meta = load_meta(step_name)
    if meta is not None:
        print("  Step 2: ParCa (cached)")
        return meta

    print("  Step 2: ParCa")

    sim_data_cache = os.path.join(CACHE_DIR, 'sim_data_cache.dill')
    sim_data_path = SIM_DATA_PATH

    parca_ran = False
    simdata_source = 'unknown'
    sim_data = None
    if os.path.exists(sim_data_cache):
        print(f"    Cache exists at {CACHE_DIR}")
        parca_time = 0.0
        cache_time = 0.0
        if sim_data_path and os.path.normpath(sim_data_path) == os.path.normpath('out/kb/simData.cPickle'):
            simdata_source = 'vecoli_pickle'
        elif sim_data_path and os.path.exists(sim_data_path):
            simdata_source = 'workflow_pickle'
        else:
            simdata_source = 'cache'
        sim_data_path = sim_data_path or '(from cache)'
    elif sim_data_path and os.path.exists(sim_data_path):
        print(f"    Using existing simData at {sim_data_path}")
        parca_time = 0.0
        if os.path.normpath(sim_data_path) == os.path.normpath('out/kb/simData.cPickle'):
            simdata_source = 'vecoli_pickle'
        else:
            simdata_source = 'workflow_pickle'
    else:
        fixture_path = os.path.join('models', 'parca', 'parca_state.pkl.gz')
        sim_data_path = os.path.join(WORKFLOW_DIR, 'simData.cPickle')
        os.makedirs(WORKFLOW_DIR, exist_ok=True)

        if os.path.exists(fixture_path) and not _OPTIONS.get('parca_rerun'):
            print(f"    Using shipped ParCa fixture at {fixture_path}")
            t0 = time.time()
            from v2ecoli.processes.parca.data_loader import (
                hydrate_sim_data_from_state, load_parca_state,
            )
            state = load_parca_state(fixture_path)
            sim_data = hydrate_sim_data_from_state(state)
            with open(sim_data_path, 'wb') as f:
                dill.dump(sim_data, f)
            parca_time = time.time() - t0
            parca_ran = False
            simdata_source = 'parca_fixture'
            print(f"    Fixture hydrated in {parca_time:.1f}s → {sim_data_path}")
        else:
            print("    Running v2ecoli ParCa composite "
                  "(this takes ~70 min in fast mode)...")
            from v2ecoli.processes.parca.composite import build_parca_composite
            from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
                KnowledgeBaseEcoli,
            )
            raw_data = KnowledgeBaseEcoli(
                operons_on=True, remove_rrna_operons=False,
                remove_rrff=False, stable_rrna=False,
            )
            t0 = time.time()
            composite = build_parca_composite(
                raw_data, debug=True,
                cpus=_OPTIONS.get('parca_cpus', 4),
                cache_dir=os.path.join('out', 'cache'),
            )
            from v2ecoli.processes.parca.data_loader import (
                hydrate_sim_data_from_state,
            )
            sim_data = hydrate_sim_data_from_state(composite.state)
            with open(sim_data_path, 'wb') as f:
                dill.dump(sim_data, f)
            parca_time = time.time() - t0
            parca_ran = True
            simdata_source = 'parca_composite'
            print(f"    ParCa composite completed in {parca_time:.1f}s")

    if not os.path.exists(sim_data_cache):
        print("    Generating cache (initial_state.json + sim_data_cache.dill)...")
        t0 = time.time()
        if sim_data is not None:
            save_sim_input(sim_data, CACHE_DIR)
        else:
            save_cache(sim_data_path, CACHE_DIR)
        cache_time = time.time() - t0
        print(f"    Cache generated in {cache_time:.1f}s")
    else:
        cache_time = 0.0
        print("    Cache already exists")

    stats = {}
    try:
        with open(sim_data_cache, 'rb') as f:
            cache = dill.load(f)
        configs = cache.get('configs', {})
        unique_names = cache.get('unique_names', [])
        stats['n_process_configs'] = len(configs)
        stats['process_names'] = sorted(configs.keys())
        stats['n_unique_types'] = len(unique_names)
        stats['unique_types'] = unique_names
        init_state_path = os.path.join(CACHE_DIR, 'initial_state.json')
        if os.path.exists(init_state_path):
            init = load_initial_state(init_state_path)
            bulk = init.get('bulk')
            if bulk is not None and hasattr(bulk, '__len__'):
                stats['n_bulk_molecules'] = len(bulk)
    except Exception as e:
        stats['note'] = f'Could not extract stats: {e}'

    meta = {
        'sim_data_path': sim_data_path,
        'parca_ran': parca_ran,
        'simdata_source': simdata_source,
        'parca_time': parca_time,
        'cache_time': cache_time,
        'cache_dir': CACHE_DIR,
        'stats': stats,
    }
    save_meta(step_name, meta)
    return meta


def step_load_model():
    """Step 3: Build composite from cache."""
    step_name = 'load_model'
    meta = load_meta(step_name)

    print("  Step 3: Load Model", end='')
    t0 = time.time()
    composite = _OPTIONS['composite_factory'](cache_dir=CACHE_DIR)
    build_time = time.time() - t0

    n_steps = len(composite.step_paths)
    n_processes = len(composite.process_paths)

    cell = composite.state['agents']['0']
    bulk = cell.get('bulk')
    n_bulk = len(bulk) if bulk is not None and hasattr(bulk, '__len__') else 0
    unique = cell.get('unique', {})
    n_unique_types = len(unique)
    mass = cell.get('listeners', {}).get('mass', {})
    initial_dry_mass = _mfloat(mass.get('dry_mass', 0))

    if meta is not None:
        print(f" (cached metadata, rebuilt composite in {build_time:.2f}s)")
    else:
        print(f" ({build_time:.2f}s)")

    meta = {
        'build_time': build_time,
        'n_steps': n_steps,
        'n_processes': n_processes,
        'n_bulk': n_bulk,
        'n_unique_types': n_unique_types,
        'initial_dry_mass': initial_dry_mass,
    }
    save_meta(step_name, meta)

    print(f"    {n_steps} steps, {n_processes} processes, "
          f"{n_bulk} bulk molecules, {n_unique_types} unique types, "
          f"dry_mass={initial_dry_mass:.1f}fg")

    return meta, composite


def step_single_cell():
    """Step 4: Run single-cell simulation to division.

    Runs in chunks (SNAPSHOT_INTERVAL seconds each), capturing chromosome
    state at every interval. Stops when division condition is met (2+
    chromosomes AND dry mass >= 2x initial) or MAX_LONG_DURATION is reached.
    Saves pre-division cell state as JSON and .pbg for downstream use.
    """
    step_name = 'single_cell'
    meta = load_meta(step_name)
    if meta is not None:
        print("  Step 2: Single Cell Simulation (cached)")
        return meta

    max_dur = _OPTIONS['max_duration']
    print(f"  Step 2: Single Cell Simulation (to division, max {max_dur}s)")
    composite = _OPTIONS['composite_factory'](cache_dir=CACHE_DIR)

    cell = composite.state['agents']['0']
    bulk_before = np.array(cell['bulk']['count'], copy=True)
    initial_dry_mass = _mfloat(cell.get('listeners', {}).get('mass', {}).get('dry_mass', 380))

    # Plasmid mode: full_plasmid array seeded with active entries means the
    # plasmid-aware cache (out/cache_plasmid) was used. Captured snapshots
    # feed scripts/plasmid_report.py via out/plasmid/timeseries.json.
    has_plasmids = False
    fp = cell.get('unique', {}).get('full_plasmid')
    if fp is not None and hasattr(fp, 'dtype') and '_entryState' in fp.dtype.names:
        has_plasmids = int(fp['_entryState'].view(np.bool_).sum()) > 0
    plasmid_snaps = []
    plasmid_ids = {'trimer': [], 'monomer': [], 'dntp': []}
    plasmid_snapshot_fn = None
    if has_plasmids:
        from scripts.run_plasmid_experiment import snapshot as plasmid_snapshot_fn
        try:
            with open(os.path.join(CACHE_DIR, 'sim_data_cache.dill'), 'rb') as _f:
                _cache = dill.load(_f)
            _cfg = _cache.get('configs', {}).get('ecoli-plasmid-replication', {})
            plasmid_ids['trimer'] = _cfg.get('replisome_trimers_subunits', [])
            plasmid_ids['monomer'] = _cfg.get('replisome_monomers_subunits', [])
            plasmid_ids['dntp'] = _cfg.get('dntps', [])
        except Exception as _e:
            print(f"    plasmid id lookup failed: {_e}")
        plasmid_snaps.append(plasmid_snapshot_fn(
            0, cell, plasmid_ids['trimer'], plasmid_ids['monomer'], plasmid_ids['dntp']))
        print(f"    plasmid mode: t=0 plasmids={plasmid_snaps[0]['n_full_plasmids']}, "
              f"oriV={plasmid_snaps[0]['n_oriV']}")

    em_edge = cell.get('emitter', {})

    t0 = time.time()
    divided = False
    last_cell_data = None
    total_run = 0
    # Per-chunk time-series the report plots afterward (mass, chromosomes, RNAP
    # positions, ppGpp/AA/NTP counts). This replaces the old
    # `emitter_instance.history` read: no currently-supported emitter (parquet,
    # sqlite, xarray, null) keeps every timestep in memory — ParquetEmitter, the
    # default, streams each tick to disk. We snapshot the live state here instead,
    # which works for every emitter type. Copies are taken because the live-state
    # arrays are mutated in place across chunks.
    history_snaps = []

    chunk_interval = PLASMID_SNAPSHOT_INTERVAL if has_plasmids else SNAPSHOT_INTERVAL
    while total_run < max_dur:
        chunk = min(chunk_interval, max_dur - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            total_run += chunk
            err_str = str(e)
            if 'divide' in err_str.lower() or '_add' in err_str or '_remove' in err_str:
                print(f"    Cell divided at ~t={total_run}s ({total_run/60:.0f}min)")
                divided = True
                break
            # The structural divide can complete before an unrelated error
            # fires later in the same composite.run (e.g. realize Array on a
            # daughter's listener store). If the mother agent is gone, the
            # divide already happened — treat as terminal. Still print the
            # traceback so the deeper bug (blocks multigen) is diagnosable.
            if composite.state.get('agents', {}).get('0') is None:
                import traceback
                print(f"    Cell divided at ~t={total_run}s ({total_run/60:.0f}min) "
                      f"({type(e).__name__} during post-add; daughter state discarded)")
                print(f"    post-add error: {err_str[:200]}")
                traceback.print_exc()
                divided = True
                break
            # Non-division error — log full traceback so deeper bugs
            # (e.g. growth_limits realize failure) are diagnosable.
            import traceback
            print(f"    Warning at ~t={total_run}s: {type(e).__name__}: {err_str[:200]}")
            traceback.print_exc()
            continue
        total_run += chunk

        cell = composite.state.get('agents', {}).get('0')
        if cell is None:
            divided = True
            break

        # Plasmid snapshot from live state (process_state.plasmid_rna_control
        # isn't carried in emitter_history, so capture here per chunk).
        if has_plasmids and plasmid_snapshot_fn is not None:
            plasmid_snaps.append(plasmid_snapshot_fn(
                total_run, cell,
                plasmid_ids['trimer'], plasmid_ids['monomer'], plasmid_ids['dntp']))

        _data_keys = {'bulk', 'unique', 'listeners', 'environment', 'boundary',
                      'global_time', 'timestep', 'divide', 'division_threshold',
                      'process_state', 'allocator_rng'}
        last_cell_data = {k: v for k, v in cell.items()
                          if k in _data_keys or k.startswith('request_') or k.startswith('allocate_')}

        unique = cell.get('unique', {})

        # Snapshot the fields the downstream plots read, flattening the unique
        # molecules to the top level exactly as the old emitter history did.
        _snap = {'global_time': total_run}
        _lmass = cell.get('listeners', {}).get('mass', {})
        _snap['listeners'] = {'mass': dict(_lmass) if isinstance(_lmass, dict) else {}}
        _bulk = cell.get('bulk')
        if _bulk is not None and hasattr(_bulk, 'copy'):
            _snap['bulk'] = _bulk.copy()
        for _uk in ('full_chromosome', 'active_replisome', 'active_RNAP',
                    'chromosome_domain'):
            _uv = unique.get(_uk)
            if _uv is not None and hasattr(_uv, 'copy'):
                _snap[_uk] = _uv.copy()
        history_snaps.append(_snap)

        fc = unique.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        mass = cell.get('listeners', {}).get('mass', {})
        dry_mass = _mfloat(mass.get('dry_mass', 0))

        threshold = cell.get('division_threshold', float('inf'))
        if isinstance(threshold, str):
            threshold = float('inf')

        if total_run % 500 == 0:
            rep = unique.get('active_replisome')
            fork_count = 0
            if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
                fork_count = int(rep['_entryState'].view(np.bool_).sum())
            thresh_str = f'{threshold:.0f}fg' if threshold < float('inf') else '?'
            print(f"    t={total_run}s ({total_run/60:.0f}min): {n_chrom} chroms, "
                  f"dry_mass={dry_mass:.0f}/{thresh_str}, forks={fork_count}")

    # Time-series captured live per chunk above (see history_snaps). The previous
    # `emitter_instance.history` assumed an in-memory emitter that no supported
    # emitter provides (ParquetEmitter streams to disk), which crashed report
    # generation with AttributeError after an otherwise-successful run.
    emitter_history = history_snaps

    ppgpp_idx = None
    aa_idxs = {}
    ntp_idxs = {}
    if emitter_history:
        first_bulk = emitter_history[0].get('bulk')
        if first_bulk is not None and hasattr(first_bulk, 'dtype') and 'id' in first_bulk.dtype.names:
            bulk_ids = first_bulk['id']

            def _find_idx(mol_id):
                mask = np.where(bulk_ids == mol_id)[0]
                if len(mask) == 0:
                    mask = np.where(bulk_ids == mol_id.encode())[0]
                return mask[0] if len(mask) else None

            ppgpp_idx = _find_idx('GUANOSINE-5DP-3DP[c]')

            _AA_IDS = [
                'L-ALPHA-ALANINE[c]', 'ARG[c]', 'ASN[c]', 'L-ASPARTATE[c]',
                'CYS[c]', 'GLT[c]', 'GLN[c]', 'GLY[c]', 'HIS[c]', 'ILE[c]',
                'LEU[c]', 'LYS[c]', 'MET[c]', 'PHE[c]', 'PRO[c]', 'SER[c]',
                'THR[c]', 'TRP[c]', 'TYR[c]', 'VAL[c]', 'L-SELENOCYSTEINE[c]',
            ]
            for aa_id in _AA_IDS:
                idx = _find_idx(aa_id)
                if idx is not None:
                    aa_idxs[aa_id] = idx

            for ntp in ['ATP[c]', 'GTP[c]', 'CTP[c]', 'UTP[c]']:
                idx = _find_idx(ntp)
                if idx is not None:
                    ntp_idxs[ntp] = idx

    snapshots = []
    for snap in emitter_history:
        t = snap.get('global_time', 0)
        if int(t) % SNAPSHOT_INTERVAL != 0 and t != 1:
            continue

        mass = snap.get('listeners', {}).get('mass', {}) if isinstance(snap.get('listeners'), dict) else {}

        fc = snap.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        rep = snap.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active_rep = rep[rep['_entryState'].view(np.bool_)]
            if len(active_rep) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active_rep['coordinates'].tolist()

        domains = snap.get('chromosome_domain')
        n_domains = 0
        if domains is not None and hasattr(domains, 'dtype') and '_entryState' in domains.dtype.names:
            n_domains = int(domains['_entryState'].view(np.bool_).sum())

        rnap = snap.get('active_RNAP')
        rnap_coords = []
        n_rnap = 0
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            active_rnap = rnap[rnap['_entryState'].view(np.bool_)]
            n_rnap = len(active_rnap)
            if n_rnap > 0 and 'coordinates' in rnap.dtype.names:
                rnap_coords = active_rnap['coordinates'].tolist()

        ppgpp_count = 0
        aa_counts_dict = {}
        ntp_counts = {}
        bulk = snap.get('bulk')
        if bulk is not None and hasattr(bulk, 'dtype'):
            bc = bulk['count']
            if ppgpp_idx is not None:
                ppgpp_count = int(bc[ppgpp_idx])
            for aa_id, aa_idx in aa_idxs.items():
                aa_counts_dict[aa_id] = int(bc[aa_idx])
            for ntp_name, ntp_idx in ntp_idxs.items():
                ntp_counts[ntp_name] = int(bc[ntp_idx])

        snapshots.append({
            'time': float(t),
            'n_chromosomes': n_chrom,
            'n_domains': n_domains,
            'fork_coords': fork_coords,
            'rnap_coords': rnap_coords,
            'n_rnap': n_rnap,
            'dna_mass': _mfloat(mass.get('dna_mass', 0)),
            'dry_mass': _mfloat(mass.get('dry_mass', 0)),
            'protein_mass': _mfloat(mass.get('protein_mass', 0)),
            'rna_mass': _mfloat(mass.get('rRna_mass', 0)) + _mfloat(mass.get('tRna_mass', 0)) + _mfloat(mass.get('mRna_mass', 0)),
            'rRna_mass': _mfloat(mass.get('rRna_mass', 0)),
            'tRna_mass': _mfloat(mass.get('tRna_mass', 0)),
            'mRna_mass': _mfloat(mass.get('mRna_mass', 0)),
            'smallMolecule_mass': _mfloat(mass.get('smallMolecule_mass', 0)),
            'instantaneous_growth_rate': _mfloat(mass.get('instantaneous_growth_rate', 0)),
            'volume': _mfloat(mass.get('volume', 0)),
            'ppgpp_count': ppgpp_count,
            'aa_counts': aa_counts_dict,
            'ntp_counts': ntp_counts,
        })

    wall_time = time.time() - t0

    agents = composite.state.get('agents', {})
    cell = agents.get('0')
    if cell is None:
        for aid, astate in agents.items():
            if isinstance(astate, dict) and 'bulk' in astate:
                cell = astate
                break

    if last_cell_data and 'bulk' in last_cell_data:
        bulk_after = last_cell_data['bulk']['count']
        changed = int((bulk_before != bulk_after).sum())
    else:
        changed = 0

    final_snap = snapshots[-1] if snapshots else {}

    save_state_data(step_name, {
        'cell_state': last_cell_data,
        'global_time': final_snap.get('time', 0.0),
    })

    pre_div_dir = os.path.join(WORKFLOW_DIR, 'pre_division')
    os.makedirs(pre_div_dir, exist_ok=True)
    if last_cell_data and 'bulk' in last_cell_data:
        from v2ecoli.cache import save_initial_state
        save_initial_state(last_cell_data, os.path.join(pre_div_dir, 'pre_division_state.json.gz'))
        print(f"    Saved pre-division state: {pre_div_dir}/pre_division_state.json.gz")

    try:
        from v2ecoli.pbg import save_pbg
        pbg_path = os.path.join(pre_div_dir, 'pre_division.pbg')
        save_pbg(composite, pbg_path)
        print(f"    Saved pre-division .pbg: {pbg_path}")
    except Exception as e:
        print(f"    Warning: could not save .pbg: {e}")

    # Plasmid timeseries — written in the schema scripts/plasmid_report.py
    # consumes, so the same renderer works for workflow + standalone runs.
    plasmid_timeseries_path = None
    if has_plasmids and plasmid_snaps:
        plasmid_timeseries_path = 'out/plasmid/timeseries.json'
        os.makedirs(os.path.dirname(plasmid_timeseries_path), exist_ok=True)
        with open(plasmid_timeseries_path, 'w', encoding='utf-8') as _f:
            json.dump({
                'duration': total_run,
                'interval': PLASMID_SNAPSHOT_INTERVAL,
                'wall_time': wall_time,
                'trimer_subunit_ids': plasmid_ids['trimer'],
                'monomer_subunit_ids': plasmid_ids['monomer'],
                'dntp_ids': plasmid_ids['dntp'],
                'snapshots': plasmid_snaps,
            }, _f, indent=2, cls=NumpyJSONEncoder)
        print(f"    wrote {plasmid_timeseries_path} ({len(plasmid_snaps)} snapshots)")

    meta = {
        'duration': total_run,
        'wall_time': wall_time,
        'bulk_changed': changed,
        'total_bulk': len(bulk_before),
        'final_dry_mass': final_snap.get('dry_mass', 0),
        'final_cell_mass': 0,
        'final_volume': 0,
        'rate': total_run / wall_time if wall_time > 0 else 0,
        'division_reached': divided,
        'initial_dry_mass': initial_dry_mass,
        'chromosome_snapshots': snapshots,
        'pre_division_dir': pre_div_dir,
        'has_plasmids': has_plasmids,
        'plasmid_timeseries_path': plasmid_timeseries_path,
    }
    save_meta(step_name, meta)

    print(f"    {wall_time:.0f}s wall, {changed} molecules changed, "
          f"{meta['rate']:.1f}x realtime")
    print(f"    dry_mass={meta['final_dry_mass']:.0f}fg, "
          f"division={'reached' if divided else 'not reached'}")
    return meta


def step_v1_comparison():
    """Step 4b: Run v1 lifecycle comparison (cached independently)."""
    step_name = 'v1_comparison'
    meta = load_meta(step_name)
    if meta is not None:
        n = len(meta.get('v1_snapshots', []))
        print(f"  Step 4b: v1 Comparison (cached, {n} snapshots)")
        return meta

    single_cell_meta = load_meta('single_cell')
    if single_cell_meta is None:
        print("  Step 4b: v1 Comparison (skipped, no long sim data)")
        meta = {'skipped': True, 'reason': 'no long sim data', 'v1_snapshots': []}
        save_meta(step_name, meta)
        return meta

    duration = single_cell_meta.get('duration', 0)

    v1_cache_path = os.path.join(WORKFLOW_DIR, f'v1_lifecycle_{duration}s.json')
    v1_snapshots = []
    v1_wall_time = 0

    if os.path.exists(v1_cache_path):
        with open(v1_cache_path) as f:
            cached = json.load(f)
        if isinstance(cached, list):
            v1_snapshots = cached
        elif isinstance(cached, dict):
            v1_snapshots = cached.get('snapshots', [])
            v1_wall_time = cached.get('wall_time', 0)
        print(f"  Step 4b: v1 Comparison (loaded {len(v1_snapshots)} cached snapshots)")
    elif V1_AVAILABLE:
        print(f"  Step 4b: Running v1 for {duration}s...")
        try:
            result = _collect_v1_lifecycle(duration)
            if isinstance(result, dict):
                v1_snapshots = result.get('snapshots', [])
                v1_wall_time = result.get('wall_time', 0)
            else:
                v1_snapshots = result
            if v1_snapshots:
                with open(v1_cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'snapshots': v1_snapshots, 'wall_time': v1_wall_time,
                               'duration': duration}, f)
        except Exception as e:
            print(f"    v1 failed: {e}")
    else:
        print("  Step 4b: v1 Comparison (v1 not available)")

    meta = {
        'duration': duration,
        'v1_available': len(v1_snapshots) > 0,
        'v1_snapshots': v1_snapshots,
        'v1_wall_time': v1_wall_time,
        'n_snapshots': len(v1_snapshots),
    }
    save_meta(step_name, meta)
    return meta


def step_division():
    """Step 5: Test cell division, conservation, and daughter viability."""
    step_name = 'division'
    meta = load_meta(step_name)
    if meta is not None:
        print("  Step 3: Division (cached)")
        return meta

    print("  Step 3: Division")

    prediv_state = None
    prediv_time = 0.0
    long_state_path = os.path.join(WORKFLOW_DIR, 'single_cell.dill')
    if os.path.exists(long_state_path):
        try:
            with open(long_state_path, 'rb') as f:
                checkpoint = dill.load(f)
            cell_data = checkpoint.get('cell_state', {})
            if cell_data.get('unique', {}).get('full_chromosome') is not None:
                prediv_state = cell_data
                prediv_time = checkpoint.get('global_time', 0.0)
                print(f"    Using pre-division state from long sim (t={prediv_time:.0f})")
        except Exception as e:
            print(f"    Could not load long sim state: {e}")

    if prediv_state is None:
        old_prediv = 'out/predivision.dill'
        if os.path.exists(old_prediv):
            try:
                with open(old_prediv, 'rb') as f:
                    checkpoint = dill.load(f)
                cell_data = checkpoint.get('cell_state', {})
                if cell_data.get('unique', {}).get('full_chromosome') is not None:
                    prediv_state = cell_data
                    prediv_time = checkpoint.get('global_time', 0.0)
                    print(f"    Using pre-division checkpoint (t={prediv_time:.0f})")
            except Exception as e:
                print(f"    Could not load pre-division state: {e}")

    if prediv_state is not None:
        cell = prediv_state
    else:
        print("    No pre-division checkpoint -- using initial state (t=0)")
        composite = _OPTIONS['composite_factory'](cache_dir=CACHE_DIR)
        cell = composite.state['agents']['0']

    d1_bulk, d2_bulk = divide_bulk(cell['bulk'])
    mother_count = int(cell['bulk']['count'].sum())
    d1_count = int(d1_bulk['count'].sum())
    d2_count = int(d2_bulk['count'].sum())
    conserved = (d1_count + d2_count == mother_count)

    t0 = time.time()
    d1_state, d2_state = divide_cell(cell)
    split_time = time.time() - t0

    unique_conservation = {}
    for name in d1_state.get('unique', {}):
        d1_arr = d1_state['unique'][name]
        d2_arr = d2_state['unique'][name]
        mother_arr = cell['unique'][name]
        if hasattr(d1_arr, 'shape') and hasattr(mother_arr, 'dtype'):
            if '_entryState' in mother_arr.dtype.names:
                m = int(mother_arr['_entryState'].view(np.bool_).sum())
                d1 = d1_arr.shape[0]
                d2 = d2_arr.shape[0]
                unique_conservation[name] = {
                    'mother': m, 'd1': d1, 'd2': d2,
                    'conserved': d1 + d2 == m
                }

    div_step = cell.get('division', {}).get('instance')
    configs = getattr(div_step, '_configs', None)
    unique_names = getattr(div_step, '_unique_names', None)
    dry_mass_inc = getattr(div_step, 'dry_mass_inc_dict', None)

    if configs is None and os.path.isdir(CACHE_DIR):
        cache_path = os.path.join(CACHE_DIR, 'sim_data_cache.dill')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = dill.load(f)
            configs = cache.get('configs', {})

    can_build_daughters = configs is not None and bool(configs)
    daughter_build_time = 0
    daughter_viable = False

    if can_build_daughters:
        t0 = time.time()
        try:
            _core = build_core()
            d1_doc = _baseline_doc(core=_core, seed=1, cache_dir=CACHE_DIR)
            _agent = d1_doc['state']['agents']['0']
            for _key in ('bulk', 'unique', 'environment', 'boundary'):
                if _key in d1_state:
                    _agent[_key] = d1_state[_key]
            _agent['listeners']['mass'] = {'dry_mass': 0.0, 'cell_mass': 0.0}
            seed_mass_listener(_agent, _core)
            daughter_build_time = time.time() - t0
            d1_composite = Composite(d1_doc, core=_core)
            d1_composite.run(1.0)
            daughter_viable = True
        except Exception as e:
            daughter_build_time = time.time() - t0
            print(f"    Daughter build error: {e}")

    fc = cell.get('unique', {}).get('full_chromosome')
    n_chromosomes = 0
    if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
        n_chromosomes = int(fc['_entryState'].view(np.bool_).sum())
    mass = cell.get('listeners', {}).get('mass', {})
    dry_mass = _mfloat(mass.get('dry_mass', 0))

    meta = {
        'prediv_time': prediv_time,
        'dry_mass': dry_mass,
        'n_chromosomes': n_chromosomes,
        'mother_bulk_count': mother_count,
        'd1_bulk_count': d1_count,
        'd2_bulk_count': d2_count,
        'bulk_conserved': conserved,
        'split_time': split_time,
        'unique_conservation': unique_conservation,
        'can_build_daughters': can_build_daughters,
        'daughter_build_time': daughter_build_time,
        'daughter_viable': daughter_viable,
    }
    save_meta(step_name, meta)

    print(f"    Bulk conserved: {conserved}, split: {split_time*1000:.0f}ms")
    print(f"    Daughters buildable: {can_build_daughters}, viable: {daughter_viable}")
    if daughter_build_time > 0:
        print(f"    Daughter build: {daughter_build_time:.1f}s")

    return meta


def _extract_snapshots_from_emitter(composite, label=''):
    """Extract snapshot data from a composite's emitter history."""
    cell = composite.state['agents']['0']
    em_edge = cell.get('emitter')
    _inst = em_edge.get('instance') if isinstance(em_edge, dict) else None
    history = getattr(_inst, 'history', None) or []
    if not history:
        # No supported emitter keeps an in-memory .history (ParquetEmitter, the
        # default, streams to disk) — degrade to a single snapshot from the live
        # final state so callers that only need final values (daughter dry_mass)
        # work instead of crashing on a missing attribute.
        _lmass = cell.get('listeners', {}).get('mass', {})
        history = [{
            'global_time': 0,
            'listeners': {'mass': dict(_lmass) if isinstance(_lmass, dict) else {}},
            'bulk': cell.get('bulk'),
            'full_chromosome': cell.get('unique', {}).get('full_chromosome'),
            'active_replisome': cell.get('unique', {}).get('active_replisome'),
        }]

    snapshots = []
    for snap in history:
        t = snap.get('global_time', 0)
        if int(t) % SNAPSHOT_INTERVAL != 0 and t != 1:
            continue

        mass = snap.get('listeners', {}).get('mass', {}) if isinstance(snap.get('listeners'), dict) else {}

        fc = snap.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        rep = snap.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active_rep = rep[rep['_entryState'].view(np.bool_)]
            if len(active_rep) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active_rep['coordinates'].tolist()

        domains = snap.get('chromosome_domain')
        n_domains = 0
        if domains is not None and hasattr(domains, 'dtype') and '_entryState' in domains.dtype.names:
            n_domains = int(domains['_entryState'].view(np.bool_).sum())

        rnap = snap.get('active_RNAP')
        rnap_coords = []
        n_rnap = 0
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            active_rnap = rnap[rnap['_entryState'].view(np.bool_)]
            n_rnap = len(active_rnap)
            if n_rnap > 0 and 'coordinates' in rnap.dtype.names:
                rnap_coords = active_rnap['coordinates'].tolist()

        snapshots.append({
            'time': float(t),
            'n_chromosomes': n_chrom,
            'n_domains': n_domains,
            'fork_coords': fork_coords,
            'rnap_coords': rnap_coords,
            'n_rnap': n_rnap,
            'dna_mass': _mfloat(mass.get('dna_mass', 0)),
            'dry_mass': _mfloat(mass.get('dry_mass', 0)),
            'protein_mass': _mfloat(mass.get('protein_mass', 0)),
            'rna_mass': _mfloat(mass.get('rRna_mass', 0)) + _mfloat(mass.get('tRna_mass', 0)) + _mfloat(mass.get('mRna_mass', 0)),
            'rRna_mass': _mfloat(mass.get('rRna_mass', 0)),
            'tRna_mass': _mfloat(mass.get('tRna_mass', 0)),
            'mRna_mass': _mfloat(mass.get('mRna_mass', 0)),
            'smallMolecule_mass': _mfloat(mass.get('smallMolecule_mass', 0)),
        })

    return snapshots


def step_daughters():
    """Step 6: Divide pre-division cell into 2 daughters, run both for half a generation."""
    step_name = 'daughters'
    meta = load_meta(step_name)
    if meta is not None:
        print("  Step 4: Daughter Simulations (cached)")
        return meta

    single_cell_meta = load_meta('single_cell') or {}
    generation_time = single_cell_meta.get('duration', 2500)
    default_dur = int(generation_time / 2)
    daughter_dur = _OPTIONS.get('daughter_duration', default_dur)
    print(f"  Step 4: Daughter Simulations ({daughter_dur}s per daughter, "
          f"half generation = {default_dur}s)")

    long_state_path = os.path.join(WORKFLOW_DIR, 'single_cell.dill')
    if not os.path.exists(long_state_path):
        print("    No pre-division state available — skipping")
        meta = {'skipped': True, 'reason': 'no pre-division state'}
        save_meta(step_name, meta)
        return meta

    with open(long_state_path, 'rb') as f:
        checkpoint = dill.load(f)
    cell_data = checkpoint.get('cell_state', {})

    if not cell_data or 'bulk' not in cell_data:
        print("    No valid cell state (mother did not divide) — skipping")
        meta = {'skipped': True, 'reason': 'no valid pre-division cell state'}
        save_meta(step_name, meta)
        return meta

    print("    Dividing mother cell...")
    d1_state, d2_state = divide_cell(cell_data)

    def _run_daughter(label, dstate, seed):
        """Build and run a single daughter composite, return results dict."""
        t0 = time.time()
        _core = build_core()
        doc = _baseline_doc(core=_core, seed=seed, cache_dir=CACHE_DIR)
        _agent = doc['state']['agents']['0']
        for _key in ('bulk', 'unique', 'environment', 'boundary'):
            if _key in dstate:
                _agent[_key] = dstate[_key]
        _agent['listeners']['mass'] = {'dry_mass': 0.0, 'cell_mass': 0.0}
        seed_mass_listener(_agent, _core)
        comp = Composite(doc, core=_core)
        build_time = time.time() - t0

        d_cell = comp.state['agents']['0']
        d_mass = d_cell.get('listeners', {}).get('mass', {})
        initial_dry = _mfloat(d_mass.get('dry_mass', 0))

        t0 = time.time()
        try:
            comp.run(daughter_dur)
            run_ok = True
        except Exception:
            run_ok = False
        wall_time = time.time() - t0

        snaps = _extract_snapshots_from_emitter(comp, label)
        final_snap = snaps[-1] if snaps else {}
        final_dry = final_snap.get('dry_mass', 0)

        if final_dry == 0:
            d_cell_post = comp.state.get('agents', {}).get('0', {})
            d_mass_post = d_cell_post.get('listeners', {}).get('mass', {})
            final_dry = _mfloat(d_mass_post.get('dry_mass', 0))

        return {
            'build_time': build_time,
            'wall_time': wall_time,
            'run_ok': run_ok,
            'initial_dry_mass': initial_dry,
            'final_dry_mass': final_dry,
            'fold_change': final_dry / initial_dry if initial_dry > 0 else 0,
            'n_snapshots': len(snaps),
            'snapshots': snaps,
        }

    daughters = {}
    for label, dstate, seed in [('daughter_1', d1_state, 1), ('daughter_2', d2_state, 2)]:
        print(f"    Building {label}...")
        d = _run_daughter(label, dstate, seed)
        daughters[label] = d
        if d['initial_dry_mass'] > 0:
            print(f"    {label}: {d['wall_time']:.0f}s wall, "
                  f"dry_mass {d['initial_dry_mass']:.0f} -> {d['final_dry_mass']:.0f}fg "
                  f"({d['fold_change']:.2f}x)")

    meta = {
        'duration': daughter_dur,
        'daughter_1': {k: v for k, v in daughters.get('daughter_1', {}).items() if k != 'snapshots'},
        'daughter_2': {k: v for k, v in daughters.get('daughter_2', {}).items() if k != 'snapshots'},
        'daughter_1_snapshots': daughters.get('daughter_1', {}).get('snapshots', []),
        'daughter_2_snapshots': daughters.get('daughter_2', {}).get('snapshots', []),
    }
    save_meta(step_name, meta)
    return meta


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def _load_study(study_path):
    """Load + lightly validate a v3-shape study.yaml driving this report.

    workflow_report is single-cell + single-architecture (baseline) by
    design, so the spec is checked against the narrow shape that fits
    that intent::

        baseline:
        - name: <any>
          composite: v2ecoli.composites.ecoli_baseline.ecoli_baseline
          params: {seed?, cache_dir?, duration?, daughter_duration?}
        visualizations:
        - name: <any>
          address: local:WorkflowVisualization
          config: {title: str, ...}

    Reject:
      - more than one ``baseline`` entry (workflow is not multi-arch),
      - a composite ref other than ``...baseline.baseline``,
      - a ``lineage`` block (workflow is one generation; use
        multigeneration_report for chained generations).
    """
    import yaml as _yaml
    with open(study_path) as fh:
        spec = _yaml.safe_load(fh) or {}

    baseline_entries = spec.get("baseline") or []
    if len(baseline_entries) != 1:
        raise ValueError(
            f"study {study_path!r}: workflow_report needs exactly one "
            f"`baseline:` entry; got {len(baseline_entries)}"
        )
    composite_ref = baseline_entries[0].get("composite") or ""
    if not composite_ref.endswith(".baseline.baseline"):
        raise ValueError(
            f"study {study_path!r}: workflow_report only handles the "
            f"partitioned `baseline` architecture (e.g. "
            f"v2ecoli.composites.ecoli_baseline.ecoli_baseline); got {composite_ref!r}"
        )
    if spec.get("lineage"):
        raise ValueError(
            f"study {study_path!r}: workflow_report is single-generation; "
            f"the `lineage:` block belongs to multigeneration_report"
        )
    return spec


def run_workflow(out_path=None, viz_config=None):
    """Execute the full workflow pipeline with caching.

    ``viz_config`` (when provided) overrides the WorkflowVisualization
    config dict used at the report-render step. Defaults to
    ``{"title": "v2ecoli Simulation Report"}``.
    """
    os.makedirs(WORKFLOW_DIR, exist_ok=True)

    print("=" * 60)
    print("v2ecoli Workflow Pipeline")
    print("=" * 60)
    pipeline_t0 = time.time()

    step_results = {}

    # Step 0: EcoCyc API (skipped by default, use --fetch-biocyc to enable)
    if _OPTIONS['fetch_biocyc']:
        biocyc_meta = step_biocyc()
    else:
        biocyc_meta = load_meta('biocyc') or {'skipped': True, 'files': {}}
        print("  Step 0: EcoCyc API (skipped, use --fetch-biocyc to refresh)")
    step_results['biocyc'] = biocyc_meta

    # Step 1: Raw Data
    raw_meta = step_raw_data()
    step_results['raw_data'] = raw_meta

    # Step 2: ParCa
    parca_meta = step_parca()
    step_results['parca'] = parca_meta

    # Step 3: Load Model (always builds composite for later steps)
    model_meta, composite = step_load_model()
    step_results['load_model'] = model_meta

    # Step 2: Single Cell Simulation (to division)
    single_cell_meta = step_single_cell()
    step_results['single_cell'] = single_cell_meta

    # Step 3: Division
    div_meta = step_division()
    step_results['division'] = div_meta

    # Step 4: Daughter Simulations (skip with --no-daughters)
    if _OPTIONS.get('skip_daughters'):
        daughters_meta = load_meta('daughters') or {'skipped': True, 'reason': '--no-daughters'}
        print("  Step 4: Daughter Simulations (skipped)")
    else:
        daughters_meta = step_daughters()
    step_results['daughters'] = daughters_meta

    # Step Diagnostics (always run, uses the composite from step 3)
    print("  Diagnostics: Step analysis")
    diag_composite = _OPTIONS['composite_factory'](cache_dir=CACHE_DIR)
    diagnostics = bench_step_diagnostics(diag_composite)
    print(f"    {len(diagnostics)} steps analyzed")

    # Update .pbg model files
    print("  Updating .pbg model files...")
    from v2ecoli.pbg import save_pbg, save_pbg_doc
    from v2ecoli.processes.parca.composite import build_parca_document
    os.makedirs('models', exist_ok=True)
    save_pbg(diag_composite, 'models/partitioned.pbg')
    print("    models/partitioned.pbg updated")
    save_pbg_doc(build_parca_document(), 'models/parca.pbg')
    print("    models/parca.pbg updated")

    # Network Visualization (Cytoscape.js interactive viewer)
    print("  Generating interactive network visualization...")
    network_data = build_graph(diag_composite, build_execution_layers(DEFAULT_FEATURES))
    _, network_html_path = write_outputs(
        network_data,
        out_dir=WORKFLOW_DIR,
        name='network',
        title='v2ecoli Baseline Composite',
        subtitle='Interactive Cytoscape.js view (built from the workflow composite)',
    )
    network_html_rel = os.path.relpath(network_html_path, WORKFLOW_DIR)

    # ParCa Composition Diagram (the upstream 9-Step ParCa pipeline).
    parca_network_html_rel = None
    try:
        from v2ecoli.processes.parca.viz import (
            build_graph as build_parca_graph,
            write_outputs as write_parca_outputs,
        )
        parca_graph = build_parca_graph()
        _, parca_network_html_path = write_parca_outputs(
            parca_graph,
            out_dir=WORKFLOW_DIR,
            name='parca_network',
            title='ParCa Composition Diagram',
            subtitle='Interactive Cytoscape.js view of the 9-Step ParCa pipeline',
        )
        parca_network_html_rel = os.path.relpath(
            parca_network_html_path, WORKFLOW_DIR)
    except Exception as e:
        print(f"    Skipped ParCa composition diagram: {e}")

    # Collect trajectory for rendering
    chrom_snaps = single_cell_meta.get('chromosome_snapshots', [])

    # Dispatch to WorkflowVisualization Step for HTML rendering
    print("  Generating HTML report (via WorkflowVisualization)...")
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.workflow import WorkflowVisualization

    # Pass network links through metadata for wrapper-level insertion
    step_results['_network_html_rel'] = network_html_rel
    if parca_network_html_rel:
        step_results['_parca_network_html_rel'] = parca_network_html_rel

    viz = WorkflowVisualization(
        config=viz_config or {"title": "v2ecoli Simulation Report"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": chrom_snaps,
        "metadata": step_results,
    })
    html_content = result["html"]

    network_section = f"""
<!-- ===== Process-Bigraph Network (Cytoscape.js viewer) ===== -->
<h2 id="sec-network">Process-Bigraph Network</h2>
<div style="background:white;border-radius:8px;padding:15px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.08);">
  <p>Interactive Cytoscape.js viewer of the composite — stores on the left, processes on the right,
  sorted by execution layer.
  Full-screen viewer: <a href="{network_html_rel}" target="_blank"><code>{network_html_rel}</code></a>.</p>
</div>
<iframe src="{network_html_rel}" style="width:100%;height:900px;border:1px solid #e2e8f0;border-radius:6px;"></iframe>
"""
    if parca_network_html_rel:
        network_section += f"""
<!-- ===== ParCa Composition Diagram ===== -->
<h2 id="sec-parca-network">ParCa Composition Diagram</h2>
<div style="background:white;border-radius:8px;padding:15px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.08);">
  <p>Interactive Cytoscape.js view of the upstream <strong>ParCa</strong> 9-Step pipeline.
  Full-screen: <a href="{parca_network_html_rel}" target="_blank"><code>{parca_network_html_rel}</code></a>
  &middot; <a href="parca_workflow_report.html">ParCa Workflow Report &rarr;</a></p>
</div>
<iframe src="{parca_network_html_rel}" style="width:100%;height:750px;border:1px solid #e2e8f0;border-radius:6px;"></iframe>
"""

    html_content = html_content.replace('</body>', network_section + '\n</body>', 1)

    report_path = out_path or os.path.join(WORKFLOW_DIR, 'workflow_report.html')
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Plasmid report (only if simulation was plasmid-enabled). Renders via
    # scripts/plasmid_report.py — separate from the main WorkflowVisualization.
    plasmid_report_path = None
    if single_cell_meta.get('has_plasmids'):
        print("  Generating plasmid replication report...")
        try:
            from scripts.plasmid_report import main as plasmid_report_main, OUT as PLASMID_OUT
            plasmid_report_main()
            plasmid_report_path = PLASMID_OUT
        except Exception as e:
            print(f"    plasmid report failed: {type(e).__name__}: {e}")

    pipeline_time = time.time() - pipeline_t0
    print("=" * 60)
    print(f"Pipeline complete in {pipeline_time:.0f}s")
    print(f"Report: {report_path}")
    if plasmid_report_path:
        print(f"Plasmid report: {plasmid_report_path}")
    print(f"Cache:  {WORKFLOW_DIR}/")

    if _OPTIONS.get('open_reports'):
        import subprocess
        for p in [report_path, plasmid_report_path]:
            if p and os.path.exists(p):
                subprocess.run(['open', p], check=False)

    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='v2ecoli workflow pipeline')
    parser.add_argument('--study', default=None,
                        help='Path to a v3-shape study.yaml driving the report '
                             '(baseline[0].params seeds cache_dir / duration / '
                             'daughter_duration; visualizations[0].config seeds '
                             'the WorkflowVisualization title). CLI flags '
                             'below still override.')
    parser.add_argument('--clean', action='store_true',
                        help='Clear cached metadata and re-run all steps')
    parser.add_argument('--fetch-biocyc', action='store_true',
                        help='Fetch fresh data from EcoCyc API (slow, skipped by default)')
    parser.add_argument('--parca-rerun', action='store_true',
                        help='Re-run the v2ecoli.processes.parca composite '
                             'instead of hydrating models/parca/parca_state.pkl.gz '
                             '(adds ~70 min in fast mode)')
    parser.add_argument('--parca-cpus', type=int, default=4,
                        help='Parallelism for ParCa steps 4 + 5 when --parca-rerun '
                             'is set (default 4)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Override max sim duration in seconds (default: run to division)')
    parser.add_argument('--no-daughters', action='store_true',
                        help='Skip the daughters simulation step')
    parser.add_argument('--daughter-duration', type=int, default=None,
                        help='Override daughter sim duration in seconds')
    parser.add_argument('--cache-dir', default=None,
                        help='Directory holding ParCa cache (default: '
                             'out/cache or out/workflow/cache; or study param; '
                             'use e.g. out/cache_plasmid for a plasmid-enabled run).')
    parser.add_argument('--out', default=None,
                        help='Output HTML path (default: out/workflow/workflow_report.html)')
    parser.add_argument('--open', dest='open_reports', action='store_true',
                        help='Open generated HTML report(s) at end of run (macOS)')
    args = parser.parse_args()

    # Resolve study-derived defaults before CLI overrides land on top.
    study_spec = _load_study(args.study) if args.study else None
    study_params = ((study_spec or {}).get('baseline') or [{}])[0].get('params') or {}
    study_viz_config = None
    if study_spec is not None:
        for v in (study_spec.get('visualizations') or []):
            if isinstance(v, dict) and 'WorkflowVisualization' in (v.get('address') or ''):
                study_viz_config = dict(v.get('config') or {})
                break

    # Apply CLI overrides on top of study values.
    _OPTIONS['fetch_biocyc'] = args.fetch_biocyc
    _OPTIONS['parca_rerun'] = args.parca_rerun
    _OPTIONS['parca_cpus'] = args.parca_cpus
    duration = args.duration if args.duration is not None \
        else study_params.get('duration')
    if duration is not None:
        _OPTIONS['max_duration'] = int(duration)
    _OPTIONS['skip_daughters'] = args.no_daughters
    daughter_duration = args.daughter_duration if args.daughter_duration is not None \
        else study_params.get('daughter_duration')
    if daughter_duration is not None:
        _OPTIONS['daughter_duration'] = int(daughter_duration)

    if args.cache_dir is not None:
        CACHE_DIR = args.cache_dir
    elif 'cache_dir' in study_params:
        CACHE_DIR = study_params['cache_dir']



    _OPTIONS['open_reports'] = args.open_reports

    if args.clean:
        import glob as glob_mod
        for f in glob_mod.glob(os.path.join(WORKFLOW_DIR, '*_meta.json')):
            os.remove(f)
            print(f"  Removed {f}")
        for f in glob_mod.glob(os.path.join(WORKFLOW_DIR, '*.dill')):
            os.remove(f)
            print(f"  Removed {f}")

    if study_spec is not None:
        print(f"Study: {study_spec.get('name', '(unnamed)')}")
    run_workflow(out_path=args.out, viz_config=study_viz_config)
