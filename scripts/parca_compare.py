#!/usr/bin/env python
"""
compare_parca.py — per-step side-by-side report comparing v2parca (port-
first composite) against the original ``fitSimData_1`` in vivarium-ecoli.

Walks every available step checkpoint in both directories and emits a
single self-contained HTML report with a sticky left navigation.  Each
Step gets its own section with:

  * runtime (v2parca vs vEcoli, ratio)
  * Input / Output port manifest (documents the step's declared data flow)
  * State comparison — scalars, distributions, and cell_specs entries
    that the step produced, each with max |Δ|, max rel Δ, and KS p-value
  * overlaid histograms for array-valued outputs

Inputs
------
  --v2parca-outdir       DIR  contains checkpoint_step_N.pkl + runtimes.json
                             (produced by scripts/parca_bigraph.py)
  --original-intermediates DIR  contains sim_data_<step>.cPickle +
                             cell_specs_<step>.cPickle (produced by
                             vivarium-ecoli's runscripts/parca.py
                             --save-intermediates)
  -o                    PATH output HTML file

Missing checkpoints on either side render as "not compared" with the
reason listed; partial pipelines don't break the report.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Per-step metadata — name, v2parca checkpoint filename, vEcoli filenames,
# and the port module we pull INPUT_PORTS / OUTPUT_PORTS from.
# ---------------------------------------------------------------------------

STEPS: List[Dict[str, str]] = [
    dict(n=1, name='initialize',        long='Initialize sim_data + scatter',
         v2parca='checkpoint_step_1.pkl',  vecoli_stub='initialize',
         module='v2ecoli.processes.parca.steps.step_01_initialize'),
    dict(n=2, name='input_adjustments', long='Pre-fitted adjustments to expression + deg rates',
         v2parca='checkpoint_step_2.pkl',  vecoli_stub='input_adjustments',
         module='v2ecoli.processes.parca.steps.step_02_input_adjustments'),
    dict(n=3, name='basal_specs',       long='Build basal cell specifications',
         v2parca='checkpoint_step_3.pkl',  vecoli_stub='basal_specs',
         module='v2ecoli.processes.parca.steps.step_03_basal_specs'),
    dict(n=4, name='tf_condition_specs',long='Per-TF + combined condition cell specs',
         v2parca='checkpoint_step_4.pkl',  vecoli_stub='tf_condition_specs',
         module='v2ecoli.processes.parca.steps.step_04_tf_condition_specs'),
    dict(n=5, name='fit_condition',     long='Bulk distributions + translation supply rates',
         v2parca='checkpoint_step_5.pkl',  vecoli_stub='fit_condition',
         module='v2ecoli.processes.parca.steps.step_05_fit_condition'),
    dict(n=6, name='promoter_binding',  long='TF-promoter binding probabilities (CVXPY)',
         v2parca='checkpoint_step_6.pkl',  vecoli_stub='promoter_binding',
         module='v2ecoli.processes.parca.steps.step_06_promoter_binding'),
    dict(n=7, name='adjust_promoters',  long='Ligand concentrations + RNAP recruitment',
         v2parca='checkpoint_step_7.pkl',  vecoli_stub='adjust_promoters',
         module='v2ecoli.processes.parca.steps.step_07_adjust_promoters'),
    dict(n=8, name='set_conditions',    long='Per-nutrient dicts + mass rescaling',
         v2parca='checkpoint_step_8.pkl',  vecoli_stub='set_conditions',
         module='v2ecoli.processes.parca.steps.step_08_set_conditions'),
    dict(n=9, name='final_adjustments', long='ppGpp kinetics + amino-acid supply constants',
         v2parca='checkpoint_step_9.pkl',  vecoli_stub='final_adjustments',
         module='v2ecoli.processes.parca.steps.step_09_final_adjustments'),
]


# Fields to compare as scalars (sim_data attr paths).
SCALARS: List[Tuple[str, Tuple[str, ...]]] = [
    ('mass.avg_cell_dry_mass_init',       ('mass', 'avg_cell_dry_mass_init')),
    ('mass.avg_cell_dry_mass',            ('mass', 'avg_cell_dry_mass')),
    ('mass.avg_cell_water_mass_init',     ('mass', 'avg_cell_water_mass_init')),
    ('mass.fitAvgSolubleTargetMolMass',   ('mass', 'fitAvgSolubleTargetMolMass')),
    ('constants.darkATP',                 ('constants', 'darkATP')),
]

# Array-valued distributions.
DISTRIBUTIONS: List[Tuple[str, Tuple[str, ...]]] = [
    ('RNA expression — basal',    ('process', 'transcription', 'rna_expression', 'basal')),
    ('RNA synthesis prob — basal',('process', 'transcription', 'rna_synth_prob', 'basal')),
    ('RNA deg rates',              ('process', 'transcription', 'rna_data', 'deg_rate')),
    ('Cistron deg rates',          ('process', 'transcription', 'cistron_data', 'deg_rate')),
    ('Protein deg rates',          ('process', 'translation', 'monomer_data', 'deg_rate')),
    ('Translation efficiencies',   ('process', 'translation', 'translation_efficiencies_by_monomer')),
    ('Km endoRNase (transcribed)', ('process', 'transcription', 'rna_data', 'Km_endoRNase')),
    ('Km endoRNase (mature)',      ('process', 'transcription', 'mature_rna_data', 'Km_endoRNase')),
]

# Per-step narrative — what the step does and why it matters in the
# wider pipeline.  Surfaced at the top of each step section.
STEP_EXPLANATIONS: Dict[int, str] = {
    1: ("Bootstraps <code>SimulationDataEcoli</code> from the flat-file "
        "knowledge base.  Loads TSVs into typed tables (genes, RNAs, "
        "proteins, reactions, metabolite concentrations, genome sequence) "
        "and scatters them across the 20 subsystem objects (Mass, "
        "Constants, Transcription, Translation, Metabolism, …) that the "
        "rest of the pipeline reads from."),
    2: ("Applies hand-curated corrections before any fitting: edits to "
        "RNA expression (rRNA + mRNA), protein and RNA degradation rates, "
        "translation efficiencies, and the small-molecule concentrations "
        "that the fitting loop uses as targets.  Many of these are "
        "Keasling-lab / Covert-lab adjustments that reflect empirical "
        "observations missing from pure bioinformatic sources."),
    3: ("Builds the <em>basal</em> cell specification — expression levels, "
        "synthesis probabilities, bulk-molecule counts, and doubling time "
        "for a reference growth condition (M9 + glucose, no TF "
        "perturbations).  This is the spec every later condition "
        "perturbs away from."),
    4: ("Replays step 3 once per transcription-factor activation state, "
        "producing a cell spec for each (condition, TF) pair.  The bulk "
        "of the wall time (~50 s even in fast mode) goes into parallel "
        "evaluation of the TF conditions required to anchor the "
        "binding-probability fit in step 6."),
    5: ("Solves for the bulk-molecule distribution and translation supply "
        "rates that make the basal-plus-condition cell specs "
        "self-consistent.  Runs a fixed-point iteration inside CVXPY; "
        "~70 min in fast mode — the single most expensive step and the "
        "reason checkpointing was added."),
    6: ("Fits TF-promoter binding probabilities.  For each (RNA, "
        "condition) pair, solves a small convex program so that "
        "<code>synth_prob = basal + Σ r · P</code> reproduces the "
        "synthesis probability step 4 computed, with non-negative "
        "recruitment strengths <code>r</code> and equilibrium-derived "
        "binding probabilities <code>P</code>."),
    7: ("Adjusts ligand concentrations and RNAP recruitment strengths to "
        "account for the TF-promoter fit.  Produces the final per-"
        "condition RNA synthesis probabilities that the online model "
        "will use."),
    8: ("Materializes everything into the per-nutrient dictionaries the "
        "online simulation indexes by <code>condition</code>: external "
        "concentrations, doubling times, rescaled masses.  Thin wrapper "
        "over the step 3-7 outputs."),
    9: ("Last-mile adjustments: ppGpp kinetics (for the stringent "
        "response), amino-acid supply constants, and mechanistic "
        "kcat/KM corrections for amino-acid export, uptake, and "
        "supply.  Some of these fail in fast mode because the reduced "
        "TF-condition set leaves key metabolites unconstrained — those "
        "are caught and logged as warnings so the pipeline still "
        "produces a comparable pickle."),
}


# EcoCyc (BioCyc) API — the 10 TSVs that can be refreshed from BioCyc's
# web service.  Mirrors ``v2ecoli.workflow.BIOCYC_FILE_IDS``; kept in
# sync manually because the file list rarely changes.
BIOCYC_FILE_IDS: List[str] = [
    "complexation_reactions", "dna_sites", "equilibrium_reactions",
    "genes", "metabolic_reactions", "metabolites", "proteins",
    "rnas", "transcription_units", "trna_charging_reactions",
]
BIOCYC_BASE_URL = "https://websvc.biocyc.org/wc-get?type="


def _section_biocyc(repo_root: str, out_dir: str, do_fetch: bool) -> str:
    """Render the EcoCyc API section.  When ``do_fetch`` is True, actually
    calls the web service; otherwise loads the last cached meta from
    ``out/compare/biocyc_meta.json`` (or shows a hint when absent)."""
    meta_path = os.path.join(out_dir, 'biocyc_meta.json')
    flat_dir = os.path.join(
        repo_root, 'v2ecoli', 'processes', 'parca', 'reconstruction', 'ecoli', 'flat')

    meta: Optional[Dict[str, Any]] = None
    if do_fetch:
        try:
            import requests  # type: ignore
        except Exception as e:
            return (
                '<section id="biocyc"><h2>0. EcoCyc API</h2>'
                '<p class="meta">Fetch requested but <code>requests</code> '
                f'unavailable: <code>{e}</code>.</p></section>')
        results: Dict[str, Dict[str, Any]] = {}
        print("  Fetching EcoCyc files ...")
        for fid in BIOCYC_FILE_IDS:
            out_tsv = os.path.join(flat_dir, fid + ".tsv")
            try:
                r = requests.get(BIOCYC_BASE_URL + fid, timeout=30)
                r.raise_for_status()
                with open(out_tsv, 'w') as f:
                    f.write(r.text)
                results[fid] = dict(bytes=len(r.text),
                                    lines=r.text.count('\n'), status='ok')
                print(f"    {fid}: {len(r.text):,} bytes")
            except Exception as e:
                results[fid] = dict(bytes=0, lines=0, status=str(e))
                print(f"    {fid}: FAILED ({e})")
            time.sleep(1)
        meta = {
            'when': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_files': len(BIOCYC_FILE_IDS),
            'n_fetched': sum(1 for v in results.values() if v['status'] == 'ok'),
            'files': results,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    elif os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = None

    header = '<section id="biocyc"><h2>0. EcoCyc API</h2>'
    intro = (
        '<p>Ten TSVs in <code>reconstruction/ecoli/flat/</code> are '
        'refreshable from <a href="https://biocyc.org" target="_blank" '
        'rel="noopener">BioCyc</a> via '
        f'<code>{BIOCYC_BASE_URL}&lt;file_id&gt;</code>.  '
        'These are the upstream source of truth for genes, RNAs, proteins, '
        'reactions, DNA binding sites, and equilibrium/complexation '
        'reactions.  The other 120-odd files under <code>flat/</code> are '
        'either curated by the Covert lab or derived from other databases '
        '(EcoCyc + ModelSEED + literature).  Rerun the comparison report '
        'with <code>--fetch-biocyc</code> to refresh these files '
        'in-place.</p>')

    gh_blob = ('https://github.com/vivarium-collective/v2parca/blob/main/'
               'v2ecoli/processes/parca/reconstruction/ecoli/flat')

    if meta is None:
        inner = (
            '<table><tr><th>file</th></tr>'
            + ''.join(
                f'<tr><td><a href="{gh_blob}/{fid}.tsv" target="_blank" '
                f'rel="noopener"><code>{fid}.tsv</code></a></td></tr>'
                for fid in BIOCYC_FILE_IDS)
            + '</table>')
        body = (
            '<p class="meta">No cached metadata at '
            f'<code>{meta_path}</code>.  Run '
            '<code>scripts/parca_compare.py --fetch-biocyc</code> to '
            'populate this section (takes ~15-30 s; hits the BioCyc web '
            'service).</p>'
            '<details><summary>The 10 EcoCyc-refreshable files</summary>'
            f'<div class="details-body">{inner}</div></details>')
        return header + intro + body + '</section>'

    rows = []
    for fid in BIOCYC_FILE_IDS:
        info = meta['files'].get(fid, {})
        status = info.get('status', 'unknown')
        ok = status == 'ok'
        cls = 'pass' if ok else 'fail'
        sz = f"{info.get('bytes', 0):,}" if ok else '—'
        lines = info.get('lines', 0) if ok else '—'
        rows.append(
            f'<tr class="{cls}">'
            f'<td><a href="{gh_blob}/{fid}.tsv" target="_blank" '
            f'rel="noopener"><code>{fid}.tsv</code></a></td>'
            f'<td>{lines}</td><td>{sz}</td><td>{status}</td></tr>')

    body = (
        f'<p class="meta">Last fetched: <strong>{meta.get("when", "?")}</strong> '
        f'— <strong>{meta["n_fetched"]}/{meta["n_files"]}</strong> files OK.</p>'
        '<details open><summary>EcoCyc files fetched from the API '
        f'({meta["n_fetched"]}/{meta["n_files"]} OK — click to browse, '
        'links to GitHub)</summary>'
        '<div class="details-body">'
        '<table><tr><th>file</th><th>lines</th><th>bytes</th><th>status</th></tr>'
        + ''.join(rows) + '</table>'
        '</div></details>')
    return header + intro + body + '</section>'


def _parse_run_warnings(outdir: str) -> Dict[str, List[str]]:
    """Scan out/run*.log for per-step warnings emitted by parca_bigraph.
    Returns a dict mapping 'step_N' -> list of warning strings."""
    logs_dir = os.path.dirname(os.path.abspath(outdir))
    candidates = [os.path.join(logs_dir, 'run_resume.log'),
                  os.path.join(logs_dir, 'run.log')]
    warnings: Dict[str, List[str]] = {}
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            text = open(path).read()
        except Exception:
            continue
        import re
        for m in re.finditer(r'Step (\d)\s*WARNING[:\s]+([^\n]+)', text):
            msg = m.group(2).strip()
            key = f'step_{m.group(1)}'
            bucket = warnings.setdefault(key, [])
            if msg not in bucket:
                bucket.append(msg)
    return warnings


def _section_run_outcome(v2parca_outdir: str, v2parca_rt: Dict[str, float],
                         original_rt: Dict[str, float],
                         step_status: Dict[int, str]) -> str:
    """Summarize what happened on the most recent v2parca run:
    checkpoints written, runtimes, and any per-step warnings."""
    warnings = _parse_run_warnings(v2parca_outdir)

    ok_steps = [n for n in range(1, 10)
                if os.path.exists(os.path.join(
                    v2parca_outdir, f'checkpoint_step_{n}.pkl'))]
    total_v = sum(v2parca_rt.get(f'step_{n}', 0) or 0 for n in range(1, 10))
    state_pkl = os.path.join(v2parca_outdir, 'parca_state.pkl')
    state_size = (os.path.getsize(state_pkl) / 1024 / 1024
                  if os.path.exists(state_pkl) else 0.0)

    cards = [
        ('Checkpoints written', f'{len(ok_steps)} / 9'),
        ('Total v2parca runtime', f'{total_v:.1f} s ({total_v / 60:.1f} min)'),
        ('Final state pickle', f'{state_size:.1f} MB'
         if state_size else '—'),
        ('Steps compared to vEcoli',
         f'{sum(1 for s in step_status.values() if s == "pass")} / '
         f'{len(step_status)}'),
    ]
    cards_html = (
        '<div class="cards">'
        + ''.join(f'<div class="card"><div class="card-label">{lbl}</div>'
                  f'<div class="card-value">{val}</div></div>'
                  for lbl, val in cards)
        + '</div>')

    warn_html = ''
    if warnings:
        rows = []
        for step_key in sorted(warnings.keys()):
            for msg in warnings[step_key]:
                rows.append(
                    f'<tr class="warn"><td>{step_key.replace("_", " ")}</td>'
                    f'<td><code>{msg}</code></td></tr>')
        warn_html = (
            '<h3>Warnings caught (non-fatal)</h3>'
            '<table><tr><th>step</th><th>message</th></tr>'
            + ''.join(rows) + '</table>'
            '<p class="meta">These were logged and tolerated so the '
            'pipeline continued to produce a comparable pickle.  Fast '
            'mode (<code>--mode fast</code>) uses a reduced TF-condition '
            'set which can leave some mechanistic kcat/KM fits '
            'unconstrained — acceptable for comparison but not for '
            'production runs.</p>')

    return (
        '<section id="run_outcome">'
        '<h2>This Run</h2>'
        '<p>Summary of the most recent <code>parca_bigraph.py</code> '
        'invocation (checkpoints + runtimes from '
        f'<code>{v2parca_outdir}</code>).</p>'
        + cards_html
        + warn_html
        + '</section>')


CELL_SPECS_FIELDS = [
    'expression', 'synthProb', 'fit_cistron_expression',
    'doubling_time', 'avgCellDryMassInit', 'fitAvgSolubleTargetMolMass',
    'bulkContainer',
]


# ---------------------------------------------------------------------------
# vEcoli pickle compatibility
# ---------------------------------------------------------------------------

def _alias_vivarium_ecoli_modules() -> None:
    """Register the canonical aliases used by older pickles.  Delegates
    to ``data_loader._install_legacy_pickle_aliases`` which already
    handles:
        v2parca.*  → v2ecoli.processes.parca.*
        vparca.*   → v2ecoli.processes.parca.*
        reconstruction.*/wholecell.*/ecoli.* → v2ecoli.processes.parca.*
    The last set is what lets vivarium-ecoli pickles (top-level
    ``reconstruction.ecoli.*`` module refs) unpickle against the
    vendored subpackage."""
    from v2ecoli.processes.parca.data_loader import (
        _install_legacy_pickle_aliases,
    )
    _install_legacy_pickle_aliases()


def _load_pickle(path: Optional[str]) -> Any:
    if path is None or not os.path.exists(path):
        return None
    _alias_vivarium_ecoli_modules()
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Data-shape navigation (flat v2parca state vs nested SimulationDataEcoli)
# ---------------------------------------------------------------------------

def _get(obj, attr):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    val = getattr(obj, attr, None)
    if val is not None:
        return val
    try:
        return obj[attr]
    except (KeyError, IndexError, TypeError):
        return None


def _reach(obj, path: Tuple[str, ...]):
    if obj is None:
        return None
    if isinstance(obj, dict) and 'transcription' in obj and 'process' not in obj:
        if path and path[0] == 'process' and len(path) > 1:
            path = path[1:]
        if path and path[0] == 'internal_state' and len(path) > 1:
            if path[1] == 'bulk_molecules':
                path = ('bulk_molecules',) + tuple(path[2:])
    for p in path:
        if obj is None:
            return None
        obj = _get(obj, p)
    return obj


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _as_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        if hasattr(x, 'asNumber'):
            x = x.asNumber()
    except Exception:
        pass
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.dtype.kind not in ('i', 'f', 'u'):
        if arr.dtype.names:
            for name in ('count', 'counts', 'deg_rate'):
                if name in arr.dtype.names:
                    return np.asarray(arr[name], dtype=float)
            for name in arr.dtype.names:
                sub = arr[name]
                if sub.dtype.kind in ('i', 'f', 'u'):
                    return np.asarray(sub, dtype=float)
        return None
    return arr.astype(float, copy=False)


def _safe_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float('nan')
    eps = 1e-30
    denom = np.maximum(np.abs(a) + np.abs(b) + eps, eps)
    return float(np.max(np.abs(a - b) / denom))


def _safe_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float('nan')
    return float(np.max(np.abs(a - b)))


def _b64fig(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _hist_overlay(a, b, title, log=False) -> str:
    aa = _as_array(a); bb = _as_array(b)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    if aa is not None and aa.size:
        v = aa[np.isfinite(aa)]
        if log and (v > 0).any():
            v = np.log10(v[v > 0])
        ax.hist(v, bins=60, alpha=0.55, label='v2ecoli.processes.parca', color='#2563eb', density=True)
    if bb is not None and bb.size:
        v = bb[np.isfinite(bb)]
        if log and (v > 0).any():
            v = np.log10(v[v > 0])
        ax.hist(v, bins=60, alpha=0.55, label='vEcoli',  color='#dc2626', density=True)
    ax.set_title(title + (' (log10)' if log else ''))
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    return _b64fig(fig)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _row_class(rel_diff: Optional[float],
               tol_pass: float = 1e-6, tol_warn: float = 1e-3) -> str:
    if rel_diff is None or not np.isfinite(rel_diff):
        return 'warn'
    if rel_diff < tol_pass:
        return 'pass'
    if rel_diff < tol_warn:
        return 'warn'
    return 'fail'


def _fmt(x):
    if x is None:
        return ''
    if isinstance(x, float):
        if not np.isfinite(x):
            return 'n/a'
        if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
            return f'{x:.3e}'
        return f'{x:.4g}'
    return str(x)


# ---------------------------------------------------------------------------
# Per-step section builders
# ---------------------------------------------------------------------------

_STORE_PATH_CACHE: Optional[Dict[str, List[str]]] = None

def _store_paths() -> Dict[str, List[str]]:
    global _STORE_PATH_CACHE
    if _STORE_PATH_CACHE is None:
        try:
            from v2ecoli.processes.parca.composite import STORE_PATH as _SP
            _STORE_PATH_CACHE = dict(_SP)
        except Exception:
            _STORE_PATH_CACHE = {}
    return _STORE_PATH_CACHE


def _port_rows(ports: Dict[str, Any], store_paths: Dict[str, List[str]]) -> str:
    """Render a port dict as a table: port name, store path, description."""
    visible = [(k, v) for k, v in ports.items() if not k.startswith('tick_')]
    tick_ports = [k for k in ports if k.startswith('tick_')]
    if not visible:
        return ('<p class="meta">(none — this step has no upstream '
                'subsystem reads; it only consumes ordering tokens: '
                f'<code>{", ".join(tick_ports) or "—"}</code>)</p>')
    rows = ['<table class="ports"><tr>'
            '<th>port</th><th>store path</th><th>role</th></tr>']
    for k, v in visible:
        path = store_paths.get(k)
        path_str = '/' + '/'.join(path) if path else '<span class="meta">(not in STORE_PATH)</span>'
        rows.append(
            f'<tr><td><code>{k}</code></td>'
            f'<td><code>{path_str}</code></td>'
            f'<td class="meta">{v}</td></tr>'
        )
    rows.append('</table>')
    if tick_ports:
        rows.append(f'<p class="meta">Plus ordering tokens: '
                    f'<code>{", ".join(tick_ports)}</code> — these enforce '
                    'Step execution order in the bigraph but carry no data.</p>')
    return '\n'.join(rows)


def _port_table(module_name: str) -> str:
    """Render a step's declared data flow: prose header + per-direction
    port tables showing port name, store path, and role."""
    try:
        m = importlib.import_module(module_name)
    except Exception as e:
        return f'<p class="meta">(port manifest unavailable: {e})</p>'
    ins  = getattr(m, 'INPUT_PORTS',  {}) or {}
    outs = getattr(m, 'OUTPUT_PORTS', {}) or {}
    doc = (m.__doc__ or '').strip()
    # Keep the first paragraph of the module docstring (if any) as context.
    doc_para = doc.split('\n\n', 1)[0].replace('\n', ' ').strip() if doc else ''

    intro = [
        '<p class="meta">Each Step declares <code>INPUT_PORTS</code> '
        '(stores it reads) and <code>OUTPUT_PORTS</code> (stores it writes). '
        'Port names are resolved to absolute store paths via '
        '<code>STORE_PATH</code> in <code>v2parca/composite.py</code>, and '
        'the composite wires each port to the corresponding location in the '
        'nested bigraph. The <em>role</em> column shows the description '
        'given in the step module (e.g. <code>sim_data.transcription</code> '
        'means "mirrors the <code>transcription</code> subsystem on '
        '<code>SimulationDataEcoli</code>"; <code>overwrite</code> means '
        'the port replaces whatever value lives at that store).</p>'
    ]
    if doc_para:
        intro.append(f'<p><strong>Step purpose:</strong> {doc_para}</p>')

    sp = _store_paths()
    parts = list(intro)
    parts.append(f'<h4>Inputs — reads ({sum(1 for k in ins if not k.startswith("tick_"))})</h4>')
    parts.append(_port_rows(ins, sp))
    parts.append(f'<h4>Outputs — writes ({sum(1 for k in outs if not k.startswith("tick_"))})</h4>')
    parts.append(_port_rows(outs, sp))
    return '\n'.join(parts)


def _section_runtime(vt: Optional[float], ot: Optional[float]) -> str:
    ratio = (vt / ot) if (vt and ot and ot > 0) else None
    return (
        '<h3>Runtime</h3>'
        '<table><tr><th>v2parca</th><th>vEcoli</th><th>ratio (v2parca/vEcoli)</th></tr>'
        f'<tr><td>{_fmt(vt)} s</td><td>{_fmt(ot)} s</td><td>{_fmt(ratio)}</td></tr>'
        '</table>'
    )


def _section_scalars(v2parca, original) -> str:
    out = ['<h3>Scalar state</h3>',
           '<table><tr><th>path</th><th>v2parca</th><th>vEcoli</th>'
           '<th>rel Δ</th></tr>']
    any_row = False
    for label, path in SCALARS:
        a = _reach(v2parca, path); b = _reach(original, path)
        a_num = float(a.asNumber()) if a is not None and hasattr(a, 'asNumber') else (
            float(a) if isinstance(a, (int, float, np.number)) else None)
        b_num = float(b.asNumber()) if b is not None and hasattr(b, 'asNumber') else (
            float(b) if isinstance(b, (int, float, np.number)) else None)
        if a_num is None and b_num is None:
            continue
        any_row = True
        rd = (_safe_rel_diff(np.array([a_num]), np.array([b_num]))
              if (a_num is not None and b_num is not None) else float('nan'))
        out.append(
            f'<tr class="{_row_class(rd)}">'
            f'<td><code>{label}</code></td>'
            f'<td>{_fmt(a_num)}</td><td>{_fmt(b_num)}</td>'
            f'<td>{_fmt(rd)}</td></tr>')
    out.append('</table>')
    return '\n'.join(out) if any_row else ''


def _section_distributions(v2parca, original) -> str:
    figs_html = []
    tbl = ['<h3>Distribution numerical summary</h3>',
           '<table><tr><th>distribution</th><th>shape</th>'
           '<th>max |Δ|</th><th>max rel Δ</th><th>KS p-value</th></tr>']
    any_dist = False
    for label, path in DISTRIBUTIONS:
        va = _reach(v2parca, path); oa = _reach(original, path)
        a = _as_array(va); b = _as_array(oa)
        if a is None and b is None:
            continue
        any_dist = True
        shape = (a.shape if a is not None else b.shape)
        if a is not None and b is not None and a.shape == b.shape:
            rd = _safe_rel_diff(a, b); ma = _safe_max_abs(a, b)
            ks = (_scipy_stats.ks_2samp(a.ravel(), b.ravel()).pvalue
                  if HAVE_SCIPY else None)
        else:
            rd, ma, ks = float('nan'), float('nan'), None
        log = (a is not None and a.size > 0 and (np.asarray(a) > 0).any()
               and (float(a.max()) / max(float(a[a > 0].min()) if (a > 0).any() else 1, 1e-30) > 100))
        img = _hist_overlay(va, oa, label, log=log)
        figs_html.append(f'<div><strong>{label}</strong><br>'
                         f'<img src="data:image/png;base64,{img}"/></div>')
        tbl.append(
            f'<tr class="{_row_class(rd)}">'
            f'<td><code>{label}</code></td>'
            f'<td>{shape}</td>'
            f'<td>{_fmt(ma)}</td><td>{_fmt(rd)}</td><td>{_fmt(ks)}</td></tr>')
    tbl.append('</table>')
    if not any_dist:
        return ''
    return ('<h3>Distributions</h3>'
            '<div class="grid">' + '\n'.join(figs_html) + '</div>' + '\n' + '\n'.join(tbl))


def _section_cell_specs(v2parca, original_cell_specs) -> str:
    cs_v = _reach(v2parca, ('cell_specs',)) or {}
    cs_o = original_cell_specs or {}
    if not cs_v and not cs_o:
        return ''
    common = sorted(set(cs_v.keys()) & set(cs_o.keys()))
    only_v = sorted(set(cs_v.keys()) - set(cs_o.keys()))
    only_o = sorted(set(cs_o.keys()) - set(cs_v.keys()))
    out = ['<h3>cell_specs (per-condition max rel Δ)</h3>']
    if only_v or only_o:
        out.append(
            f'<p class="meta">Only v2parca: <code>{only_v}</code>. '
            f'Only vEcoli: <code>{only_o}</code>.</p>')
    if not common:
        return '\n'.join(out) + '<p class="meta">(no shared conditions)</p>'
    out.append('<table><tr><th>condition</th>'
               + ''.join(f'<th>{f}</th>' for f in CELL_SPECS_FIELDS)
               + '</tr>')
    for cond in common:
        v_spec = cs_v[cond]; o_spec = cs_o[cond]
        cells = []
        for field in CELL_SPECS_FIELDS:
            va = v_spec.get(field) if isinstance(v_spec, dict) else None
            oa = o_spec.get(field) if isinstance(o_spec, dict) else None
            if va is None or oa is None:
                cells.append('<td class="meta">—</td>'); continue
            aa = _as_array(va); ob = _as_array(oa)
            if aa is None or ob is None:
                try:
                    av = float(va.asNumber()) if hasattr(va, 'asNumber') else float(va)
                    bv = float(oa.asNumber()) if hasattr(oa, 'asNumber') else float(oa)
                    rd = _safe_rel_diff(np.array([av]), np.array([bv]))
                    cells.append(f'<td class="{_row_class(rd)}">{_fmt(rd)}</td>')
                except Exception:
                    cells.append('<td class="meta">?</td>')
                continue
            if aa.shape != ob.shape:
                cells.append(f'<td class="fail">shape {aa.shape}≠{ob.shape}</td>')
                continue
            rd = _safe_rel_diff(aa, ob)
            cells.append(f'<td class="{_row_class(rd)}">{_fmt(rd)}</td>')
        out.append(f'<tr><td><code>{cond}</code></td>{"".join(cells)}</tr>')
    out.append('</table>')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>v2ecoli ParCa Workflow Report</title>
<style>
  :root {{ --accent: #2563eb; --pass: #ecfdf5; --warn: #fffbeb; --fail: #fef2f2; }}
  * {{ box-sizing: border-box; }}
  body  {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; color: #111; background: #fafafa; }}
  #layout {{ display: grid; grid-template-columns: 220px 1fr; min-height: 100vh; }}
  nav    {{ position: sticky; top: 0; height: 100vh; overflow-y: auto;
            background: #111; color: #eee; padding: 18px 12px; }}
  nav h2  {{ color: #fff; font-size: 14px; margin: 0 0 12px 4px;
             letter-spacing: .5px; text-transform: uppercase; }}
  nav a   {{ display: block; color: #ccc; text-decoration: none;
             padding: 6px 10px; border-radius: 4px; font-size: 13px;
             margin-bottom: 2px; }}
  nav a:hover {{ background: #1f2937; color: #fff; }}
  nav a.active {{ background: var(--accent); color: #fff; }}
  nav .status-pass {{ color: #34d399; }}
  nav .status-warn {{ color: #fbbf24; }}
  nav .status-fail {{ color: #f87171; }}
  nav .status-na   {{ color: #6b7280; }}
  main   {{ padding: 24px 40px; max-width: 1100px; }}
  h1     {{ border-bottom: 3px solid var(--accent); padding-bottom: 6px; }}
  section {{ margin-bottom: 48px; padding: 24px 28px;
             background: #fff; border-radius: 8px;
             box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  section h2 {{ margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  section h3 {{ margin-top: 24px; color: #374151; }}
  table  {{ border-collapse: collapse; width: 100%; font-size: 13px;
            margin-top: 8px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 5px 8px; text-align: left; }}
  th     {{ background: #f3f4f6; }}
  tr.pass {{ background: var(--pass); }}
  tr.warn {{ background: var(--warn); }}
  tr.fail {{ background: var(--fail); }}
  td.pass {{ background: var(--pass); }}
  td.warn {{ background: var(--warn); }}
  td.fail {{ background: var(--fail); }}
  img    {{ max-width: 100%; border: 1px solid #eee; margin: 6px 0;
            background: #fff; }}
  .grid  {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px;
            margin-top: 10px; }}
  code   {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
  .meta  {{ color: #6b7280; font-size: 12px; }}
  .banner-na {{ background: #f9fafb; border-left: 4px solid #9ca3af;
                padding: 10px 14px; color: #6b7280; }}
  .intro {{ background: #eff6ff; border-left: 4px solid var(--accent);
            padding: 14px 18px; border-radius: 4px; margin-bottom: 20px; }}
  .intro p {{ margin: 0 0 8px 0; }}
  .intro p:last-child {{ margin-bottom: 0; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px; margin: 12px 0; }}
  .card  {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
            padding: 10px 12px; }}
  .card-label {{ font-size: 11px; text-transform: uppercase;
                 letter-spacing: .04em; color: #64748b; }}
  .card-value {{ font-size: 18px; font-weight: 600; color: #0f172a;
                 margin-top: 2px; }}
  details {{ margin: 8px 0; border: 1px solid #e5e7eb; border-radius: 6px;
             background: #fff; }}
  details > summary {{ cursor: pointer; font-weight: 600; color: #374151;
                       padding: 10px 14px; user-select: none; }}
  details > summary:hover {{ color: #111; background: #f9fafb; }}
  details[open] > summary {{ border-bottom: 1px solid #e5e7eb;
                             background: #f9fafb; }}
  details .details-body {{ padding: 10px 14px; max-height: 480px;
                           overflow-y: auto; }}
  .src-biocyc {{ background: #3b82f6; color: #fff; }}
  .src-curated {{ background: #64748b; color: #fff; }}
  .src-badge {{ padding: 1px 6px; border-radius: 3px; font-size: 11px;
                font-weight: 600; letter-spacing: .02em; }}
</style>
<script>
  // Highlight the current section in the left nav while scrolling.
  document.addEventListener('DOMContentLoaded', () => {{
    const sections = document.querySelectorAll('section[id]');
    const links    = document.querySelectorAll('nav a[href^="#"]');
    const byId = {{}};
    links.forEach(a => byId[a.getAttribute('href').slice(1)] = a);
    const io = new IntersectionObserver(entries => {{
      entries.forEach(e => {{
        if (e.isIntersecting) {{
          links.forEach(a => a.classList.remove('active'));
          const a = byId[e.target.id]; if (a) a.classList.add('active');
        }}
      }});
    }}, {{ rootMargin: '-45% 0px -55% 0px' }});
    sections.forEach(s => io.observe(s));
  }});
</script>
</head><body>
<div id='layout'>
<nav>
  <h2>ParCa Workflow</h2>
  <a href="workflow_report.html" style="color:#60a5fa;border:1px solid #374151;margin-bottom:6px;">Simulation Report &rarr;</a>
  <a href="#overview">Overview</a>
  <a href="#run_outcome">This Run</a>
  <a href="#biocyc">0. EcoCyc API</a>
  <a href="#raw_data">Raw Data</a>
  {nav_links}
  <a href="#network">Composition Diagram</a>
</nav>
<main>
{banner}
<h1>v2ecoli ParCa Workflow Report</h1>
<p class='meta'>{meta}</p>
{sections}
</main>
</div></body></html>"""


def _section_raw_data() -> str:
    """Catalog the flat-file knowledge base that feeds step 1.

    Walks ``v2parca/reconstruction/ecoli/flat/`` — the TSVs that
    ``KnowledgeBaseEcoli`` reads at composite-construction time and
    passes into ``InitializeStep``'s config.  This is what ultimately
    feeds the whole pipeline, even though it never appears on step 1's
    declared port table (raw_data is a config, not a port).
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    flat_dir = os.path.join(
        repo_root, 'v2ecoli', 'processes', 'parca', 'reconstruction', 'ecoli', 'flat')
    gh_tree = ('https://github.com/vivarium-collective/v2parca/tree/main/'
               'v2ecoli/processes/parca/reconstruction/ecoli/flat')

    if not os.path.isdir(flat_dir):
        return ('<section id="raw_data"><h2>Raw Data</h2>'
                f'<p class="meta">flat directory not found: <code>{flat_dir}</code></p>'
                '</section>')

    by_subdir: Dict[str, Dict[str, int]] = {}
    file_list: List[Dict[str, Any]] = []
    n_files = 0
    total_bytes = 0
    for root, _dirs, files in os.walk(flat_dir):
        rel = os.path.relpath(root, flat_dir)
        label = 'root' if rel == '.' else rel
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            n_files += 1
            total_bytes += sz
            by_subdir.setdefault(label, {'count': 0, 'size': 0})
            by_subdir[label]['count'] += 1
            by_subdir[label]['size'] += sz
            # BioCyc-sourced files are the 10 top-level TSVs in BIOCYC_FILE_IDS;
            # everything else (subdirs + other top-level) is curated/derived.
            base, ext = os.path.splitext(fn)
            is_biocyc = (label == 'root' and ext.lower() == '.tsv'
                         and base in BIOCYC_FILE_IDS)
            rel_path = fn if label == 'root' else f'{label}/{fn}'
            file_list.append({
                'name': fn, 'subdir': label, 'rel_path': rel_path,
                'size': sz, 'source': 'biocyc' if is_biocyc else 'curated',
            })

    # KB stats — optional; import lazily so the rest of the report still
    # renders if reconstruction deps fail to import.
    stats_html = ''
    try:
        sys.path.insert(0, repo_root)
        from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
        raw = KnowledgeBaseEcoli(
            operons_on=True, remove_rrna_operons=False,
            remove_rrff=False, stable_rrna=False)
        n_genes = len(raw.genes) if hasattr(raw, 'genes') else 0
        n_rnas = len(raw.rnas) if hasattr(raw, 'rnas') else 0
        n_proteins = len(raw.proteins) if hasattr(raw, 'proteins') else 0
        n_metabolites = len(raw.metabolites) if hasattr(raw, 'metabolites') else 0
        genome_length = len(raw.genome_sequence) if hasattr(raw, 'genome_sequence') else 0
        stats_html = (
            '<h3>Knowledge base statistics</h3>'
            '<table><tr><th>genes</th><th>RNAs</th><th>proteins</th>'
            '<th>metabolites</th><th>genome length (bp)</th></tr>'
            f'<tr><td>{n_genes:,}</td><td>{n_rnas:,}</td><td>{n_proteins:,}</td>'
            f'<td>{n_metabolites:,}</td><td>{genome_length:,}</td></tr></table>'
        )
    except Exception as e:
        stats_html = (
            '<p class="meta">Knowledge base stats unavailable '
            f'(<code>{type(e).__name__}: {e}</code>).</p>')

    subdir_rows = ['<table><tr><th>subdir</th><th>files</th><th>size</th></tr>']
    for sub in sorted(by_subdir.keys()):
        c = by_subdir[sub]['count']
        sz = by_subdir[sub]['size']
        href = gh_tree if sub == 'root' else f'{gh_tree}/{sub}'
        subdir_rows.append(
            f'<tr><td><a href="{href}" target="_blank" rel="noopener">'
            f'<code>{sub}</code></a></td>'
            f'<td>{c}</td><td>{sz / 1024:.1f} KB</td></tr>')
    subdir_rows.append('</table>')

    gh_blob = gh_tree.replace('/tree/main/', '/blob/main/')
    file_rows = ['<table><tr><th>file</th><th>size</th><th>source</th></tr>']
    for fi in sorted(file_list, key=lambda x: x['rel_path']):
        sz = fi['size']
        sz_str = f'{sz / 1024:.1f} KB' if sz >= 1024 else f'{sz} B'
        badge = ('<span class="src-badge src-biocyc">EcoCyc API</span>'
                 if fi['source'] == 'biocyc'
                 else '<span class="src-badge src-curated">Curated</span>')
        file_rows.append(
            f'<tr><td><a href="{gh_blob}/{fi["rel_path"]}" '
            f'target="_blank" rel="noopener"><code>{fi["rel_path"]}</code>'
            f'</a></td><td>{sz_str}</td><td>{badge}</td></tr>')
    file_rows.append('</table>')

    n_biocyc = sum(1 for f in file_list if f['source'] == 'biocyc')
    n_curated = n_files - n_biocyc

    return (
        '<section id="raw_data">'
        '<h2>Raw Data — flat-file knowledge base</h2>'
        '<p>'
        f'Loaded by <code>KnowledgeBaseEcoli</code> at composite-construction '
        f'time and passed into <code>InitializeStep</code>\'s config (not a '
        f'port, which is why it does not appear on step 1\'s declared data '
        f'flow table).  Total: <strong>{n_files}</strong> files '
        f'(<strong>{n_biocyc}</strong> EcoCyc-refreshable, '
        f'<strong>{n_curated}</strong> curated/derived), '
        f'<strong>{total_bytes / 1024 / 1024:.1f} MB</strong>.  '
        f'Source: <a href="{gh_tree}" target="_blank" rel="noopener">'
        f'v2ecoli / processes / parca / reconstruction / ecoli / flat &#8599;</a>.</p>'
        + stats_html
        + '<details><summary>File catalog by subdirectory '
        f'({len(by_subdir)} dirs)</summary>'
        '<div class="details-body">'
        + '\n'.join(subdir_rows)
        + '</div></details>'
        + '<details><summary>All raw data files '
        f'({n_files} files — click to browse, links to GitHub)</summary>'
        '<div class="details-body">'
        + '\n'.join(file_rows)
        + '</div></details>'
        + '</section>'
    )


def _runtime_table(v2parca_times: Dict[str, float],
                   original_times: Dict[str, float]) -> str:
    """Render per-step runtime as a table alongside the bar chart."""
    rows = ['<table><tr><th>step</th><th>v2parca (s)</th>'
            '<th>vEcoli (s)</th><th>ratio (v2parca/vEcoli)</th></tr>']
    total_v = 0.0
    total_o = 0.0
    for step in STEPS:
        n = step['n']
        v = v2parca_times.get(f'step_{n}')
        o = original_times.get(step['vecoli_stub']) or original_times.get(f'step_{n}')
        if v is not None:
            total_v += v
        if o is not None:
            total_o += o
        ratio = (v / o) if (v and o and o > 0) else None
        rows.append(
            f'<tr><td>step {n} — {step["name"]}</td>'
            f'<td>{_fmt(v)}</td><td>{_fmt(o)}</td><td>{_fmt(ratio)}</td></tr>')
    total_ratio = (total_v / total_o) if total_o > 0 else None
    rows.append(
        f'<tr><th>total</th>'
        f'<th>{_fmt(total_v) if total_v else "—"}</th>'
        f'<th>{_fmt(total_o) if total_o else "—"}</th>'
        f'<th>{_fmt(total_ratio)}</th></tr>')
    rows.append('</table>')
    return '\n'.join(rows)


def _runtime_bar(v2parca_times: Dict[str, float],
                 original_times: Dict[str, float]) -> str:
    steps = [f'step_{n}' for n in range(1, 10)]
    v = [v2parca_times.get(s, 0.0)  for s in steps]
    o = [original_times.get(s, 0.0) for s in steps]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(steps)); w = 0.38
    ax.bar(x - w/2, v, w, label='v2ecoli.processes.parca', color='#2563eb')
    ax.bar(x + w/2, o, w, label='vEcoli',  color='#dc2626')
    ax.set_xticks(x); ax.set_xticklabels([f'step {n}' for n in range(1, 10)],
                                          rotation=30)
    ax.set_ylabel('seconds')
    ax.set_title('Per-step runtime')
    ax.set_yscale('symlog', linthresh=1)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=9)
    return _b64fig(fig)


def build_report(v2parca_outdir: str, vecoli_dir: Optional[str],
                 output_path: str, fetch_biocyc: bool = False) -> None:
    # Load runtimes.
    v2parca_rt_path = os.path.join(v2parca_outdir, 'runtimes.json')
    # Fall back to the shipped fixture's runtimes when the outdir copy
    # doesn't exist (common for fresh clones that haven't re-run the
    # pipeline).
    if not os.path.exists(v2parca_rt_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        v2parca_rt_path = os.path.join(repo_root, 'models', 'parca', 'runtimes.json')
    v2parca_rt = json.load(open(v2parca_rt_path)) if os.path.exists(v2parca_rt_path) else {}
    # vEcoli's --save-intermediates doesn't write a runtimes.json; we
    # parse its per-stage "Ran X in Ys" prints if a log is available.
    original_rt = _maybe_parse_vecoli_runtimes(vecoli_dir)

    # Per-step section data — status ('pass' / 'warn' / 'fail' / 'na')
    # is a quick summary for the left-nav colored dots.
    section_items = []
    for step in STEPS:
        n = step['n']
        v2parca_pkl = os.path.join(v2parca_outdir, step['v2parca'])
        vecoli_sd  = (os.path.join(vecoli_dir, f"sim_data_{step['vecoli_stub']}.cPickle")
                      if vecoli_dir else None)
        vecoli_cs  = (os.path.join(vecoli_dir, f"cell_specs_{step['vecoli_stub']}.cPickle")
                      if vecoli_dir else None)

        v2parca_state = _load_pickle(v2parca_pkl)
        original     = _load_pickle(vecoli_sd)
        original_cs  = _load_pickle(vecoli_cs)

        # Attach cell_specs to the original sim_data if present.
        if original is not None and original_cs is not None:
            try:
                original.cell_specs = original_cs
            except Exception:
                pass

        vt = v2parca_rt.get(f'step_{n}')
        ot = original_rt.get(step['vecoli_stub'])

        section_id = f'step_{n}'
        parts = []
        parts.append(f'<section id="{section_id}">')
        parts.append(f'<h2>Step {n} — {step["name"]}</h2>')
        parts.append(f'<p class="meta">{step["long"]}</p>')
        if n in STEP_EXPLANATIONS:
            parts.append(f'<p>{STEP_EXPLANATIONS[n]}</p>')

        # Availability banner.
        have_vp = v2parca_state is not None
        have_ve = original is not None
        if not have_vp and not have_ve:
            parts.append('<div class="banner-na">No checkpoint on either side. '
                         'Run `scripts/parca_bigraph.py` for v2parca and '
                         '`runscripts/parca.py --save-intermediates` for vEcoli.</div>')
            status = 'na'
        elif not have_ve:
            parts.append('<div class="banner-na">vEcoli reference pickle unavailable '
                         '— showing v2parca side only.</div>')
            status = 'warn'
        elif not have_vp:
            parts.append('<div class="banner-na">v2parca checkpoint unavailable '
                         '— showing vEcoli side only.</div>')
            status = 'warn'
        else:
            status = 'pass'

        parts.append(_section_runtime(vt, ot))
        parts.append('<h3>Declared data flow</h3>')
        parts.append(_port_table(step['module']))

        if have_vp and have_ve:
            parts.append(_section_scalars(v2parca_state, original))
            parts.append(_section_distributions(v2parca_state, original))
            parts.append(_section_cell_specs(v2parca_state, original_cs))

        parts.append('</section>')
        section_items.append({
            'n': n, 'name': step['name'], 'status': status,
            'html': '\n'.join(parts),
        })

    # Overview section.
    overview = ['<section id="overview">', '<h2>Overview</h2>']
    overview.append(f'<p>Generated {time.strftime("%Y-%m-%d %H:%M:%S")}</p>')
    overview.append(f'<p class="meta">v2parca checkpoints: <code>{v2parca_outdir}</code>. '
                    f'vEcoli intermediates: <code>{vecoli_dir}</code>.</p>')
    overview.append('<h3>Per-step runtime</h3>')
    overview.append(f'<img src="data:image/png;base64,{_runtime_bar(v2parca_rt, original_rt)}"/>')
    overview.append(_runtime_table(v2parca_rt, original_rt))
    if not original_rt:
        overview.append(
            '<p class="meta">vEcoli runtimes unavailable — run '
            '<code>runscripts/parca.py --save-intermediates</code> in '
            '<code>vivarium-ecoli</code> and drop a <code>runtimes.json</code> '
            'into <code>out/original_intermediates/</code> '
            '(keys: <code>initialize</code>, <code>input_adjustments</code>, …) '
            'to populate the vEcoli column.</p>')
    overview.append('<h3>Step-by-step availability</h3>')
    overview.append('<table><tr><th>step</th><th>v2parca</th><th>vEcoli</th>'
                    '<th>compared</th></tr>')
    for step, item in zip(STEPS, section_items):
        n = step['n']
        vp_ok = os.path.exists(os.path.join(v2parca_outdir, step['v2parca']))
        ve_ok = (vecoli_dir is not None and
                 os.path.exists(os.path.join(vecoli_dir,
                                             f"sim_data_{step['vecoli_stub']}.cPickle")))
        overview.append(
            f'<tr><td>step {n} — {step["name"]}</td>'
            f'<td>{"✓" if vp_ok else "—"}</td>'
            f'<td>{"✓" if ve_ok else "—"}</td>'
            f'<td>{"yes" if vp_ok and ve_ok else "no"}</td></tr>')
    overview.append('</table>')
    overview.append('</section>')

    nav_links = '\n'.join(
        f'<a href="#step_{it["n"]}"><span class="status-{it["status"]}">●</span> '
        f'Step {it["n"]} — {it["name"]}</a>'
        for it in section_items
    )

    meta = (f'Generated {time.strftime("%Y-%m-%d %H:%M:%S")} — '
            f'tolerance: pass &lt; 1e-6, warn &lt; 1e-3')

    # Intro — what this report is and how to read it.
    intro_html = (
        '<section id="intro"><h2>What this report shows</h2>'
        '<div class="intro">'
        '<p><strong>v2parca</strong> is a port of the '
        '<a href="https://github.com/CovertLab/vEcoli">vEcoli</a> '
        '<em>Parameter Calculator</em> (ParCa) onto '
        '<a href="https://github.com/vivarium-collective/process-bigraph">'
        'process-bigraph</a>.  ParCa is the once-per-simulation build '
        'step that turns the E. coli knowledge base (~130 TSVs of genes, '
        'RNAs, proteins, reactions, concentrations) into a fitted '
        '<code>SimulationDataEcoli</code> pickle that the online model '
        'consumes.  Nine compute-heavy steps, top to bottom: initialize '
        '&rarr; adjust inputs &rarr; build basal specs &rarr; expand per-TF '
        'conditions &rarr; fit bulk distributions &rarr; fit '
        'promoter binding &rarr; adjust promoters &rarr; set per-nutrient '
        'conditions &rarr; final mechanistic adjustments.</p>'
        '<p>This report runs each step as a process-bigraph <code>Step</code> '
        'and diffs every checkpointed state against the original '
        'vivarium-ecoli pickles.  Per-step sections show scalar / '
        'distribution / cell-spec differences with max |Δ|, max rel Δ, '
        'and KS p-values; tolerance bands are <span style="background:'
        '#ecfdf5;padding:1px 5px;border-radius:3px">pass &lt; 1e-6</span>, '
        '<span style="background:#fffbeb;padding:1px 5px;border-radius:3px">'
        'warn &lt; 1e-3</span>, <span style="background:#fef2f2;'
        'padding:1px 5px;border-radius:3px">fail &gt; 1e-3</span>.</p>'
        '<p>Sections below: <strong>Overview</strong> (runtimes, '
        'availability matrix) &middot; <strong>This Run</strong> '
        '(what just happened) &middot; <strong>EcoCyc API</strong> '
        '(refreshable upstream source) &middot; <strong>Raw Data</strong> '
        '(the flat-file knowledge base) &middot; <strong>Steps 1-9</strong> '
        '(per-step diffs).</p>'
        '</div></section>')

    # Additional sections between Overview and the per-step sections.
    run_outcome_html = _section_run_outcome(
        v2parca_outdir, v2parca_rt, original_rt,
        {it['n']: it['status'] for it in section_items})
    out_dir = os.path.dirname(os.path.abspath(output_path))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    biocyc_html = _section_biocyc(repo_root, out_dir, fetch_biocyc)
    raw_data_html = _section_raw_data()

    # Network visualization — embed the interactive Cytoscape.js diagram
    # of the 9-Step pipeline as an iframe.  Generate it on-the-fly so the
    # report is always up-to-date with the current port manifests.
    network_html = ''
    try:
        from v2ecoli.processes.parca.viz import build_graph, render_html
        graph_data = build_graph()
        network_path = os.path.join(
            os.path.dirname(output_path) or '.', 'parca_network.html')
        with open(network_path, 'w') as nf:
            nf.write(render_html(
                graph_data,
                'ParCa Composition Diagram',
                'Interactive Cytoscape.js view of the 9-Step ParCa pipeline',
            ))
        # Embed as an iframe
        network_rel = os.path.basename(network_path)
        network_html = (
            '<section id="network">'
            '<h2>ParCa Composition Diagram</h2>'
            '<p class="meta">Interactive Cytoscape.js network of the 9-Step '
            'ParCa pipeline and 34 stores it wires through.  Click any node '
            'to inspect its ports, class, and docstring.  Switch layouts from '
            'the dropdown.</p>'
            f'<iframe src="{network_rel}" '
            'style="width:100%;height:750px;border:1px solid #e5e7eb;'
            'border-radius:6px;" loading="lazy"></iframe>'
            '</section>')
    except Exception as e:
        network_html = (
            '<section id="network"><h2>ParCa Composition Diagram</h2>'
            f'<p class="meta">Network generation failed: <code>{e}</code></p>'
            '</section>')

    sections_html = (intro_html + '\n' +
                     '\n'.join(overview) + '\n' +
                     run_outcome_html + '\n' +
                     biocyc_html + '\n' +
                     raw_data_html + '\n' +
                     '\n'.join(it['html'] for it in section_items) + '\n' +
                     network_html)

    # Banner is best-effort — if git / platform probing fails, render nothing.
    try:
        from v2ecoli.processes.parca.viz.repro_banner import banner_html
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        banner = banner_html(repo_root=repo_root)
    except Exception:
        banner = ''

    html = _HTML_TEMPLATE.format(
        banner=banner, nav_links=nav_links, meta=meta, sections=sections_html)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'wrote {output_path} ({size_kb:.1f} KB)')


def _maybe_parse_vecoli_runtimes(vecoli_dir: Optional[str]) -> Dict[str, float]:
    """Best-effort: look for a 'Ran X in Ys' log next to the pickles."""
    if not vecoli_dir:
        return {}
    for candidate in (os.path.join(vecoli_dir, 'runtimes.json'),
                      os.path.join(os.path.dirname(vecoli_dir), 'runtimes.json')):
        if os.path.exists(candidate):
            try:
                return json.load(open(candidate))
            except Exception:
                pass
    # Parse from vecoli_parca logs in /tmp if present.
    for log in ('/tmp/vecoli_intermediates.log',
                '/tmp/vecoli_solo.log', '/tmp/vecoli_parca_run3.log'):
        if os.path.exists(log):
            import re
            out = {}
            for m in re.finditer(r'Ran (\S+) in (\d+) s', open(log).read()):
                out[m.group(1)] = float(m.group(2))
            if out:
                return out
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v2parca-outdir', default='out/sim_data',
                   help='dir with checkpoint_step_N.pkl + runtimes.json')
    p.add_argument('--original-intermediates', default='out/original_intermediates',
                   help='dir with sim_data_<step>.cPickle + cell_specs_<step>.cPickle')
    p.add_argument('-o', '--output', default='out/workflow/parca_workflow_report.html')
    p.add_argument('--fetch-biocyc', action='store_true',
                   help='Refresh the 10 BioCyc-sourced TSVs under '
                        'reconstruction/ecoli/flat/ by calling the BioCyc '
                        'web service (~15-30 s).  Off by default.')
    args = p.parse_args()

    vecoli_dir = args.original_intermediates if os.path.isdir(
        args.original_intermediates) else None
    build_report(args.v2parca_outdir, vecoli_dir, args.output,
                 fetch_biocyc=args.fetch_biocyc)


if __name__ == '__main__':
    main()
