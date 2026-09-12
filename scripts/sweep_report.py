"""Self-contained HTML report for a v2ecoli workflow sweep, with a provenance
banner (per AGENTS.md "HTML reports with provenance banners").

Reads a sweep's hive-partitioned parquet, embeds matplotlib plots as base64
PNGs, prepends a git/run provenance banner, appends how-to-run + framework
docs, and writes TWO files: a "latest" copy (overwritten) and a
timestamp+commit archival copy (never overwritten). Both are portable —
plots are base64-embedded, so they survive the parquet being deleted.

    .venv/bin/python scripts/sweep_report.py <sweep_dir> [--open]

``<sweep_dir>`` holds the hive-partitioned parquet
(``experiment_id=…/variant=…/lineage_seed=…/generation=…/agent_id=…``).
Output defaults to ``reports/figures/<experiment_id>/sweep_report.html`` plus
``sweep_report_<YYYYMMDDTHHMMSS>_<git_short>.html`` alongside it. ``reports/``
is gitignored, so commit the archive with ``git add -f``.

Requires the ``[parquet]`` extra (duckdb) and matplotlib.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import glob
import io
import os
import platform
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO = "https://github.com/vivarium-collective/v2ecoli"
PR = "#95"


def _git(*args, default=""):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def collect_provenance(extra: dict | None = None) -> dict:
    """Identifying metadata for the report header (mirrors the AGENTS.md
    pattern in scripts/compare_pdmp_vs_baseline.py::collect_provenance)."""
    sha = _git("rev-parse", "HEAD")
    prov = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": sha,
        "git_short": sha[:8] if sha else "",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_last_commit_msg": _git("log", "-1", "--format=%s"),
        "git_last_commit_author": _git("log", "-1", "--format=%an"),
        "git_last_commit_when": _git("log", "-1", "--format=%ai"),
        "host": platform.node(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
    }
    if extra:
        prov.update(extra)
    return prov


MASS_COLS = ["global_time", "listeners__mass__dry_mass", "listeners__mass__protein_mass",
             "listeners__mass__rRna_mass", "listeners__mass__dna_mass"]

_SEED_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
_VARIANT_STYLES = ["-", "--", ":", "-."]


def discover_experiment_id(sweep_dir) -> str:
    for f in glob.glob(os.path.join(sweep_dir, "**", "*.pq"), recursive=True):
        m = re.search(r"experiment_id=([^/]+)", f)
        if m:
            return m.group(1)
    return "sweep"


def load_cells(sweep_dir):
    """Return ``{(variant, seed, gen): numpy array of MASS_COLS}`` from parquet."""
    import duckdb
    import numpy as np

    by_cell = {}
    for f in glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                       recursive=True):
        m = re.search(r"variant=([^/]+)/lineage_seed=(\d+)/generation=(\d+)", f)
        if m:
            v = m.group(1)
            key = (int(v) if v.isdigit() else v, int(m.group(2)), int(m.group(3)))
        else:
            m2 = re.search(r"lineage_seed=(\d+)/generation=(\d+)", f)
            if not m2:
                continue
            key = (0, int(m2.group(1)), int(m2.group(2)))
        by_cell.setdefault(key, []).append(f)

    cells = {}
    for key, files in by_cell.items():
        union = " UNION ALL ".join(
            f"SELECT {','.join(MASS_COLS)} FROM read_parquet('{f}')" for f in files)
        recs = duckdb.sql(f"SELECT * FROM ({union}) t ORDER BY global_time").fetchall()
        cells[key] = np.array(recs, dtype=float)
    return cells


def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _plots(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = sorted({v for v, s, g in cells})
    seeds = sorted({s for v, s, g in cells})
    gens = sorted({g for v, s, g in cells})

    frac, div_rows = {}, []
    for (v, s, g), a in cells.items():
        t, dry, prot, rrna, dna = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        frac[(v, s, g)] = {"protein": float(np.mean(prot / dry)),
                           "rRNA": float(np.mean(rrna / dry)),
                           "DNA": float(np.mean(dna / dry))}
        # ⛔ A CYCLE DURATION IS ``t[-1] - t[0]``, NEVER ``t[-1]``.
        # ``time`` is written under two different conventions depending on how
        # the sweep was produced, and only the DIFFERENCE is correct under both:
        #   * one continuous invocation -> an ABSOLUTE clock that keeps rising
        #     across generations (``run_comparison_ensemble.py`` refuses
        #     ``--initial-generation > 1`` on the wrapped arm, so that arm is
        #     always this case).
        #   * chained/resumed invocations (``--initial-generation`` /
        #     ``--daughter-state-out``) -> an INVOCATION-RELATIVE clock that
        #     restarts near zero.
        # ``t[-1]`` is a span only when the generation's clock happened to start
        # at ~0 -- true for the first generation of an invocation, and never
        # true past generation 1 on an absolute clock.
        # `[m@an 8-generation two-arm campaign]` the absolute arm reported
        # 43/88/132/178/230/288/357/424 as "cycles" where the true spans are
        # 42/44/43/45/51/57/68/66.
        div_rows.append((v, s, g, dry[0], dry[-1], t[-1] - t[0]))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (v, s) in sorted({(v, s) for v, s, g in cells}):
        offset, first = 0.0, True
        for g in gens:
            a = cells.get((v, s, g))
            if a is None:
                continue
            t, dry = a[:, 0], a[:, 1]
            label = (f"seed {s}" + (f" / v{v}" if len(variants) > 1 else "")) if first else None
            # Rebase each generation onto its own start -- see the note above.
            ax.plot((t - t[0] + offset) / 60.0, dry,
                    color=_SEED_COLORS[seeds.index(s) % len(_SEED_COLORS)],
                    ls=_VARIANT_STYLES[variants.index(v) % len(_VARIANT_STYLES)],
                    lw=1.4, label=label)
            ax.axvline(offset / 60.0, color="#9ca3af", ls=":", lw=0.7)
            offset += t[-1] - t[0]
            first = False
    ax.set_xlabel("time across lineage (min)")
    ax.set_ylabel("dry mass (fg)")
    ax.set_title("Multigeneration dry-mass trajectory (grow → divide)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    p1 = _b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for (v, s) in sorted({(v, s) for v, s, g in cells}):
        xs = [g for g in gens if (v, s, g) in frac]
        ys = [frac[(v, s, g)]["protein"] for g in xs]
        axes[0].plot(xs, ys, marker="o",
                     color=_SEED_COLORS[seeds.index(s) % len(_SEED_COLORS)],
                     ls=_VARIANT_STYLES[variants.index(v) % len(_VARIANT_STYLES)])
    axes[0].set_title("Protein mass fraction vs generation")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("protein / dry mass")
    axes[0].set_xticks(gens)
    axes[0].grid(alpha=0.25)
    width = 0.25
    for i, comp in enumerate(["protein", "rRNA", "DNA"]):
        vals = [np.mean([frac[k][comp] for k in frac if k[2] == g]) for g in gens]
        axes[1].bar(np.array(gens) + (i - 1) * width, vals, width, label=comp)
    axes[1].set_title("Mean mass fractions (all cells)")
    axes[1].set_xlabel("generation")
    axes[1].set_xticks(gens)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, which="both")
    p2 = _b64(fig)
    return p1, p2, frac, sorted(div_rows)


HOWTO_HTML = """
<h2>How to run your own sweep</h2>
<p>Install the console script once, then drive a sweep from a vEcoli-style JSON config:</p>
<pre>uv pip install --python .venv/bin/python -e .          # registers v2ecoli-workflow
v2ecoli-workflow --config v2ecoli/configs/two_generations.json --out out/myrun
v2ecoli-workflow --config &lt;cfg&gt; --build-only            # write sweep.pbg without running
v2ecoli-workflow --config &lt;cfg&gt; --max-sim-time 20000   # sim-time cap (seconds)</pre>
<p>…then build this report from the run's parquet:</p>
<pre>.venv/bin/python scripts/sweep_report.py out/myrun/parquet --open</pre>

<h3>Config knobs</h3>
<table class="doc">
<tr><th>key</th><th>meaning</th></tr>
<tr><td><code>n_init_sims</code></td><td>number of <b>seeds</b> — independent replicate lineages</td></tr>
<tr><td><code>generations</code></td><td>cell divisions per lineage</td></tr>
<tr><td><code>lineage_seed</code></td><td>first seed; seeds = <code>lineage_seed … lineage_seed+n_init_sims-1</code></td></tr>
<tr><td><code>single_daughters</code></td><td><code>true</code>: keep one daughter per division. <code>false</code> (binary tree) is not yet implemented — raises a clear error.</td></tr>
<tr><td><code>variants</code></td><td>declarative parameter sweep (see below)</td></tr>
<tr><td><code>different_seeds_per_variant</code></td><td>give each variant a non-overlapping seed range</td></tr>
<tr><td><code>skip_baseline</code></td><td>omit the unmodified baseline variant from the grid</td></tr>
<tr><td><code>cache_dir</code> / <code>out_dir</code></td><td>ParCa cache in; parquet + <code>sweep.pbg</code> out</td></tr>
<tr><td><code>time_step</code></td><td>composite tick cadence (s)</td></tr>
<tr><td><code>max_duration_per_gen</code></td><td>per-generation safety cap (s)</td></tr>
<tr><td><code>inherit_from</code></td><td>list of parent configs to merge (priority: current &gt; first &gt; …)</td></tr>
<tr><td><code>analysis_options</code></td><td>declared analyses; <code>mass_fraction_summary</code> implemented, cross-cell scales pending</td></tr>
</table>

<h3>Changing seeds &amp; generations</h3>
<p>The sweep grid is <b>variants × seeds</b>, and each branch walks <code>generations</code> divisions. To run 4 seeds × 2 generations:</p>
<pre>{ "inherit_from": ["default.json"], "experiment_id": "4seed_2gen",
  "n_init_sims": 4, "generations": 2 }</pre>

<h3>Variants (parameter sweep)</h3>
<p>A variant overrides a process <code>config_schema</code> value by dotted path. Single parameter:</p>
<pre>"variants": { "kcat": { "target": "ecoli-metabolism.kcat", "value": [1, 2, 3] } }</pre>
<p>Multiple parameters combine via <code>op</code> (<code>prod</code> | <code>zip</code> | <code>add</code>); <code>linspace</code> (or any numpy generator) is also accepted:</p>
<pre>"variants": { "sweep": {
    "a": { "target": "p.a", "value": [1, 2] },
    "b": { "target": "p.b", "linspace": { "start": 0.0, "stop": 1.0, "num": 3 } },
    "op": "prod" } }</pre>
<p>This yields 2&times;3 = 6 variant points; crossed with <code>n_init_sims</code> seeds it produces one
<b>branch</b> per (variant, seed). Each branch emits hive-partitioned parquet keyed by
<code>experiment_id / variant / lineage_seed / generation / agent_id</code>.
<span class="meta">(Deferred: <code>nested</code> grammar and <code>sim_data</code>-recomputing variants such as gene knockouts.)</span></p>
"""

PR_HTML = """
<h2>What this framework adds (PR {pr})</h2>
<p>Brings vEcoli's multiseed / multigeneration / variant workflow framework into v2ecoli as
<b>pure process-bigraph</b> — the whole sweep is one inspectable meta-composite document, replacing NextFlow.</p>
<ul>
 <li><b>Config loader</b> (<code>workflow/config.py</code>) — vEcoli <code>inherit_from</code> chains, list-merge, cycle detection.</li>
 <li><b>Variant grammar</b> (<code>workflow/variants.py</code>) — <code>value</code>/<code>linspace</code> + <code>op: prod|zip|add</code> → declarative process-config overrides → branch grid.</li>
 <li><b>Declarative variants</b> (<code>composites/baseline.py</code>) — <code>config_overrides</code> patch process config at build time without mutating the shared ParCa cache.</li>
 <li><b>LineageProcess</b> (<code>workflow/lineage.py</code>) — one (variant, seed) lineage as an embeddable Process; single-daughter generation walk with per-generation parquet metadata.</li>
 <li><b>Meta-composite + registration</b> (<code>workflow/meta_composite.py</code>) — one <code>LineageProcess</code> branch per (variant, seed); <code>local:</code> address resolution.</li>
 <li><b>Driver + CLI</b> (<code>workflow/run.py</code>) — <code>v2ecoli-workflow</code>; ticks until all branches complete, saves <code>sweep.pbg</code>.</li>
 <li><b>Analyses as Steps</b> (<code>workflow/analysis.py</code>) — <code>AnalysisStep</code> base + five-scale registry + the <code>MassFractionSummary</code> example.</li>
 <li><b>Ported configs</b> — <code>configs/default.json</code>, <code>configs/two_generations.json</code>.</li>
 <li><b>Parquet emits bulk molecules</b> — the parquet path captures the ~16k-molecule <code>bulk__id</code>/<code>bulk__count</code> arrays (RAM-only before).</li>
 <li><b>Fix: double-divide</b> — daughters were halved twice (started ~¼ mother mass, grew too slowly to divide). Now the inner Division's daughter is carried directly; generations divide normally.</li>
</ul>
<p class="meta">Remaining follow-ups: cross-cell analysis Steps (multiseed / multigeneration / multivariant), <code>single_daughters=false</code> binary-tree lineages, resume, and parallel execution.</p>
""".replace("{pr}", PR)


# Shared report CSS (single-brace; substituted verbatim, not f-string source).
# Reused by scripts/sweep_report_xarray.py so both variants look identical.
REPORT_CSS = """
 body{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1f2937;line-height:1.55}
 h1{font-size:1.5rem;margin:0 0 6px} h2{font-size:1.15rem;margin-top:2.2rem;border-bottom:1px solid #e5e7eb;padding-bottom:.3rem}
 h3{font-size:1rem;margin-top:1.4rem}
 .meta{color:#6b7280;font-size:.9rem} img{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}
 .provenance{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin:14px 0 20px;font-size:.85em;line-height:1.55}
 .provenance dt{display:inline-block;min-width:120px;color:#475569;font-weight:600}
 .provenance dd{display:inline;margin:0;font-family:ui-monospace,Menlo,monospace}
 .provenance .row{margin:1px 0}
 table{border-collapse:collapse;width:100%;font-size:.9rem;margin-top:.5rem}
 th,td{border:1px solid #e5e7eb;padding:.35rem .6rem;text-align:left;vertical-align:top} th{background:#f3f4f6}
 td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
 .doc tr td:first-child{width:230px;white-space:nowrap}
 pre{background:#0f172a;color:#e2e8f0;padding:.7rem .9rem;border-radius:6px;overflow-x:auto;font-size:.82rem;line-height:1.4}
 code{background:rgba(0,0,0,.04);padding:1px 5px;border-radius:3px;font-size:.88em} pre code{background:none;padding:0}
 .note{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:.6rem .9rem;font-size:.9rem;margin-top:1rem}
 ul{font-size:.93rem} li{margin:.25rem 0}
"""


def provenance_banner(provenance) -> str:
    """The shared git/run provenance banner ``<div>`` (AGENTS.md convention)."""
    dirty_badge = ('<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
                   if provenance.get("git_dirty") else "")
    sha, short = provenance.get("git_sha", ""), provenance.get("git_short", "")
    return f"""<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{provenance.get('generated_at','')}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="{GITHUB_REPO}/commit/{sha}" style="color:#0369a1;text-decoration:none">{short}</a>
        &nbsp;<code>{sha}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{provenance.get('git_branch','')} &middot; PR {PR}</dd></div>
  <div class="row"><dt>last commit</dt><dd>{provenance.get('git_last_commit_msg','')} — {provenance.get('git_last_commit_author','')} ({provenance.get('git_last_commit_when','')})</dd></div>
  <div class="row"><dt>script</dt><dd>{provenance.get('script','')}</dd></div>
  <div class="row"><dt>host</dt><dd>{provenance.get('host','')} &nbsp;<span style="color:#94a3b8">{provenance.get('platform','')}, Python {provenance.get('python','')}</span></dd></div>
  <div class="row"><dt>sweep</dt><dd>{provenance.get('n_variants','?')} variant(s) &times; {provenance.get('n_seeds','?')} seed(s) &times; {provenance.get('n_gens','?')} generation(s) = {provenance.get('n_cells','?')} cells &middot; <code>{provenance.get('sweep_dir','')}</code></dd></div>
</div>"""


def render_html(provenance, plot1, plot2, frac, div_rows):
    """Assemble the standalone HTML document (pure: no IO)."""
    short = provenance.get("git_short", "")
    has_variant = provenance.get("n_variants", 1) > 1
    prov = provenance_banner(provenance)

    rows = ""
    for (v, s, g, m0, m1, t1) in div_rows:
        fr = frac[(v, s, g)]
        vcell = f"<td>{v}</td>" if has_variant else ""
        rows += (f"<tr>{vcell}<td>{s}</td><td>{g}</td><td class='num'>{m0:.0f}</td>"
                 f"<td class='num'>{m1:.0f}</td><td class='num'>{t1:.0f}</td>"
                 f"<td class='num'>{fr['protein']:.3f}</td><td class='num'>{fr['rRNA']:.3f}</td>"
                 f"<td class='num'>{fr['DNA']:.3f}</td></tr>")
    vhdr = "<th>variant</th>" if has_variant else ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>v2ecoli workflow sweep — {short or 'report'}</title>
<style>{REPORT_CSS}</style></head><body>
<h1>v2ecoli workflow sweep report</h1>
<p class="meta">Pure process-bigraph meta-composite sweep</p>
{prov}
<h2>Multigeneration mass trajectory</h2>
<img src="data:image/png;base64,{plot1}">
<p class="meta">Each lineage grows from a newborn to its division threshold, divides, and one daughter
is carried forward (dotted lines = generation boundaries).</p>
<h2>Mass-fraction analysis (<code>mass_fraction_summary</code>, single scale)</h2>
<img src="data:image/png;base64,{plot2}">
<h2>Per-cell summary</h2>
<table>
<tr>{vhdr}<th>seed</th><th>gen</th><th>newborn (fg)</th><th>division (fg)</th><th>cycle (s)</th>
<th>protein</th><th>rRNA</th><th>DNA</th></tr>
{rows}
</table>
{HOWTO_HTML}
{PR_HTML}
<div class="note"><b>Scope:</b> <code>mass_fraction_summary</code> (single scale) is the one implemented
analysis Step. Raw bulk molecule counts are in the emitted parquet (<code>bulk__id</code> / <code>bulk__count</code>).</div>
</body></html>"""


def build_report(sweep_dir, out=None):
    """Render the report. Returns (latest_path, archive_path)."""
    cells = load_cells(sweep_dir)
    if not cells:
        raise SystemExit(f"no sweep parquet found under {sweep_dir!r} "
                         f"(expected …/history/…/*.pq)")
    plot1, plot2, frac, div_rows = _plots(cells)
    prov = collect_provenance(extra={
        "sweep_dir": sweep_dir,
        "n_variants": len({v for v, s, g in cells}),
        "n_seeds": len({s for v, s, g in cells}),
        "n_gens": len({g for v, s, g in cells}),
        "n_cells": len(cells),
    })
    html = render_html(prov, plot1, plot2, frac, div_rows)

    if out is None:
        study = discover_experiment_id(sweep_dir).replace("_", "-")
        out = REPO_ROOT / "reports" / "figures" / study / "sweep_report.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    # Archival copy: timestamp + commit short, never overwritten.
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    short = prov.get("git_short") or "nogit"
    archive = out.with_name(f"{out.stem}_{stamp}_{short}.html")
    archive.write_bytes(out.read_bytes())
    return out, archive


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", help="run dir holding the hive-partitioned parquet")
    p.add_argument("--out", default=None,
                   help="latest html (default reports/figures/<experiment_id>/sweep_report.html)")
    p.add_argument("--open", action="store_true", help="open the report when done")
    args = p.parse_args()
    latest, archive = build_report(args.sweep_dir, out=args.out)
    print(f"Wrote latest  {latest}")
    print(f"Wrote archive {archive}")
    print(f"  commit the archive with: git add -f {archive.relative_to(REPO_ROOT)}")
    if args.open:
        subprocess.run(["open", str(latest)], check=False)


if __name__ == "__main__":
    main()
