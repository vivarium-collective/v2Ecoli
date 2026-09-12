"""Self-contained HTML report for a v2ecoli workflow sweep that emitted **xarray
/ zarr** (the parquet variant is scripts/sweep_report.py).

Reads the per-lineage ``.zarr`` stores a sweep emits (``emitter: "xarray"``),
embeds matplotlib plots as base64 PNGs, prepends the shared git/run provenance
banner, appends the how-to-run + framework docs, and writes a latest + a
timestamp+commit archival copy under ``reports/figures/<experiment>/`` (per
AGENTS.md). Plots are base64-embedded so the report is portable.

    .venv/bin/python scripts/sweep_report_xarray.py <sweep_dir> [--open]

``<sweep_dir>`` is the directory holding the ``<exp>_v<variant>_s<seed>.zarr``
stores (i.e. the sweep's ``emitter_arg.out_dir``). Output defaults to
``reports/figures/<experiment_id>/sweep_report_xarray.html``.

Requires the ``[xarray]`` extra (xarray + zarr) and matplotlib.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling
import sweep_report as sr  # noqa: E402  (REPORT_CSS, provenance_banner, docs, ...)

# Mass components captured by the default xarray view (see LineageProcess
# DEFAULT_XARRAY_VIEW). The fraction analysis uses whichever are present.
_FRACTION_COMPONENTS = [("protein", "protein_mass"),
                        ("RNA", "rna_mass"),
                        ("DNA", "dna_mass")]
_MASS_VARS = ["dry_mass", "cell_mass", "protein_mass", "rna_mass", "dna_mass"]
_SEED_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
_VARIANT_STYLES = ["-", "--", ":", "-."]


def discover_experiment_id(sweep_dir) -> str:
    stores = glob.glob(os.path.join(sweep_dir, "*.zarr"))
    for s in stores:
        m = re.match(r"^(.*)_v\d+_s\d+\.zarr$", os.path.basename(s))
        if m:
            return m.group(1)
    return "sweep"


def load_cells(sweep_dir):
    """Return ``{(variant, seed, gen): {var: ndarray}}`` from the .zarr stores.

    Each store is one lineage. Within it, ``time_gen=N`` lives on the lineage
    node and each mass variable is a child group holding ``generation=N`` data
    vars (dims ``emitstep_gen=N``). Generations are 1-based as stored.
    """
    import numpy as np
    import xarray as xr

    cells = {}
    for store in sorted(glob.glob(os.path.join(sweep_dir, "*.zarr"))):
        dt = xr.open_datatree(store, engine="zarr")
        groups = set(dt.groups)  # multi-level paths; dt.get() only sees children
        for path in dt.groups:
            if not re.search(r"lineage_seed=\d+$", path):
                continue
            v = int(re.search(r"variant=([^/]+)", path).group(1))
            s = int(re.search(r"lineage_seed=(\d+)", path).group(1))
            node = dt[path].to_dataset()
            gens = sorted(int(re.search(r"time_gen=(\d+)", n).group(1))
                          for n in node.data_vars if n.startswith("time_gen="))
            for g in gens:
                cell = {"time": np.asarray(node[f"time_gen={g}"]).ravel()}
                key = f"generation={g}"
                for var in _MASS_VARS:
                    vpath = f"{path}/{var}"
                    if vpath not in groups:
                        continue
                    ds = dt[vpath].to_dataset()
                    if key in ds.data_vars:
                        cell[var] = np.asarray(ds[key]).ravel()
                cells[(v, s, g)] = cell
    return cells


def _plots(cells):
    """Return (trajectory_b64, fractions_b64, frac, div_rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = sorted({v for v, s, g in cells})
    seeds = sorted({s for v, s, g in cells})
    gens = sorted({g for v, s, g in cells})

    frac, div_rows = {}, []
    for (v, s, g), c in cells.items():
        dry = c.get("dry_mass")
        f = {}
        for label, var in _FRACTION_COMPONENTS:
            comp = c.get(var)
            if comp is not None and dry is not None and np.all(dry > 0):
                f[label] = float(np.mean(comp / dry))
            else:
                f[label] = 0.0
        frac[(v, s, g)] = f
        t = c.get("time")
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
        cycle = float(t[-1] - t[0]) if t is not None and len(t) else 0.0
        div_rows.append((v, s, g,
                         float(dry[0]) if dry is not None and len(dry) else 0.0,
                         float(dry[-1]) if dry is not None and len(dry) else 0.0,
                         cycle))

    # Plot 1: dry-mass trajectory across generations (offset cumulatively).
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (v, s) in sorted({(v, s) for v, s, g in cells}):
        offset, first = 0.0, True
        for g in gens:
            c = cells.get((v, s, g))
            if not c or c.get("dry_mass") is None:
                continue
            t, dry = c["time"], c["dry_mass"]
            label = (f"seed {s}" + (f" / v{v}" if len(variants) > 1 else "")) if first else None
            # Rebase each generation onto its own start before applying the
            # cumulative offset. Adding a cumulative offset to an already-
            # absolute axis double-counts: `[m]` the absolute arm accumulated
            # 1740 min of "lineage time" against a true elapsed 416.
            ax.plot((t - (t[0] if len(t) else 0.0) + offset) / 60.0, dry,
                    color=_SEED_COLORS[seeds.index(s) % len(_SEED_COLORS)],
                    ls=_VARIANT_STYLES[variants.index(v) % len(_VARIANT_STYLES)],
                    lw=1.4, marker=".", label=label)
            ax.axvline(offset / 60.0, color="#9ca3af", ls=":", lw=0.7)
            offset += float(t[-1] - t[0]) if len(t) else 0.0
            first = False
    ax.set_xlabel("time across lineage (min)")
    ax.set_ylabel("dry mass (fg)")
    ax.set_title("Multigeneration dry-mass trajectory (from zarr)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    p1 = sr._b64(fig)

    # Plot 2: mass fractions vs generation + grouped bars.
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
    labels = [lbl for lbl, _ in _FRACTION_COMPONENTS]
    for i, lbl in enumerate(labels):
        vals = [np.mean([frac[k][lbl] for k in frac if k[2] == g]) for g in gens]
        axes[1].bar(np.array(gens) + (i - 1) * width, vals, width, label=lbl)
    axes[1].set_title("Mean mass fractions (all cells)")
    axes[1].set_xlabel("generation")
    axes[1].set_xticks(gens)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, which="both")
    p2 = sr._b64(fig)
    return p1, p2, frac, sorted(div_rows)


def render_html(provenance, plot1, plot2, frac, div_rows):
    """Assemble the standalone HTML (pure). Reuses the shared CSS / banner /
    docs from sweep_report; labels the RNA-total fraction (not rRNA) and notes
    the zarr source."""
    short = provenance.get("git_short", "")
    has_variant = provenance.get("n_variants", 1) > 1
    prov = sr.provenance_banner(provenance)
    labels = [lbl for lbl, _ in _FRACTION_COMPONENTS]

    rows = ""
    for (v, s, g, m0, m1, t1) in div_rows:
        fr = frac[(v, s, g)]
        vcell = f"<td>{v}</td>" if has_variant else ""
        cols = "".join(f"<td class='num'>{fr[lbl]:.3f}</td>" for lbl in labels)
        rows += (f"<tr>{vcell}<td>{s}</td><td>{g}</td><td class='num'>{m0:.0f}</td>"
                 f"<td class='num'>{m1:.0f}</td><td class='num'>{t1:.0f}</td>{cols}</tr>")
    vhdr = "<th>variant</th>" if has_variant else ""
    frac_hdr = "".join(f"<th>{lbl}</th>" for lbl in labels)

    body = f"""<h1>v2ecoli workflow sweep report <span class="meta">(xarray / zarr)</span></h1>
<p class="meta">Pure process-bigraph meta-composite sweep — emitter <code>xarray</code></p>
{prov}
<h2>Multigeneration mass trajectory</h2>
<img src="data:image/png;base64,{plot1}">
<p class="meta">Read from the per-lineage zarr stores (dotted lines = generation
boundaries). Generations are stored as <code>generation=N</code> vars under each
mass variable, sharing one lineage node.</p>
<h2>Mass-fraction analysis</h2>
<img src="data:image/png;base64,{plot2}">
<h2>Per-cell summary</h2>
<table>
<tr>{vhdr}<th>seed</th><th>gen</th><th>newborn (fg)</th><th>division (fg)</th><th>cycle (s)</th>{frac_hdr}</tr>
{rows}
</table>
{sr.HOWTO_HTML}
{sr.PR_HTML}
<div class="note"><b>Source:</b> read from the sweep's hive-partitioned zarr stores
(<code>&lt;exp&gt;_v&lt;variant&gt;_s&lt;seed&gt;.zarr</code>). Only the variables the
sweep's xarray <code>view</code> emitted are available — the default view captures
scalar mass gauges (<code>dry_mass / cell_mass / protein_mass / rna_mass /
dna_mass</code>), so the fraction here is total RNA, not rRNA.</div>"""
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>v2ecoli workflow sweep (xarray) — {short or "report"}</title>'
            f'<style>{sr.REPORT_CSS}</style></head><body>{body}</body></html>')


def build_report(sweep_dir, out=None):
    """Render the report. Returns (latest_path, archive_path)."""
    cells = load_cells(sweep_dir)
    if not cells:
        raise SystemExit(f"no .zarr sweep stores found under {sweep_dir!r} "
                         f"(expected <exp>_v<variant>_s<seed>.zarr)")
    plot1, plot2, frac, div_rows = _plots(cells)
    prov = sr.collect_provenance(extra={
        "sweep_dir": sweep_dir,
        "n_variants": len({v for v, s, g in cells}),
        "n_seeds": len({s for v, s, g in cells}),
        "n_gens": len({g for v, s, g in cells}),
        "n_cells": len(cells),
    })
    prov["script"] = "scripts/sweep_report_xarray.py"
    html = render_html(prov, plot1, plot2, frac, div_rows)

    if out is None:
        study = discover_experiment_id(sweep_dir).replace("_", "-")
        out = REPO_ROOT / "reports" / "figures" / study / "sweep_report_xarray.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = out.with_name(f"{out.stem}_{stamp}_{prov.get('git_short') or 'nogit'}.html")
    archive.write_bytes(out.read_bytes())
    return out, archive


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", help="dir holding the <exp>_v<v>_s<s>.zarr stores")
    p.add_argument("--out", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    latest, archive = build_report(args.sweep_dir, out=args.out)
    try:
        rel = archive.relative_to(REPO_ROOT)
    except ValueError:
        rel = archive  # custom --out outside the repo
    print(f"Wrote latest  {latest}")
    print(f"Wrote archive {archive}")
    print(f"  commit the archive with: git add -f {rel}")
    if args.open:
        import subprocess
        subprocess.run(["open", str(latest)], check=False)


if __name__ == "__main__":
    main()
