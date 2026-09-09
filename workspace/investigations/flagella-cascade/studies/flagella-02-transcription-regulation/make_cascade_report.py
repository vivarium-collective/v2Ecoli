"""Generate the flagella-cascade evidence report (HTML + provenance banner).

Runs the baseline composite with the opt-in `flagella_regulation` feature and
records the Kalir & Alon SUM-gate trajectory: mean Class II / Class III
init_prob_override alongside free FliA (EG11355-MONOMER[c]), cytoplasmic FlgM
(G369-MONOMER[c]), and complete flagella (CPLX0-7452[j]). Renders a two-panel
figure with a self-describing provenance banner.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-01-cascade-timing/make_cascade_report.py \
        --seconds 600 --out reports/figures/flagella_cascade/cascade_report.html
"""
import argparse
import base64
import io
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

import v2ecoli
from v2ecoli.core import load_cache_bundle
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "?"


def collect_provenance(script_path):
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain") != ""
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sha": sha,
        "short": sha[:8],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "last_commit": _git("log", "-1", "--pretty=%s (%an, %ad)", "--date=short"),
        "script": script_path,
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def run_trajectory(cache_dir, seconds, sample, seed):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    cII = [rna_ids.index(r) for r in cfg["flg_classII_rnaids"]]
    cIII = [rna_ids.index(r) for r in cfg["flg_classIII_rnaids"]]

    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    bulk = comp.state["agents"]["0"]["bulk"]
    bulk = bulk["_data"] if isinstance(bulk, dict) and "_data" in bulk else bulk
    bids = bulk["id"]
    i_fliA = bulk_name_to_idx("EG11355-MONOMER[c]", bids)
    i_flgM = bulk_name_to_idx("G369-MONOMER[c]", bids)
    i_flag = bulk_name_to_idx("CPLX0-7452[j]", bids)

    def snap():
        a = comp.state["agents"]["0"]
        p = a["unique"]["promoter"]
        p = p["_data"] if isinstance(p, dict) and "_data" in p else p
        m = p["_entryState"].view(bool)
        tu, ov = p["TU_index"][m], p["init_prob_override"][m]
        cii = ov[np.isin(tu, cII)]
        ciii = ov[np.isin(tu, cIII)]
        b = a["bulk"]
        b = b["_data"] if isinstance(b, dict) and "_data" in b else b
        return (
            float(cii.mean()) if len(cii) else 0.0,
            float(ciii.mean()) if len(ciii) else 0.0,
            int(b["count"][i_fliA]), int(b["count"][i_flgM]), int(b["count"][i_flag]),
        )

    ts, rows = [], []
    rows.append(snap()); ts.append(0)
    for t in range(sample, seconds + 1, sample):
        comp.run(sample)
        rows.append(snap()); ts.append(t)
    return np.array(ts), np.array(rows)


def render_png(ts, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cII, cIII, fliA, flgM, flag = rows.T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(ts, cII, "-o", color="#1f77b4", label="Class II ⟨override⟩ (FlhDC+FliA)")
    ax1.plot(ts, cIII, "-s", color="#d62728", label="Class III ⟨override⟩ (FliA / σ28)")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("mean init_prob_override")
    ax1.set_title("Kalir & Alon SUM-gate output"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ts, fliA, "-o", color="#2ca02c", label="free FliA  EG11355-MONOMER[c]")
    ax2.plot(ts, flgM, "-s", color="#ff7f0e", label="FlgM  G369-MONOMER[c]")
    ax2b = ax2.twinx()
    ax2b.plot(ts, flag, "-^", color="#9467bd", label="complete flagella CPLX0-7452[j]")
    ax2b.set_ylabel("flagella count", color="#9467bd")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("molecule count")
    ax2.set_title("FlgM secretion → FliA release")
    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render_html(prov, png_b64, ts, rows):
    dirty = ' <span style="color:#fff;background:#c0392b;padding:1px 6px;border-radius:3px">DIRTY TREE</span>' if prov["dirty"] else ""
    url = f"https://github.com/vivarium-collective/v2ecoli/commit/{prov['sha']}"
    final = rows[-1]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Flagella transcriptional cascade — evidence</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#222}}
 .banner{{background:#1b2838;color:#cfd8e3;font-size:12px;padding:10px 18px;font-family:ui-monospace,monospace}}
 .banner a{{color:#7fb3ff}} .wrap{{max-width:1080px;margin:24px auto;padding:0 18px}}
 h1{{font-size:22px}} table{{border-collapse:collapse;font-size:13px;margin-top:8px}}
 td,th{{border:1px solid #ccc;padding:3px 8px;text-align:right}} th{{background:#eef}}
 .note{{background:#fff8e1;border-left:4px solid #f0ad4e;padding:10px 14px;font-size:13px;margin:14px 0}}
 img{{max-width:100%;background:#fff;border:1px solid #ddd;border-radius:6px}}
</style></head><body>
<div class="banner">
 generated {prov['generated']} · branch <b>{prov['branch']}</b>{dirty} ·
 commit <a href="{url}">{prov['short']}</a> · last: {prov['last_commit']}<br>
 script {prov['script']} · {prov['host']} · {prov['os']} · py{prov['python']}
</div>
<div class="wrap">
 <h1>Flagella transcriptional cascade — Kalir &amp; Alon SUM-gate in v2ecoli</h1>
 <p>Baseline composite with the opt-in <code>flagella_regulation</code> feature
 (seed 0, fast-mode cache). Ported from Maya Abdalla's vEcoli
 <code>biofilm</code> branch.</p>
 <img src="data:image/png;base64,{png_b64}">
 <div class="note">
  <b>Mechanism shown.</b> The SUM-gate writes a per-promoter
  <code>init_prob_override</code> for Class II and Class III flagella TUs (left).
  Complete flagella (CPLX0-7452) drive type-III secretion of cytoplasmic FlgM
  (G369-MONOMER, orange, right); as FlgM falls, the FlgM·FliA equilibrium
  releases free FliA / σ28 (green), which raises the Class III override —
  the Class II → Class III coupling.
 </div>
 <p><b>Final state (t={int(ts[-1])} s):</b> Class II ⟨override⟩={final[0]:.3e},
  Class III ⟨override⟩={final[1]:.3e}, FliA={int(final[2])}, FlgM={int(final[3])},
  flagella={int(final[4])}.</p>
 <table><tr><th>t (s)</th><th>⟨CII⟩</th><th>⟨CIII⟩</th><th>FliA</th><th>FlgM</th><th>flagella</th></tr>
 {''.join(f"<tr><td>{int(t)}</td><td>{r[0]:.3e}</td><td>{r[1]:.3e}</td><td>{int(r[2])}</td><td>{int(r[3])}</td><td>{int(r[4])}</td></tr>" for t, r in zip(ts, rows))}
 </table>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--out", default="reports/figures/flagella_cascade/cascade_report.html")
    args = ap.parse_args()

    rel_script = os.path.relpath(os.path.abspath(__file__))
    prov = collect_provenance(rel_script)
    ts, rows = run_trajectory(args.cache_dir, args.seconds, args.sample, args.seed)
    png = render_png(ts, rows)
    html = render_html(prov, png, ts, rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archive = args.out.replace(".html", f"_{stamp}_{prov['short']}.html")
    with open(archive, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}\nwrote {archive}")


if __name__ == "__main__":
    main()
