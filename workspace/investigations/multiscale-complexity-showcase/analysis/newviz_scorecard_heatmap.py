#!/usr/bin/env python
"""Investigation scorecard — verdict heatmap across the 8 mcs studies.

Rows = studies mcs-01..08, columns = their behavior_tests (left-aligned; ragged).
Cells colored by verdict read live from each study.yaml behavior_tests[].verdict:
  pass = green, fail/refute = red (annotated where the failure is the intended
  finding), inconclusive = amber, pending = grey. Hover gives test name + observed.
"""
from __future__ import annotations
import os, glob
import yaml
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies"))

ORDER = [
    "mcs-01-ppgpp-replication-coupling",
    "mcs-02-metabolism-energy-balance",
    "mcs-03-single-cell-heterogeneity",
    "mcs-04-capstone-downshift",
    "mcs-05-etc-stoichiometry-fix",
    "mcs-06-stochastic-heterogeneity",
    "mcs-07-ribosome-allocation-law",
    "mcs-08-rna-protein-composition",
]

# studies whose "fail" verdict is the intended finding (the defect later fixed)
INTENDED_FAIL = {"mcs-02-metabolism-energy-balance"}

VMAP = {"pass": 4, "inconclusive": 2, "fail": 1, "refute": 1, "pending": 0}
# discrete colorscale: 0 grey, 1 red, 2 amber, 4 green (with an unused 3 slot)
COLORS = {0: "#B0B0B0", 1: "#E45756", 2: "#F2A93B", 4: "#4FA24F"}

rows = []
for slug in ORDER:
    y = yaml.safe_load(open(os.path.join(STUDIES, slug, "study.yaml")))
    tests = y.get("behavior_tests") or []
    rows.append((slug, y.get("title") or slug, tests))

maxc = max(len(t) for _, _, t in rows)
ncols = maxc

z, text, hover = [], [], []
for slug, title, tests in rows:
    zr, tr, hr = [], [], []
    intended = slug in INTENDED_FAIL
    for j in range(ncols):
        if j < len(tests):
            t = tests[j]
            v = str(t.get("verdict", "pending")).lower()
            code = VMAP.get(v, 0)
            zr.append(code)
            sym = {"pass": "✓", "fail": "✗", "refute": "✗",
                   "inconclusive": "~", "pending": "·"}.get(v, "·")
            if v in ("fail", "refute") and intended:
                sym = "✗*"
            tr.append(sym)
            obs = t.get("observed") or t.get("observed_value") or t.get("value") or ""
            note = "  (intended finding — the defect fixed in mcs-05)" if (v in ("fail", "refute") and intended) else ""
            hr.append(f"<b>{t.get('name','?')}</b><br>verdict: {v}{note}<br>observed: {obs}")
        else:
            zr.append(None); tr.append(""); hr.append("")
    z.append(zr); text.append(tr); hover.append(hr)

ylabels = [f"{slug.split('-')[0]}-{slug.split('-')[1]}  {title.split('—')[0].strip()[:22]}"
           for slug, title, _ in rows]
# short y labels: mcs-01 .. mcs-08 with a phrase
ylabels = []
for slug, title, _ in rows:
    code = "-".join(slug.split("-")[:2])
    short = title.replace("Arc ", "").split("—")[-1].strip()
    ylabels.append(f"<b>{code}</b>  {short[:30]}")

xlabels = [f"test {i+1}" for i in range(ncols)]

# build a discrete colorscale
vals = sorted(COLORS)
n = len(vals)
colorscale = []
for i, v in enumerate(vals):
    colorscale.append([i / n, COLORS[v]])
    colorscale.append([(i + 1) / n, COLORS[v]])

# remap z to contiguous indices for the discrete scale
remap = {v: i for i, v in enumerate(vals)}
z_idx = [[None if c is None else remap[c] for c in row] for row in z]

fig = go.Figure(go.Heatmap(
    z=z_idx, x=xlabels, y=ylabels, text=text, customdata=hover,
    texttemplate="%{text}", textfont=dict(size=18, color="white"),
    hovertemplate="%{customdata}<extra></extra>",
    colorscale=colorscale, zmin=-0.5, zmax=n - 0.5, showscale=False,
    xgap=3, ygap=3))

# manual legend via invisible scatter
leg = [("pass", COLORS[4]), ("inconclusive", COLORS[2]),
       ("fail / refute (✗* = intended)", COLORS[1]), ("pending", COLORS[0])]
for name, col in leg:
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(size=13, color=col, symbol="square"), name=name, showlegend=True))

CAPTION = ("<b>Interpretation:</b> One-glance status of the whole investigation. Arc-1/4/7/8 growth-law claims all "
           "pass; mcs-02's three red cells are the <i>intended</i> energetic defect (✗*), which mcs-05 then repairs "
           "(five greens). Open work is the single-cell heterogeneity CV, still under-dispersed (mcs-03 red / "
           "inconclusive) with its restoration fix designed but not yet run (mcs-06 grey).")

fig.update_layout(
    title=dict(text="<b>Investigation scorecard — what is proven vs. open</b><br>"
                    "<sub>rows = 8 mcs studies · cells = behavior tests · hover for test name + observed value</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=560, width=1050,
    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    xaxis=dict(side="top"),
    legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="center", x=0.5),
    margin=dict(t=110, b=120, l=250),
)
fig.add_annotation(text=CAPTION, xref="paper", yref="paper", x=0.5, y=-0.20,
    showarrow=False, align="center", font=dict(size=11.5, color="#333"),
    xanchor="center", yanchor="top", width=1000,
    bordercolor="#ccc", borderwidth=1, borderpad=8, bgcolor="#f7f7f7")

out = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
    "mcs-04-capstone-downshift", "viz"))
os.makedirs(out, exist_ok=True)
p = os.path.join(out, "investigation_scorecard.html")
fig.write_html(p, include_plotlyjs="cdn", full_html=True)
print("wrote", p)
