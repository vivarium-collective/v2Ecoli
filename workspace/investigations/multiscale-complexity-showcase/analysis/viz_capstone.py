#!/usr/bin/env python
"""Capstone figure — one nutrient axis moves every scale together.

Synthesises the arcs across the nutrient axis (minimal glucose <-> rich +AA):
a molecular signal (ppGpp), the cell cycle (origins, growth), and metabolism
(respiration RQ) all shift in concert. Steady-state contrast (NOT a temporal
transient). Reads arc1_minimal/rich.json and arc2_baseline/rich_with_aa.json.
"""
from __future__ import annotations
import json, os
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


a1min, a1rich = load("arc1_minimal.json"), load("arc1_rich.json")
a2min, a2rich = load("arc2_baseline.json"), load("arc2_rich_with_aa.json")

# Build the multiscale rows: (scale, label, minimal, rich, hint)
rows = [
    ("molecular",  "ppGpp signal",        a1min["ppgpp_conc"]["mean"],       a1rich["ppgpp_conc"]["mean"],       "stringent alarmone"),
    ("cell cycle", "origins / cell",       a1min["origins_per_cell"]["mean"], a1rich["origins_per_cell"]["mean"], "replication rounds"),
    ("cell cycle", "growth rate (1/h)",    a1min["growth_rate_per_h"]["mean"],a1rich["growth_rate_per_h"]["mean"],"single-cell"),
]
if a2min and a2rich:
    gmin = a2min["exchange_mmol_gDW_h"]["glucose"]
    grich = a2rich["exchange_mmol_gDW_h"]["glucose"]
    rows.append(("metabolism", "glucose uptake (mmol/gDW/h)", gmin, grich, "flux reallocation"))

# Normalize each row to [0,1] across its own (min,rich) for a slopegraph.
fig = go.Figure()
colors = {"molecular": "#B279A2", "cell cycle": "#4C78A8", "metabolism": "#F58518"}
xL, xR = 0, 1
for i, (scale, label, vmin, vrich, hint) in enumerate(rows):
    y = len(rows) - i
    c = colors[scale]
    # slope line between the two nutrient states, annotated with real values
    fig.add_trace(go.Scatter(
        x=[xL, xR], y=[y, y], mode="lines+markers",
        line=dict(color=c, width=3), marker=dict(size=14, color=c),
        showlegend=False,
        hovertemplate=f"{label}<br>minimal %{{customdata[0]}}<br>rich %{{customdata[1]}}<extra></extra>",
        customdata=[[f"{vmin:.2f}", f"{vrich:.2f}"], [f"{vmin:.2f}", f"{vrich:.2f}"]]))
    fig.add_annotation(x=xL, y=y, text=f"<b>{vmin:.2g}</b>", showarrow=False, xshift=-26, font=dict(color=c, size=13))
    fig.add_annotation(x=xR, y=y, text=f"<b>{vrich:.2g}</b>", showarrow=False, xshift=26, font=dict(color=c, size=13))
    fig.add_annotation(x=-0.30, y=y, text=f"{label}<br><sub>{scale} · {hint}</sub>", showarrow=False,
                       xanchor="left", font=dict(size=12))
    # arrow direction cue
    up = vrich > vmin
    fig.add_annotation(x=0.5, y=y, text=("▲ rises" if up else "▼ falls") + " with nutrients",
                       showarrow=False, yshift=16, font=dict(size=10, color=c))

fig.add_annotation(x=xL, y=len(rows) + 0.6, text="<b>MINIMAL</b><br><sub>glucose</sub>", showarrow=False, font=dict(size=13))
fig.add_annotation(x=xR, y=len(rows) + 0.6, text="<b>RICH</b><br><sub>+ amino acids</sub>", showarrow=False, font=dict(size=13))

fig.update_xaxes(visible=False, range=[-0.75, 1.4])
fig.update_yaxes(visible=False, range=[0.3, len(rows) + 1.1])
fig.update_layout(
    title=dict(text="<b>Capstone — one nutrient axis moves every scale together</b><br>"
                    "<sub>A molecular signal, the cell cycle, and metabolism shift in concert (steady-state contrast, not a transient)</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=460, width=920,
    margin=dict(t=100, b=40, l=210, r=90),
)

out_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
                          "mcs-04-capstone-downshift", "viz"))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "capstone_multiscale.html")
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print("wrote", out)
