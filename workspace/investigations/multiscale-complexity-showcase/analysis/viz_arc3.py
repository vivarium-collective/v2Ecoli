#!/usr/bin/env python
"""Arc 3 figure — single-cell heterogeneity is under-dispersed.

Reads arc3_heterogeneity.json and renders a 2-panel interactive figure:
  A. interdivision-time distribution across the ensemble (per-cell), mean marked
  B. CV accumulating across generations vs the biological 10-30% band
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "arc3_heterogeneity.json")))
cells = D["cells"]
taus = [c["division_time_min"] for c in cells]
by_gen = defaultdict(list)
for c in cells:
    by_gen[c["gen"]].append(c["division_time_min"])
gens = sorted(by_gen)
cv_by_gen = [(np.std(by_gen[g]) / np.mean(by_gen[g])) if len(by_gen[g]) > 1 else 0.0 for g in gens]
ACC = "#4C78A8"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>A</b>  Interdivision times cluster tightly<br><sub>18 cells, 6 seeds — mean 49 min, CV 7%</sub>",
        "<b>B</b>  Heterogeneity stays below the biological band<br><sub>CV accumulates across generations but never reaches 10-30%</sub>",
    ),
    horizontal_spacing=0.13,
)

# Panel A — tau distribution (strip + box)
fig.add_trace(go.Box(y=taus, name="", boxpoints="all", jitter=0.5, pointpos=0,
                     marker=dict(color=ACC, size=8, opacity=0.6),
                     line=dict(color=ACC), fillcolor="rgba(76,120,168,0.15)",
                     hovertemplate="tau=%{y:.1f} min<extra></extra>"), row=1, col=1)
fig.add_hline(y=float(np.mean(taus)), line=dict(color="#333", dash="dot"),
              annotation_text=f"mean {np.mean(taus):.0f} min", annotation_position="right", row=1, col=1)

# Panel B — CV vs generation, with biological band
fig.add_hrect(y0=0.10, y1=0.30, line_width=0, fillcolor="#2CA02C", opacity=0.18, row=1, col=2)
fig.add_annotation(x=gens[len(gens)//2], y=0.20, text="biological E. coli band (10-30%)",
                   showarrow=False, font=dict(size=11, color="#2CA02C"), row=1, col=2)
fig.add_trace(go.Scatter(x=gens, y=cv_by_gen, mode="lines+markers+text",
                         line=dict(color="#E45756", width=3), marker=dict(size=12),
                         text=[f"{v:.1%}" for v in cv_by_gen], textposition="top center",
                         name="model CV",
                         hovertemplate="gen %{x}<br>CV %{y:.3f}<extra></extra>"), row=1, col=2)

fig.update_yaxes(title_text="interdivision time (min)", row=1, col=1)
fig.update_xaxes(title_text="generation", tickmode="array", tickvals=gens, row=1, col=2)
fig.update_yaxes(title_text="CV of interdivision time", range=[0, 0.34], row=1, col=2)

fig.update_layout(
    title=dict(text="<b>Arc 3 — The model under-produces single-cell heterogeneity</b><br>"
                    "<sub>Order from noise, but too little of it: CV ~7% vs the biological 10-30%</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=520, width=1050, showlegend=False,
    margin=dict(t=110, b=70),
)

out_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
                          "mcs-03-single-cell-heterogeneity", "viz"))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "arc3_heterogeneity.html")
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print("wrote", out)
