#!/usr/bin/env python
"""Arc 2 figure — the maintenance-ATP fix is refuted.

Reads arc2_sweep.json and renders a 2-panel interactive Plotly figure:
  A. biomass yield vs maintenance-ATP scale (GAM & NGAM), against the measured band
  B. O2:glucose and RQ vs scale, against the healthy-respiration region
Both flat -> the FBA is insensitive to maintenance ATP; the defect is ATP SUPPLY.
"""
from __future__ import annotations
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "arc2_sweep.json")))
pts = S["sweep"]
band = S["measured_band"]

gam = sorted([p for p in pts if p["knob"] in ("GAM", "baseline")], key=lambda p: p["scale"])
ngam = sorted([p for p in pts if p["knob"] in ("NGAM", "baseline")], key=lambda p: p["scale"])
C_GAM, C_NGAM = "#E45756", "#4C78A8"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>A</b>  Biomass yield ignores maintenance ATP<br><sub>target = measured band (green); model sits ~2× above</sub>",
        "<b>B</b>  Respiration never turns on<br><sub>O2:glucose stays ~0.14; RQ stays ~2.6 (healthy ≈ 1)</sub>",
    ),
    horizontal_spacing=0.12,
)

# Panel A — yield vs scale
fig.add_hrect(y0=band[0], y1=band[1], line_width=0, fillcolor="#2CA02C", opacity=0.18, row=1, col=1)
fig.add_hline(y=S["theoretical_max"], line=dict(color="#2CA02C", dash="dash", width=1), row=1, col=1)
fig.add_annotation(x=1.0, y=(band[0]+band[1])/2, text="measured band", showarrow=False,
                   font=dict(size=11, color="#2CA02C"), xref="x", yref="y", yshift=0, xshift=0)
fig.add_annotation(x=8, y=S["theoretical_max"], text="1st-principles ceiling 0.538", showarrow=False,
                   font=dict(size=10, color="#2CA02C"), yshift=10, row=1, col=1)
for grp, c, name in ((gam, C_GAM, "GAM (dark_atp) ×"), (ngam, C_NGAM, "NGAM ×")):
    fig.add_trace(go.Scatter(x=[p["scale"] for p in grp], y=[p["biomass_yield_gDW_g_glucose"] for p in grp],
                             mode="lines+markers", line=dict(color=c, width=2),
                             marker=dict(size=10), name=name,
                             hovertemplate=name+"%{x}<br>yield=%{y:.3f}<extra></extra>"), row=1, col=1)

# Panel B — O2:glucose and RQ vs scale
for grp, c, name, key in ((gam, C_GAM, "O2:glc GAM×", "O2_glucose"),
                          (ngam, C_NGAM, "O2:glc NGAM×", "O2_glucose")):
    fig.add_trace(go.Scatter(x=[p["scale"] for p in grp], y=[p[key] for p in grp],
                             mode="lines+markers", line=dict(color=c, width=2), marker=dict(size=9),
                             name=name, showlegend=False,
                             hovertemplate=name+"%{x}<br>O2:glc=%{y:.3f}<extra></extra>"), row=1, col=2)
fig.add_hrect(y0=1.0, y1=2.0, line_width=0, fillcolor="#2CA02C", opacity=0.12, row=1, col=2)
fig.add_annotation(x=5, y=1.5, text="healthy O2:glucose (≈1–2)", showarrow=False,
                   font=dict(size=10, color="#2CA02C"), row=1, col=2)

fig.update_xaxes(title_text="maintenance-ATP scale factor (×)", type="log", row=1, col=1)
fig.update_yaxes(title_text="biomass yield (gDW/g glucose)", range=[0, 0.9], row=1, col=1)
fig.update_xaxes(title_text="maintenance-ATP scale factor (×)", type="log", row=1, col=2)
fig.update_yaxes(title_text="O2 : glucose (mol/mol)", range=[0, 2.2], row=1, col=2)

fig.update_layout(
    title=dict(text="<b>Arc 2 — The scoped maintenance-ATP fix is refuted</b><br>"
                    "<sub>NGAM to 10× (83.9 mmol/gDW/h, verified at the FBA) barely moves yield: the defect is ATP SUPPLY, not demand</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=520, width=1080,
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.25),
    margin=dict(t=110, b=90),
)

out_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
                          "mcs-02-metabolism-energy-balance", "viz"))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "arc2_energy_balance.html")
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print("wrote", out)
