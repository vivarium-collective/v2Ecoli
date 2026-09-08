#!/usr/bin/env python
"""Cross-scale phase portrait — one signal moves three scales together.

x = ppGpp concentration, y = origins per cell, marker color = growth rate,
marker size = ribosome content. One point per nutrient condition (minimal/basal,
rich/+AA), joined by an arrow showing the coordinated trajectory: lowering ppGpp
simultaneously raises replication, growth and ribosome allocation.
"""
from __future__ import annotations
import json, os
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
MIN = json.load(open(os.path.join(HERE, "arc1_minimal.json")))
RICH = json.load(open(os.path.join(HERE, "arc1_rich.json")))


def pt(d):
    return dict(
        ppgpp=d["ppgpp_conc"]["mean"],
        origins=d["origins_per_cell"]["mean"],
        growth=d["growth_rate_per_h"]["mean"],
        ribo=d["ribosome_conc"]["mean"],
    )


m, r = pt(MIN), pt(RICH)
fig = go.Figure()

# faint per-cell cloud for context
for d, label in ((MIN, "minimal"), (RICH, "rich")):
    fig.add_trace(go.Scatter(
        x=d["ppgpp_conc"]["values"], y=d["origins_per_cell"]["values"],
        mode="markers", marker=dict(color="#bbb", size=7, opacity=0.5),
        showlegend=False, hoverinfo="skip"))

# size scaling for ribosome content
smin, smax = min(m["ribo"], r["ribo"]), max(m["ribo"], r["ribo"])
def size(v):
    return 34 + (v - smin) / (smax - smin + 1e-9) * 26

for p, name, cond in ((m, "minimal (glucose, starved)", "minimal"),
                      (r, "rich (+amino acids, fed)", "rich")):
    fig.add_trace(go.Scatter(
        x=[p["ppgpp"]], y=[p["origins"]], mode="markers",
        marker=dict(size=size(p["ribo"]), color=[p["growth"]],
                    colorscale="Viridis", cmin=m["growth"], cmax=r["growth"],
                    showscale=(name.startswith("rich")),
                    colorbar=dict(title="growth μ<br>(1/h)", x=1.02, len=0.7),
                    line=dict(color="white", width=2)),
        name=name, showlegend=True,
        hovertemplate=(f"<b>{name}</b><br>ppGpp=%{{x:.1f}}<br>origins=%{{y:.2f}}"
                       f"<br>μ={p['growth']:.2f}/h<br>ribosome={p['ribo']:.1f} a.u.<extra></extra>")))

# arrow from minimal -> rich showing the coordinated trajectory
fig.add_annotation(x=r["ppgpp"], y=r["origins"], ax=m["ppgpp"], ay=m["origins"],
    xref="x", yref="y", axref="x", ayref="y", showarrow=True,
    arrowhead=3, arrowsize=1.4, arrowwidth=2.5, arrowcolor="#333")
fig.add_annotation(x=(m["ppgpp"] + r["ppgpp"]) / 2, y=(m["origins"] + r["origins"]) / 2,
    text="<b>nutrient upshift</b><br>ppGpp ↓  →  origins ↑, μ ↑, ribosomes ↑",
    showarrow=False, font=dict(size=12, color="#333"), xshift=70, yshift=-8,
    align="left", bgcolor="rgba(255,255,255,0.75)")

fig.add_annotation(x=m["ppgpp"], y=m["origins"], text="high ppGpp<br>= stringent / slow",
    showarrow=False, yshift=-42, font=dict(size=10, color="#888"))
fig.add_annotation(x=r["ppgpp"], y=r["origins"], text="low ppGpp<br>= relaxed / fast",
    showarrow=False, yshift=38, font=dict(size=10, color="#888"))

fig.update_xaxes(title_text="ppGpp concentration  (the master signal →)",
                 autorange="reversed")  # reversed so 'fed/fast' is to the right
fig.update_yaxes(title_text="origins per cell (replication scale)", rangemode="tozero")

CAPTION = ("<b>Interpretation:</b> A single intracellular signal, ppGpp, sets the operating point across three "
           "scales at once. As ppGpp falls (nutrient upshift, right→left is starved→fed here), replication origins, "
           "growth rate (color) and ribosome content (marker size) all move together — the coordination is an "
           "emergent property of the whole-cell model, not an imposed correlation.")

fig.update_layout(
    title=dict(text="<b>Cross-scale phase portrait — ppGpp coordinates replication, growth and ribosomes</b><br>"
                    "<sub>each point = one nutrient condition · color = growth rate · size = ribosome content</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=620, width=980,
    legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.45),
    margin=dict(t=100, b=150, r=120),
)
fig.add_annotation(text=CAPTION, xref="paper", yref="paper", x=0.5, y=-0.30,
    showarrow=False, align="center", font=dict(size=11.5, color="#333"),
    xanchor="center", yanchor="top", width=900,
    bordercolor="#ccc", borderwidth=1, borderpad=8, bgcolor="#f7f7f7")

out = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
    "mcs-04-capstone-downshift", "viz"))
os.makedirs(out, exist_ok=True)
p = os.path.join(out, "cross_scale_phase_portrait.html")
fig.write_html(p, include_plotlyjs="cdn", full_html=True)
print("wrote", p)
