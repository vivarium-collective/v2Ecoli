#!/usr/bin/env python
"""Arc 1 figure — ppGpp coordinates growth and replication.

Reads arc1_minimal.json + arc1_rich.json (produced by observables.py --json) and
renders a 3-panel interactive Plotly figure:
  A. origins-per-cell vs growth rate (Cooper-Helmstetter scaling) with per-cell points
  B. ppGpp vs growth rate (stringent-response anti-correlation)
  C. initiation mass constancy (Donachie) across the 43% growth change
"""
from __future__ import annotations
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
MIN = json.load(open(os.path.join(HERE, "arc1_minimal.json")))
RICH = json.load(open(os.path.join(HERE, "arc1_rich.json")))

C_MIN, C_RICH = "#4C78A8", "#E45756"  # minimal / rich


def gr(d):  # growth per hour
    return d["growth_rate_per_h"]["mean"]


fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "<b>A</b>  Replication scales with growth<br><sub>origins/cell vs μ (Cooper-Helmstetter)</sub>",
        "<b>B</b>  ppGpp anti-correlates with growth<br><sub>the stringent-response signal</sub>",
        "<b>C</b>  Initiation mass is ~constant<br><sub>mass per origin (Donachie)</sub>",
    ),
    horizontal_spacing=0.08,
)

# Panel A — origins vs growth, per-cell points + condition means + connecting line
for d, c, name in ((MIN, C_MIN, "minimal (glucose)"), (RICH, C_RICH, "rich (+amino acids)")):
    g = gr(d)
    fig.add_trace(go.Scatter(
        x=[g] * len(d["origins_per_cell"]["values"]), y=d["origins_per_cell"]["values"],
        mode="markers", marker=dict(color=c, size=9, opacity=0.45, line=dict(width=0)),
        name=name, legendgroup=name, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[g], y=[d["origins_per_cell"]["mean"]], mode="markers",
        marker=dict(color=c, size=18, symbol="diamond", line=dict(color="white", width=1.5)),
        name=name, legendgroup=name, showlegend=False,
        hovertemplate=f"{name}<br>μ=%{{x:.2f}}/h<br>origins=%{{y:.2f}}<extra></extra>"),
        row=1, col=1)
fig.add_trace(go.Scatter(
    x=[gr(MIN), gr(RICH)], y=[MIN["origins_per_cell"]["mean"], RICH["origins_per_cell"]["mean"]],
    mode="lines", line=dict(color="#888", dash="dot"), showlegend=False,
    hoverinfo="skip"), row=1, col=1)
slope = (RICH["origins_per_cell"]["mean"] - MIN["origins_per_cell"]["mean"]) / (gr(RICH) - gr(MIN))
fig.add_annotation(x=(gr(MIN)+gr(RICH))/2, y=(MIN["origins_per_cell"]["mean"]+RICH["origins_per_cell"]["mean"])/2,
                   text=f"slope +{slope:.1f}/(1/h)", showarrow=False, yshift=22,
                   font=dict(size=12, color="#444"), row=1, col=1)

# Panel B — ppGpp vs growth
for d, c, name in ((MIN, C_MIN, "minimal (glucose)"), (RICH, C_RICH, "rich (+amino acids)")):
    g = gr(d)
    fig.add_trace(go.Scatter(
        x=[g] * len(d["ppgpp_conc"]["values"]), y=d["ppgpp_conc"]["values"],
        mode="markers", marker=dict(color=c, size=9, opacity=0.45), showlegend=False,
        legendgroup=name), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[g], y=[d["ppgpp_conc"]["mean"]], mode="markers",
        marker=dict(color=c, size=18, symbol="diamond", line=dict(color="white", width=1.5)),
        showlegend=False, legendgroup=name,
        hovertemplate=f"{name}<br>μ=%{{x:.2f}}/h<br>ppGpp=%{{y:.1f}}<extra></extra>"), row=1, col=2)
ratio = MIN["ppgpp_conc"]["mean"] / RICH["ppgpp_conc"]["mean"]
fig.add_annotation(x=gr(MIN), y=MIN["ppgpp_conc"]["mean"], text=f"{ratio:.1f}× higher<br>when starved",
                   showarrow=True, arrowhead=0, ax=40, ay=-10, font=dict(size=11, color=C_MIN), row=1, col=2)

# Panel C — initiation mass bars with a "constant" band
im_min, im_rich = MIN["initiation_mass_fg"]["mean"], RICH["initiation_mass_fg"]["mean"]
fig.add_trace(go.Bar(x=["minimal", "rich"], y=[im_min, im_rich],
                     marker_color=[C_MIN, C_RICH], showlegend=False,
                     text=[f"{im_min:.0f} fg", f"{im_rich:.0f} fg"], textposition="outside",
                     hovertemplate="%{x}<br>init mass %{y:.0f} fg<extra></extra>"), row=1, col=3)
mean_im = (im_min + im_rich) / 2
fig.add_hrect(y0=mean_im*0.9, y1=mean_im*1.1, line_width=0, fillcolor="#2CA02C", opacity=0.12, row=1, col=3)
fig.add_annotation(x=0.5, y=mean_im, text="±10% band", showarrow=False, font=dict(size=11, color="#2CA02C"),
                   xref="x3", yref="y3", yshift=32)

fig.update_xaxes(title_text="growth rate μ (1/h)", row=1, col=1)
fig.update_yaxes(title_text="origins per cell", row=1, col=1)
fig.update_xaxes(title_text="growth rate μ (1/h)", row=1, col=2)
fig.update_yaxes(title_text="ppGpp (conc)", row=1, col=2)
fig.update_yaxes(title_text="mass per origin (fg)", range=[0, max(im_min, im_rich)*1.25], row=1, col=3)

fig.update_layout(
    title=dict(text="<b>Arc 1 — A single signal (ppGpp) coordinates growth and replication</b><br>"
                    "<sub>v2ecoli reproduces the bacterial growth law natively: no correction needed</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=520, width=1150,
    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    margin=dict(t=110, b=90),
)

out_dir = os.path.join(HERE, "..", "..", "..", "studies",
                       "mcs-01-ppgpp-replication-coupling", "viz")
out_dir = os.path.abspath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "arc1_ppgpp_replication.html")
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print("wrote", out)
