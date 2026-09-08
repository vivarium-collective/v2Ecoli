#!/usr/bin/env python
"""Population heterogeneity — the model is under-dispersed vs. biology.

Left: per-cell interdivision-time distribution (histogram + strip) with the model
CV (~7%) annotated and the biological 10-30% CV band shaded for contrast, so the
under-dispersion is visually obvious.
Right: adder/size-homeostasis plot (added mass vs birth mass) with the fitted slope.
"""
from __future__ import annotations
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(os.path.join(HERE, "arc3_heterogeneity.json")))
cells = H["cells"]
dt = [c["division_time_min"] for c in cells]
birth = [c["birth_mass_fg"] for c in cells]
added = [c["added_mass_fg"] for c in cells]

mean = H["division_time"]["mean_min"]
cv_model = H["division_time"]["cv"]
slope = H["adder"]["slope"]
icpt = H["adder"]["intercept_fg"]

fig = make_subplots(rows=1, cols=2,
    subplot_titles=(
        f"<b>a</b>  Interdivision time is under-dispersed<br>"
        f"<sub>model CV {cv_model*100:.0f}% vs. biological 10–30%</sub>",
        "<b>b</b>  Size homeostasis (adder plot)<br>"
        "<sub>added mass vs birth mass</sub>"),
    horizontal_spacing=0.12, column_widths=[0.52, 0.48])

# ---- Panel a: division-time distribution + CV bands ----
# shade what 30% and 10% CV would span (mean ± 1 sd) to expose under-dispersion
for cv, col, lbl in ((0.30, "#E45756", "30% CV"), (0.10, "#F2A93B", "10% CV")):
    sd = cv * mean
    fig.add_vrect(x0=mean - sd, x1=mean + sd, line_width=0, fillcolor=col, opacity=0.12,
        row=1, col=1)
    fig.add_annotation(x=mean + sd, y=1.0, xref="x", yref="paper", text=lbl,
        showarrow=False, textangle=-90, xshift=8, font=dict(size=10, color=col))
# model spread band (mean ± 1 sd)
sd_model = H["division_time"]["sd_min"]
fig.add_vrect(x0=mean - sd_model, x1=mean + sd_model, line_width=0, fillcolor="#4C78A8",
    opacity=0.28, row=1, col=1)

fig.add_trace(go.Histogram(x=dt, nbinsx=12, marker_color="#4C78A8",
    marker_line=dict(color="white", width=1), opacity=0.9, name="cells",
    hovertemplate="τ=%{x:.1f} min<br>count=%{y}<extra></extra>", showlegend=False),
    row=1, col=1)
# strip of individual cells
fig.add_trace(go.Scatter(x=dt, y=[-0.6] * len(dt), mode="markers",
    marker=dict(color="#1f3b5c", size=8, opacity=0.6, symbol="line-ns-open"),
    showlegend=False, hovertemplate="cell τ=%{x:.1f} min<extra></extra>"), row=1, col=1)
fig.add_vline(x=mean, line=dict(color="#333", dash="dash", width=1.5), row=1, col=1)
fig.add_annotation(x=mean, y=1.0, xref="x", yref="paper", yshift=-2,
    text=f"<b>model: {mean:.1f} min, CV {cv_model*100:.0f}%</b><br>"
         f"(biology: CV 10–30%)", showarrow=False, font=dict(size=11, color="#1f3b5c"),
    bgcolor="rgba(255,255,255,0.75)")

# ---- Panel b: adder plot ----
fig.add_trace(go.Scatter(x=birth, y=added, mode="markers",
    marker=dict(color="#4C78A8", size=10, opacity=0.75, line=dict(color="white", width=1)),
    showlegend=False, hovertemplate="birth=%{x:.0f} fg<br>added=%{y:.0f} fg<extra></extra>"),
    row=1, col=2)
xr = [min(birth) * 0.98, max(birth) * 1.02]
fig.add_trace(go.Scatter(x=xr, y=[icpt + slope * x for x in xr], mode="lines",
    line=dict(color="#E45756", width=2.5), showlegend=False,
    hovertemplate="fit<extra></extra>"), row=1, col=2)
# reference: perfect adder (slope 0) and timer (slope 1) for orientation
madd = sum(added) / len(added)
fig.add_trace(go.Scatter(x=xr, y=[madd, madd], mode="lines",
    line=dict(color="#2CA02C", width=1.2, dash="dot"), showlegend=False,
    hovertemplate="perfect adder (slope 0)<extra></extra>"), row=1, col=2)
fig.add_annotation(x=xr[1], y=madd, xref="x2", yref="y2", text="ideal adder (slope 0)",
    showarrow=False, font=dict(size=9, color="#2CA02C"), xshift=-58, yshift=10)
fig.add_annotation(x=sum(birth) / len(birth), y=max(added),
    xref="x2", yref="y2", text=f"<b>fitted slope {slope:.2f}</b><br>"
    "(between adder 0 and timer 1;<br>birth-size spread too narrow<br>to resolve — inconclusive)",
    showarrow=False, align="left", font=dict(size=10, color="#B26A00"))

fig.update_xaxes(title_text="interdivision time τ (min)", row=1, col=1)
fig.update_yaxes(title_text="number of cells", row=1, col=1)
fig.update_xaxes(title_text="birth mass (fg)", row=1, col=2)
fig.update_yaxes(title_text="added mass Δ (fg)", row=1, col=2)

CAPTION = ("<b>Interpretation:</b> The whole-cell model divides far too regularly — an interdivision-time CV of "
           f"~{cv_model*100:.0f}% against a biological 10–30% (blue band sits well inside the amber/red bands). Real "
           "single-cell noise is missing. The adder regression (slope "
           f"{slope:.2f}) is inconclusive because the model's birth-size spread is itself too narrow to resolve "
           "the homeostasis mode.")

fig.update_layout(
    title=dict(text="<b>Single-cell heterogeneity is under-dispersed vs. biology</b><br>"
                    f"<sub>{H['n_cells']} cells across seeds · mcs-03</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=560, width=1120, bargap=0.05,
    margin=dict(t=100, b=120),
)
fig.add_annotation(text=CAPTION, xref="paper", yref="paper", x=0.5, y=-0.19,
    showarrow=False, align="center", font=dict(size=11.5, color="#333"),
    xanchor="center", yanchor="top", width=1050,
    bordercolor="#ccc", borderwidth=1, borderpad=8, bgcolor="#f7f7f7")

out = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
    "mcs-03-single-cell-heterogeneity", "viz"))
os.makedirs(out, exist_ok=True)
p = os.path.join(out, "heterogeneity_variance_band.html")
fig.write_html(p, include_plotlyjs="cdn", full_html=True)
print("wrote", p)
