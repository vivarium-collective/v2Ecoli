#!/usr/bin/env python
"""ATP-synthase-cap fix — respiration turns ON (before vs after flux balance).

Before = mcs05_etc_diagnostic.json (ATP synthase runs in reverse; near-zero O2).
After  = mcs05_integrated_validation.json (reverse flux capped; forward synthesis
         restored, O2 consumed, acetate overflow appears, yield in measured band).
Grouped before/after bars for biomass yield, RQ, O2 uptake, acetate secretion, plus
the ATP-synthase forward/reverse diagnosis. Measured biomass-yield band 0.355-0.444
and RQ~1 target are shaded.
"""
from __future__ import annotations
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = json.load(open(os.path.join(HERE, "mcs05_etc_diagnostic.json")))
AFT = json.load(open(os.path.join(HERE, "mcs05_integrated_validation.json")))
FIX = json.load(open(os.path.join(HERE, "mcs05_fix_result.json")))

before = DIAG["baseline"]
aps = DIAG["atp_synthase"]

C_BEF, C_AFT = "#E45756", "#2CA02C"  # before (broken) / after (fixed)
YIELD_LO, YIELD_HI = 0.355, 0.444

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "<b>a</b>  Biomass yield<br><sub>gDW / g glucose  (measured band shaded)</sub>",
        "<b>b</b>  Respiratory quotient<br><sub>CO₂/O₂  (respiration → RQ≈1)</sub>",
        "<b>c</b>  O₂ uptake<br><sub>mmol/gDW/h  (the cell starts breathing)</sub>",
        "<b>d</b>  Acetate secretion<br><sub>mmol/gDW/h  (overflow appears)</sub>",
        "<b>e</b>  ATP-synthase flux (diagnosis)<br><sub>before: runs backwards, hydrolysing ATP</sub>",
        None,
    ),
    horizontal_spacing=0.09, vertical_spacing=0.18,
    specs=[[{}, {}, {}], [{}, {"colspan": 2}, None]],
)

def pair(rr, cc, bef, aft, unit, hov):
    fig.add_trace(go.Bar(x=["before"], y=[bef], marker_color=C_BEF, width=0.5,
        text=[f"{bef:.3g}"], textposition="outside", showlegend=False,
        hovertemplate=f"before<br>{hov}=%{{y:.3f}} {unit}<extra></extra>"), row=rr, col=cc)
    fig.add_trace(go.Bar(x=["after"], y=[aft], marker_color=C_AFT, width=0.5,
        text=[f"{aft:.3g}"], textposition="outside", showlegend=False,
        hovertemplate=f"after<br>{hov}=%{{y:.3f}} {unit}<extra></extra>"), row=rr, col=cc)

# a. biomass yield with measured band
pair(1, 1, before["biomass_yield"], AFT["biomass_yield_gDW_g_glucose"], "gDW/g", "yield")
fig.add_hrect(y0=YIELD_LO, y1=YIELD_HI, line_width=0, fillcolor="#4C78A8", opacity=0.15, row=1, col=1)
fig.add_annotation(x=0.5, y=YIELD_HI, xref="x", yref="y", text="measured 0.355–0.444",
    showarrow=False, yshift=10, font=dict(size=9, color="#4C78A8"), row=1, col=1)

# b. RQ with ~1 target
pair(1, 2, before["RQ"], AFT["RQ"], "", "RQ")
fig.add_hrect(y0=0.9, y1=1.15, line_width=0, fillcolor="#4C78A8", opacity=0.15, row=1, col=2)
fig.add_annotation(x=0.5, y=1.15, xref="x2", yref="y2", text="respiratory RQ≈1",
    showarrow=False, yshift=10, font=dict(size=9, color="#4C78A8"), row=1, col=2)

# c. O2 uptake
pair(1, 3, before["o2_uptake_mmol_gDW_h"], AFT["exchange_mmol_gDW_h"]["o2"], "mmol/gDW/h", "O₂")

# d. acetate secretion — before value not measured in diagnostic; show after only
fig.add_trace(go.Bar(x=["before"], y=[0.0], marker_color=C_BEF, width=0.5,
    text=["n/a"], textposition="outside", showlegend=False,
    hovertemplate="before<br>acetate not reported in diagnostic<extra></extra>"), row=2, col=1)
fig.add_trace(go.Bar(x=["after"], y=[AFT["exchange_mmol_gDW_h"]["acetate"]], marker_color=C_AFT,
    width=0.5, text=[f"{AFT['exchange_mmol_gDW_h']['acetate']:.1f}"], textposition="outside",
    showlegend=False, hovertemplate="after<br>acetate=%{y:.2f} mmol/gDW/h<extra></extra>"), row=2, col=1)
fig.add_annotation(x=0, y=0, xref="x4", yref="y4", text="not reported",
    showarrow=False, yshift=14, font=dict(size=9, color="#999"), row=2, col=1)

# e. ATP synthase forward/reverse (before diagnosis)
fig.add_trace(go.Bar(x=["forward<br>(synthesis)", "reverse<br>(hydrolysis)"],
    y=[aps["forward_synthesis_flux"], aps["reverse_hydrolysis_flux"]],
    marker_color=["#4C78A8", C_BEF],
    text=[f"{aps['forward_synthesis_flux']:.1f}", f"{aps['reverse_hydrolysis_flux']:.1f}"],
    textposition="outside", showlegend=False,
    hovertemplate="%{x}<br>flux=%{y:.2f} a.u.<extra></extra>"), row=2, col=2)
fig.add_annotation(x=1, y=aps["reverse_hydrolysis_flux"], xref="x5", yref="y5",
    text="<b>fix caps this reverse flux →</b><br>net synthesis turns forward,<br>"
         "O₂ uptake rises 0.9 → 16.9 (panel c)",
    showarrow=True, arrowhead=2, ax=-120, ay=-30, align="left",
    font=dict(size=10, color=C_AFT), row=2, col=2)

fig.update_yaxes(rangemode="tozero")
fig.update_yaxes(title_text="gDW/g glucose", range=[0, 0.95], row=1, col=1)
fig.update_yaxes(title_text="RQ", row=1, col=2)
fig.update_yaxes(title_text="mmol/gDW/h", row=1, col=3)
fig.update_yaxes(title_text="mmol/gDW/h", row=2, col=1)
fig.update_yaxes(title_text="FBA flux (a.u.)", row=2, col=2)

CAPTION = ("<b>Interpretation:</b> The defect was a mis-signed constraint — ATP synthase ran in reverse (7.5 a.u. "
           "hydrolysis, zero synthesis), so ATP came from substrate-level phosphorylation and the model behaved "
           "fermentatively despite oxygen. Capping the reverse flux flips it: O₂ uptake jumps ~19×, RQ falls 2.94→1.11, "
           "acetate overflow appears, and biomass yield drops from an unphysical 0.82 into the measured 0.355–0.444 band.")

fig.update_layout(
    title=dict(text="<b>Capping ATP-synthase reversal turns respiration ON</b><br>"
                    "<sub>mcs-05 — before (broken, red) vs after fix (green)</sub>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=760, width=1150, bargap=0.25,
    margin=dict(t=100, b=120),
)
fig.add_annotation(text=CAPTION, xref="paper", yref="paper", x=0.5, y=-0.14,
    showarrow=False, align="center", font=dict(size=11.5, color="#333"),
    xanchor="center", yanchor="top", width=1080,
    bordercolor="#ccc", borderwidth=1, borderpad=8, bgcolor="#f7f7f7")

out = os.path.abspath(os.path.join(HERE, "..", "..", "..", "studies",
    "mcs-05-etc-stoichiometry-fix", "viz"))
os.makedirs(out, exist_ok=True)
p = os.path.join(out, "atp_fix_flux_balance.html")
fig.write_html(p, include_plotlyjs="cdn", full_html=True)
print("wrote", p)
