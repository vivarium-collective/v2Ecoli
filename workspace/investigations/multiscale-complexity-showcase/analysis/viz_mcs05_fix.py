import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
HERE=os.path.dirname(os.path.abspath(__file__))
before={"biomass_yield":0.816,"RQ":2.94,"O2_glucose":0.137,"growth":0.956}
# integrated, calibrated (cap=4.5) fix validated natively
fix=json.load(open(os.path.join(HERE,"mcs05_integrated_validation.json")))
after={"biomass_yield":fix["biomass_yield_gDW_g_glucose"],"RQ":fix["RQ"],"O2_glucose":fix["O2_glucose"],"growth":fix["growth_rate_per_h"]}
panels=[("biomass yield (gDW/g glc)","biomass_yield",[0.355,0.444],1.0),
        ("respiratory quotient RQ","RQ",[0.8,1.3],3.2),
        ("O2 : glucose","O2_glucose",[1.0,2.0],2.0),
        ("growth rate (1/h)","growth",None,1.4)]
fig=make_subplots(rows=1,cols=4,subplot_titles=[p[0] for p in panels],horizontal_spacing=0.06)
for i,(title,key,band,ymax) in enumerate(panels,1):
    if band:
        fig.add_hrect(y0=band[0],y1=band[1],line_width=0,fillcolor="#2CA02C",opacity=0.18,row=1,col=i)
    fig.add_trace(go.Bar(x=["before","after fix"],y=[before[key],after[key]],
        marker_color=["#E45756","#4C78A8"],showlegend=False,
        text=[f"{before[key]:.2f}",f"{after[key]:.2f}"],textposition="outside",
        hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"),row=1,col=i)
    fig.update_yaxes(range=[0,ymax],row=1,col=i)
fig.update_layout(
    title=dict(text="<b>mcs-05 — the ETC fix lands: ATP synthase net-forward restores respiration</b><br>"
        "<sub>Green = healthy band. Integrated fix (cap=4.5): yield centred IN-BAND, RQ→physiological, growth preserved (held-out)</sub>",
        x=0.5,xanchor="center"),
    template="plotly_white",height=460,width=1150,margin=dict(t=100,b=40))
out=os.path.abspath(os.path.join(HERE,"..","..","..","studies","mcs-05-etc-stoichiometry-fix","viz","mcs05_fix_before_after.html"))
fig.write_html(out,include_plotlyjs="cdn",full_html=True)
print("wrote",out)
