import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
HERE=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(HERE,'mcs05_etc_diagnostic.json')))
aps=d['atp_synthase']
fig=make_subplots(rows=1,cols=2,subplot_titles=(
  "<b>A</b>  ATP synthase runs BACKWARDS<br><sub>zero respiratory ATP; it hydrolyses ATP to pump protons</sub>",
  "<b>B</b>  The cell barely breathes<br><sub>O2 use near zero, RQ impossible for respiration</sub>"),horizontal_spacing=0.16)
fig.add_trace(go.Bar(x=['forward<br>(synthesis)','reverse<br>(hydrolysis)'],
  y=[aps['forward_synthesis_flux'],aps['reverse_hydrolysis_flux']],
  marker_color=['#2CA02C','#E45756'], text=[f"{aps['forward_synthesis_flux']:.1f}",f"{aps['reverse_hydrolysis_flux']:.1f}"],
  textposition='outside', showlegend=False),row=1,col=1)
b=d['baseline']
fig.add_trace(go.Bar(x=['O2 uptake','glucose uptake','RQ (CO2/O2)'],
  y=[b['o2_uptake_mmol_gDW_h'],b['glucose_uptake_mmol_gDW_h'],b['RQ']],
  marker_color=['#4C78A8','#F58518','#B279A2'],
  text=[f"{b['o2_uptake_mmol_gDW_h']:.2f}",f"{b['glucose_uptake_mmol_gDW_h']:.2f}",f"{b['RQ']:.2f}"],
  textposition='outside', showlegend=False),row=1,col=2)
fig.add_hrect(y0=0.7,y1=1.3,line_width=0,fillcolor='#2CA02C',opacity=0.12,row=1,col=2)
fig.add_annotation(x=2,y=1.0,text='healthy RQ ~1',showarrow=False,font=dict(size=10,color='#2CA02C'),row=1,col=2)
fig.update_yaxes(title_text='FBA flux (a.u.)',row=1,col=1)
fig.update_yaxes(title_text='mmol/gDW/h  ·  RQ',row=1,col=2)
fig.update_layout(title=dict(text="<b>mcs-05 — Why maintenance ATP was irrelevant: ATP synthase runs in reverse</b><br>"
  "<sub>ATP is over-supplied by substrate-level phosphorylation and thrown away; the model behaves fermentatively despite O2</sub>",x=0.5,xanchor='center'),
  template='plotly_white',height=480,width=1000,margin=dict(t=105,b=50))
out=os.path.abspath(os.path.join(HERE,'..','..','..','studies','mcs-05-etc-stoichiometry-fix','viz'))
os.makedirs(out,exist_ok=True)
p=os.path.join(out,'mcs05_atp_synthase_reversal.html')
fig.write_html(p,include_plotlyjs='cdn',full_html=True); print('wrote',p)
