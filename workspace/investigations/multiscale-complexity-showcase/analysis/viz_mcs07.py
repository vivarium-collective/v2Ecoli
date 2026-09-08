import json, os
import plotly.graph_objects as go
HERE=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(HERE,'mcs07_ribosome_ladder.json')))
L=sorted(d['ladder'], key=lambda x:x['growth_per_h'])
x=[p['growth_per_h'] for p in L]; y=[p['ribosome_conc'] for p in L]; names=[p['condition'] for p in L]
colors=['#8C564B','#E45756','#F58518','#4C78A8']
fig=go.Figure()
fig.add_trace(go.Scatter(x=x,y=y,mode='lines+markers+text',line=dict(color='#333',dash='dot',width=1.5),
  marker=dict(size=16,color=colors),text=names,textposition='top center',
  hovertemplate='%{text}<br>growth %{x:.3f}/h<br>ribosome %{y:.2f}<extra></extra>',showlegend=False))
fig.add_annotation(x=0.6,y=11,text='ribosome fraction rises with growth<br>(Scott-Hwa C-line)',showarrow=False,font=dict(size=12,color='#2CA02C'))
fig.update_xaxes(title_text='growth rate μ (1/h)')
fig.update_yaxes(title_text='ribosome concentration (a.u.)')
fig.update_layout(title=dict(text="<b>mcs-07 — v2ecoli reproduces the ribosome growth law (Scott 2010)</b><br>"
  "<sub>Carbon-quality ladder acetate→succinate→glucose→+AA; enabled by a landed biotin-clamp fix. Absolute μ for acetate/succinate is a gen-1 under-estimate.</sub>",x=0.5,xanchor='center'),
  template='plotly_white',height=500,width=920,margin=dict(t=100,b=60))
out=os.path.abspath(os.path.join(HERE,'..','..','..','studies','mcs-07-ribosome-allocation-law','viz'))
os.makedirs(out,exist_ok=True)
p=os.path.join(out,'mcs07_ribosome_growth_law.html')
fig.write_html(p,include_plotlyjs='cdn',full_html=True); print('wrote',p)
