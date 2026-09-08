import os
import plotly.graph_objects as go
HERE=os.path.dirname(os.path.abspath(__file__))
# axis, pre, post, kind(target/heldout/held)
rows=[
 ("biomass yield (gDW/g glc)",0.816,0.390,"target"),
 ("RQ (CO2/O2)",2.94,1.11,"target"),
 ("growth rate (1/h)",0.896,0.883,"held"),
 ("division time (min)",45.0,45.3,"held"),
 ("cell mass (fg)",1575,1608,"held"),
 ("origins / cell",2.43,2.47,"held"),
 ("protein fraction",0.425,0.440,"held"),
 ("RNA fraction",0.132,0.135,"held"),
]
labels=[r[0] for r in rows]
pct=[(r[2]-r[1])/r[1]*100 for r in rows]
colors=["#2CA02C" if r[3]=="target" else "#9DA7B3" for r in rows]
fig=go.Figure(go.Bar(y=labels[::-1],x=pct[::-1],orientation="h",
    marker_color=colors[::-1],
    text=[f"{r[1]:.3g} → {r[2]:.3g}" for r in rows][::-1],textposition="outside",
    hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>"))
fig.add_vrect(x0=-5,x1=5,line_width=0,fillcolor="#9DA7B3",opacity=0.10)
fig.add_annotation(x=0,y=-0.7,text="±5% 'held' zone",showarrow=False,font=dict(size=10,color="#888"),yref="paper")
fig.update_layout(
    title=dict(text="<b>mcs-05 regression check — the fix corrects energetics, everything else holds</b><br>"
        "<sub>Green = target axes moved into band (yield -52%, RQ -62%); grey = held within a few percent (no regression)</sub>",
        x=0.5,xanchor="center"),
    xaxis_title="% change with fix (pre-fix → with-fix)", template="plotly_white",
    height=460,width=920,margin=dict(t=95,b=50,l=200))
out=os.path.abspath(os.path.join(HERE,"..","..","..","studies","mcs-05-etc-stoichiometry-fix","viz","mcs05_regression.html"))
fig.write_html(out,include_plotlyjs="cdn",full_html=True)
print("wrote",out)
