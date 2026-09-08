import json, os
import plotly.graph_objects as go
HERE=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(HERE,"composition.json")))
m,r=d["minimal"],d["rich"]
fig=go.Figure()
# the two model points
fig.add_trace(go.Scatter(x=[m["growth_per_h"],r["growth_per_h"]],
    y=[m["rna_protein_ratio"],r["rna_protein_ratio"]],
    mode="lines+markers+text", line=dict(color="#4C78A8",width=3),
    marker=dict(size=16,color=["#4C78A8","#E45756"]),
    text=["minimal<br>glucose","rich<br>+amino acids"], textposition="top center",
    name="v2ecoli", hovertemplate="μ=%{x:.2f}/h<br>RNA/protein=%{y:.3f}<extra></extra>"))
# Bremer-Dennis reference band (illustrative steep C-line: RNA/protein ~0.1 at slow -> ~0.45 at fast)
gx=[0.4,1.6]
fig.add_trace(go.Scatter(x=gx,y=[0.12,0.42],mode="lines",line=dict(color="#2CA02C",dash="dash",width=2),
    name="Bremer-Dennis C-line (illustrative)", hovertemplate="expected steep rise<extra></extra>"))
fig.add_annotation(x=1.0,y=0.30,text="model slope +0.021/h<br>(shallow — R-line axis)",showarrow=False,font=dict(size=11,color="#4C78A8"))
fig.add_annotation(x=1.35,y=0.40,text="classic C-line (steep)",showarrow=False,font=dict(size=11,color="#2CA02C"))
fig.update_layout(
    title=dict(text="<b>Composition growth law — RNA/protein rises with growth (shallow)</b><br>"
        "<sub>v2ecoli reproduces the Bremer-Dennis DIRECTION; a carbon-quality ladder is needed for the C-line slope</sub>",
        x=0.5,xanchor="center"),
    xaxis_title="growth rate μ (1/h)", yaxis_title="RNA / protein mass ratio",
    template="plotly_white", height=500, width=850,
    yaxis=dict(range=[0,0.5]), xaxis=dict(range=[0.3,1.7]),
    legend=dict(orientation="h",yanchor="bottom",y=-0.22,xanchor="center",x=0.5), margin=dict(t=90,b=80))
out=os.path.abspath(os.path.join(HERE,"..","..","..","studies","cgl-01-rna-protein-ratio","viz","rna_protein_law.html"))
fig.write_html(out,include_plotlyjs="cdn",full_html=True)
print("wrote",out)
