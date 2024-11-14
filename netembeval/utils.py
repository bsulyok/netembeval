import numpy as np
import networkx as nx
from plotly import graph_objects as go


def draw(G: nx.Graph, coords: dict) -> go.Figure:
    
    edge_x, edge_y = [], []
    for u, v in G.edges:
        edge_x += [coords[u][0], coords[v][0], None]
        edge_y += [coords[u][1], coords[v][1], None]
    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line_width=1.0,
        line_color='gray',
        showlegend=False,
    )
    
    vert_x = np.array([coords[v][0] for v in G.nodes])
    vert_y = np.array([coords[v][1] for v in G.nodes])
    vertex_trace = go.Scattergl(
        x=vert_x,
        y=vert_y,
        mode='markers',
        marker_size=10.0,
        showlegend=False,
        marker_line_width=1.0,
    )
    
    layout = go.Layout(
        width=800,
        height=800,
        xaxis_scaleanchor='y',
        margin=dict(t=0,r=0,b=0,l=0),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        xaxis_linecolor='black',
        yaxis_linecolor='black',
        xaxis_linewidth=5.0,
        yaxis_linewidth=5.0,
        xaxis_mirror=True,
        yaxis_mirror=True,
    )
    
    fig = go.Figure(data=[edge_trace, vertex_trace], layout=layout)
    return fig