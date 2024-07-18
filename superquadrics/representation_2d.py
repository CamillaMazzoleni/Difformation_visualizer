
import plotly.graph_objects as go
import numpy as np

def create_plot(magnitudes, title=None, withgrid=False, withlabels=False):
    fig = go.Figure()
    norm_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    
    fig.add_trace(go.Heatmap(
        z=norm_magnitudes,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        x=np.linspace(-180, 180, norm_magnitudes.shape[1]),
        y=np.linspace(-90, 90, norm_magnitudes.shape[0])
    ))

    if title is not None:
        fig.update_layout(title=title)

    if not withgrid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    if withlabels:
        fig.update_xaxes(title_text="β (degrees)")
        fig.update_yaxes(title_text="α (degrees)")

    return fig