from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_history(
    history: Dict[str, np.ndarray],
    metrics: List[str] = ["loss"],
    ncols: int = 2,
    name: str = "model",
):
    nrows = int(len(metrics) / ncols + 0.5)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        subplot_titles=metrics,
    )
    epochs = np.arange(1, len(history["loss"]) + 1)
    colors = px.colors.qualitative.Plotly
    args = lambda x: dict(
        x=epochs,
        mode="lines+markers",
        marker=dict(color=colors[x]),
        line=dict(color=colors[x]),
    )
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(y=history[metric], name="train", showlegend=(i == 0), **args(0)),
            row=i // ncols + 1,
            col=i % ncols + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=history[f"val_{metric}"], name="val", showlegend=(i == 0), **args(1)
            ),
            row=i // ncols + 1,
            col=i % ncols + 1,
        )
        fig.update_yaxes(title_text=metric)
        fig.update_xaxes(title_text="epochs")
    fig.update_layout(
        template="seaborn",
        height=400 * nrows,
        width=600 * ncols,
        title=f"Training and validation of {name}",
        margin=dict(t=50, b=10, r=10, l=10),
    )
    return fig
