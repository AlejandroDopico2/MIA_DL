import tensorflow as tf 
from keras.metrics import Metric
from keras.models import Model
from keras.optimizers import Optimizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.data import Dataset
import plotly.graph_objects as go 
import plotly.express as px 
from data import WalmartDataset
import numpy as np 
from typing import List 
from model import WalmartModel




def plot_series(
        model: WalmartModel,
        data: WalmartDataset,
        title: str = 'Predictions'
    ) -> go.Figure:
    """Returns a Plotly figure to display the time series.

    Args:
        model (Model): Walmart model.
        data (Dataset): Walmart Dataset.
        batch_size (int): Batch size.
        title (str): Figure title.

    Returns:
        go.Figure: Plotly figure.
    """
    stores = np.split(model.predict(data), len(data.stores))
    
    fig = go.Figure()
    colors = px.colors.qualitative.D3

    for i, store in enumerate(stores):
        fig.add_traces([
            go.Scatter(x=data.dates[model.seq_len:], y=data[i].to_numpy(model.seq_len)[1], 
                       name=f'Store {data[i].ID}', legendgroup=f'{data[i].ID}', 
                       line=dict(color=colors[i])),
            go.Scatter(x=data.dates[model.seq_len:], y=store, 
                    legendgroup=f'{data[i].ID}', name=f'Store {data[i].ID} [pred]',
                    line=dict(color=colors[i], dash='dot'), showlegend=False)
        ])

    fig.update_layout(
        title_text=title, height=600, width=1200, template='plotly_dark', 
        margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(title_text='Sales')
    )
    return fig 

def plot_errors(
        models: List[WalmartModel],
        data: WalmartDataset,
        title: str = 'Difference between real and predicted values with different models'
    ) -> go.Figure:
    """Returns a Plotly figure to display the time series.

    Args:
        model (Model): Keras model.
        data (Dataset): Walmart Dataset.

    Returns:
        go.Figure: Plotly figure.
    """
    errors = []
    for model in models:
        stores = np.split(model.predict(data), len(data.stores))
        errors.append(
            np.mean(np.stack([(data[i].to_numpy(model.seq_len)[1]-store) for i, store in enumerate(stores)]), axis=0),
        )

    fig = go.Figure()
    colors = px.colors.qualitative.D3
    for error, model in zip(errors, models):
        fig.add_trace(
            go.Scatter(x=data.dates[model.seq_len:], y=error, 
                       name=f'{model.seq_len}',
                       line=dict(color=colors[model.seq_len])),
        )
    fig.update_layout(
        title_text=title, height=600, width=1200, template='plotly_dark', 
        margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(title_text='difference')
    )
    return fig 