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
        datas: List[WalmartDataset],
        title: str = 'Predictions',
        n_stores: int = 10
    ) -> go.Figure:
    
    fig = go.Figure()
    colors = px.colors.qualitative.D3
    
    show = True 
    for data in datas:
        preds = np.split(model.predict(data), len(data.stores))
        golds = np.split(data.to_numpy(model.seq_len, norm=False)[1], len(data.stores))
        delay = model.seq_len + WalmartDataset.TARGET_TIMESTEP - 1
        dates = data.dates[delay:]

        for i, (pred, gold) in enumerate(zip(preds, golds)):
            fig.add_traces([
                # gold targets
                go.Scatter(x=dates, y=gold, mode='lines',
                        name=f'Store {data[i].ID}', legendgroup=f'{data[i].ID}', 
                        line=dict(color=colors[i%len(colors)]), showlegend=show),
                # predicted targets 
                go.Scatter(x=dates, y=pred, mode='lines',
                        legendgroup=f'{data[i].ID}', name=f'Store {data[i].ID} [pred]',
                        line=dict(color=colors[i%len(colors)], dash='dot'), showlegend=False)
            ])
            if i >= n_stores:
                break 
        show = False

    fig.update_layout(
        title_text=title, height=600, width=1200, template='seaborn', 
        margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(title_text='Sales')
    )
    return fig 


def plot_errors(
        models: List[WalmartModel],
        datas: List[WalmartDataset],
        title: str = 'Difference between real and predicted values with different models'
    ) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.D3
    show = True
    for data in datas:
        errors = []
        for model in models:
            preds = np.split(model.predict(data), len(data.stores))
            golds = np.split(data.to_numpy(model.seq_len, norm=False)[1], len(data.stores))
            errors.append(np.mean(np.stack([gold-pred for pred, gold in zip(preds, golds)]), axis=0))

        for error, model in zip(errors, models):
            delay = model.seq_len + WalmartDataset.TARGET_TIMESTEP - 1
            dates = data.dates[delay:]
            fig.add_trace(
                go.Scatter(x=dates, y=error, mode='lines', name=f'{model.seq_len}',
                        line=dict(color=colors[model.seq_len]), showlegend=show),
            )
        show = False
    fig.update_layout(
        title_text=title, height=600, width=1200, template='seaborn', 
        margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(title_text='difference')
    )
    return fig 