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


class DenormalizedMAE(Metric):
    def __init__(self, std: float, name='dmae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.std = std 
        self.value = self.add_weight(shape=(), initializer='zeros', name='value')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.multiply(y_true, self.std)
        y_pred = tf.multiply(y_pred, self.std)
        result = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)))
        self.value.assign(result)
        
    def result(self):
        return self.value

def train_walmart(
        model: Model, 
        path: str,
        train_tf: Dataset, 
        val_tf: Dataset,
        optimizer: Optimizer, 
        target_std: float, 
        epochs: int = 100,
        patience: int = 20, 
    ) -> Model:
    """Trains a Walmart model.

    Args:
        model (Model): Keras model.
        path (str): Path to store the model weights.
        optimizer (Optimizer): Optimizer.
        target_std (float): Standard deviation of the target variable.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        patience (int, optional): Number of epochs with no validation improvement. Defaults to 20.
    
    Returns:
        Trained Keras model.
    """
    model.compile(optimizer, 'mse', metrics=[DenormalizedMAE(target_std)])
    callbacks = [
        EarlyStopping(monitor='val_dmae', patience=patience, mode='min', verbose=0),
        ModelCheckpoint(path, monitor='val_dmae', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
    ]
    model.fit(train_tf, epochs=epochs, validation_data=val_tf, callbacks=callbacks)
    model.load_weights(path)
    return model 


def plot_series(
        model: Model,
        data: WalmartDataset,
        means: np.ndarray,
        stds: np.ndarray,
        seq_len: int,
        batch_size: int = 30,
        title: str = 'Predictions'
    ) -> go.Figure:
    """Returns a Plotly figure to display the time series.

    Args:
        model (Model): Keras model.
        data (Dataset): Walmart Dataset.
        means (np.ndarray): Input means.
        stds (np.ndarray): Input standard deviations.

    Returns:
        go.Figure: Plotly figure.
    """
    data_ds = data.to_tf(seq_len, means, stds, batch_size)
    preds = model.predict(data_ds, verbose=0).flatten()*stds[WalmartDataset.TARGET_INDEX]+means[WalmartDataset.TARGET_INDEX]
    stores = np.split(preds, len(data.stores))
    
    fig = go.Figure()
    colors = px.colors.qualitative.D3

    for i, store in enumerate(stores):
        fig.add_traces([
            go.Scatter(x=data.dates[seq_len:], y=data[i].to_numpy(seq_len)[1], name=f'Store {data[i].ID}',
                    legendgroup=f'{data[i].ID}', 
                    line=dict(color=colors[i])),
            go.Scatter(x=data.dates[seq_len:], y=store, 
                    legendgroup=f'{data[i].ID}', name=f'Store {data[i].ID} [pred]',
                    line=dict(color=colors[i], dash='dot'), showlegend=False)
        ])

    fig.update_layout(
        title_text=title, height=600, width=1200, template='plotly_dark', 
        margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(title_text='Sales')
    )
    return fig 

def plot_errors(
        models: List[Model],
        data: WalmartDataset,
        means: np.ndarray,
        stds: np.ndarray,
        seq_lens: List[int],
        batch_size: int = 30,
        title: str = 'Difference between real and predicted values with different models'
    ) -> go.Figure:
    """Returns a Plotly figure to display the time series.

    Args:
        model (Model): Keras model.
        data (Dataset): Walmart Dataset.
        means (np.ndarray): Input means.
        stds (np.ndarray): Input standard deviations.

    Returns:
        go.Figure: Plotly figure.
    """
        
    errors = []
    for model, seq_len in zip(models, seq_lens):
        data_ds = data.to_tf(seq_len, means, stds, batch_size)
        preds = model.predict(data_ds, verbose=0).flatten()*stds[WalmartDataset.TARGET_INDEX]+means[WalmartDataset.TARGET_INDEX]
        stores = np.split(preds, len(data.stores))
        errors.append(
            np.mean(np.stack([(data[i].to_numpy(seq_len)[1]-store) for i, store in enumerate(stores)]), axis=0),
        )

    fig = go.Figure()
    colors = px.colors.qualitative.D3
    for error, seq_len in zip(errors, seq_lens):
        fig.add_trace(
            go.Scatter(x=data.dates[seq_len:], y=error, name=f'{seq_len}',line=dict(color=colors[seq_len])),
        )
    fig.update_layout(
        title_text=title, height=600, width=1200, template='plotly_dark', 
        margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(title_text='difference')
    )
    return fig 