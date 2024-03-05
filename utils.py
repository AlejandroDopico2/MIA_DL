import gc
from keras.backend import clear_session
from keras.utils import image_dataset_from_directory
from tensorflow.data import Dataset
from keras.optimizers import Optimizer
from keras import Model
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Layer 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "vscode"
import plotly.express as px
import pickle


def reset():
    clear_session()
    gc.collect()

def freeze(model: Layer, maintain: int) -> Layer:
    if maintain == 0: # do not defreeze
        layers = model.layers
    elif maintain == -1: # maitain all 
        return model
    else:
        layers = model.layers[:-maintain]
    for layer in layers:
        layer.trainable = False
    return model 

def load_data(
        base_dir: str, img_size: int = 128, batch_size: int = 64
    ) -> Tuple[Dataset, Dataset, Dataset]:

    train_dataset, val_dataset = image_dataset_from_directory(
        base_dir + "/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        validation_split=0.15,
        subset="both",
        seed=42,
        shuffle=True,
    )

    test_dataset = image_dataset_from_directory(
        base_dir + "/val", image_size=(img_size, img_size), batch_size=batch_size
    )

    return train_dataset, val_dataset, test_dataset

def train_model(
    model: Model,
    train: Dataset,
    val: Dataset,
    path: str,
    epochs: int = 20,
    batch_size: int = 64,
    val_patience: int = 10, 
    lr_patience: int = 5, 
    verbose: Union[str, int] = "auto"
) -> History:
    callbacks = [ 
        EarlyStopping(monitor="val_loss", mode="min", patience=val_patience),
        ReduceLROnPlateau('val_loss', factor=0.2, patience=lr_patience, min_delta=1e-5, min_lr=0)
    ]
    if path:
        callbacks.append(ModelCheckpoint(f'{path}/{model.name}.keras', 'val_loss', save_best_only=True))
    
    history = model.fit(train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, 
                                 verbose=verbose, validation_data=val)
    with open(f'{path}/history.pickle', 'wb') as writer:
        pickle.dump(history.history, writer)
    return history


def plot_history(history: Dict[str, np.ndarray], metrics: List[str] = ['loss'], ncols: int = 2, name: str = 'model'):
    nrows = int(len(metrics)/ncols + 0.5)
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.05, vertical_spacing=0.05,
                        subplot_titles=metrics)
    epochs = np.arange(1, len(history['loss']) + 1)
    colors = px.colors.qualitative.Plotly
    args = lambda x: dict(x=epochs, mode='lines+markers', marker=dict(color=colors[x]), line=dict(color=colors[x]))
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(y=history[metric], name='train', showlegend=(i==0), **args(0)), row=i//ncols+1, col=i%ncols+1)
        fig.add_trace(go.Scatter(y=history[f'val_{metric}'], name=f'val', showlegend=(i==0), **args(1)), row=i//ncols+1, col=i%ncols+1)
        fig.update_yaxes(title_text=metric)
        fig.update_xaxes(title_text='epochs')
    fig.update_layout(
        template='seaborn', height=400*nrows, width=600*ncols, 
        title=f'Training and validation of {name}', margin=dict(t=50, b=10, r=10, l=10)
    )    
    return fig


def number_params(model: Model) -> Tuple[int, int]:
    trainable = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    return trainable, non_trainable

def evaluate(models: List[Model], test: Dataset, batch_size: int) -> List[float]:
    return [model.evaluate(test, batch_size=batch_size, verbose=False)[1] for  model in models]


def comparison(
    models: List[Model], 
    datasets: Tuple[Dataset, Dataset],
    colors: List[str] = px.colors.qualitative.D3, 
    symbols: List[str] = ['circle', 'square']
) -> go.Figure:
    params = [number_params(model)[0] for model in models]
    # order by params 
    order = np.argsort(params)
    models = [models[i] for i in order]
    params = [params[i] for i in order]
    names = [model.name for model in models]
    accs = [evaluate(models, data, BATCH_SIZE) for data in datasets]
    fig = make_subplots(
        rows=1, cols=2, vertical_spacing=0, subplot_titles=['Custom models accuracy', 'Overfitting comparison'])
    for i, (acc, sett) in enumerate(zip(accs, ('train', 'test'))):
        fig.add_trace(
            go.Scatter(x=params, y=acc, mode='lines+markers+text', name=sett, text=names, 
                       textposition='top center', marker=dict(color=colors[i], size=12, symbol=symbols[i])
            ), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=params, y=np.subtract(*accs), mode='lines+markers+text', text=names, name='difference',
                   textposition='top center', marker=dict(color=colors[2], size=12, symbol='diamond'), 
    ), row=1, col=2)
    fig.update_yaxes(title_text='acc', row=1, col=2)
    fig.update_xaxes(title_text='#params', row=1, col=2)
    fig.update_layout(
        height=500, width=1500, template='seaborn',
        yaxis=dict(range=[0.7, 1.05], title_text='acc'), 
        xaxis=dict(range=[-1e6, max(params)+max(params)*0.15], title_text='#params'),
        margin=dict(t=30, b=10, l=0, r=0), 
        )
    return fig