import os
from typing import Dict, List, Tuple, Union, Callable, Optional
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import Callback, ModelCheckpoint
from numpy import cov, iscomplexobj, trace
from PIL import Image
from scipy.linalg import sqrtm
from keras.models import Model 
from skimage.transform import resize
from data import CelebADataset
from tensorflow.data import Dataset 
from keras.metrics import Metric 

from tqdm import tqdm 

    
    
class FID(Callback):
    INCEPTION_SIZE = (299, 299, 3)
    
    def __init__(
            self, 
            model: Model,
            val: CelebADataset,
            hidden_size: int, 
            NORM: Callable,
            DENORM: Callable, 
            batch_size: int = 10,
            n_samples: int = 500,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.model = model 
        self.hidden_size = hidden_size
        self.NORM = NORM 
        self.DENORM = DENORM
        self.fmodel = InceptionV3(include_top=False, pooling='avg', input_shape=self.INCEPTION_SIZE)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.best, self.best_epoch = np.Inf, 0

        # precompute mean and standard deviation from the ground truth 
        mu, sigma, n = 0, 0, n_samples//batch_size
        stream = val.stream(None, batch_size=batch_size)
        for _ in tqdm(range(n), desc='inception', total=n, leave=False):
            real = self.preprocess(next(stream)[1])
            act = self.fmodel.predict(real, verbose=0)
            mu += act.mean(axis=0)
            sigma += cov(act, rowvar=False)
        self.mu = mu/n
        self.sigma = sigma/n
        self.step = 0 

        
    def on_epoch_end(self, epoch, logs):
        mu, sigma, n = 0, 0, self.n_samples//self.batch_size
        for _ in tqdm(range(n), desc='val', total=n, leave=False):
            latent = tf.random.normal(shape=(self.batch_size, self.hidden_size))
            fake = self.preprocess(self.model.predict(latent, verbose=0))
            act = self.fmodel.predict(fake, verbose=0)
            mu += act.mean(axis=0)
            sigma += cov(act, rowvar=False)
        mu /= n
        sigma /= n 
        ssdiff = np.sum((self.mu - mu) ** 2.0)
        covmean = sqrtm(self.sigma.dot(sigma))
        if iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + trace(self.sigma + sigma - 2.0 * covmean)
        logs['fid'] = fid 

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        return np.stack(
            [preprocess_input(resize(img, self.INCEPTION_SIZE, 0)) for img in self.DENORM(imgs)], 0
        )


class SaveImagesCallback(Callback):
    def __init__(
        self,
        model,
        val: CelebADataset,
        output_folder: str,
        NORM: Callable, 
        DENORM: Callable,
        hidden_size: int, 
        save_frequency: int = 5,
        n_samples: int = 100,
        from_latent: bool = False 
    ):
        super(SaveImagesCallback, self).__init__()
        self.model = model
        self.files, self.data = next(val.stream(NORM, n_samples))
        self.output_folder = output_folder + "/val_pred"
        self.save_frequency = save_frequency
        self.DENORM = DENORM
        self.from_latent = from_latent
        self.hidden_size = hidden_size

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_frequency != 0:
            return
        output_folder = f"{self.output_folder}/epoch_{epoch}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if self.from_latent:
            latent = tf.random.normal(shape=(len(self.files), self.hidden_size))
            output_batch = self.DENORM(self.model.predict(latent, verbose=0))
        else:
            output_batch = self.DENORM(self.model.predict(self.data, verbose=0))
            
        for i, file in enumerate(self.files):
            img = Image.fromarray(output_batch[i])
            img.save(f"{output_folder}/{file}.jpg")
        print(f"Saved validation predictions for epoch {epoch + 1}.")



def plot_history(
    history: Dict[str, List[float]], name: Union[str, List[str]] = "loss"
) -> go.Figure:
    fig = go.Figure()
    if isinstance(name, str):
        fig.add_trace(go.Scatter(y=history[name], name="train"))
        fig.add_trace(go.Scatter(y=history[f"val_{name}"], name="val"))
    else:
        for x in name:
            fig.add_trace(go.Scatter(y=history[x], name=x))
    fig.update_layout(
        template="seaborn",
        height=500,
        width=800,
        margin=dict(t=30, b=10, l=30, r=10),
        title_text=f'Training {name if isinstance(name, str) else ""} evolution',
    )
    fig.update_xaxes(title_text="epochs")
    if isinstance(name, str):
        fig.update_yaxes(title_text=name)
    return fig
