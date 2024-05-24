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
            train: CelebADataset,
            val: CelebADataset,
            NORM: Callable,
            DENORM: Callable, 
            batch_size: int = 10,
            n_samples: int = 500,
            **kwargs
        ):
        """
        Callback to track the FID score in a CelebA dataset.
        
        Args:
            model (Model): Keras model to generate images.
            train (CelebADataset): Ground truth of the train set.
            val (CelebADataset): Ground truth of the validation set.
            NORM (Callable): Normalization function.
            DENORM (Callable): Denormalization function.
            batch_size (int): Batch size to feed the Inception network. Default to 10.
            n_samples (int): Number of samples to compute the FID metrics (mean and std). Defaults to 500.
        """
        super().__init__(**kwargs)
        self.model = model 
        self.NORM = NORM 
        self.DENORM = DENORM
        self.fmodel = InceptionV3(include_top=False, pooling='avg', input_shape=self.INCEPTION_SIZE)
        self.n_samples = n_samples
        self.batch_size = batch_size
        
        # get batches 
        self.train, self.val = [], []
        train_stream = train.stream(NORM, batch_size)
        val_stream = val.stream(NORM, batch_size)
        for _ in range(n_samples//batch_size):
            self.train.append(next(train_stream)[1])
            self.val.append(next(val_stream)[1])

        # precompute mean and standard deviation from the ground truth 
        self.mu, self.sigma = 0, 0
        for real in tqdm(self.train, desc='inception', total=len(self.train), leave=False):
            act = self.fmodel.predict(self.preprocess(real), verbose=0)
            self.mu += act.mean(axis=0)
            self.sigma += cov(act, rowvar=False)
        self.mu /= len(self.train)
        self.sigma /= len(self.train)

        
    def on_epoch_end(self, epoch, logs):
        logs['fid'] = self.eval(self.train)
        logs['val_fid'] = self.eval(self.val)

    def eval(self, data: List[np.ndarray]) -> float:
        mu, sigma = 0, 0
        for real in tqdm(data, desc='saving-data', total=len(data), leave=False):
            fake = self.preprocess(self.DENORM(self.model.predict(real, verbose=0)))
            act = self.fmodel.predict(fake, verbose=0)
            mu += act.mean(axis=0)
            sigma += cov(act, rowvar=False)
        mu, sigma = mu/len(data), sigma/len(data)
        ssdiff = np.sum((self.mu - mu) ** 2.0)
        covmean = sqrtm(self.sigma.dot(sigma))
        if iscomplexobj(covmean):
            covmean = covmean.real
        return ssdiff + trace(self.sigma + sigma - 2.0 * covmean)
        

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        return np.stack(
            [preprocess_input(resize(img, self.INCEPTION_SIZE, 0)) for img in imgs], 0
        )


class SaveImagesCallback(Callback):
    def __init__(
        self,
        model,
        data: CelebADataset,
        output_folder: str,
        NORM: Callable, 
        DENORM: Callable,
        save_frequency: int = 5,
        n_samples: int = 100
    ):
        super(SaveImagesCallback, self).__init__()
        self.model = model
        self.files, self.data = next(data.stream(NORM, n_samples))
        self.output_folder = output_folder
        self.save_frequency = save_frequency
        self.DENORM = DENORM

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_frequency != 0:
            return
        output_folder = f"{self.output_folder}/epoch_{epoch}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_batch = self.DENORM(self.model.predict(self.data, verbose=0))
        for i, file in enumerate(self.files):
            img = Image.fromarray(output_batch[i])
            img.save(f"{output_folder}/{file}")

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
