import os
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from data import CelebADataset
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import Callback
from keras.models import Model
from numpy import cov, iscomplexobj, trace
from PIL import Image
from scipy.linalg import sqrtm
from skimage.transform import resize
from tqdm import tqdm


class FID(Callback):
    IMG_SIZE = (256, 256, 3)

    def __init__(
        self,
        model: Model,
        inception: Model,
        train: CelebADataset,
        val: CelebADataset,
        NORM: Callable,
        DENORM: Callable,
        batch_size: int = 10,
        n_samples: int = 100,
        **kwargs,
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
        self.inception = inception
        self.n_samples = n_samples
        self.batch_size = batch_size

        # get batches
        self.train, self.val = [], []
        train_stream = train.stream(NORM, batch_size)
        val_stream = val.stream(NORM, batch_size)
        for _ in range(n_samples // batch_size):
            self.train.append(next(train_stream)[1])
            self.val.append(next(val_stream)[1])

        # precompute mean and standard deviation from the ground truth
        act = []
        for real in tqdm(
            self.train, desc="inception", total=len(self.train), leave=False
        ):
            act.append(self.inception.predict(self.preprocess(real), verbose=0))
        act = np.concatenate(act, 0)
        self.mu = act.mean(axis=0)
        self.sigma = cov(act, rowvar=False)

    def on_epoch_end(self, epoch, logs):
        logs["fid"] = self.eval(self.train)
        logs["val_fid"] = self.eval(self.val)

    def eval(self, data: List[np.ndarray]) -> float:
        act = []
        for real in tqdm(data, desc="fid-score", total=len(data), leave=False):
            fake = self.preprocess(self.DENORM(self.model.predict(real, verbose=0)))
            act.append(self.inception.predict(fake, verbose=0))
        act = np.concatenate(act, 0)
        mu, sigma = act.mean(axis=0), cov(act, rowvar=False)
        ssdiff = np.sum((self.mu - mu) ** 2.0)
        covmean = sqrtm(self.sigma.dot(sigma))
        if iscomplexobj(covmean):
            covmean = covmean.real
        return ssdiff + trace(self.sigma + sigma - 2.0 * covmean)

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        return np.stack(
            [preprocess_input(resize(img, self.IMG_SIZE, 0)) for img in imgs], 0
        )


def get_acts(
    inception: Model,
    data: Union[CelebADataset, np.ndarray],
    batch_size: int = 10,
    img_size: Tuple[int, int, int] = (256, 256, 3),
):
    if isinstance(data, np.ndarray):
        acts = []
        for i in range(0, data.shape[0], batch_size):
            input = np.stack(
                [
                    preprocess_input(resize(img, img_size, 0))
                    for img in data[i : (i + batch_size)]
                ]
            )
            acts.append(inception.predict(input, verbose=0))
    else:
        acts = []
        for _, input in data.stream(None, batch_size):
            input = np.stack(
                [preprocess_input(resize(img, img_size, 0)) for img in input]
            )
            acts.append(inception.predict(input, verbose=0))
    return np.concatenate(acts, 0)


def fid_score(real_act: np.ndarray, fake_act: np.ndarray) -> float:
    """
    Computes the FID score between a set of real and generated images.

    Args:
        real_act (np.ndarray): Set of real activations.
        fake_act (np.ndarray): Set of generated activations.
        batch_size (int): Batch size.
    """
    real_mu, real_sigma = real_act.mean(0), cov(real_act, rowvar=False)
    fake_mu, fake_sigma = fake_act.mean(0), cov(fake_act, rowvar=False)
    ssdiff = np.sum((real_mu - fake_mu) ** 2.0)
    covmean = sqrtm(real_sigma.dot(fake_sigma))
    if iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + trace(real_sigma + fake_sigma - 2.0 * covmean)


class SaveImagesCallback(Callback):
    def __init__(
        self,
        model,
        data: CelebADataset,
        output_folder: str,
        NORM: Callable,
        DENORM: Callable,
        save_frequency: int = 5,
        n_samples: int = 20,
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


def display(real: np.ndarray, fake: np.ndarray, max_cols: int = 5):
    assert real.shape[0] <= max_cols
    assert real.shape[0] == fake.shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=max_cols, figsize=(5 * max_cols, 10))
    for i in range(real.shape[0]):
        ax[0, i].imshow(real[i])
        ax[1, i].imshow(fake[i])
        ax[1, i].set_axis_off()
        ax[0, i].set_axis_off()
    fig.show()


def plot_embeds(
    embeds: List[np.ndarray], groups: List[str], title: str = "Embedding projection"
) -> go.Figure:
    n_samples = len(embeds) // len(groups)
    fig = go.Figure()
    for i, group in enumerate(groups):
        _embed = embeds[(i * n_samples) : (i * n_samples + n_samples)]
        fig.add_trace(
            go.Scatter3d(
                x=_embed[:, 0],
                y=_embed[:, 1],
                z=_embed[:, 2],
                name=group,
                mode="markers",
            )
        )
    fig.update_layout(
        width=1000,
        height=600,
        template="seaborn",
        title_text=title,
        margin=dict(b=20, l=20, r=20),
    )
    return fig
