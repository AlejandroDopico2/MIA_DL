from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from data import CelebADataset
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers.legacy import Adam, Optimizer
from models.layers import vae
from PIL import Image
from tqdm import tqdm
from utils import FID, SaveImagesCallback, fid_score, get_acts


class VAE:
    NORM = lambda x: x / 255
    DENORM = lambda x: np.uint8(x * 255)
    INCEPTION_SIZE = (256, 256, 3)

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        hidden_size: int,
        pool: str,
        residual: bool,
        loss_factor: int = 1000,
    ):
        self.hidden_size = hidden_size
        args = (
            dict(dilation=2, strides=1, pool=True)
            if pool == "dilation"
            else dict(strides=2, dilation=1)
        )
        (
            encoder_input,
            encoder_output,
            decoder_input,
            decoder_output,
            mean_mu,
            log_var,
        ) = vae(img_size, hidden_size, act="sigmoid", residual=residual, **args)

        def r_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        def kl_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return -0.5 * K.sum(
                1 + log_var - K.square(mean_mu) - K.exp(log_var), axis=1
            )

        def total_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return loss_factor * r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

        encoder = Model(encoder_input, encoder_output, name="encoder")
        decoder = Model(decoder_input, decoder_output, name="decoder")
        self.model = Model(encoder_input, decoder(encoder_output), name="vae")
        self.model.METRICS = [r_loss, kl_loss]
        self.model.LOSS = total_loss
        self.model.encoder = encoder
        self.model.decoder = decoder
        self.inception = InceptionV3(
            include_top=False, pooling="avg", input_shape=self.INCEPTION_SIZE
        )
        print(encoder.summary())
        print(decoder.summary())

    def train(
        self,
        train: CelebADataset,
        val: CelebADataset,
        test: CelebADataset,
        path: str,
        batch_size: int = 10,
        epochs: int = 10,
        train_patience: int = 10,
        val_patience: int = 5,
        steps_per_epoch: int = 1500,
        optimizer: Optimizer = Adam(1e-4),
    ) -> Dict[str, List[float]]:
        self.model.compile(
            optimizer=optimizer, loss=self.model.LOSS, metrics=self.model.METRICS
        )
        train_tf = train.to_tf(VAE.NORM, batch_size, targets=True)
        val_tf = val.to_tf(VAE.NORM, batch_size, targets=True)

        callbacks = [
            SaveImagesCallback(
                self.model,
                val,
                f"{path}/epoch-preds",
                VAE.NORM,
                VAE.DENORM,
                save_frequency=1,
            ),
            FID(
                self.model,
                self.inception,
                train,
                val,
                VAE.NORM,
                VAE.DENORM,
                batch_size=64,
            ),
            EarlyStopping("loss", patience=train_patience),
            EarlyStopping("val_loss", patience=val_patience),
            ModelCheckpoint(
                f"{path}/model.h5",
                "val_fid",
                save_weights_only=True,
                save_best_only=True,
                verbose=0,
            ),
        ]
        history = self.model.fit(
            train_tf,
            validation_data=val_tf,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_epoch,
        ).history
        with open(f"{path}/history.pkl", "wb") as f:
            pickle.dump(history, f)
        self.model.load_weights(f"{path}/model.h5")
        self.predict(test, f"{path}/test-preds", batch_size=64)
        train_score = self.evaluate(train, batch_size=64)
        val_score = self.evaluate(val, batch_size=64)
        test_score = self.evaluate(test, batch_size=64)
        with open(f"{path}/results.pkl", "wb") as writer:
            pickle.dump(dict(train=train_score, val=val_score, test=test_score), writer)
        return history

    def predict(self, data: CelebADataset, out: str, batch_size: int = 128):
        if not os.path.exists(out):
            os.makedirs(out)
        inputs, outputs = [], []
        for files, input in tqdm(
            data.stream(VAE.NORM, batch_size),
            total=len(data) // batch_size,
            desc="predict",
            leave=False,
        ):
            output = VAE.DENORM(self.model.predict(input, verbose=0))
            for j in range(output.shape[0]):
                img = Image.fromarray(output[j])
                img.save(f"{out}/{files[j]}")
            inputs.append(input)
            outputs.append(output)

    def evaluate(self, data: CelebADataset, batch_size: int = 128) -> float:
        fake = []
        for _, input in tqdm(
            data.stream(VAE.NORM, batch_size),
            total=len(data) // batch_size,
            desc="eval",
            leave=False,
        ):
            fake.append(VAE.DENORM(self.model.predict(input, verbose=0)))
        fake = get_acts(self.inception, np.concatenate(fake, 0), batch_size)
        real = get_acts(self.inception, data, batch_size)
        return fid_score(real, fake)

    def latent(self, paths: List[str], batch_size: int = 128) -> List[np.ndarray]:
        """
        Obtain the representation in the latent space of input imges.

        Args:
            paths (List[str]): Image paths.

        Returns:
            np.ndarray: Latent embeddings.
        """
        embed = []
        for i in tqdm(
            range(0, len(paths), batch_size),
            total=len(paths) // batch_size,
            desc="latent",
            leave=False,
        ):
            _paths = paths[i : (i + batch_size)]
            imgs = VAE.NORM(
                np.stack(
                    [
                        np.array(Image.open(path).resize(CelebADataset.IMG_SIZE[:2]))
                        for path in _paths
                    ]
                )
            )
            embed += list(self.model.encoder.predict(imgs, verbose=0))
        return embed

    def from_latent(
        self, latent: List[np.ndarray], batch_size: int = 128
    ) -> List[np.ndarray]:
        """
        Generate images from latent space.

        Args:
            latent (List[np.ndarray]): Latent representation.

        Returns:
            List[np.ndarray]: Generated images.
        """
        fake = []
        for i in tqdm(
            range(0, len(latent), batch_size),
            total=len(latent) // batch_size,
            desc="from-latent",
            leave=False,
        ):
            _latent = latent[i : (i + batch_size)]
            fake += list(self.model.decoder.predict(_latent, verbose=0))
        return fake

    def generate(self, paths: List[str], batch_size: int = 10) -> List[np.ndarray]:
        fake = []
        for i in tqdm(
            range(0, len(paths), batch_size),
            total=len(paths) // batch_size,
            desc="from-latent",
            leave=False,
        ):
            _paths = paths[i : (i + batch_size)]
            imgs = VAE.NORM(
                np.stack(
                    [
                        np.array(Image.open(path).resize(CelebADataset.IMG_SIZE[:2]))
                        for path in _paths
                    ]
                )
            )
            fake += list(VAE.DENORM(self.model.predict(imgs, verbose=0)))
        return fake
