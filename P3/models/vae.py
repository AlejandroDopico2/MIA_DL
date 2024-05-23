from __future__ import annotations

import os
import pickle
import shutil
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from data import CelebADataset
from keras import backend as K
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Activation,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    LeakyReLU,
    Reshape,
)
from keras.models import Model
from keras.optimizers.legacy import Adam, Optimizer
from PIL import Image
from utils import FID, SaveImagesCallback


class VariationalAutoEncoder:
    LOSS_FACTOR = 1000
    NORM = lambda _, x: x / 255
    DENORM = lambda _, x: np.uint8(x * 255)
    INCEPTION_SIZE = (299, 299, 3)

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        hidden_size: int,
        kernels: List[int],
        strides: List[int],
        filters: List[int],
        dilation: bool = False,
    ):
        self.hidden_size = hidden_size

        # ---------------------- ENCODER ----------------------
        encoder_input = Input(shape=img_size, name="encoder-input")
        x = encoder_input
        shapes = [img_size]
        for i, (n, k, s) in enumerate(zip(filters, kernels, strides)):
            if dilation:
                x = Conv2D(
                    n, k, dilation_rate=s, padding="same", name=f"encoder-conv-{i}"
                )(x)
            else:
                x = Conv2D(n, k, strides=s, padding="same", name=f"encoder-conv-{i}")(x)
            x = LeakyReLU()(x)
            shapes.append(K.int_shape(x)[1:])

        x = Flatten()(x)
        mean_mu = Dense(hidden_size, name="mu")(x)
        log_var = Dense(hidden_size, name="log-var")(x)

        def sampling(args):
            mean_mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0.0, stddev=1.0)
            return mean_mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name="encoder-output")([mean_mu, log_var])
        self.encoder = Model(encoder_input, encoder_output, name="encoder")

        # ---------------------- DECODER ----------------------
        decoder_input = Input(shape=(hidden_size,), name="decoder-input")
        filters = [img_size[-1]] + filters[1:]
        x = Dense(np.prod(shapes[-1]))(decoder_input)
        x = Reshape(shapes[-1])(x)
        for i, (n, k, s) in enumerate(zip(filters[::-1], kernels[::-1], strides[::-1])):
            if dilation:  # Conditionally choose Conv2DTranspose with dilation
                x = Conv2DTranspose(
                    n, k, dilation_rate=s, padding="same", name=f"decoder-conv-{i}"
                )(x)
            else:
                x = Conv2DTranspose(
                    n, k, strides=s, padding="same", name=f"decoder-conv-{i}"
                )(x)

            h, w = x.shape[1] - shapes[-i - 2][0], x.shape[2] - shapes[-i - 2][1]
            x = Cropping2D(
                cropping=(
                    (h // 2 + (h % 2 != 0), h // 2),
                    (w // 2 + (w % 2 != 0), w // 2),
                )
            )(x)
            # check if dimensions are the same
            if i == (len(filters) - 1):
                x = Activation(sigmoid)(x)
            else:
                x = LeakyReLU()(x)
        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
        self.model = Model(encoder_input, self.decoder(encoder_output), name="VAE")

        def r_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        def kl_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            kl_loss = -0.5 * K.sum(
                1 + log_var - K.square(mean_mu) - K.exp(log_var), axis=1
            )
            return kl_loss

        def total_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return self.LOSS_FACTOR * r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

        self.METRICS = [r_loss, kl_loss]
        self.LOSS = total_loss

    def train(
        self,
        train: CelebADataset,
        val: CelebADataset,
        test: CelebADataset,
        path: str,
        batch_size: int = 10,
        epochs: int = 10,
        train_patience: int = 10,
        dev_patience: int = 5,
        steps_per_epoch: int = 1500,
        optimizer: Optimizer = Adam(1e-4),
    ) -> Dict[str, List[float]]:
        self.model.compile(optimizer=optimizer, loss=self.LOSS, metrics=self.METRICS)
        train_tf = train.to_tf(self.NORM, batch_size, targets=True)
        val_tf = val.to_tf(self.NORM, batch_size, targets=True)

        callbacks = [
            SaveImagesCallback(
                self.model,
                val,
                path,
                self.NORM,
                self.DENORM,
                self.hidden_size,
                save_frequency=1,
                from_latent=False,
            ),
            FID(self.decoder, val, self.hidden_size, self.NORM, self.DENORM),
            EarlyStopping("loss", patience=train_patience),
            EarlyStopping("val_loss", patience=dev_patience),
            ModelCheckpoint(
                f"{path}/model.h5",
                "fid",
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
        )
        with open(f"{path}/history.pkl", "wb") as f:
            pickle.dump(history, f)
        self.model.load_weights(f"{path}/model.h5")
        self.predict(test, f"{path}/preds/", batch_size)
        return history

    def predict(
        self,
        data: CelebADataset,
        out: str,
        batch_size: int = 10,
    ):
        if not os.path.exists(out):
            os.makedirs(out)
        inputs, outputs = [], []
        for files, input in data.stream(self.NORM, batch_size):
            output = self.DENORM(self.model.predict(input))
            for j in range(output.shape[0]):
                img = Image.fromarray(output[j])
                img.save(f"{out}/{files[j]}")
            inputs.append(input)
            outputs.append(output)
