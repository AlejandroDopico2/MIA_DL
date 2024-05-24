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
from models.layers import vae, skip_vae


class VAE:
    NORM = lambda _, x: x / 255
    DENORM = lambda _, x: np.uint8(x * 255)
    INCEPTION_SIZE = (299, 299, 3)

    def __init__(
            self,
            img_size: Tuple[int, int, int],
            hidden_size: int,
            pool: str,
            residual: bool,
            loss_factor: int = 1000
        ):
        self.hidden_size = hidden_size
        args = dict(dilation=2, strides=1, pool=True) if pool == 'dilation' else dict(strides=2, dilation=1)
        encoder_input, encoder_output, decoder_input, decoder_output, mean_mu, log_var = \
            vae(img_size, hidden_size, act='sigmoid', residual=residual, **args)
        
        def r_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        def kl_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return -0.5 * K.sum(1 + mean_mu - K.square(mean_mu) - K.exp(log_var), axis=1)

        def total_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return loss_factor * r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)
        
        encoder = Model(encoder_input, encoder_output, name='encoder')
        decoder = Model(decoder_input, decoder_output, name='decoder')
        self.model = Model(encoder_input, decoder(encoder_output), name='vae')
        self.model.METRICS = [r_loss, kl_loss]
        self.model.LOSS = total_loss
        self.model.encoder = encoder
        self.model.decoder = decoder 

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
        self.model.compile(optimizer=optimizer, loss=self.model.LOSS, metrics=self.model.METRICS)
        train_tf = train.to_tf(self.NORM, batch_size, targets=True)
        val_tf = val.to_tf(self.NORM, batch_size, targets=True)

        callbacks = [
            SaveImagesCallback(self.model, val, f'{path}/val-preds', self.NORM, self.DENORM, save_frequency=1),
            FID(self.model, train, val, self.NORM, self.DENORM),
            EarlyStopping("loss", patience=train_patience),
            EarlyStopping("val_loss", patience=val_patience),
            ModelCheckpoint(f"{path}/model.h5", "val_fid", save_weights_only=True, save_best_only=True, verbose=0),
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
        self.predict(test, f"{path}/test-preds/", batch_size)
        return history

    def predict(self, data: CelebADataset, out: str, batch_size: int = 10):
        if not os.path.exists(out):
            os.makedirs(out)
        inputs, outputs = [], []
        for files, input in data.stream(self.NORM, batch_size):
            output = self.DENORM(self.model.predict(input, verbose=0))
            for j in range(output.shape[0]):
                img = Image.fromarray(output[j])
                img.save(f"{out}/{files[j]}")
            inputs.append(input)
            outputs.append(output)
            
            
            
