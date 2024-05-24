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
from models.layers import VariationalAutoEncoder


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
            skips: bool
        ):
        self.hidden_size = hidden_size
        self.model = VariationalAutoEncoder(
            img_size, hidden_size, pool, residual, skips, act='sigmoid', 
            name='skip-'*skips + 'vae', loss_factor=1000
        )

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
            ModelCheckpoint(f"{path}/model.weights.h5", "val_fid", save_weights_only=True, save_best_only=True, verbose=0),
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
        self.model.load_weights(f"{path}/model.weights.h5")
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
            
            
            
