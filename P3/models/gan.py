from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from data import CelebADataset
from keras.applications.inception_v3 import InceptionV3
from models.layers import ConvBlock, DeconvBlock, ResidualDeconvBlock
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Input,
    InputLayer,
    Reshape,
)
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer
from tqdm import tqdm
from utils import FID, SaveImagesCallback, fid_score, get_acts


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.history = {"c_loss": [], "g_loss": [], "c_acc": [], "g_acc": [], "fid": []}

    def on_epoch_end(self, epoch, logs=None):
        self.history["c_loss"].append(logs["c_loss"])
        self.history["g_loss"].append(logs["g_loss"])
        self.history["c_acc"].append(logs["c_acc"])
        self.history["g_acc"].append(logs["g_acc"])


class GAN:
    NORM = lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5
    DENORM = lambda x: np.uint8(x * 127.5 + 127.5)
    INCEPTION_SIZE = (256, 256, 3)

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        hidden_size: int,
        pool: str,
        residual: bool,
        critic_steps: int,
        gp_weight: float,
    ):
        self.hidden_size = hidden_size
        self.model = WGANGP(
            img_size, hidden_size, pool, residual, critic_steps, gp_weight
        )
        self.inception = InceptionV3(
            include_top=False, pooling="avg", input_shape=self.INCEPTION_SIZE
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
        steps_per_epoch: int = 1500,
        c_optimizer: Optimizer = Adam(2e-4, 0.5, 0.999),
        g_optimizer: Optimizer = Adam(2e-4, 0.5, 0.999),
    ) -> Dict[str, List[float]]:
        self.model.compile(c_optimizer, g_optimizer)
        train_tf = train.to_tf(GAN.NORM, batch_size, targets=False)
        val_tf = val.to_tf(GAN.NORM, batch_size, targets=False)
        callbacks = [
            EarlyStopping("g_loss", patience=train_patience),
            LossHistory(),
            FID(
                self.model,
                self.inception,
                train,
                val,
                GAN.NORM,
                GAN.DENORM,
                batch_size=64,
            ),
            SaveImagesCallback(
                self.model,
                val,
                f"{path}/epoch-preds",
                GAN.NORM,
                GAN.DENORM,
                save_frequency=1,
            ),
            ModelCheckpoint(
                f"{path}/checkpoint/checkpoint.ckpt",
                "val_fid",
                save_weights_only=True,
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = self.model.fit(
            train_tf,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_tf,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_epoch,
        ).history
        self.model.load_weights(f"{path}/checkpoint/checkpoint.ckpt")
        with open(f"{path}/history.pkl", "wb") as writer:
            pickle.dump(history, writer)
        self.predict(test, f"{path}/test-preds", batch_size=64)

        # evaluate
        train_score = self.evaluate(train, batch_size=64)
        val_score = self.evaluate(val, batch_size=64)
        test_score = self.evaluate(test, batch_size=64)
        with open(f"{path}/results.pkl", "wb") as writer:
            pickle.dump(dict(train=train_score, val=val_score, test=test_score), writer)
        return history

    def predict(self, data: CelebADataset, out: str, batch_size: int = 10):
        if not os.path.exists(out):
            os.makedirs(out)
        inputs, outputs = [], []
        for files, input in tqdm(
            data.stream(GAN.NORM, batch_size),
            total=len(data) // batch_size,
            desc="predict",
        ):
            latent = tf.random.normal(shape=(input.shape[0], self.hidden_size))
            output = GAN.DENORM(self.model.generator.predict(latent, verbose=0))
            for j in range(output.shape[0]):
                img = Image.fromarray(output[j])
                img.save(f"{out}/{files[j]}")
            outputs.append(output)
            inputs.append(input)

    def evaluate(self, data: CelebADataset, batch_size: int = 10) -> float:
        fake = []
        for _, input in tqdm(
            data.stream(GAN.NORM, batch_size),
            total=len(data) // batch_size,
            desc="eval",
        ):
            fake.append(GAN.DENORM(self.model.predict(input, verbose=0)))
        fake = get_acts(self.inception, np.concatenate(fake, 0), batch_size)
        real = get_acts(self.inception, data, batch_size)
        return fid_score(real, fake)


class WGANGP(Model):
    def __init__(
        self,
        img_size: Tuple[int, int, int],
        hidden_size: int,
        pool: str,
        residual: bool,
        critic_steps: int,
        gp_weight: float,
    ):
        super(WGANGP, self).__init__()

        self.critic = Sequential(
            [
                InputLayer(img_size),
                ConvBlock(32, 4, 2),
                ConvBlock(32, 4, 2),
                ConvBlock(32, 4, 2, dropout=0.3),
                Conv2D(1, kernel_size=4, strides=1, padding="valid"),
                Flatten(),
            ],
            name="discriminator",
        )

        # generator
        args = dict(bias=False, batch_norm=False) | (
            dict(dilation=1, strides=2)
            if pool == "strdes"
            else dict(dilation=2, pool=True)
        )
        deconv = DeconvBlock if not residual else ResidualDeconvBlock
        self.generator = Sequential(
            [
                Input(shape=(hidden_size,), name="latent-input"),
                Reshape((1, 1, hidden_size), name="input-reshape"),
                deconv(
                    256,
                    4,
                    1,
                    padding="valid",
                    bias=False,
                    batch_norm=True,
                    name="deconv1",
                ),
                deconv(128, 4, **args, name="deconv2"),
                deconv(64, 4, **args, name="deconv3"),
                deconv(32, 4, **args, name="deconv4"),
                Conv2DTranspose(
                    img_size[-1], 4, 2, padding="same", activation="tanh", name="output"
                ),
            ],
            name="generator",
        )
        self.from_latent = True
        self.hidden_size = hidden_size
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer: Optimizer, g_optimizer):
        super(WGANGP, self).compile(run_eagerly=True)
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = Mean(name="c_wass_loss")
        self.c_gp_metric = Mean(name="c_gp")
        self.c_loss_metric = Mean(name="c_loss")
        self.g_loss_metric = Mean(name="g_loss")
        self._is_compiled = True

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real, fake):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real: tf.Tensor):
        batch_size = tf.shape(real)[0]

        for _ in range(self.critic_steps):
            inputs = self.get_inputs(real)
            with tf.GradientTape() as tape:
                fake = self.generator(inputs, training=True)
                fake_predictions = self.critic(fake, training=True)
                real_predictions = self.critic(real, training=True)
                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_predictions
                )
                c_gp = self.gradient_penalty(batch_size, real, fake)
                c_loss = c_wass_loss + c_gp * self.gp_weight
            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        inputs = self.get_inputs(real)
        with tf.GradientTape() as tape:
            fake = self.generator(inputs, training=True)
            fake_predictions = self.critic(fake, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Calculate Critic and Generator Accuracy
        c_acc = (
            tf.reduce_mean(tf.cast(tf.math.greater(real_predictions, 0), tf.float32))
            * 100
        )
        g_acc = (
            tf.reduce_mean(tf.cast(tf.math.less(fake_predictions, 0), tf.float32)) * 100
        )

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {
            **{m.name: m.result() for m in self.metrics},
            "c_acc": c_acc,
            "g_acc": g_acc,
        }

    def get_inputs(self, real: tf.Tensor):
        return (
            tf.random.normal(shape=(tf.shape(real)[0], self.hidden_size))
            if self.from_latent
            else real
        )

    def test_step(self, real: tf.Tensor):
        inputs = self.get_inputs(real)
        fake = self.generator(inputs, training=False)
        fake_preds = self.critic(fake, training=False)
        real_preds = self.critic(real, training=False)

        c_acc = (
            tf.reduce_mean(tf.cast(tf.math.greater(real_preds, 0), tf.float32)) * 100
        )
        g_acc = tf.reduce_mean(tf.cast(tf.math.less(fake_preds, 0), tf.float32)) * 100

        return {"c_acc": c_acc, "g_acc": g_acc}

    def call(self, real: tf.Tensor):
        return self.generator(self.get_inputs(real))
