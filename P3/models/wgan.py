
from __future__ import annotations
import os
import shutil
from typing import Dict, List, Tuple, Callable

import cv2
import numpy as np
import tensorflow as tf
from data import CelebADataset
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    Activation
)
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm
from utils import SaveImagesCallback, FID
from keras.activations import tanh

class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.history = {"c_loss": [], "g_loss": [], "c_acc": [], "g_acc": [], 'fid': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history["c_loss"].append(logs["c_loss"])
        self.history["g_loss"].append(logs["g_loss"])
        self.history["c_acc"].append(logs["c_acc"])
        self.history["g_acc"].append(logs["g_acc"])

        
class WGAN:
    NORM = lambda _, x: (tf.cast(x, tf.float32) - 127.5) / 127.5
    DENORM = lambda _, x: np.uint8(x * 127.5 + 127.5)
    INCEPTION_SIZE = (299, 299, 3)

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        hidden_size: int,
        critic_steps: int,
        gp_weight: float,
    ):
        self.hidden_size = hidden_size
        # discriminator
        critic_input = Input(shape=img_size)
        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(critic_input)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Conv2D(1024, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
        critic_output = Flatten()(x)

        critic = Model(critic_input, critic_output, name="discriminator")

        # generator
        generator_input = Input(shape=(hidden_size,))
        x = Reshape((1, 1, hidden_size))(generator_input)
        x = Conv2DTranspose(
            1024, kernel_size=4, strides=1, padding="valid", use_bias=False
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2DTranspose(
            512, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2DTranspose(
            256, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(tanh)(x)
        generator_output = Conv2DTranspose(
            img_size[-1], kernel_size=4, strides=2, padding="same", activation="tanh"
        )(x)
        generator = Model(generator_input, generator_output, name="generator")
        self.model = WGANGP(critic, generator, hidden_size, critic_steps, gp_weight)

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
        if os.path.exists(path):
            shutil.rmtree(path)
        train_tf = train.to_tf(self.NORM, batch_size, targets=False)
        val_tf = val.to_tf(self.NORM, batch_size, targets=False)
        callbacks = [
            EarlyStopping("g_loss", patience=train_patience),
            TensorBoard(log_dir=f"{path}/logs"),
            LossHistory(),
            FID(self.model, val, self.hidden_size, self.NORM, self.DENORM),
            SaveImagesCallback(self.model, val, path, self.NORM, self.DENORM, 
                               self.hidden_size, save_frequency=1, from_latent=True),
            ModelCheckpoint(f"{path}/checkpoint/checkpoint.ckpt", 'fid', save_weights_only=True, save_best_only=True, verbose=0),
            
        ]

        history = self.model.fit(train_tf, epochs=epochs, callbacks=callbacks, validation_data=val_tf,
                                 steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch
                                 ).history
        self.model.load_weights(f"{path}/checkpoint/checkpoint.ckpt")
        self.predict(test, f"{path}/preds/", batch_size)
        return history

    def predict(
        self,
        data: CelebADataset,
        out: str,
        batch_size: int = 10,
    ):
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        inputs, outputs = [], []
        for files, input in data.stream(self.NORM, batch_size):
            latent = tf.random.normal(shape=(input.shape[0], self.hidden_size))
            output = self.DENORM(self.model.generator.predict(latent, verbose=0))
            for j in range(output.shape[0]):
                img = Image.fromarray(output[j])
                img.save(f'{out}/{files[j]}')
            outputs.append(output)
            inputs.append(input)



class WGANGP(Model):
    def __init__(
        self,
        critic: Model,
        generator: Model,
        latent_dim: int,
        critic_steps: int,
        gp_weight: float,
    ):
        super(WGANGP, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
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
            self.g_loss_metric
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_predictions = self.critic(fake_images, training=True)
                real_predictions = self.critic(real_images, training=True)

                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_predictions
                )
                c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Calculate Critic and Generator Accuracy
        c_acc = (tf.reduce_mean(tf.cast(tf.math.greater(real_predictions, 0), tf.float32))* 100)
        g_acc = (tf.reduce_mean(tf.cast(tf.math.less(fake_predictions, 0), tf.float32)) * 100)

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {
            **{m.name: m.result() for m in self.metrics},
            "c_acc": c_acc,
            "g_acc": g_acc,
        }
        
    def test_step(self, real: tf.Tensor):
        batch_size = tf.shape(real)[0]

        latent = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake = self.generator(latent, training=False)
        fake_preds = self.critic(fake, training=False)
        real_preds = self.critic(real, training=False)

        c_acc = (tf.reduce_mean(tf.cast(tf.math.greater(real_preds, 0), tf.float32))* 100)
        g_acc = (tf.reduce_mean(tf.cast(tf.math.less(fake_preds, 0), tf.float32)) * 100)

        return {**{m.name: m.result() for m in self.metrics}, "c_acc": c_acc, "g_acc": g_acc}

    def call(self, latent: tf.Tensor):
        return self.generator(latent)