import gc
import tensorflow as tf
from keras.backend import clear_session
from keras.utils import image_dataset_from_directory
from tensorflow.data import Dataset
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import L1, L2, L1L2
from keras import models
from keras import Model
from keras import layers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt

# OPTIMIZERS = {"rmsprop": RMSprop, "adam": Adam, "sgd": SGD}
# REGULARIZERS = {"l1": L1, "l2": L2, "l1l2": L1L2}


def reset():
    clear_session()
    gc.collect()


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


# def build_model(
#     num_classes: int,
#     img_size: int = 128,
#     dropout: float = 0.0,
#     use_batch_normalization: bool = False,
#     weight_regularization: Optional[str] = None,
#     kernel_initializer: str = "glorot_uniform",
#     regularizer_factor: float = 0.001,
#     optimizer: str = "rmsprop",
#     learning_rate: float = 1e-3,
#     kernel_size: int = 3,
# ) -> models.Model:

#     if dropout > 0 and use_batch_normalization:
#         print(
#             "Using dropout and batch normalization together is not recommendable, switching to only batch normalization..."
#         )
#         dropout = 0

#     if weight_regularization:
#         weight_regularization = REGULARIZERS[weight_regularization.lower()](
#             regularizer_factor
#         )

#     args = dict(
#         kernel_regularizer=weight_regularization,
#         kernel_initializer=kernel_initializer,
#         bias_initializer="zeros",
#         activation="relu",
#     )

#     # Input layer
#     inputs = layers.Input(shape=(img_size, img_size, 3))
#     x = layers.Rescaling(1.0 / 255)(inputs)

#     # Convolutional layers
#     x = layers.Conv2D(32, kernel_size=kernel_size, name="hidden_1", **args)(x)
#     if dropout > 0:
#         x = layers.Dropout(dropout)(x)
#     elif use_batch_normalization:
#         x = layers.BatchNormalization()(x)

#     x = layers.MaxPooling2D(pool_size=2)(x)

#     x = layers.Conv2D(64, kernel_size=kernel_size, name="hidden_2", **args)(x)
#     if dropout > 0:
#         x = layers.Dropout(dropout)(x)
#     elif use_batch_normalization:
#         x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D(pool_size=2)(x)
#     x = layers.Conv2D(64, kernel_size=kernel_size, name="hidden_3", **args)(x)
#     x = layers.MaxPooling2D(pool_size=2)(x)
#     # Flatten and dense layers
#     x = layers.Flatten()(x)
#     # x = layers.Dense(16, name='hidden_dense', **args)(x)
#     outputs = layers.Dense(num_classes, name="output", activation="softmax")(x)

#     # Create the model
#     model = models.Model(inputs=inputs, outputs=outputs)

#     optimizer = OPTIMIZERS[optimizer.lower()](learning_rate)
#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=["sparse_categorical_accuracy"],
#     )

#     return model


def train_model(
    model: Model,
    train_data: Dataset,
    val_data: Dataset,
    n_epochs: int = 20,
    batch_size: int = 64,
    es_patience: int = 10,
    lr_patience: int = 5,
    verbose: Union[str, int] = "auto",
    model_dir: str = ".",
    name: str = "conv_net.keras",
) -> History:
    callbacks = []

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=es_patience)
    callbacks.append(early_stopping)
    model_checkpoint = ModelCheckpoint(
        model_dir + "/models/" + name, "val_loss", save_best_only=True
    )
    callbacks.append(model_checkpoint)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        min_delta=0.00001,
        patience=lr_patience,
        min_lr=0.0,
    )
    callbacks.append(reduce_lr)

    return model.fit(
        train_data,
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        validation_data=val_data,
    )


def plot(history: Dict[str, np.ndarray], metric: str = "accuracy"):
    train = history[metric]
    plt.clf()
    epochs = range(1, len(train) + 1)

    plt.plot(epochs, train, "b-o", label="Training " + metric)
    if "val_" + metric in history.keys():
        validation = history[f"val_{metric}"]
        plt.plot(epochs, validation, "r-o", label="Validation " + metric)

    plt.title("Training and validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.show()
