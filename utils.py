import gc
import tensorflow as tf
from keras.backend import clear_session
from keras.utils import image_dataset_from_directory
from tensorflow.data import Dataset
from keras.optimizers import Optimizer
from keras import models
from keras import Model
from keras import layers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt

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

def train_model(
    model: Model,
    train: Dataset,
    val: Dataset,
    optimizer: Optimizer = 'adam',
    epochs: int = 20,
    batch_size: int = 64,
    val_patience: int = 10, 
    lr_patience: int = 5, 
    verbose: Union[str, int] = "auto",
    path: Optional[str] = None,
    name: str = "conv_net.keras",
) -> History:

    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    callbacks = [ 
        EarlyStopping(monitor="val_loss", mode="min", patience=val_patience),
        ReduceLROnPlateau('val_loss', factor=0.2, patience=lr_patience, min_delta=1e-5, min_lr=0)
    ]
    if path:
        callbacks.append(ModelCheckpoint(f'{path}/{name}.keras', 'val_loss', save_best_only=True))
    
    history = model.fit(train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, 
                                 verbose=verbose, validation_data=val)
    
    return history

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
