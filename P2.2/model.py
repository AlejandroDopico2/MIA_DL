from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from data import AmazonDataset
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers import Bidirectional, Dense, Dropout, Embedding, Layer
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import Regularizer
from keras_nlp.layers import TransformerEncoder
from utils import plot_history


class AmazonReviewsModel:
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        recurrent_layer: Layer,
        num_recurrent_layers: int = 2,
        recurrent_dims: Optional[Union[List[int], int]] = None,
        ffn_dims: List[int] = [],
        num_transformers: int = 0,
        bidirectional: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
        regularizer: Optional[Regularizer] = None,
        initializer: str = "glorot_uniform",
        name: str = "AmazonReviewsModel",
    ):
        self.name = name
        if recurrent_dims is None:
            recurrent_dims = embed_size
        if isinstance(recurrent_dims, int):
            recurrent_dims = [recurrent_dims for _ in range(num_recurrent_layers)]

        if bidirectional:
            add_layer = lambda *args, **kwargs: Bidirectional(
                recurrent_layer(*args, **kwargs)
            )
        else:
            add_layer = recurrent_layer

        self.model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=embed_size, mask_zero=True),
                # transformer layers
                *[
                    TransformerEncoder(embed_size, num_heads=4)
                    for _ in range(num_transformers)
                ],
                # recurrent encoder
                *[
                    add_layer(
                        dim,
                        kernel_regularizer=regularizer,
                        kernel_initializer=initializer,
                        return_sequences=True,
                    )
                    for dim in recurrent_dims[:-1]
                ],
                add_layer(recurrent_dims[-1]),
                Dropout(dropout),
                # ffn decoder
                *[Dense(dim, activation=activation) for dim in ffn_dims],
                Dense(1, activation="sigmoid"),
            ],
            name=name,
        )

    def train(
        self,
        data: AmazonDataset,
        path: str,
        lr: float = 1e-3,
        opt: Callable = Adam,
        epochs: int = 10,
        batch_size: int = 30,
        dev_patience: int = 10,
        train_patience: int = 5,
        clipnorm: Optional[int] = None,
        clipvalue: Optional[int] = None,
    ) -> Tuple[History, go.Figure]:
        optimizer = opt(lr, clipnorm=clipnorm, clipvalue=clipvalue)
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=dev_patience, mode="min", verbose=0
            ),
            EarlyStopping(
                monitor="loss", patience=train_patience, mode="min", verbose=0
            ),
            ModelCheckpoint(
                path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]
        self.model.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        history = self.model.fit(
            data.X_train,
            data.y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(data.X_val, data.y_val),
        )
        self.model.load_weights(path)
        return history, plot_history(
            history.history, metrics=["loss", "accuracy"], name=self.model.name
        )

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, batch_size: int = 1000):
        return self.model.evaluate(inputs, targets, verbose=0, batch_size=batch_size)
