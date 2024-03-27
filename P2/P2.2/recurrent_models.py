from typing import List, Union

from keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout, Embedding, SimpleRNN
from keras.models import Model, Sequential


def create_recurrent_model(
    recurrent_layer_type: str,
    recurrent_units: Union[List[int], int],
    num_recurrent_layers: int = 2,
    max_features: int = 1000,
    embedding_dim: int = 32,
    include_hidden: bool = False,
) -> Model:
    model = Sequential()
    model.add(
        Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True)
    )

    for i in range(num_recurrent_layers):
        return_sequences = False if i == num_recurrent_layers - 1 else True
        if recurrent_layer_type == "LSTM":
            model.add(LSTM(recurrent_units, return_sequences=return_sequences))
        elif recurrent_layer_type == "GRU":
            model.add(GRU(recurrent_units, return_sequences=return_sequences))
        elif recurrent_layer_type == "SimpleRNN":
            model.add(SimpleRNN(recurrent_units, return_sequences=return_sequences))
        elif recurrent_layer_type == "BidirectionalLSTM":
            model.add(
                Bidirectional(LSTM(recurrent_units, return_sequences=return_sequences))
            )
        else:
            raise ValueError(
                "Invalid recurrent_layer_type. Please choose from 'LSTM', 'GRU', 'SimpleRNN', or 'BidirectionalLSTM'."
            )

        model.add(Dropout(0.25))

    if include_hidden:
        model.add(Dense(64, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    return model
