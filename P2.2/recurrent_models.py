from typing import List, Union

from keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout, Embedding, SimpleRNN, Layer
from keras.models import Model, Sequential


def create_recurrent_model(
    max_features: int,
    embedding_dim: int,
    recurrent_layer: Layer,
    num_recurrent_layers: int,
    recurrent_units: Union[List[int], int],
    num_ffn_layers: int = 0, 
    ffn_units: Union[List[int], int] = [],
    bidirectional: bool = False
) -> Model:
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units for _ in range(num_recurrent_layers)]
    if isinstance(ffn_units, int):
        ffn_units = [ffn_units for _ in range(num_ffn_layers)]
    
    if bidirectional:
        add_layer = Bidirectional()
    
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True),
        
        # recurrent encoder
        *[
            recurrent_layer()
        ]
    ])
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
