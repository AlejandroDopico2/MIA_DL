from keras.layers import * 
from keras.models import Sequential, Model
from keras.regularizers import Regularizer, L2
from keras.optimizers import Optimizer, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import Metric, MeanAbsoluteError
from keras import ops
import numpy as np 
import tensorflow as tf
from data import WalmartDataset
from typing import Optional


class DenormalizedMAE(MeanAbsoluteError):
    def __init__(self, std: float, name='dmae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.std = std
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.scalar_mul(self.std, y_true), tf.scalar_mul(self.std, y_pred), sample_weight)
        
        

class WalmartModel:
    def __init__(
        self,
        seq_len: int,
        base_layer: Layer = LSTM,
        num_encoder_layers: int = 2, 
        num_decoder_layers: int = 2,
        hidden_size: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = False,
        activation: str = 'tanh',
        regularizer: Optional[Regularizer] = None,
        initializer: str = 'random_normal'
    ):
        add_lstm = lambda enc: base_layer(
            hidden_size, return_sequences=enc, 
            kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer,
            kernel_initializer=initializer, bias_initializer='zeros', recurrent_initializer=initializer
        )
        if bidirectional:
            add_layer = lambda enc: Bidirectional(add_lstm(enc))
        else:
            add_layer = add_lstm

        self.model = Sequential([
            # encoder part
            *[add_layer(True) for _ in range(num_encoder_layers-1)],
            Dropout(dropout),
            add_layer(False),
            # decoder part 
            *[Dense(hidden_size, activation=activation, kernel_regularizer=regularizer) for _ in range(num_decoder_layers-1)],
            Dense(1, activation='linear')
        ])
        self.seq_len = seq_len 
        self.means, self.stds = None, None 
        
    def train(
        self,
        train: WalmartDataset,
        val: WalmartDataset,
        path: str,
        optimizer: Optimizer = Adam(1e-4),
        batch_size: int = 50,
        epochs: int = 2000,
        patience: int = 5,
        verbose: int = 1
    ) -> Model:
        train_tf = train.to_tf(self.seq_len, batch_size, shuffle=True)
        self.means, self.stds = train.means, train.stds
        val_tf = val.to_tf(self.seq_len, batch_size, shuffle=False, norm=True, params=(self.means, self.stds))
        self.model.compile(optimizer, 'mse', metrics=[MeanAbsoluteError(name='mae'), DenormalizedMAE(train.target_std)])
        callbacks = [
            EarlyStopping(monitor='val_mae', patience=patience, mode='min', verbose=1),
            ModelCheckpoint(path, monitor='val_mae', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
        ]
        self.model.fit(train_tf, epochs=epochs, validation_data=val_tf, callbacks=callbacks, verbose=verbose)
        self.model.load_weights(path)
        return self.model 
    
    def evaluate(self, test: WalmartDataset, batch_size: int = 100):
        test_tf = test.to_tf(self.seq_len, batch_size, shuffle=False, norm=True, params=(self.means, self.stds))
        return self.model.evaluate(test_tf)
    
    def predict(self, test: WalmartDataset, batch_size: int = 100) -> np.ndarray:
        test_tf = test.to_tf(self.seq_len, batch_size, shuffle=False, norm=True, params=(self.means, self.stds))
        target_mean, target_std = self.means[WalmartDataset.TARGET_INDEX], self.stds[WalmartDataset.TARGET_INDEX]
        return self.model.predict(test_tf, verbose=0).flatten()*target_std+target_mean
        
