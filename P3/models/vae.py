
from __future__ import annotations
from typing import Tuple, List, Dict
from keras.optimizers.legacy import Optimizer, Adam
from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, Layer, Activation, ZeroPadding2D, Cropping2D
from keras.activations import sigmoid 
from keras.models import Model
from keras.optimizers import Optimizer
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K
import numpy as np 
import tensorflow as tf 
from data import CelebADataset
from utils import FID
import os, shutil, cv2

class VariationalAutoEncoder:
    LOSS_FACTOR = 1000
    NORM = lambda _, x: x/255
    DENORM = lambda _, x: np.uint8(x*255)
    
    def __init__(
        self, 
        img_size: Tuple[int, int, int], 
        hidden_size: int, 
        kernels: List[int],  
        strides: List[int],
        filters: List[int], 
    ):
        
        # ---------------------- ENCODER ----------------------
        encoder_input = Input(shape=img_size, name='encoder-input')
        x = encoder_input 
        shapes = [img_size]
        for i, (n, k, s) in enumerate(zip(filters, kernels, strides)):
            x = Conv2D(n, k, strides=s, padding='same', name=f'encoder-conv-{i}')(x)
            x = LeakyReLU()(x)
            shapes.append(K.int_shape(x)[1:] )
            
        x = Flatten()(x)
        mean_mu = Dense(hidden_size, name = 'mu')(x)
        log_var = Dense(hidden_size, name = 'log-var')(x)
        
        def sampling(args):
            mean_mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
            return mean_mu + K.exp(log_var/2)*epsilon   

        encoder_output = Lambda(sampling, name='encoder-output')([mean_mu, log_var])
        self.encoder = Model(encoder_input, encoder_output, name='encoder')
        
        # ---------------------- DECODER ----------------------
        decoder_input = Input(shape=(hidden_size,), name='decoder-input')
        filters = [img_size[-1]] + filters[1:]
        x = Dense(np.prod(shapes[-1]))(decoder_input)
        x = Reshape(shapes[-1])(x)
        for i, (n, k, s) in enumerate(zip(filters[::-1], kernels[::-1], strides[::-1])):
            x = Conv2DTranspose(n, k, strides=s, padding='same', name=f'decoder-conv-{i}')(x)
            
            h, w = x.shape[1] - shapes[-i-2][0], x.shape[2] - shapes[-i-2][1]
            x = Cropping2D(cropping=((h//2+(h%2 != 0), h//2), (w//2+(w%2 != 0), w//2)))(x)
            # check if dimensions are the same 
            if i == (len(filters) - 1):
                x = Activation(sigmoid)(x)
            else:
                x = LeakyReLU()(x)
        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output, name='decoder')    
        self.model = Model(encoder_input, self.decoder(encoder_output), name='VAE')
    
        def r_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        def kl_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
            return kl_loss

        def total_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return self.LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)
        
        self.METRICS = [r_loss, kl_loss]
        self.LOSS = total_loss
        
        
    def train(
        self, 
        train: CelebADataset,
        val: CelebADataset,
        test: CelebADataset,
        path: str, 
        batch_size: int = 10,
        epochs: int = 10,
        train_patience: int = 10,
        dev_patience: int = 5,
        optimizer: Optimizer = Adam(1e-4)
    ) -> Dict[str, List[float]]:
        self.model.compile(optimizer=optimizer, loss=self.LOSS, metrics=self.METRICS)
        if os.path.exists(path):
            shutil.rmtree(path)
        callbacks = [
            ModelCheckpoint(f'{path}/model.h5', save_weights_only=True, save_best_only=True, verbose=0), 
            EarlyStopping('loss', patience=train_patience),
            EarlyStopping('val_loss', patience=dev_patience)
        ] 
        
        train_tf = train.to_tf(self.NORM, batch_size, targets=True)
        val_tf = val.to_tf(self.NORM, batch_size, targets=True)
        history = self.model.fit(train_tf, validation_data=val_tf, epochs=epochs, callbacks=callbacks).history 
        self.model.load_weights(f'{path}/model.h5')
        self.predict(test, f'{path}/preds/', batch_size)
        return history 


    def predict(
        self,
        data: CelebADataset,
        out: str, 
        batch_size: int = 10,
    ) -> float:
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        inputs, outputs = [], []
        for files, input in data.stream(self.NORM, batch_size):
            output = self.DENORM(self.model.predict(input))
            for j in range(output.shape[0]):
                cv2.imwrite(f'{out}/{files[j]}', output[j])
            inputs.append(input)
            outputs.append(output)
        return FID(CelebADataset.IMG_SIZE)(np.concatenate(inputs, 0), np.concatenate(outputs, 0))
            
    def evaluate(
        self, 
        data: CelebADataset,
        batch_size: int = 10
    ) -> float:
        inputs, outputs = [], []
        for _, input in data.stream(self.NORM, batch_size):
            output = self.DENORM(self.model.predict(inputs))
            outputs.append(output)
            inputs.append(input)
        return FID(CelebADataset.IMG_SIZE)(np.concatenate(inputs, 0), np.concatenate(outputs, 0))



        