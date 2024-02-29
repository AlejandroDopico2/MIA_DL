import tensorflow as tf
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Layer 
from keras.models import Sequential
from typing import Tuple

class Inception(Layer):
    def __init__(self, n_filters: Tuple[int]):
        super().__init__()
        self.block1 = Conv2D(n_filters[0], 1)
        self.block2 = Sequential([
            Conv2D(n_filters[1], 1),
            Conv2D(n_filters[1], 3, padding='same')
        ])
        self.block3 = Sequential([
            MaxPooling2D(2, strides=1, padding='same'),
            Conv2D(n_filters[2], 1, padding='same')
        ])
        self.block4 = Sequential([
            Conv2D(n_filters[3], 1),
            Conv2D(n_filters[3], 3, padding='same'),
            Conv2D(n_filters[3], 3, padding='same')
        ])
        self.concat = Concatenate(-1)

    def call(self, x: tf.Tensor):
        b1 = self.block1(x)
        b2 = self.block2(x)
        b3 = self.block3(x)
        b4 = self.block4(x)
        return self.concat([b1, b2, b3, b4])