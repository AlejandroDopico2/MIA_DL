import tensorflow as tf
from keras.layers import Add, BatchNormalization, Conv2D, Layer, ReLU


class ConvolutionBlock(Layer):
    def __init__(self, filters: int):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=3, strides=2, padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.bn2 = BatchNormalization()

        self.skip_conv = Conv2D(filters, kernel_size=1, strides=2)

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x_skipped = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x_skipped = self.skip_conv(x_skipped)

        x = Add()([x, x_skipped])
        x = ReLU()(x)
        return x


class IdentityBlock(Layer):
    def __init__(self, filters: int):
        super(IdentityBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.bn2 = BatchNormalization()

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x_skipped = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = ReLU()(x)
        x = self.conv2(inputs)
        x = self.bn2(x, training=training)

        x = Add()([x, x_skipped])
        x = ReLU()(x)
        return x
