from keras.layers import Layer, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Identity, LeakyReLU, ReLU, Add
import tensorflow as tf

class ConvBlock(Layer):
    def __init__(
        self, 
        n_filters: int, 
        kernel_size: int, 
        strides: int = 1, 
        padding: str = 'same',
        dilation: int = 1,
        bias: bool = True,
        slope: float = 0.2,
        momentum: float = 0.9,
        batch_norm: bool = False,
        dropout: float = 0,
        **kwargs
    ):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv2D(n_filters, kernel_size, strides, padding, dilation_rate=(dilation, dilation), use_bias=bias)
        self.batch = BatchNormalization(momentum=momentum) if batch_norm else Identity()
        self.act = LeakyReLU(slope)
        self.dropout = Dropout(dropout)
        
    def call(self, inputs: tf.Tensor):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.act(x)
        return self.dropout(x)
        
        
class DeconvBlock(Layer):
    def __init__(
        self, 
        n_filters: int, 
        kernel_size: int, 
        strides: int = 1, 
        padding: str = 'same',
        dilation: int = 1,
        bias: bool = True,
        slope: float = 0.2,
        momentum: float = 0.9,
        batch_norm: bool = False,
        dropout: float = 0,
        **kwargs
    ):
        super(DeconvBlock, self).__init__(**kwargs)
        self.conv = Conv2DTranspose(n_filters, kernel_size, strides, padding, dilation_rate=(dilation, dilation), use_bias=bias)
        self.batch = BatchNormalization(momentum=momentum) if batch_norm else Identity()
        self.act = LeakyReLU(slope)
        self.dropout = Dropout(dropout)
        
    def call(self, inputs: tf.Tensor):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.act(x)
        return self.dropout(x)
        
        
        
class ResidualBlock(Layer):
    def __init__(self, n_filters: int, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(n_filters, kernel_size=3, strides=2, padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(n_filters, kernel_size=3, strides=1, padding="same")
        self.bn2 = BatchNormalization()
        self.skip_conv = Conv2D(n_filters, kernel_size=1, strides=2)

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