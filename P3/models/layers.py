from keras.layers import Layer, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Identity, LeakyReLU, ReLU, Add, Concatenate, Input 
import tensorflow as tf
from keras.models import Model 
from typing import Tuple 
from keras import Sequential


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
        
    def build(self, input_shape):
        self.conv.build(input_shape)
        
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
    
    def build(self, input_shape):
        self.conv.build(input_shape)
    
    def call(self, inputs: tf.Tensor):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.act(x)
        return self.dropout(x)
        
        
        
class ResidualBlock(Layer):
    def __init__(self, n_filters: int, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(n_filters, kernel_size=4, strides=2, padding="same")
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
    
    
    
class SkipAutoEncoder(Model):
    def __init__(self, img_size: Tuple[int, int, int], **kwargs):
        super(SkipAutoEncoder, self).__init__(**kwargs)
        # encoder 
        self.conv1 = ConvBlock(64, 4, 2, bias=False, batch_norm=True, name='conv1')
        self.conv2 = ConvBlock(128, 4, 2, bias=False, batch_norm=True, name='conv2')
        self.conv3 = ConvBlock(256, 4, 2, bias=False, batch_norm=True, name='conv3')
        self.conv4 = ConvBlock(512, 4, 2, bias=False, batch_norm=True, name='conv4')
        self.conv5 = ConvBlock(512, 4, 2, bias=False, batch_norm=True, name='conv5')
        
        # decoder 
        self.deconv5 = DeconvBlock(512, 4, 2, bias=False, name='deconv5')
        self.deconv4 = Sequential([Concatenate(-1), DeconvBlock(512, 4, 2, bias=False)], name='deconv4')
        self.deconv3 = Sequential([Concatenate(-1), DeconvBlock(512, 4, 2, bias=False)], name='deconv3')
        self.deconv2 = Sequential([Concatenate(-1), DeconvBlock(256, 4, 2, bias=False)], name='deconv2')
        self.deconv1 = Sequential([Concatenate(-1), DeconvBlock(128, 4, 2, bias=False)], name='deconv1')
        self.out = Sequential([
            DeconvBlock(512, 5, 2, bias=False),
            Conv2D(3, 1, activation='tanh')
        ], name='output')

    def build(self, input_shape):
        self.conv1.build(input_shape)
        
    def call(self, real: tf.Tensor):
        x1 = self.conv1(real)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        y4 = self.deconv5(x5)
        y3 = self.deconv4([y4, x4])
        y2 = self.deconv3([y3, x3])
        y1 = self.deconv1([y2, x2])
        return self.out(y1)
        
        
def skip_autoencoder(img_size: Tuple[int, int, int], name: str):
    inputs = Input(img_size)
    conv1 = ConvBlock(64, 4, 2, bias=False, batch_norm=True, name='conv1')(inputs)
    conv2 = ConvBlock(128, 4, 2, bias=False, batch_norm=True, name='conv2')(conv1)
    conv3 = ConvBlock(256, 4, 2, bias=False, batch_norm=True, name='conv3')(conv2)
    conv4 = ConvBlock(512, 4, 2, bias=False, batch_norm=True, name='conv4')(conv3)
    conv5 = ConvBlock(512, 4, 2, bias=False, batch_norm=True, name='conv5')(conv4)
    
    # decoder 
    deconv5 = DeconvBlock(512, 4, 2, bias=False, name='deconv5')(conv5)
    deconv4 = Sequential([Concatenate(-1), DeconvBlock(512, 4, 2, bias=False)], name='deconv4')([deconv5, conv4])
    deconv3 = Sequential([Concatenate(-1), DeconvBlock(512, 4, 2, bias=False)], name='deconv3')([deconv4, conv3])
    deconv2 = Sequential([Concatenate(-1), DeconvBlock(256, 4, 2, bias=False)], name='deconv2')([deconv3, conv2])
    deconv1 = Sequential([Concatenate(-1), DeconvBlock(128, 4, 2, bias=False)], name='deconv1')([deconv2, conv1])
    deconv0 = DeconvBlock(512, 5, 2, bias=False)(deconv1)
    out = Conv2D(3, 1, activation='tanh', name='output')(deconv0)
    
    model = Model(inputs, out, name=name)
    return model 