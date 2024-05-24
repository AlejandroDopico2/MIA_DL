from keras.layers import Layer, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Identity, LeakyReLU, ReLU, Add, Concatenate, Input, Flatten, Dense, Lambda, Reshape, UpSampling2D, MaxPooling2D
import tensorflow as tf
from keras.models import Model 
from typing import Tuple 
from keras import Sequential
import keras.backend as K
import numpy as np 

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
        pool: bool = False,
        **kwargs
    ):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv2D(n_filters, kernel_size, strides, padding, dilation_rate=(dilation, dilation), use_bias=bias)
        self.batch = BatchNormalization(momentum=momentum) if batch_norm else Identity()
        self.act = LeakyReLU(slope)
        self.dropout = Dropout(dropout)
        self.down = MaxPooling2D(2) if pool else Identity()
        
    def build(self, input_shape):
        self.conv.build(input_shape)
        
    def call(self, inputs: tf.Tensor):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.down(x)
        
        
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
        pool: bool = False,
        **kwargs
    ):
        super(DeconvBlock, self).__init__(**kwargs)
        self.conv = Conv2DTranspose(n_filters, kernel_size, strides, padding, dilation_rate=(dilation, dilation), use_bias=bias)
        self.batch = BatchNormalization(momentum=momentum) if batch_norm else Identity()
        self.act = LeakyReLU(slope)
        self.dropout = Dropout(dropout)
        self.up = UpSampling2D(2) if pool else Identity()
    
    def build(self, input_shape):
        self.conv.build(input_shape)
    
    def call(self, inputs: tf.Tensor):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.up(x)
        
        
        
class ResidualConvBlock(Layer):
    def __init__(
        self, 
        n_filters: int, 
        kernel_size: int, 
        strides: int = 1, 
        dilation: int = 1,
        bias: bool = True,
        slope: float = 0.2,
        momentum: float = 0.9,
        dropout: float = 0,
        padding: str = 'same',
        batch_norm: bool = True,
        pool: bool = False,
        **kwargs
    ):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, dilation_rate=(dilation, dilation), padding=padding, use_bias=bias)
        self.bn1 = BatchNormalization(momentum=momentum)  if batch_norm else Identity()
        self.conv2 = Conv2D(n_filters, kernel_size=3, strides=1, padding="same", use_bias=bias)
        self.bn2 = BatchNormalization(momentum=momentum) if batch_norm else Identity()
        self.skip_conv = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, dilation_rate=(dilation, dilation), padding=padding, use_bias=bias)
        self.act = LeakyReLU(slope)
        self.dropout = Dropout(dropout)
        self.down = MaxPooling2D(2) if pool else Identity()
        
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x_skipped = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x_skipped = self.skip_conv(x_skipped)
        x = Add()([x, x_skipped])
        x = self.act(x)
        return self.down(x)
    

    
        
class ResidualDeconvBlock(Layer):
    def __init__(
        self, 
        n_filters: int, 
        kernel_size: int, 
        strides: int = 1, 
        dilation: int = 1,
        bias: bool = True,
        slope: float = 0.2,
        momentum: float = 0.9,
        dropout: float = 0,
        padding: str = 'same',
        batch_norm: bool = True,
        pool: bool = False,
        **kwargs
    ):
        super(ResidualDeconvBlock, self).__init__(**kwargs)
        self.deconv1 = Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding=padding, use_bias=bias)
        self.bn1 = BatchNormalization(momentum=momentum)  if batch_norm else Identity()
        self.deconv2 = Conv2DTranspose(n_filters, kernel_size=3, strides=1, padding="same", use_bias=bias)
        self.bn2 = BatchNormalization(momentum=momentum) if batch_norm else Identity()
        self.skip_conv = Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding=padding, use_bias=bias)
        self.act = LeakyReLU(slope)
        self.dropout = Dropout(dropout)
        self.up = UpSampling2D(2) if pool else Identity()
        
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x_skipped = inputs
        x = self.deconv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.bn2(x, training=training)

        x_skipped = self.skip_conv(x_skipped)
        x = Add()([x, x_skipped])
        x = self.act(x)
        return self.up(x)
    
def sampling(args):
    mean_mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0.0, stddev=1.0)
    return mean_mu + K.exp(log_var / 2) * epsilon

def vae(img_size: Tuple[int, int, int], hidden_size: int, act: str, residual: bool, **args):
    conv = ConvBlock if not residual else ResidualConvBlock
    deconv = DeconvBlock if not residual else ResidualDeconvBlock
    
    encoder_input = Input(img_size, name='encoder-input')
    conv1 = conv(32, 3, **args, name='conv1')(encoder_input)
    conv2 = conv(32, 3, **args, name='conv2')(conv1)
    conv3 = conv(32, 3, **args, name='conv3')(conv2)
    conv4 = conv(32, 3, **args, name='conv4')(conv3)
    latent = Flatten()(conv4)
    mean_mu = Dense(hidden_size, name="mu")(latent)
    log_var = Dense(hidden_size, name="log-var")(latent)
    encoder_output = Lambda(sampling, name="encoder-output")([mean_mu, log_var])
    
    shape = (img_size[0]//2**4, img_size[1]//2**4, 32)
    decoder_input = Input((hidden_size,))
    transform = Sequential([Dense(np.prod(shape)), Reshape(shape)], name='transform')(decoder_input)
    deconv4 = deconv(32, 3, **args, name='deconv4')(transform)
    deconv3 = deconv(32, 3, **args, name='deconv3')(deconv4)
    deconv2 = deconv(32, 3, **args, name='deconv2')(deconv3)
    deconv1 = deconv(32, 3, **args, name='deconv1')(deconv2)
    decoder_output = Conv2D(3, 1, activation=act, name='output')(deconv1)
    return encoder_input, encoder_output, decoder_input, decoder_output, mean_mu, log_var


def skip_vae(img_size: Tuple[int, int, int], hidden_size: int, act: str, residual: bool, **args):
    conv = ConvBlock if not residual else ResidualConvBlock
    deconv = DeconvBlock if not residual else ResidualDeconvBlock

    encoder_input = Input(img_size, name='encoder-input')
    conv1 = conv(32, 3, **args, name='conv1')(encoder_input)
    conv2 = conv(32, 3, **args, name='conv2')(conv1)
    conv3 = conv(32, 3, **args, name='conv3')(conv2)
    conv4 = conv(32, 3, **args, name='conv4')(conv3)
    latent = Flatten()(conv4)
    mean_mu = Dense(hidden_size, name="mu")(latent)
    log_var = Dense(hidden_size, name="log-var")(latent)
    encoder_output = Lambda(sampling, name="encoder-output")([mean_mu, log_var])
    
    # decoder 
    shape = (img_size[0]//2**4, img_size[1]//2**4, 32)
    # decoder_input = Input((hidden_size,))
    transform = Sequential([Dense(np.prod(shape)), Reshape(shape)], name='transform')(encoder_output)
    deconv4 = deconv(32, 3, **args, name='deconv4')(transform)
    deconv3 = Sequential([Concatenate(-1), deconv(32, 3, **args)], name='deconv3')([deconv4, conv3])
    deconv2 = Sequential([Concatenate(-1), deconv(32, 3, **args)], name='deconv2')([deconv3, conv2])
    deconv1 = Sequential([Concatenate(-1), deconv(32, 3, **args)], name='deconv1')([deconv2, conv1])
    decoder_output = Conv2D(3, 1, activation=act, name='output')(deconv1)
    return encoder_input, encoder_output, encoder_output, decoder_output, mean_mu, log_var



class VariationalAutoEncoder(Model):
    def __init__(
            self, 
            img_size: Tuple[int, int, int], 
            hidden_size: int, 
            pool: str, 
            residual: bool, 
            skips: bool, 
            act: str,
            loss_factor: float,
            name: str,
        ):
        super().__init__(name=name)
        args = dict(dilation=2, strides=1, pool=True) if pool == 'dilation' else dict(strides=2, dilation=1)
        
        conv = ConvBlock if not residual else ResidualConvBlock
        
        # -------------------- encoder ------------------
        self.conv1 = conv(32, 3, **args, name='conv1')
        self.conv2 = conv(32, 3, **args, name='conv2')
        self.conv3 = conv(32, 3, **args, name='conv3')
        self.conv4 = conv(32, 3, **args, name='conv4')
        self.flatten = Flatten()
        self.mean = Dense(hidden_size, name="mu")
        self.logvar = Dense(hidden_size, name="log-var")
        self.latent = Lambda(sampling, name="encoder-output")
        
        # -------------------- decoder -----------------------
        deconv = DeconvBlock if not residual else ResidualDeconvBlock
        shape = (img_size[0]//2**4, img_size[1]//2**4, 32)
        self.transform = Sequential([Dense(np.prod(shape)), Reshape(shape)], name='transform')
        self.deconv4 = deconv(32, 3, **args, name='deconv5')
        if skips:
            self.deconv3 = Sequential([Concatenate(-1), deconv(32, 3, **args)], name='deconv3')
            self.deconv2 = Sequential([Concatenate(-1), deconv(32, 3, **args)], name='deconv2')
            self.deconv1 = Sequential([Concatenate(-1), deconv(32, 3, **args)], name='deconv1')
        else:
            self.deconv = DeconvBlock if not residual else ResidualDeconvBlock
            self.deconv3 = deconv(32, 3, **args, name='deconv3')
            self.deconv2 = deconv(32, 3, **args, name='deconv2')
            self.deconv1 = deconv(32, 3, **args, name='deconv1')
        self.out = Conv2D(3, 1, activation=act, name='output')

        def r_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        def kl_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            kl_loss = -0.5 * K.sum(1 + self.__mean - K.square(self.__mean) - K.exp(self.__logvar), axis=1)
            return kl_loss

        def total_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return loss_factor * r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)
        
        self.skips = skips 
        self.METRICS = [r_loss, kl_loss]
        self.LOSS = total_loss 
        self.build((None, *img_size))
        
    def build(self, input_shape):
        super(VariationalAutoEncoder, self).build(input_shape)
        dummy = tf.random.normal((1, *input_shape[1:]))
        self.call(dummy)
        
    def call(self, real: tf.Tensor):
        # encoder pass 
        conv1 = self.conv1(real)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        flat = self.flatten(conv4)
        self.__mean, self.__logvar = self.mean(flat), self.logvar(flat)
        latent = self.latent([self.__mean, self.__logvar])
        
        # decoder pass 
        reshape = self.transform(latent)
        deconv4 = self.deconv4(reshape)
        deconv3 = self.deconv3([deconv4, conv3] if self.skips else deconv4)
        deconv2 = self.deconv2([deconv3, conv2] if self.skips else deconv3)
        deconv1 = self.deconv1([deconv2, conv1] if self.skips else deconv2)
        return self.out(deconv1)