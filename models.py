import tensorflow as tf
from keras.models import Model
from keras import layers 
from keras.applications import ResNet50V2, MobileNetV2, EfficientNetB0
from layers import IdentityBlock, ConvolutionBlock, Inception
from typing import List, Tuple
from utils import freeze

class SimpleModel(Model):
    def __init__(self, num_classes: int, name: str = 'SimpleModel'):
        super().__init__(name=name)

        self.rescaling = layers.Rescaling(1.0 / 255)
        self.conv1 = layers.Conv2D(32, kernel_size=3)
        self.max1 = layers.MaxPooling2D(pool_size=2)
        self.conv2 = layers.Conv2D(64, kernel_size=3)
        self.max2 = layers.MaxPooling2D(pool_size=2)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)


class MidModel(Model):
    def __init__(self, num_classes: int, kernel_size: int = 3, dropout: float = 0.25, name: str = 'MidModel'):
        super().__init__(name=name)

        self.rescaling = layers.Rescaling(1.0 / 255)
        self.conv1 = layers.Conv2D(32, kernel_size, activation="relu")
        self.conv2 = layers.Conv2D(32, kernel_size, activation="relu")
        self.max_pool = layers.MaxPool2D(2)
        self.drop = layers.Dropout(dropout)

        self.conv3 = layers.Conv2D(64, kernel_size, activation="relu")
        self.conv4 = layers.Conv2D(64, kernel_size, activation="relu")

        self.conv5 = layers.Conv2D(128, kernel_size, activation="relu")
        self.conv6 = layers.Conv2D(128, kernel_size, activation="relu")

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(128, activation="relu")
        self.output_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)

class CustomResNet(Model):
    def __init__(self, num_classes: int, num_blocks: int = 4, filters_size: int = 64, name: str = 'ResNet'):
        super(CustomResNet, self).__init__(name=name)

        self.rescaling = layers.Rescaling(1.0 / 255)
        self.zpad = layers.ZeroPadding2D(3)
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.max_pool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        self.residual_blocks = []

        for i in range(num_blocks):
            if i == 0:
                self.residual_blocks.append(IdentityBlock(filters=filters_size))
            else:
                self.residual_blocks.append(ConvolutionBlock(filters=filters_size))
            self.residual_blocks.append(IdentityBlock(filters=filters_size))
            filters_size *= 2

        self.dense = layers.Dense(1000, activation="relu")
        self.output_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):

        x = self.rescaling(inputs)
        x = self.zpad(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = layers.ReLU()(x)
        x = self.max_pool(x)
        for block in self.residual_blocks:
            x = block(x)

        x = self.dense(x)
        x = layers.GlobalAveragePooling2D()(x)
        return self.output_layer(x)

class InceptionModel(Model):
    def __init__(
        self, 
        num_classes: int, 
        num_blocks: int, 
        n_filters = List[Tuple[int]], 
        kernel_size: int = 3,
        dropout: float = 0.2,
        name: str = 'InceptionModel'
    ):
        super(InceptionModel, self).__init__(name=name)

        self.rescaling = layers.Rescaling(1.0 / 255)
    
        self.inception_blocks = []
        self.convolutional_blocks = []
        self.batch_norm_layers = []

        for i in range(num_blocks):
            self.inception_blocks.append(Inception(n_filters=n_filters[i]))
            self.convolutional_blocks.append(layers.Conv2D(n_filters[i][-1], kernel_size = kernel_size))
            self.batch_norm_layers.append(layers.BatchNormalization())

        self.dense = layers.Dense(1000, activation="relu")
        self.dropout = layers.Dropout(dropout)
        self.output_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs: tf.Tensor, training=False):
        x = self.rescaling(inputs)
        
        for i, (block, conv, bn) in enumerate(zip(self.inception_blocks, self.convolutional_blocks, self.batch_norm_layers)):
            x = block(x)
            x = bn(x, training = training)
            x = conv(x)

            if i != len(self.inception_blocks)-1:
                x = layers.MaxPool2D()(x)
        
        x = layers.Flatten()(x)
        x = self.dropout(x)
        x = self.dense(x)
        
        return self.output_layer(x)

class PretrainedModel(Model):
    AVAILABLE = {'ResNet50': ResNet50V2, 'MobileNet': MobileNetV2, 'EfficientNetB0': EfficientNetB0}
    
    def __init__(self, num_classes: int, img_size: int,  pretrained: str, proj: int = 1000, defreeze: int = -1):
        """Image classifier model with pretrained weights.

        Args:
            num_classes (int): Number of target classes.
            pretrained (str): Pretrained model on Imagenet.
            img_size (int): Dimension of the image.
            proj (int, optional): Hidden dense projection.
            defreeze (int, optional): Number of layers to defreeze.. Defaults to -1.
        """
        assert pretrained in self.AVAILABLE.keys(), 'Pretrained model not available for this implementation'
        super().__init__(name=pretrained)
        
        self.rescaling = layers.Rescaling(1.0 / 255)
        self.model = freeze(self.AVAILABLE[pretrained](weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3)), defreeze)
        self.flatten = layers.Flatten()
        self.proj = layers.Dense(proj, activation='relu')
        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        x = self.model(x)
        x = self.flatten(x)
        x = self.proj(x)
        x = self.dense(x)
        return x