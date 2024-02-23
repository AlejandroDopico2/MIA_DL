from keras import Model
from keras import layers


class SimpleModel(Model):
    def __init__(self, num_classes: int):
        super().__init__()

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
    def __init__(self, num_classes: int, kernel_size: int = 3, dropout: float = 0.25):
        super().__init__()

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


class ConvolutionBlock(layers.Layer):
    def __init__(self, filters):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=2, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.skip_conv = layers.Conv2D(filters, kernel_size=1, strides=2)

    def call(self, inputs, training=False):
        x_skipped = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x_skipped = self.skip_conv(x_skipped)

        x = layers.Add()([x, x_skipped])
        x = layers.ReLU()(x)
        return x


class IdentityBlock(layers.Layer):
    def __init__(self, filters):
        super(IdentityBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x_skipped = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = layers.ReLU()(x)
        x = self.conv2(inputs)
        x = self.bn2(x, training=training)

        x = layers.Add()([x, x_skipped])
        x = layers.ReLU()(x)
        return x


class CustomResNet(Model):
    def __init__(self, num_classes: int, num_blocks: int = 4, filters_size: int = 64):
        super(CustomResNet, self).__init__()

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
