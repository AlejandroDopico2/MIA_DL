from keras.applications import EfficientNetB0, MobileNetV2, ResNet50V2
from keras.layers import Dense, Flatten, Rescaling
from keras.models import Model
from utils import freeze


class PretrainedModel(Model):
    AVAILABLE = {
        "resnet": ResNet50V2,
        "mobilenet": MobileNetV2,
        "efficientnet": EfficientNetB0,
    }

    def __init__(
        self,
        num_classes: int,
        img_size: int,
        pretrained: str,
        proj: int = 1000,
        defreeze: int = -1,
    ):
        """Image classifier model with pretrained weights.

        Args:
            num_classes (int): Number of target classes.
            pretrained (str): Pretrained model on Imagenet.
            img_size (int): Dimension of the image.
            proj (int, optional): Hidden dense projection.
            defreeze (int, optional): Number of layers to defreeze.. Defaults to -1.
        """
        assert (
            pretrained in self.AVAILABLE.keys()
        ), "Pretrained model not available for this implementation"
        super().__init__(name="PretrainedModel")

        self.rescaling = Rescaling(1.0 / 255)
        self.model = freeze(
            self.AVAILABLE[pretrained](
                weights="imagenet",
                include_top=False,
                input_shape=(img_size, img_size, 3),
            ),
            defreeze,
        )
        self.flatten = Flatten()
        self.proj = Dense(proj, activation="relu")
        self.dense = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        x = self.model(x)
        x = self.flatten(x)
        x = self.proj(x)
        x = self.dense(x)
        return x
