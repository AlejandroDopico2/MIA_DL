import tensorflow as tf 
from keras.layers import Layer 
from transformers import TFAutoModel, AutoImageProcessor

class PretrainedLayer(Layer):
    def __init__(self, pretrained: str):
        self.pretrained = pretrained 
        self.processor = AutoImageProcessor.from_pretrained(pretrained)
        self.model = TFAutoModel.from_pretrained(pretrained)
        
    def forward(self, x: tf.Tensor):
        inputs = self.processor(x, return_tensors='tf')
        return self.model(**inputs).last_hidden_state