from keras.metrics import mean_absolute_error, Metric 
import tensorflow as tf 

class DenormalizedMAE(Metric):
    def __init__(self, std: float, name='dmae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.std = std 
        self.value = self.add_weight(shape=(), initializer='zeros', name='value')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.multiply(y_true, self.std)
        y_pred = tf.multiply(y_pred, self.std)
        self.value.assign(tf.squeeze(mean_absolute_error(y_true, y_pred)))
        
        
    def result(self):
        return self.value