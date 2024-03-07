from keras import Model
from tensorflow.data import Dataset
from typing import Union, Optional, List 
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Optimizer
from utils import reset 

class AnimalImageClassifier:
    def __init__(self, name: str, model: Model, img_size: int = 300):
        self.name = name 
        print(model.summary())
        self.model = model 
        self.img_size = img_size 
        
    def train(
        self, 
        train: Dataset,
        val: Dataset,
        optimizer: Optimizer = 'adam',
        path: Optional[str] = None,
        epochs: int = 20,
        batch_size: int = 64, 
        val_patience: int = 10, 
        lr_patience: int = 5, 
        verbose: Union[str, int] = 'auto',
        metrics: List[str] = ['sparse_categorical_accuracy']
    ) -> History:
        self.model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=metrics)
        callbacks = [
            EarlyStopping('val_loss', mode='min', patience=val_patience),
            ReduceLROnPlateau('val_loss', factor=0.2, patience=lr_patience, min_delta=1e-5, min_lr=0)
        ]
        if path: 
            callbacks.append(ModelCheckpoint(f'{path}/{self.name}', 'val_loss', save_best_only=True))
        history = self.model.fit(train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, 
                                 verbose=verbose, validation_data=val)
        reset()
        return history 
    
    
