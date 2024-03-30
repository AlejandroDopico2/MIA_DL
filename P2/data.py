from __future__ import annotations 
from typing import List, Tuple, Optional
import pandas as pd 
import numpy as np 
import random 
import tensorflow as tf 


class StoreInstance:
    def __init__(self, ID: int, inputs: np.ndarray):
        self.ID = ID 
        self.inputs = inputs 
        
    def __len__(self) -> int:
        return self.inputs.shape[0]
    
    def to_numpy(
            self, 
            seq_len: int,
            means: Optional[np.ndarray] = None,
            stds: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        raw_inputs = (self.inputs-means)/stds if means is not None else self.inputs
            
        inputs, targets = zip(*[
            (
                raw_inputs[i:(i+seq_len)],                          # input
                raw_inputs[i+seq_len, WalmartDataset.TARGET_INDEX]  # target
            ) 
            for i in range(len(self)-seq_len)
        ])
        return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)
        
        
class WalmartDataset:
    TARGET_INDEX = 0
    DELAY = 2
    N_FEATURES = 6
    N_SAMPLES = 143
    
    def __init__(self, stores: List[StoreInstance], dates: pd.Series):
        self.stores = stores 
        self.dates = dates
        
    def split(self, ptest: float, shuffle: bool = True, seed: int = 123) -> Tuple[WalmartDataset, WalmartDataset]:
        if shuffle:
            random.seed(seed)
            random.shuffle(self.stores)
        ntest = int(len(self.stores)*ptest)
        return WalmartDataset(self.stores[ntest:], self.dates), \
            WalmartDataset(self.stores[:ntest], self.dates)
    
    @property 
    def mean(self) -> np.ndarray:
        return np.mean(np.concatenate([store.inputs for store in self.stores]), axis=0)
    
    @property 
    def std(self) -> np.ndarray:
        return np.std(np.concatenate([store.inputs for store in self.stores]), axis=0)
    
    @property
    def target_std(self) -> float:
        return self.std[self.TARGET_INDEX]
    
    @property
    def target_mean(self) -> float:
        return self.mean[self.TARGET_INDEX]
    
    def __len__(self) -> int:
        return sum(map(len, self.stores))
    
    def __getitem__(self, item: int) -> StoreInstance:
        return self.stores[item]
    
    def to_numpy(
            self, 
            seq_len: int,
            means: Optional[np.ndarray] = None, 
            stds: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
        inputs, targets = map(np.concatenate, zip(*[store.to_numpy(seq_len, means, stds) for store in self.stores]))
        return inputs, targets
        
    def to_tf(
            self, 
            seq_len: int,
            means: Optional[np.ndarray] = None, 
            stds: Optional[np.ndarray] = None,
            batch_size: int = 128,
            shuffle: bool = False
        ) -> tf.data.Dataset:
        inputs, targets = self.to_numpy(seq_len, means, stds)
        data = tf.data.Dataset.from_tensor_slices((inputs, targets))
        if shuffle:
            data = data.shuffle(len(self))
        return data.batch(batch_size)
    
    
    @classmethod 
    def load(cls, path: str) -> WalmartDataset:
        # read adata 
        data = pd.read_csv(path)
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data.index = pd.MultiIndex.from_frame(data[['Store', 'Date']])
        dates = sorted(data['Date'].unique())
        data = data.sort_index().drop(['Store', 'Date'], axis=1)
        
        store_idxs = data.index.get_level_values(0)
        stores = [StoreInstance(idx, data[store_idxs == idx].values) for idx in store_idxs.unique()]
        return WalmartDataset(stores, dates)
        
        
    