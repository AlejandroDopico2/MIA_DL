from itertools import product 
from collections import OrderedDict
import pandas as pd
from keras.initializers import *
from keras.optimizers import * 
from keras.regularizers import * 
from keras.layers import *
from model import WalmartModel

from utils import * 
from data import * 

Regularizer.__str__ = lambda x: str(x.__class__.__name__)
Optimizer.__str__ = lambda x: str(x.__class__.__name__) + f'({float(x.learning_rate.numpy()):1.0e})'

# global parameters 
TEST_RATIO = 0.2
VAL_RATIO = 0.15
BATCH_SIZE = 200

# load data 
data = WalmartDataset.load('Walmart.csv')
train, val, test = data.split(VAL_RATIO, TEST_RATIO)

if __name__ == '__main__':
    
    grid = OrderedDict(
        regularizer = [L1(1e-3), L2(1e-3), L1L2(1e-3)],
        initializer=['random_normal', 'glorot_uniform'],
        activation=['tanh', 'relu']
    )

    def applydeep(lists, func):
        result = []
        for item in lists:
            result.append(list(map(func, item)))
        return result

    df = pd.DataFrame(columns=['train', 'val', 'test'], index=pd.MultiIndex.from_product(applydeep(grid.values(), str)))
    for i, params in enumerate(product(*grid.values())):
        params = dict(zip(grid.keys(), params))
        model = WalmartModel(seq_len=3, base_layer=GRU, num_encoder_layers=3, num_decoder_layers=2,  bidirectional=False, dropout=0.1,**params)
        model.train(train, test, f'results/walmart.weights.h5', Adam(1e-4), batch_size=BATCH_SIZE)
        (_, train_mae, _), (_, val_mae, _), (_, test_mae, _) = map(model.evaluate, (train, val, test))
        df.loc[tuple(map(str, params.values()))] = [train_mae, val_mae, test_mae]
        df.to_csv('grid.csv')
    df.index.names = grid.keys()