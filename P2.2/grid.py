from collections import OrderedDict
from keras.regularizers import L1, L2, L1L2, Regularizer
from keras.optimizers import Adam, RMSprop
from itertools import product 
import pandas as pd 
import sys 
from model import *
from data import *


# global parameters 
MAX_FEATURES = 1000
MODEL_PATH = 'results/'
train_default = dict(epochs=30, batch_size=1300, lr=1e-3, dev_patience=5)
path_dir = 'AmazonDataset/'
dataset = AmazonDataset.load(train_path=path_dir + "train_small.txt", test_path=path_dir + "test_small.txt", max_features=MAX_FEATURES)

grid = OrderedDict(
    regularizer = [L1(1e-4), L2(1e-3), L1L2(1e-4)],
    initializer=['random_normal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'orthogonal'],
    optimizer=[Adam, RMSprop]
)
Regularizer.__repr__ = lambda x: x.__class__.__name__

def tostring(x):
    if isinstance(x, type):
        return x.__name__
    else:
        return repr(x)

def applydeep(lists, func):
    result = []
    for item in lists:
        result.append(list(map(func, item)))
    return result




if __name__ == '__main__':
    df = pd.DataFrame(columns=['train', 'val', 'test'], 
                        index=pd.MultiIndex.from_product(applydeep(grid.values(), tostring)))
    search = list(product(*grid.values()))
    path = 'grid.csv'
    df.index.names = ['regularizer', 'initializer', 'optimizer']
    
    if len(sys.argv) > 1:
        start, end = map(int, sys.argv[1:])
        search = search[start:end]
        path = f'grid[{start}-{end}].csv'
        
    for i, params in enumerate(search):
        params = dict(zip(grid.keys(), params))
        model_params = params.copy()
        model_params.pop('optimizer')
        
        model = AmazonReviewsModel(
            2000, 256, GRU, num_recurrent_layers=3, ffn_dims=[64], dropout=0.1, bidirectional=True,
            **model_params
        )
        model.train(dataset, f'results/amazon{i+start}.weights.h5', opt=params['optimizer'], **train_default)
        _, train_acc = model.evaluate(dataset.X_train, dataset.y_train)
        _, val_acc = model.evaluate(dataset.X_val, dataset.y_val)
        _, test_acc = model.evaluate(dataset.X_test, dataset.y_test)
        
        df.loc[tuple(map(tostring, params.values()))] = [train_acc, val_acc, test_acc]
        df.to_csv(path)
        