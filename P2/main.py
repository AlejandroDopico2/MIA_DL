from itertools import product 
from collections import OrderedDict
import os
from data import * 
from plots import *
from model import *
from keras.layers import * 
from keras.models import Sequential, Model
from keras.optimizers import Adam, Optimizer, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1, L2, L1L2
from tensorflow.data import Dataset
from tqdm import tqdm 

# global parameters 
TEST_RATIO = 0.2
BATCH_SIZE = 200

# load data 
data = WalmartDataset.load('Walmart.csv')
train, test = data.split(TEST_RATIO)

Regularizer.__str__ = lambda x: str(x.__class__.__name__)
Optimizer.__str__ = lambda x: str(x.__class__.__name__) + f'({float(x.learning_rate.numpy()):1.0e})'

grid = OrderedDict(
    seq_len = [2, 3, 4],
    num_layers = [2, 3, 4],
    hidden_size = [50, 60, 70],
    dropout = [0.1, 0.2]
)

def applydeep(lists, func):
    result = []
    for item in lists:
        result.append(list(map(func, item)))
    return result



if not os.path.exists('grid.csv'):
    df = pd.DataFrame(columns=['train', 'test'], 
                  index=pd.MultiIndex.from_product(applydeep(grid.values(), str)))
    for i, params in enumerate(product(*grid.values())):
        params = dict(zip(grid.keys(), params))
        model = WalmartModel(**params)
        model.train(train, test, f'results/walmart.weights.h5', Adam(1e-2), batch_size=BATCH_SIZE)
        (_, train_mae, _), (_, test_mae, _) = map(model.evaluate, (train, test))
        df.loc[tuple(map(str, params.values()))] = [train_mae, test_mae]
        df.to_csv('grid.csv')
else:
    df = pd.read_csv('grid.csv', index_col=list(range(len(grid.keys()))))
df.index.names = grid.keys()