from data import AmazonDataset
from recurrent_models import create_recurrent_model
from utils import *
from keras.layers import * 

# global parameters 
MAX_FEATURES = 1000
BATCH_SIZE = 30
NUM_EPOCHS = 5
MODEL_PATH = 'results/'

path_dir = 'AmazonDataset/'
# load data 
dataset = AmazonDataset.load(train_path=path_dir + "train_small.txt", test_path=path_dir + "test_small.txt", max_features=MAX_FEATURES)