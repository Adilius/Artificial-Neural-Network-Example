
import os   #Turn off annoying tensorflow cuda error message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Turn off annoying tensorflow cuda error message
from rich.console import Console    #Rich used for pretty console printing

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

RANDOM_STATE = 66 #Sets seed for splitting train and test set
FRAC = 0.85 #Set fraction of train dataset

#Initialize console object
console = Console()

#Read file into dataframe
column_names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = pd.read_csv("abalone.data", names=column_names)

#Split training set and testing set
train_dataset = df.sample(frac=FRAC, random_state=RANDOM_STATE)
test_dataset = df.drop(train_dataset.index)

#Seperate features from labels
train_features = train_dataset.copy()
train_lables = train_features.pop('Rings')

test_features = test_dataset.copy()
test_lables = test_dataset.pop('Rings')

#Normalize, TODO
#normalizer = preprocessing.Normalization()
#normalizer.adapt(np.array(train_features))

#print(train_features.describe().transpose()[['mean', 'std', 'max', 'min']])

print(tf.__version__)