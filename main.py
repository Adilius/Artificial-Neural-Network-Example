import time
import_start = time.time()

import os   #Turn off annoying tensorflow cuda error message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Turn off annoying tensorflow cuda error message
from rich.console import Console    #Rich used for pretty console printing

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import_end = time.time()
print("Importing modules took:", round(import_end - import_start,2),"seconds.")

RANDOM_STATE = 66 #Sets seed for splitting train and test set
FRAC = 0.85 #Set fraction of train dataset
EPOCHS = 5 #Number of epochs to train model

#Create and train tensorflow model of a neural network
def create_train_model(train_features):

    #Create model object
    model = tf.keras.models.Sequential()

    #Add layer containing feature columns to model
    feature_columns = []

    for i in train_features.columns[:-1]:
        feature_columns.append(tf.feature_column.numeric_column(i))
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model.add(feature_layer)

    #Add linear layer to model for simple linear regressor
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    #Compile layers into model for TensorFlow execution
    model.compile(
        optimizer='adam',
        loss="mse",
        metrics= ['accuracy']
    )

    return model

#Initialize console object
console = Console()

#Read file into dataframe
column_names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = pd.read_csv("abalone.data", names=column_names)

#Preprocessing with label encoder
categorical_feature_mask = df.dtypes==object #Categorical mask
categorical_cols = df.columns[categorical_feature_mask].tolist() #Get categories using mask
le = preprocessing.LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(lambda col:le.fit_transform(col)) #Apply mask

#Normalize column-wise in dataset 
df = (df-df.min())/(df.max()-df.min())

#Split training set and testing set
train_dataset = df.sample(frac=FRAC, random_state=RANDOM_STATE)
test_dataset = df.drop(train_dataset.index)

#Seperate features from labels
train_features = train_dataset.copy()
train_lables = train_features.pop('Rings')
test_features = test_dataset.copy()
test_lables = test_dataset.pop('Rings')

ann_model = create_train_model(train_features)
#ann_model.fit(train_features,train_lables, epochs=EPOCHS)