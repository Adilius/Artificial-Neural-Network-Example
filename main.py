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
EPOCHS = 100 #Number of epochs to train model
BATCH_SIZE = 1000

#Create tensorflow model of a neural network
def create_model(train_features):

    #Create model object
    model = tf.keras.models.Sequential()

    #Add layer containing feature columns to model
    feature_columns = []
    for i in train_features.columns:
        feature_columns.append(tf.feature_column.numeric_column(i))
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model.add(feature_layer)

    #Add linear layer to model for simple linear regressor
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    #Compile layers into model for TensorFlow execution
    model.compile(
        optimizer = 'adam',
        loss = "mean_squared_error",
        metrics = ['accuracy']
    )

    return model

#Train model
def train_model(model, train_dataset):
    #Split dataset into features and lables
    train_features = {name:np.array(value) for name, value in train_dataset.items()}
    train_labels = np.array(train_features.pop('Class'))

    ann_model_fit = ann_model.fit(
        x=train_features,
        y=train_labels,
        batch_size = None,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True)  

    return ann_model_fit


#Initialize console object
console = Console()

#Read file into dataframe
column_names = ['IndustrailRisk','ManagementRisk','FinancialFlexibility','Credibility','Competitiveness','OperatingRisk','Class']
df = pd.read_csv("Qualitative_Bankruptcy.data", names=column_names)

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
train_labels = train_features.pop('Class')
test_features = test_dataset.copy()
test_lables = test_features.pop('Class')

print("Print train")
print(train_features)
print(train_labels)
print("Print test")
print(test_features)
print(test_lables)

#Create model
ann_model = create_model(train_features)

#Train model
ann_model_fit = train_model(ann_model, train_dataset)