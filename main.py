import time
import_start = time.time()

import os   #Turn off annoying tensorflow cuda error message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Turn off annoying tensorflow cuda error message

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import_end = time.time()
print("Importing modules took:", round(import_end - import_start,2),"seconds.")

RANDOM_STATE = 66 #Sets seed for splitting train and test set
FRAC = 0.85 #Set fraction of train dataset
EPOCHS = 1000 #Number of epochs to train model

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
    model.add(tf.keras.layers.Dense(
        units=1,
        input_shape=(6,),
        activation='sigmoid'))

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

    #Train model
    ann_model_fit = ann_model.fit(
        x=train_features,
        y=train_labels,
        batch_size = None,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True)  

    return ann_model_fit

#Evalute model
def evalute_model(model, test_df):
    test_features = {name:np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop('Class')) # isolate the label

    result = model.evaluate(x=test_features, y=test_lables, batch_size=128)
    print(result)

#Plot accuracy curve from training
def plot_acc_curve(epochs, accuracy):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")

  plt.plot(epochs, accuracy, label="Gain in accuracy")
  plt.legend()
  plt.ylim([accuracy.min()*0.95, accuracy.max() * 1.03])
  plt.show()

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

#Create model
ann_model = create_model(train_features)

#Train model
train_start = time.time()
ann_model_fit = train_model(ann_model, train_dataset)
train_end = time.time()
print("Training model took:", round(train_end - train_start,2),"seconds.")

#Evalute model
evalute_model(ann_model, test_dataset)

#Print accuracy curve
history_dataframe = pd.DataFrame(ann_model_fit.history)
history_epochs = ann_model_fit.epoch
history_accuracy = history_dataframe['accuracy']
plot_acc_curve(history_epochs, history_accuracy)