# -*- coding: utf-8 -*-

# NeuralNetwork - V 1.0
# very simple neural network for the Titanic dataset using Keras
# all the datasets are from https://www.kaggle.com/c/titanic/data

import numpy as np
import pandas as pd

# start preprocessing
dataset = pd.read_csv('train.csv')

# deal with nan in Age column
dataset = dataset.fillna(-1)

# feature selection
X = dataset.iloc[:, [2,4,5]].values
Y = dataset.iloc[:, [1]].values

# encoding categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# encode sex
labelencoder_X_sex = LabelEncoder()
X[:, 1] = labelencoder_X_sex.fit_transform(X[:, 1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# import deep learning libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# initialize classifier
classifier = Sequential()

# first hidden layer with 8 nodes , activation function - relu
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3))

# second hidden layer with 8 nodes , activation function - relu
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

# third hidden layer with 16 nodes , activation function - relu
#classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

# output layer with activation function - sigmoid
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# only for the Yhat > 0.5 probability 
Yhat = classifier.predict(X_test)
Yhat = (Yhat > 0.5)

# build confusionMatrix 
confusionMatrix = confusion_matrix(Y_test, Yhat)