# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:27:39 2020

@author: Sunny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
dataset1=load_breast_cancer()
dataset1.keys()
dataset=pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
dataset.head()
dataset.shape

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(dataset)
scaled_data=sc.transform(dataset)
scaled_data

from sklearn.decomposition import PCA
pca=PCA(n_components=2) 

pca.fit(scaled_data)
fre=pca.transform(scaled_data)
fre

plt.figure(figsize=(8,6))
plt.scatter(fre[:,0],fre[:,1],c=dataset1['target'],cmap='plasma')

'''now we can apply any machine learning algorithm 
this is a part of feature eng in this techniqe we reduce the dimensions so it is a called a 
dim reduction techniqe.'''
------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Google_Stock_Price_Train.csv")

dataset

training_set = dataset.iloc[:, 1:2]


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled


X_train = [] #60 prev stock prices before the financial day
				# this is the ip to the RNN
y_train = []  # will contain the stock price the next fin day
	
for i in range(60, 1258): # upper bound is last row, lower bound is i-60
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)



print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)


# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)


# Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price= dataset_test.iloc[:, 1:2].values


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



















