# Author : Daniel Giraldo Villa
# Copyright 2017
# based on recurrent network workshop made by  (Here the name of udemy's professors)
# Available on: https://www.superdatascience.com/deep-learning/ 

# Predictor using Udemy course recurrent neural network 
# Importing packages (Se importan librerías necesarias para el proyecto)
import pandas as pd
import datetime 
import pandas_datareader.data as web
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 

# Variables and methods for real time financial data

startTrainigSet =datetime.datetime(2010,1,1) #Start date for training data
endTrainingSet = datetime.datetime(2017,1,31) #end date for training data format (year,month,day)

startTestSet = datetime.datetime(2017,2,1) #Start date for test data
endTestSet = datetime.datetime.now() # End date for test day (current day)

#Get data from yahoo finance api, using pandas_DataReader library
training_set = web.DataReader("^DJI", "yahoo", startTrainigSet, endTrainingSet).iloc[:,3:5].values
test_set = web.DataReader("^DJI","yahoo",startTestSet,endTestSet).iloc[:,3:5].values

####################################################################################

# Set variables for data normalization
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set) # Normalize training data
test_set = sc.fit_transform(test_set) # Normalize test data

####################################################################################

X_train = training_set[0:len(training_set)-1] # Training data for network
Y_train = training_set[1:len(training_set),0] # Desired output for above data

X_train = np.reshape(X_train,(len(training_set)-1,1,2)) # Reshape data for network performance

####################################################################################

# Setting neural network parameters

regressor = Sequential() # Set variable as a neural network 

regressor.add(LSTM(units = 4, activation = 'sigmoid',input_shape = (None,2))) # Recurrent layer

regressor.add(Dense(units = 1)) # Output Layer (Regression)

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Set optimizer and loss function

regressor.fit(X_train,Y_train, batch_size = 12, epochs = 10) # Network is trained, change epochs !!

####################################################################################

# Testing trained neural network

inputs = np.reshape(test_set,(len(test_set),1,2)) # Reshape test data
predictions = regressor.predict(inputs) # Predict  output for given data
predictions= np.concatenate((predictions, test_set[:,1:2]), 1) # Trick made in order to denormalize prediction output
predicted_index_price = sc.inverse_transform(predictions) # Denormalization of prediction output

real_index_price = sc.inverse_transform(test_set)[:,0:1] # get real data of test output

####################################################################################

# Plotting results 

plt.plot(real_index_price, color = 'red', label= 'Real Index price')
# Note: predictor output is shift to right, in order to get a better view. 
plt.plot(range(1,1+len(predicted_index_price[:,0:1])), predicted_index_price[:,0:1], color = 'blue', label= 'Predicted Index price')
plt.title('Dow Jones 30 Price Prediction')
plt.xlabel('Time')
plt.ylabel('Dow JOnes 30 Price')
plt.legend()
plt.show()

# TODO check other performance by changing number of epochs and LSTM units, and compare results
