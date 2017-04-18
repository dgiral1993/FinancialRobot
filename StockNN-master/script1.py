import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

stock_data = pd.read_csv('tab.csv',delimiter= ',')
close_prices = stock_data[['Close']].values[::-1]

dataset,normalised_data = [],[]
DS = SupervisedDataSet(50 ,1)
for i in range(0,len(close_prices) - 50):
  dataset.append(close_prices[i:i+51])

#normalize in a window to reflect percentage change from the start
for window in dataset:
  normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
  normalised_data.append(normalised_window)

my_data = np.array(normalised_data)
train_data = my_data[:round( .6*len(my_data))]
test_data = my_data[ round(.6*len(my_data)): len(my_data)]

for elt in train_data:
   DS.addSample(elt[:50],elt[50])

hidden_layers = [4,8,12]

for layer in hidden_layers:
  FNN = buildNetwork(DS.indim, layer, DS.outdim, bias=True)

  TRAINER = BackpropTrainer(FNN, dataset=DS, learningrate = 0.01, \
     momentum=0.1, verbose=True)
  print "No of nodes in hidden layers:" + str(layer)
  trnerr,valerr = TRAINER.trainUntilConvergence(maxEpochs = 10)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(trnerr,'b',valerr,'r')
  ax.set_title('Error curve')
  ax.set_xlabel('iterations')
  ax.set_ylabel('error')
  plt.show()
  predicted = []
  for elt in test_data:
     predicted.append(FNN.activate(elt[:50]))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Predicted vs Actual')
  ax.set_xlabel('days')
  ax.set_ylabel('value')
  plt.plot([i for i in range(0,len(test_data))] , test_data[:,50])
  plt.plot([i for i in range(0,len(test_data))] , predicted , 'r')
  plt.show()


