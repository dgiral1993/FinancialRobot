import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

data = pd.read_csv('tab.csv')
data.info()
data = data.drop('Date',1)
data = data.drop('Adj Close',1)

#normalize data in [0,1] range
min_max_scaler = preprocessing.MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data)
rev_data = normalized_data[::-1]
np.savetxt('stock_norm.csv',rev_data,delimiter=",")
stock_data = np.genfromtxt('stock_norm.csv',delimiter= ',')
new_data = []
for i in range(len(stock_data) -1):
    d = []
    for elt in stock_data[i]:
        d.append(elt)
    d.append(stock_data[i+1][0])
    d = np.array(d)
    new_data.append(d)

DS = SupervisedDataSet(5 ,1)
my_data = np.array(new_data)
train_data = my_data[: int(.6*len(my_data))]
test_data = my_data[ int(.6*len(my_data)): len(my_data)]

for elt in train_data:
   DS.addSample(elt[:5],elt[5])

hidden_layers = [4,8,12]
for layer in hidden_layers:
  FNN = buildNetwork(DS.indim, layer, DS.outdim, bias=True)

  TRAINER = BackpropTrainer(FNN, dataset=DS, learningrate = 0.001, \
     momentum=0.1, verbose=True)
  print ("No of nodes in hidden layers:" + str(layer))
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
     predicted.append(FNN.activate(elt[:5]))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Predicted vs Actual')
  ax.set_xlabel('days')
  ax.set_ylabel('value')
  plt.plot([i for i in range(0,len(test_data))] , test_data[:,5])
  plt.plot([i for i in range(0,len(test_data))] , predicted , 'r')
  plt.show()

