'''
Machine Learning Break Down:
1. Get data into a numerical representation
2. Build a model to learn pattern in that numerical representation

Tensors Represent Inputs 
Neural Networks Represent those patterns/features/weights
'''

import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib_terminal

start = 0
end = 1
step = .02
bias =  0.3
weight = 0.7
X = torch.arange(start,end, step).unsqueeze(dim =1)

y= weight * X + bias


train_split = int(.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]



def plot_predictions(train_data = X, train_label = y_train, test_data = X_test, test_label = y_test, predictions = None ):

    #plotting training data
    print(plt.scatter(train_data, train_label, c = "b", s =4, label = "Training Data" ))

    #plotting testing data
    print(plt.scatter(test_data, test_label, s =4, label = "Testing Data" ))

    if predictions is not None:
        print(plt.scatter(test_data, predictions, c= "r", s=4, label = "Predictions"))


    #show the legend
    print(plt.legend(prop={"size": 14}))

    print(plt.show())