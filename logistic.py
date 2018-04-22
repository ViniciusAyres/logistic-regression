from __future__ import division
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fun_linear(theta, x):
    return theta.dot(x)

def sigmoid(z):
    return 1 / (1 + np.e**-z)

def predict(theta, x):
    return sigmoid(fun_linear(theta, x))

def cost_function(theta, x, y):
    n_rows, n_columns = x.shape
    cost = 0.0

    for row in range(n_rows):
        h = sigmoid(fun_linear(theta, x[row, :]))
        for column in range(n_columns):
            cost += (-y[row]*np.log(h) - (1-y[row])*np.log(1-h))/n_rows 
    
    return cost

def cost_function_derivative(theta, x, y):
    n_rows, n_columns = x.shape
    errors = np.zeros(n_columns)

    for row in range(n_rows):
        h = sigmoid(fun_linear(theta, x[row, :]))
        for column in range(n_columns):
            errors[column] += (h-y[row])*x[row, column]/n_rows

    return errors

def gradient_descendent(theta, x, y, alpha):
    n_rows, n_columns = x.shape

    cost_d = cost_function_derivative(theta, x, y)
    return np.array(theta - alpha*cost_d)


def logistic(x, y, iter = 100, alpha=0.0001):
    n_rows, n_columns = x.shape
    cost = [float("inf")]
    theta = np.ones(n_columns)

    for i in range(iter):
        cost.append(cost_function(theta, x, y))
        print("cost: ", cost[i])

        theta = gradient_descendent(theta, x, y, alpha)
        print("theta: ", theta)

    return theta


#Loading the dataset with pandas
dataset = pd.read_csv("gitdata.csv")

#Adiciona uma coluna de 1's ao Dataframe
n_rows, n_columns = dataset.shape
dataset.insert(loc=0, column="COO", value=np.ones(n_rows)) #coo : Column Of Ones

X = dataset.loc[:, dataset.columns != 'y'].values
Y = dataset['y'].values

###Preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-5, 5))
X = min_max_scaler.fit_transform(X)

####TRAINING

n = 80
predictor = logistic(X[:n, :], Y[:n], iter=1000, alpha=0.1)
print(predictor)
print('PREDICAO:')
for i in range(100):
    print(predict(predictor, X[i, :]))
