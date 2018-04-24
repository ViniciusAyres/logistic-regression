from __future__ import division
from sklearn import preprocessing
import numpy as np
import pandas as pd

def fun_linear(theta, x):
    return theta.dot(x)

def sigmoid(z):
    return 1 / (1 + np.e**-z)

def hypothesis(theta, x):
    return sigmoid(fun_linear(theta, x))

def predict(theta, x):
    prediction = hypothesis(theta, x)

    if prediction > 0.5:
        return (1, prediction * 100)
    else:
        return (0, (1 - prediction)*100)

def cost_function(theta, x, y):
    n_rows, n_columns = x.shape
    cost = 0.0
    
    for row in range(n_rows):
        h = hypothesis(theta, x[row, :])
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


def logistic(x, y, max_iter = 100,
                   max_error = 0.0001,
                   alpha=0.0001):

    n_rows, n_columns = x.shape
    cost = [float("inf")]
    theta = np.ones(n_columns)
    n_iter = 0
    exit = False
    
    while not exit:
        cost.append(cost_function(theta, x, y))
        print("cost: ", cost[n_iter])

        theta = gradient_descendent(theta, x, y, alpha)
        print("theta: ", theta)
        
        #Condicao de saida
        if abs(cost[-2] - cost[-1]) <= max_error or n_iter > max_iter:
            exit = True

        n_iter += 1

    return theta
        
#Loading the dataset with pandas
dataset = pd.read_csv("grades.csv")

#Adding a column of ones(the biased values) to the Dataframe. 
n_rows, n_columns = dataset.shape
dataset.insert(loc=0, column="COO", value=np.ones(n_rows)) #coo : Column Of Ones

X = dataset.loc[:, dataset.columns != 'y'].values
Y = dataset['y'].values

#Preprocessing
#Data need to be normalized to avoid errors
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-5, 5))
X = min_max_scaler.fit_transform(X)


#Training example
train_set = int(n_rows * 0.8)
predictor = logistic(X[:train_set, :], Y[:train_set], max_iter=10000, max_error=10**-8, alpha=0.01)

#Predictions example
test_range = n_rows - train_set
print('Predictions:')
for i in range(train_set, train_set + test_range):
    print(predict(predictor, X[i, :]), Y[i])
