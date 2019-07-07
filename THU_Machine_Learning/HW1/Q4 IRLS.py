# -*- coding: utf-8 -*-
"""
Machine Learning HW1, Question 4
Author: Zhang Naifu
ID: 2018280351
"""

import math
import numpy as np
from numpy import vstack
from numpy.linalg import pinv, norm
from sklearn import datasets
import matplotlib.pyplot as plt

numFeatures = 123

def loadData(file):
    '''
    X: K x N
    y: N x 1
    '''
    data = datasets.load_svmlight_file(file) 
    X, y = data[0], data[1]
    
    X = X.toarray()
    X = X.T # change Xi to column vectors
    zeroFeatures = np.zeros((numFeatures - X.shape[0], X.shape[1])) 
    X = vstack((X, zeroFeatures)) # ensure Xi has 123 features
    
    y = y.T # change y to column vector
    y[y == -1] = 0 # replace yi=-1 with yi=0
    
    return X, y
    
    
def sigmoid(X, w):
    '''
    takes in matrix X & column vector w, returns column vector mu
    X: K x N
    w: K x 1
    mu: N x 1
    '''    
    mu_intermediate = np.dot(w.T, X)
    mu = 1/(1+np.exp(-mu_intermediate))
    return mu

def calcR(mu):
    '''
    takes in row vector mu, returns diagonal matrix R
    mu: N x 1
    R: N x N
    '''    
    R_intermediate = np.asanyarray(mu * (1-mu))
    R = np.diag(R_intermediate) # convert R_intermediate vector to diagonal matrix
    return R
    
def irls(X, y, tolerance, regularization = False, lam = 0,  maxiter = 9):
    '''
    main IRLS function, takes in matrix X, vector y, scalar tolerance, boolean regularization, scalar lam, scalar maxiter
    X: K x N
    y: N x 1
    '''   
    K, N = X.shape # K = number of features = 123, N = sample size   
    w_init = np.squeeze(np.zeros((K,1)))
    normList = []
    differenceList = []
    gradientList = []
    max_gradient = 1000
    iteration = 0
    
    if not regularization: # unregularized loss function
        lam = 0
    else: # regularized loss function
        pass
    
    while iteration <= maxiter and max_gradient > tolerance:
        iteration += 1
        # IRLS implementation
        mu = sigmoid(X, w_init)
        mu = np.squeeze(np.asarray(mu))
        R = calcR(mu)
        
        XR = np.matmul(X, R)
        XRX = np.matmul(XR, X.T)
        XRXw = np.matmul(XRX, w_init)
        XRXw = np.squeeze(np.asarray(XRXw))
        
        lamDiagonal = lam * np.eye(K) # convert lambda into diagonal matrix
        lamXRX = lamDiagonal + XRX
        lamXRX_inv = pinv(lamXRX)

        ymu = y-mu
        Xymu = np.matmul(X, ymu)
        gradient = Xymu - np.matmul(lamDiagonal, w_init)
        max_gradient = max(np.abs(gradient)) # check 1st order derivative = 0
        gradientList.append(max_gradient)
        
        # update
        w_updated = np.matmul(lamXRX_inv, (XRXw + Xymu))
        difference = np.sum(np.abs(w_updated - w_init))
        w_init = w_updated
        normList.append(norm(w_updated, 2))
        differenceList.append(difference)
        print(iteration, difference)

    return w_updated, iteration, differenceList, gradientList, normList
    

def predict(X, y, w):   
    '''
    calculate prediction accuracy, given the X, y and w inputs
    X: K x N
    y: N x 1
    w: K x 1
    '''   
    # get prediciton on test data
    prob = sigmoid(X, w)
    predict = [int(round(x)) for x in prob]
    # compute accuracy
    accuracy = np.sum(predict == y)/y.shape[0]
    return accuracy
    

def plot(X, y1, y2, X_lable, y1_lable, y2_lable):
    '''
    plot chart of difference and gradient against each iteration, this shows whether convergence
    '''   
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(X_lable)
    ax1.set_ylabel(y1_lable, color = 'b')
    ax1.plot(X, y1, color = 'b')
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_lable, color = 'g')
    ax2.plot(X, y2, color = 'g')
    fig.tight_layout()
    plt.grid(True)
    plt.show()


'''
load and process data
'''
# load training data
trainingSet = loadData('a9a')
train_X, train_y = trainingSet[0], trainingSet[1]

# load test data
testSet = loadData('a9a.t')
test_X, test_y = testSet[0], testSet[1]


'''
unregularized version
'''
# tolerance set to 0.1. This is where the weights converge
w_final, iteration, differenceList, gradientList, normList = irls(train_X, train_y, 0.1)
print(differenceList, gradientList, normList)
# training accuracy
print(predict(train_X, train_y, w_final))
# test accuracy
print(predict(test_X, test_y, w_final))

# plot Fig 1
iterationList = [i for i in range(iteration)]
plot(iterationList, differenceList, gradientList, 'Iterations', 'Wt+1 - Wt', 'gradient')


'''
regularized version
'''
# set aside 34% of training data as cross validation data
xvalidation = int(32561 * 0.66)
regularized_train_X = train_X[..., : xvalidation]
xvalid_X = train_X[..., xvalidation:]
regularized_train_y = train_y[:xvalidation]
xvalid_y = train_y[xvalidation:]

# choose lambda value
lamList = [0.001, 0.01, 0.1, 1, 10]
train_accuracyList = []
xvalid_accuracyList = []
# run regularized IRLS on each different lambda value
for lam in lamList:
    w_final, _, _, _ = irls(regularized_train_X, regularized_train_y, 0.1, True, lam)
    train_acc = predict(xvalid_X, xvalid_y, w_final)
    xvalid_acc = predict(xvalid_X, xvalid_y, w_final)
    train_accuracyList.append(train_acc)
    xvalid_accuracyList.append(xvalid_acc)
print(train_accuracyList, xvalid_accuracyList)


# lambda set to 0.1, for the best test accuracy
w_final, iteration, differenceList, gradientList, normList = irls(train_X, train_y, 0.1, True, 0.1)
print(differenceList, gradientList, normList)
# training accuracy
print(predict(train_X, train_y, w_final))
# test accuracy
print(predict(test_X, test_y, w_final))

# plot Fig 2
iterationList = [i for i in range(iteration)]
plot(iterationList, differenceList, gradientList, 'Iterations', 'Wt+1 - Wt', 'gradient')
