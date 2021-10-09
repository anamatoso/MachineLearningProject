#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:06:50 2021

@author: inessarana
"""

# MACHINE LEARNING PROJECT
# PART 1

import numpy as np
import matplotlib.pyplot as plt

x_train1 = np.load('/Users/inessarana/Documents/Faculdade/5º ano /1º Semestre/ML/Project/Xtrain_Regression_Part1.npy')
x_train2 = np.load('/Users/inessarana/Documents/Faculdade/5º ano /1º Semestre/ML/Project/Xtrain_Regression_Part2.npy')
y_train1 = np.load('/Users/inessarana/Documents/Faculdade/5º ano /1º Semestre/ML/Project/Ytrain_Regression_Part1.npy')
y_train2 = np.load('/Users/inessarana/Documents/Faculdade/5º ano /1º Semestre/ML/Project/Ytrain_Regression_Part2.npy')
x_test1 = np.load('/Users/inessarana/Documents/Faculdade/5º ano /1º Semestre/ML/Project/Xtest_Regression_Part1.npy')
x_test2 = np.load('/Users/inessarana/Documents/Faculdade/5º ano /1º Semestre/ML/Project/Xtest_Regression_Part2.npy')

# PREDICTOR 1: LINEAR REGRESSION

# calculate parameters using normal equations (in lr now)
def lr_par(xt, yt):
    # given train sets xt and outcomes yt, determine beta parameters for predictor
    # design matrix
    c = len(xt)
    X = np.append(np.ones((c,1)),xt,axis=1)
    Xtrans = np.transpose(X)
    
    # use normal equation to determine beta parameters
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xtrans, X)),Xtrans),yt)
    return beta
    
def lr(beta,xt):
    # using the test set xt and the determined beta parameters, predict y
    c = len(xt)
    
    # use beta parameters to determine y using x testing set    
    X = np.append(np.ones((c,1)),xt,axis=1)
    return np.matmul(X,beta)

def lrpredictor(xt,yt,x_test): # predicts y based on training with xt and yt
    y_test=lr(lr_par(xt,yt),x_test)
    return y_test

def sse(y,yt):
    # calculate the squared erros using the training set yt when compared to a predicted set in y
    # yt: training set
    # y: test/ predicted set
    return np.array((yt-y)**2).sum()

# CROSS VALIDATION
def cross_val(xt,yt,k):
    # train the data set using k data sets obtained by dividing the training 
    # set into k sets each with a section excluded to use as a test set. This 
    # is used to evaluate the performance of the model usingthe available data set.
    # xt: training set
    # yt: test set
    # k: number of folds
    
    c = len(xt) #length of training set
    
    if (c%k)!=0:
        print("Cannot compute. Choose a divider of "+str(c))
        return
    elif k==1:
        print("Cannot perform 1-fold classification since there is no test set.")
        return
    else:
        fold = c//k
        f=len(xt[0])
        
        # create training sets with missing test element    
        x_train = np.empty((k,c-fold,f)) #each element of list is a training set with 1 section excluded
        y_train = np.empty((k,c-fold)) 
        for i in range(k):
            x_train[i,:,:] = [item for item in xt if np.where(xt == item)[0][0] not in range(i*fold,i*fold+fold)]
            y_train[i,:] = [item for item in yt if np.where(yt == item)[0][0] not in range(i*fold,i*fold+fold)]

    # using the predictor, generate the outcomes using the k different sets determined agove
    y_test = np.empty((k,c-fold))
    for i in range(k):
        y_test[i,:] = lrpredictor(xt,yt,x_train[i,:,:]) #outcomes predicted using linear regression model 

    # compute errors for each set
    errors = np.empty(k)
    for i in range(k):
        errors[i] = sse(y_train[i,:],y_test[i,:])
        
    print("The mean SSE for "+str(k)+"-folds is "+str(np.mean(errors)))
    return errors       
    
cross_val(x_train1,y_train1,5)   
        
        
        
        
        
        
        
        
        
