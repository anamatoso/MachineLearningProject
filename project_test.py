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

# calculate parameters using normal equations
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
    # given train set xt and beta parameters, predict y
    c = len(xt)
    X = np.append(np.ones((c,1)),xt,axis=1)
    return np.matmul(X,beta)

y_test1 = lr(lr_par(x_train1,y_train1),x_test1) #outcomes predicted using linear regression model 

# CROSS VALIDATION

#xtrain = [item for item in Xpart if i!=Xpart.index(item)]
def cross_val(xt,k):
    # xt: training set
    # k: number of folds
    k = 5
    c = 100
    fold = 20
    
    i=1 - 0-19 n:n+fold, n=0
    i=2 - 20-39 n:n+fold n=20
    i=3 - 40-59 n:n+fold n=40
    i=4 - 60-79 n:n+fold n=60
    i=5 - 80-99 n:n+fold n=80
    
    
    k = 5
    xt = x_train1
    fold = 20
    x_train = [] #cada elemento desta lista tem o set de treino com um excluido
    #fold_idx = range(0,c,fold) #vector with start indexes
    for i in range(k):
        x_train = x_train + [item for item in xt if np.where(xt == item)[0][0] not in range(i*fold,i*fold+fold-1)] # items do xt cujos idicis são diferentes de n:n+fold






