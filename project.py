#%% Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor

import warnings

warnings.filterwarnings('ignore')

#%% LOAD TEST AND TRAINING DATA
cd = os.getcwd()

# Part 1
x_train_1 = np.load(cd+'/Data/Xtrain_Regression_Part1.npy')
y_train_1 = np.load(cd+'/Data/Ytrain_Regression_Part1.npy')
x_test_1 = np.load(cd+'/Data/Xtest_Regression_Part1.npy')

del cd

#%%Plot each feature vs outcome
# for i in range(20):
#     x=x_train_1[:,i]
#     plt.figure()
#     plt.scatter(x,y_train_1)
#     plt.title("feature " +str(i)+" vs y")

#%% Plot each feature vs feature-check dependencies

# for i in range(20):
#     for j in range(i):
#         if i!=j:
#             x1=x_train_1[:,i]
#             x2=x_train_1[:,j]
#             plt.figure()
#             plt.scatter(x1,x2)
#             plt.title("feature " +str(i)+" vs feature " +str(j))
            
# The features seem independent
         

#%% PREDICTORS EVALUATION

# PREDICTOR 1: LINEAR REGRESSION
def lr_par(xt, yt):
    # given train sets xt and outcomes yt, determine beta parameters for predictor
    X = np.append(np.ones((len(xt),1)),xt,axis=1) # design matrix
    Xtrans = np.transpose(X)
    
    # use normal equation to determine beta parameters
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(Xtrans, X)),Xtrans),yt)
    
def lr(beta,xt):
    # using the test set xt and the determined beta parameters, predict y
    X = np.append(np.ones((len(xt),1)),xt,axis=1)
    return np.matmul(X,beta)

def lrpredictor(xt,yt,x_test): # predicts y based on training with xt and yt
    return lr(lr_par(xt,yt),x_test)

# PREDICTOR 2: RIDGE REGRESSION
def ridge_par(xt,yt,l):
    # determine beta parameters for ridge regression using training sets xt and yt with lambda l
    # lambda corresponds to a small number, >0, that minimizes the sse
    X = np.append(np.ones((len(xt),1)),xt,axis=1) #design matrix
    Xtrans = np.transpose(X)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(Xtrans,X)+l*np.identity(len(xt[0])+1)),Xtrans),yt)
     
#using the beta parameters determined with ridge regression, the y are predicted using the lr function
def ridgepredictor(xt,yt,l,x_test):
    return lr(ridge_par(xt,yt,l),x_test)


# PREDICTOR 3: LASSO REGRESSION
def lassopredictor(xt,yt,l,xtest):
    #l=lambda
    lassoreg = linear_model.Lasso(alpha=l)
    lassoreg.fit(xt,yt)
    return lassoreg.predict(xtest) #I checked and it is the same as calculating beta and doing y=X*beta

# PREDICTOR 4: SUPPORT VECTOR MACHINES
def svmlinearpredictor(xt,yt,xtest):
    regsvm = svm.LinearSVR(epsilon=0.05,random_state=2,tol=1e-6)
    regsvm.fit(xt, yt)
    return regsvm.predict(xtest)

# PREDICTOR 5: SGD
def sgdpredictor(xt,yt,xtest):
    sgd = SGDRegressor(random_state=0,loss='epsilon_insensitive',epsilon=0.05)
    sgd.fit(xt, yt)
    return sgd.predict(xtest)

# PREDICTOR 6: GAUSSIAN PROCESSES
def gausspredictor(xt,yt,xtest):
    kernel = DotProduct()
    gauss = GaussianProcessRegressor(kernel=kernel,random_state=0)
    gauss.fit(xt, yt)
    return gauss.predict(xtest)

# PREDICTOR 7: ELASTIC NET
def enpredictor(xt,yt,xtest):
    en = ElasticNet(random_state=0)
    en.fit(xt, yt)
    return en.predict(xtest)

# PREDICTOR 8: ORTHOGONAL MATCHING PURSUIT
def omppredictor(xt,yt,xtest):
    omp = OrthogonalMatchingPursuit(normalize=False)
    omp.fit(xt, yt)
    return omp.predict(xtest)

# PREDICTOR 9: LARS
def larspredictor(xt,yt,xtest):
    lar = linear_model.Lars(n_nonzero_coefs=1, normalize=False)
    lar.fit(xt, yt)
    return lar.predict(xtest)

# PREDICTOR 10: LARS LASSO
def larslassopredictor(xt,yt,xtest):
    lars_lasso = linear_model.LassoLars(alpha=.1, normalize=False)
    lars_lasso.fit(xt, yt)
    return lars_lasso.predict(xtest)

# PREDICTOR 11: BAYES RIDGE
def bayesridgepredictor(xt,yt,xtest):
    bayesridge = linear_model.BayesianRidge()
    bayesridge.fit(xt, yt)
    return bayesridge.predict(xtest)

# PREDICTOR 12: RANSAC REGRESSION
def ransacpredictor(xt,yt,xtest):
    regsvm = svm.LinearSVR(epsilon=0.05,random_state=2)
    ransac = RANSACRegressor(base_estimator=regsvm,random_state=2)
    ransac.fit(xt, yt)
    return ransac.predict(xtest)

# SQUARED ERRORS
def sse(y,yt):
    # calculate the squared erros using the training set yt when compared to a predicted set in y
    # yt: training set
    # y: test/ predicted set
    return np.array((y-yt)**2).sum()

# CROSS VALIDATION
def cross_val(xt,yt,k,func,*args):
    # train the data set using k data sets obtained by dividing the training 
    # set into k sets each with a section excluded to use as a test set. This 
    # is used to evaluate the performance of the model usingthe available data set.
    # xt: training set
    # yt: test set
    # k: number of folds
    
    c = len(xt) #length of training set
    
    # if (c%k)!=0:
    #     print("Cannot compute. Choose a divider of "+str(c))
    #     return
    if k==1:
        print("Cannot perform 1-fold classification since there is no test set.")
        return
    else:
        fold = c//k
        f=len(xt[0])

        # create training sets with missing test element    
        x_train = np.empty((k,c-fold,f)) #each element of list is a training set with 1 section excluded
        y_train = np.empty((k,c-fold)) 
        x_test = np.empty((k,fold,f)) # testing set with excluded section
        y_test = np.empty((k,fold)) 
        for i in range(k):
            x_train[i,:,:] = [item for item in xt if np.where(xt == item)[0][0] not in range(i*fold,i*fold+fold)]
            y_train[i,:] = [item for item in yt if np.where(yt == item)[0][0] not in range(i*fold,i*fold+fold)]
            x_test[i,:,:] = [item for item in xt if np.where(xt == item)[0][0] in range(i*fold,i*fold+fold)]
            y_test[i,:] = [item for item in yt if np.where(yt == item)[0][0] in range(i*fold,i*fold+fold)]

    if func == 'lr':
        # using the predictor, generate the outcomes using the k different sets determined agove
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = lrpredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'ridge':
        y_pred = np.empty((k,fold))
        l = args
        for i in range(k):
            y_pred[i,:] = ridgepredictor(x_train[i,:,:],y_train[i,:],l,x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'lasso':
        y_pred = np.empty((k,fold))
        l = args
        for i in range(k):
            y_pred[i,:] = lassopredictor(x_train[i,:,:],y_train[i,:],l,x_test[i,:,:]) #outcomes predicted using linear regression model

    elif func == 'svmlinear':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = svmlinearpredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model

    elif func == 'sgd':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = sgdpredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'gauss':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = gausspredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'en':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = enpredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'omp':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = omppredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'lars':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = larspredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'larslasso':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = larslassopredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'bayesridge':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = bayesridgepredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
   
    elif func == 'ransac':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = ransacpredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    # compute errors for each set
    errors = np.empty(k)
    for i in range(k):
        errors[i] = sse(y_test[i,:],y_pred[i,:])/fold
    
    # print('The mean SSE for '+str(k)+'-folds using predictor '+func+' is '+str(np.mean(errors)))
    return np.mean(errors)       

def bestlasso(lassovector,xt,yt):
    bestl=0 
    minerror=1000000
    for l in lassovector:
        error=cross_val(xt,yt,5,'lasso',l)
        if minerror>error:
            minerror=error
            bestl=l
    return bestl

#%% CHOOSE LAMBDA FOR LASSO AND RIDGE (DON'T RUN)
# Compare cross validation errors between different lambda values

l=np.logspace(-6, -3, 10000)
cv_lr_k5 = cross_val(x_train_1,y_train_1,5,'lr')   
cv_ridge_k5=[]
cv_lasso_k5=[]
for i in range(len(l)):
    cv_ridge_k5 = cv_ridge_k5 + [cross_val(x_train_1,y_train_1,5,'ridge',l[i])]  
    cv_lasso_k5 = cv_lasso_k5 + [cross_val(x_train_1,y_train_1,5,'lasso',l[i])]
    # print('\n')
np.save('Data/cv_ridge_k5_10000.npy',cv_ridge_k5)
np.save('Data/cv_lasso_k5_10000.npy',cv_lasso_k5)

#%%

l=np.logspace(-6, -3, 10000)
cv_lr_k5 = cross_val(x_train_1,y_train_1,5,'lr')   
cv_ridge_k5=np.load('Data/cv_ridge_k5_10000.npy')
cv_lasso_k5=np.load('Data/cv_lasso_k5_10000.npy')
l_lasso=l[np.where(cv_lasso_k5==np.min(cv_lasso_k5))[0][0]]
l_ridge=l[np.where(cv_ridge_k5==np.min(cv_ridge_k5))[0][0]]

plt.xscale('log')
plt.scatter(l_lasso,np.min(cv_lasso_k5),marker='x',color='k',zorder=3)
plt.scatter(l_ridge,np.min(cv_ridge_k5),marker='x',color='k',zorder=3)
plt.plot(l,cv_ridge_k5,label='Ridge')
plt.plot(l,cv_lasso_k5,label='Lasso')
plt.axhline(y=cv_lr_k5, color='darkgray', linestyle='--') #5-fold cross val using linear regression
plt.title('Evolution of \u03B2 values in Ridge and Lasso Regression')
plt.xlabel('\u03BB')
plt.ylabel('Error')
plt.xlim((1e-6, 1e-3))
plt.legend(loc='best')
plt.savefig('comparelambdaserror.eps', format="eps")

del l

#%% TEST FUNCTION
# Compute erros with chosen lambda values

cv_lr_k5 = cross_val(x_train_1,y_train_1,5,'lr')    
cv_lr_k10 = cross_val(x_train_1,y_train_1,10,'lr')    
cv_ridge_k5 = cross_val(x_train_1,y_train_1,5,'ridge',l_ridge)  
cv_ridge_k10 = cross_val(x_train_1,y_train_1,10,'ridge',l_ridge)    
cv_lasso_k5 = cross_val(x_train_1,y_train_1,5,'lasso',l_lasso)  
cv_lasso_k10 = cross_val(x_train_1,y_train_1,10,'lasso',l_lasso)  
cv_sgd_k5 = cross_val(x_train_1,y_train_1,5,'sgd')    
cv_sgd_k10 = cross_val(x_train_1,y_train_1,10,'sgd')  
cv_svmlin_k5 = cross_val(x_train_1,y_train_1,5,'svmlinear')    
cv_svmlin_k10 = cross_val(x_train_1,y_train_1,10,'svmlinear')  
cv_gauss_k5 = cross_val(x_train_1,y_train_1,5,'gauss')    
cv_gauss_k10 = cross_val(x_train_1,y_train_1,10,'gauss')  

k5 = [cv_lr_k5,cv_ridge_k5,cv_lasso_k5,cv_svmlin_k5,cv_sgd_k5,cv_gauss_k5]
k10 = [cv_lr_k10,cv_ridge_k10,cv_lasso_k10,cv_svmlin_k10,cv_sgd_k10,cv_gauss_k10]

del cv_lr_k5,cv_lr_k10,cv_ridge_k5,cv_ridge_k10,cv_lasso_k5,cv_lasso_k10,cv_svmlin_k10,cv_svmlin_k5,cv_sgd_k5,cv_sgd_k10,cv_gauss_k10,cv_gauss_k5

#%% PLOT BAR CHART 
# Compara erros between the different predictors

ind = np.arange(len(k5))
width = 0.35
plt.bar(ind, k5, width, label='5-fold')
plt.bar(ind + width, k10, width,label='10-fold')
plt.ylabel('Mean squared error')
plt.title('MSE')
plt.grid(axis='y',linestyle='--', linewidth=0.5)
plt.xticks(ind + width / 2, ('LR', 'Ridge', 'Lasso','SVMLinear','SGD','Gauss'))
plt.ylim((0.015,0.021))
plt.yticks(np.linspace(0.015, 0.021,7))
plt.legend(loc='best')
plt.savefig('comparepredictorserror.eps', format="eps")

del width, ind, k5, k10

#%% SAVE PREDICTION
y_pred = lassopredictor(x_train_1,y_train_1,l_lasso,x_test_1)
np.save('Data/YTest_Regression_Part1.npy',y_pred)

#%% COMPARE BETAS
# for the different lambda values, study the corresponding beta parameters

lambdas=np.logspace(-6, 3, 10000)
lambdasridge=np.logspace(-6, 6, 10000)

beta_lr=np.empty((21,len(lambdas)))
beta_ridge=np.empty((21,len(lambdas)))
beta_lasso=np.empty((21,len(lambdas)))
plt.figure()
for i in range(len(lambdas)):
    
    beta_ridge[:,i] = np.reshape(ridge_par(x_train_1,y_train_1,lambdasridge[i]),21)
    lassoreg = linear_model.Lasso(alpha=lambdas[i]);lassoreg.fit(x_train_1,y_train_1)
    beta_lasso[:,i] = np.hstack((lassoreg.intercept_, lassoreg.coef_))


for i in range(21):
    plt.xscale('log')
    plt.plot(lambdasridge,beta_ridge[i,:],label='beta_ridge')
    plt.title('Evolution of \u03B2 values in Ridge Regression')
    plt.xlabel('\u03BB')
    plt.ylabel('\u03B2 values')
plt.grid(linestyle='--', linewidth=0.5)

plt.savefig('lambdaridge.eps', format="eps")
plt.figure()
for i in range(21):
    plt.xscale('log')
    plt.plot(lambdas,beta_lasso[i,:],label='beta_lasso')
    plt.title('Evolution of \u03B2 values in Lasso Regression')
    plt.xlabel('\u03BB')
    plt.ylabel('\u03B2 values')
plt.grid(linestyle='--', linewidth=0.5)


plt.savefig('lambdalasso.eps', format="eps")

del i,lambdas,lassoreg,beta_ridge,beta_lr, beta_lasso



#%% PART 2
#%% OUTLIER DETECTION AND REMOVAL

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import sys

cd=os.getcwd()

x_train_2 = np.load(cd+'/Data/Xtrain_Regression_Part2.npy')
y_train_2 = np.load(cd+'/Data/Ytrain_Regression_Part2.npy')
x_test_2 = np.load(cd+'/Data/Xtest_Regression_Part2.npy')

def addyt(xt,yt):
    return np.append(xt,yt,axis=1)

def deleteyt(xt):
    xt = np.delete(xt,-1,axis=1)
    return xt

#with isolation forest
def isoforest(xt,yt,cont):
    iso = IsolationForest(contamination=cont,random_state=0)
    mask = iso.fit_predict(xt)
    isin = mask != -1
    x_train_2_iso, y_train_2_iso = xt[isin, :], yt[isin]
    return x_train_2_iso, y_train_2_iso

#with Minimum Covariance Determinant
def ellienv(xt,yt,cont):
    iso = EllipticEnvelope(contamination=cont,random_state=0)
    mask = iso.fit_predict(xt)
    isin = mask != -1
    x_train_2_ee, y_train_2_ee = xt[isin, :], yt[isin]
    return x_train_2_ee, y_train_2_ee

#Local Outlier Factor
def lof(xt,yt,cont):
    iso = LocalOutlierFactor(contamination=cont)
    mask = iso.fit_predict(xt)
    isin = mask != -1
    x_train_2_lof, y_train_2_lof = xt[isin, :], yt[isin]
    return x_train_2_lof, y_train_2_lof

#One Class SVM
def ocsvm(xt,yt,cont):
    iso = OneClassSVM(nu=cont,kernel='sigmoid')
    mask = iso.fit_predict(xt)
    isin = mask != -1
    x_train_2_ocsvm, y_train_2_ocsvm = xt[isin, :], yt[isin]
    return x_train_2_ocsvm, y_train_2_ocsvm

# DBSCAN
def dbscan(xt,yt,eps):
    dbs = DBSCAN(eps=eps, min_samples=2)
    mask = dbs.fit_predict(xt)
    isin = mask != 0
    x_train_2_dbs, y_train_2_dbs = xt[isin, :], yt[isin]
    return x_train_2_dbs, y_train_2_dbs

def outlierremoval(xt,yt,k,func):
    # remove outliers
    # xt: training set
    # yt: test set
    # k: function parameter
    if func == 'iso':
        xt,yt=isoforest(xt,yt,k)
    
    elif func == 'ee':
        xt,yt=ellienv(xt,yt,k)
    
    elif func == 'lof':
        xt,yt=lof(xt,yt,k)

    elif func == 'ocsvm':
        xt,yt=ocsvm(xt,yt,k)

    elif func == 'dbscan':
        xt,yt=dbscan(xt,yt,k)
        
    return xt,yt

# outlierfunc=['iso','ee','lof','ocsvm','dbscan']
outlierfunc = ['ee','lof']
predfunc=['huber']
cont_v=np.linspace(0.0001,0.1,1000)
nu_v = np.linspace(0.01,1,1000)
eps_v = np.linspace(3,5,1001)
lassovector = np.logspace(-6, 0, 100)

len_nu=len(nu_v)
len_cont=len(cont_v)
len_eps=len(eps_v)
#%%
list_result=[]
for outlier in outlierfunc:
    print('start',outlier)
    for pred in predfunc:
        print('start',pred)
        i=0
        if outlier=='ocsvm':
            
            for nu in nu_v:
                xtrain,ytrain=outlierremoval(addyt(x_train_2,y_train_2),y_train_2,nu,outlier)
                xtrain = deleteyt(xtrain)
                if len(xtrain)>=90:
                    if not (pred=='lasso'):
                        error=cross_val(xtrain, ytrain, 5, pred)
                        list_result.append([outlier,pred,nu,error])#save results
                    else:
                        error=cross_val(xtrain, ytrain, 5, pred, bestlasso(lassovector,xtrain,ytrain))
                        list_result.append([outlier,pred,nu,error])
                progress=i/len_nu
                i+=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-100s] %d%%" % ('='*int(progress*100), progress*100))
                sys.stdout.flush()        
        
        elif outlier=='dbscan':
            for eps in eps_v:
                xtrain,ytrain=outlierremoval(addyt(x_train_2,y_train_2),y_train_2,eps,outlier)
                xtrain = deleteyt(xtrain)
                if len(xtrain)>=90:
                    if not (pred=='lasso'):
                        error=cross_val(xtrain, ytrain, 5, pred)
                        list_result.append([outlier,pred,eps,error])
                    else:
                        error=cross_val(xtrain, ytrain, 5, pred, bestlasso(lassovector,xtrain,ytrain))
                        list_result.append([outlier,pred,eps,error])
                progress=i/len_eps
                i+=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-100s] %d%%" % ('='*int(progress*100), progress*100))
                sys.stdout.flush()  

        else:  
            for cont in cont_v:
                xtrain,ytrain=outlierremoval(addyt(x_train_2,y_train_2),y_train_2,cont,outlier)
                xtrain = deleteyt(xtrain)
                if len(xtrain)>=90:
                    if not (pred=='lasso'):
                        error=cross_val(xtrain, ytrain, 5, pred)
                        list_result.append([outlier,pred,cont,error])
                    else:
                        error=cross_val(xtrain, ytrain, 5, pred, bestlasso(lassovector,xtrain,ytrain))
                        list_result.append([outlier,pred,cont,error])
                progress=i/len_nu
                i+=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-100s] %d%%" % ('='*int(progress*100), progress*100))
                sys.stdout.flush()  
        print('\n','end',pred)
        
    print('\n','end',outlier)

# list_result_lasso = list_result  
# np.save('Data/list_result_lasso.npy',list_result)
# # np.save('Data/list_result.npy',list_result)
# list_result = np.load('Data/list_result.npy')
# list_result_ransac = np.load('Data/list_result_ransac.npy')
# # list_result_lasso = np.load('Data/list_result_lasso.npy')
# np.save('Data/list_result_ransac.npy',list_result)
#%%
# outlierfunc=['iso','ee','lof','ocsvm','dbscan']
list_result = np.array(list_result)
#best lasso
# m=10
# ind=0
# for i in range(len(list_result_lasso)):
#     if list_result_lasso[i,0]!='ocsvm' and float(list_result_lasso[i,-1])<m:
#         m=float(list_result_lasso[i,-1])
#         ind=i
# print(list_result_lasso[ind])

#best overall 
m=10
ind=0
for i in range(len(list_result)):
    if float(list_result[i,-1])<m:
        m=float(list_result[i,-1])
        ind=i
print('best:',list_result[ind])
print('\n')

list_err = np.array(list_err)
m=10
ind=0
for i in range(len(list_err)):
    if float(list_err[i,-1])<m:
        m=float(list_err[i,-1])
        ind=i
print('best:',list_err[ind])
print('\n')

#best iso
# m=10
# ind=0
# for i in range(len(list_result)):
#     if list_result[i,0]=='iso':
#         if float(list_result[i,-1])<m:
#             m=float(list_result[i,-1])
#             ind=i
# print(list_result[ind])

#best ee
m=10
ind=0
for i in range(len(list_result)):
    if list_result[i,0]=='ee':
        if float(list_result[i,-1])<m:
            m=float(list_result[i,-1])
            ind=i
print(list_result[ind])

#best lof
m=10
ind=0
for i in range(len(list_result)):
    if list_result[i,0]=='lof':
        if float(list_result[i,-1])<m:
            m=float(list_result[i,-1])
            ind=i
print(list_result[ind])

#best ransac
m=10
ind=0
for i in range(len(list_result)):
    if list_result[i,1]=='ransac' and list_result[i,0]=='lof':
        if float(list_result[i,-1])<m:
            m=float(list_result[i,-1])
            ind=i
print(list_result[ind])

#best dbscan
# m=10
# ind=0
# for i in range(len(list_result)):
#     if list_result[i,0]=='dbscsn':
#         if float(list_result[i,-1])<m:
#             m=float(list_result[i,-1])
#             ind=i
# print(list_result[ind])

# list_result_ocsvm_svmlinear = [item for item in list_result if item[0]=='ocsvm' and item[1]=='sgd']
# list_result_ocsvm_svmlinear = np.array(list_result_ocsvm_svmlinear)
# plt.figure()
# # for i in range(len(list_result_ocsvm)):
# plt.plot(list_result_ocsvm_svmlinear[:,2],list_result_ocsvm_svmlinear[:,3])
# plt.tight_layout()
# plt.show()

ran = np.linspace(0,100,101)
list_err = []
for i in ran:
    for e in ran:
        xtrain,ytrain=outlierremoval(addyt(x_train_2,y_train_2),y_train_2,0.04,'ee');xtrain=deleteyt(xtrain);cross_val(xtrain,ytrain,5,'ransac')
        list_err.append([i,e,er])

#%% BEST CASE
xtrain,ytrain=outlierremoval(addyt(x_train_2,y_train_2),y_train_2,0.04,'ee')
xtrain=deleteyt(xtrain)
print('ransac',cross_val(xtrain,ytrain,5,'ransac'))
print('svmlinear',cross_val(xtrain,ytrain,5,'svmlinear'))

#%% SAVE PREDICTION
xtrain, ytrain = outlierremoval(addyt(x_train_2, y_train_2), y_train_2,0.04,'ee')
xtrain = deleteyt(xtrain)
y_pred = ransacpredictor(xtrain, ytrain, x_test_2)
np.save('Data/YTest_Regression_Part2.npy',y_pred)

#%% Plot error vs contamination in EE
er=[]
size=[]
for cont in cont_v:
    xtrain,ytrain=outlierremoval(addyt(x_train_2,y_train_2),y_train_2,cont,'ee')
    xtrain = deleteyt(xtrain)
    er+=[cross_val(xtrain, ytrain, 5, 'huber')]
    size+=[len(xtrain)]

plt.plot(cont_v,er)
# plt.yscale('log')
plt.ylabel("error")
plt.xlabel("contamination")
plt.figure()
plt.plot(cont_v,size)
# plt.yscale('log')
plt.ylabel("size of xtrain")
plt.xlabel("contamination")
plt.show()


#%% TEST 
print('Without outlier detection: ')
cv_lasso_k5 = cross_val(x_train_2,y_train_2,5,'lasso',l_lasso)  
print('\n')
print('Testing different predictors')
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'lr')  
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'lasso',l_lasso_ocsvm)  
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'ridge',l_ridge_ocsvm)  
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'svmlinear')  #best
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'sgd')  #2nd best
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'gauss')  
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'larslasso')  
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'bayesridge')  

print('\n')

print('Testing different outliers detectors')
cv_lasso_k5_lof = cross_val(x_train_2_lof,y_train_2_lof,5,'svmlinear')  
cv_lasso_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'svmlinear')  
cv_lasso_k5_ee = cross_val(x_train_2_ee,y_train_2_ee,5,'svmlinear')  
cv_lasso_k5_iso = cross_val(x_train_2_iso,y_train_2_iso,5,'svmlinear')
cv_svmlinear_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'svmlinear') #very good

print('\n')
# cv_lasso_k5 = cross_val(x_train_2,y_train_2,5,'lasso',l_lasso)  
# cv_lasso_k5_sc = cross_val(x_train_2_sc,y_train_2_sc,5,'lasso',l_lasso)  

#%%
n_out=np.linspace(0, 0.1, 1000)
cv_lasso_k5_iso = []
for i in n_out:
    iso = IsolationForest(contamination=i)
    mask = iso.fit_predict(x_train_2)
    isin = mask != -1
    x_train_2_iso, y_train_2_iso = x_train_2[isin, :], y_train_2[isin]
    cv_lasso_k5_iso = cv_lasso_k5_iso + [cross_val(x_train_2_iso,y_train_2_iso,5,'svmlinear')]
    
# comparar: diferentes predictors, lambdas, contaminations, with and without outliers

cont_opt=n_out[np.where(cv_lasso_k5_iso==np.min(cv_lasso_k5_iso))[0][0]]

#%% CHOOSE LAMBDA FOR LASSO AND RIDGE (DON'T RUN)
# Compare cross validation errors between different lambda values

l = np.logspace(-6, 3, 10000)
cv_lr_k5 = cross_val(x_train_2,y_train_2,5,'lr')   
cv_lasso_k5_dbs =[]
for i in range(len(l)):
    dbs = DBSCAN(eps=dista[i], min_samples=2)
    mask = dbs.fit_predict(x_train_2)
    isin = mask != 0
    x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
    if len(x_train_2_dbs)>=90:
        cv_lasso_k5_dbs = cv_lasso_k5_dbs + [cross_val(x_train_2_dbs,y_train_2_dbs,5,'lasso',l[i])]
    # print('\n')
np.save('Data/cv_ridge_k5_dbs_10000.npy',cv_ridge_k5_dbs)
np.save('Data/cv_lasso_k5_dbs_10000.npy',cv_lasso_k5_dbs)
l_lasso_dbs=l[np.where(cv_lasso_k5_dbs==np.min(cv_lasso_k5_dbs))[0][0]]
l_ridge_dbs=l[np.where(cv_ridge_k5_dbs==np.min(cv_ridge_k5_dbs))[0][0]]



#%%

l = np.logspace(-6, 3, 10000)
cv_lr_k5 = cross_val(x_train_2,y_train_2,5,'lr')   
cv_ridge_k5_ocsvm = np.load('Data/cv_ridge_k5_ocsvm_10000.npy')
cv_lasso_k5_ocsvm = np.load('Data/cv_lasso_k5_ocsvm_10000.npy')
l_lasso_ocsvm=l[np.where(cv_lasso_k5_ocsvm==np.min(cv_lasso_k5_ocsvm))[0][0]]
l_ridge_ocsvm=l[np.where(cv_ridge_k5_ocsvm==np.min(cv_ridge_k5_ocsvm))[0][0]]

plt.xscale('log')
plt.scatter(l_lasso_ocsvm,np.min(cv_lasso_k5_ocsvm),marker='x',color='k',zorder=3)
plt.scatter(l_ridge_ocsvm,np.min(cv_ridge_k5_ocsvm),marker='x',color='k',zorder=3)
plt.plot(l,cv_ridge_k5_ocsvm,label='Ridge')
plt.plot(l,cv_lasso_k5_ocsvm,label='Lasso')
plt.axhline(y=cv_lr_k5, color='darkgray', linestyle='--') #5-fold cross val using linear regression
plt.title('Evolution of \u03B2 values in Ridge and Lasso Regression')
plt.xlabel('\u03BB')
plt.ylabel('Error')
plt.xlim((1e-6, 1e3))
plt.legend(loc='best')
# plt.savefig('comparelambdaserror.eps', format="eps")

del l

#%% ISOLATION FOREST
# Determine amount of contamination that minimizes error

cont = np.linspace(0.09,0.1,1001)
cv_svmlinear_k5_iso = []
for i in range(len(cont)):
    iso = IsolationForest(contamination=cont[i])
    mask = iso.fit_predict(x_train_2)
    isin = mask != -1
    x_train_2_iso, y_train_2_iso = x_train_2[isin, :], y_train_2[isin]
    cv_svmlinear_k5_iso = cv_svmlinear_k5_iso + [cross_val(x_train_2_iso,y_train_2_iso,5,'svmlinear',cont[i])]  
    
np.save('Data/cv_svmlinear_k5_iso.npy',cv_svmlinear_k5_iso)


#%%
cont = np.linspace(0.09,0.1,1001)
cv_svmlinear_k5_iso = np.load('Data/cv_svmlinear_k5_iso.npy')
cont_svmlinear_iso = cont[np.where(cv_svmlinear_k5_iso==np.min(cv_svmlinear_k5_iso))[0][0]]

plt.figure()
plt.scatter(cont_svmlinear_iso,np.min(cv_svmlinear_k5_iso),marker='x',color='k',zorder=3)
plt.scatter(cont,cv_svmlinear_k5_iso,label='Isolation')
plt.title('Cross validation error depending on contamination level') #for SVMLinear predictor and Isolation forestfor outliers
plt.xlabel('\u03BB')
plt.ylabel('Error')
plt.xlim((0.09, 0.1))
plt.legend(loc='best')
# plt.savefig('comparelambdaserror.eps', format="eps")


#%% ELEPTICAL ENVELOPE
# Determine amount of contamination that minimizes error

cont = np.linspace(0,0.1,1001)
cv_svmlinear_k5_ee = []
for i in range(len(cont)):
    ee = EllipticEnvelope(contamination=cont[i])
    mask = ee.fit_predict(x_train_2)
    isin = mask != -1
    x_train_2_ee, y_train_2_ee = x_train_2[isin, :], y_train_2[isin]
    cv_svmlinear_k5_ee = cv_svmlinear_k5_ee + [cross_val(x_train_2_ee,y_train_2_ee,5,'svmlinear',cont[i])]  
    
np.save('Data/cv_svmlinear_k5_ee.npy',cv_svmlinear_k5_ee)


#%%
cont = np.linspace(0,0.1,1001)
cv_svmlinear_k5_ee = np.load('Data/cv_svmlinear_k5_ee.npy')
cont_svmlinear_ee = cont[np.where(cv_svmlinear_k5_ee==np.min(cv_svmlinear_k5_ee))[0][0]]

plt.figure()
plt.scatter(cont_svmlinear_ee,np.min(cv_svmlinear_k5_ee),marker='x',color='k',zorder=3)
plt.plot(cont,cv_svmlinear_k5_ee,label='Eleptical Envelope')
plt.title('eliptical envelope & svmlinear') #for SVMLinear predictor and Envelope for outliers
plt.xlabel('\u03BB')
plt.ylabel('Error')
plt.xlim((0, 0.1))
plt.legend(loc='best')
# plt.savefig('comparelambdaserror.eps', format="eps")

#%% LOF
# Determine amount of contamination that minimizes error

cont = np.linspace(0.0001,0.1,1000)
cv_svmlinear_k5_lof = []
for i in range(len(cont)):
    lof = LocalOutlierFactor(contamination=cont[i])
    mask = lof.fit_predict(x_train_2)
    isin = mask != -1
    x_train_2_lof, y_train_2_lof = x_train_2[isin, :], y_train_2[isin]

    cv_svmlinear_k5_lof = cv_svmlinear_k5_lof + [cross_val(x_train_2_lof,y_train_2_lof,5,'svmlinear',cont[i])]  
    
np.save('Data/cv_svmlinear_k5_lof.npy',cv_svmlinear_k5_lof)


#%%
cont = np.linspace(0.0001,0.1,1000)
cv_svmlinear_k5_lof = np.load('Data/cv_svmlinear_k5_lof.npy')
cont_svmlinear_lof = cont[np.where(cv_svmlinear_k5_lof==np.min(cv_svmlinear_k5_lof))[0][0]]

plt.figure()
plt.scatter(cont_svmlinear_lof,np.min(cv_svmlinear_k5_lof),marker='x',color='k',zorder=3)
plt.plot(cont,cv_svmlinear_k5_lof,label='Eleptical Envelope')
plt.title('lof & svmlinear') #for SVMLinear predictor and Envelope for outliers
plt.xlabel('\u03BB')
plt.ylabel('Error')
plt.xlim((0.0001, 0.1))
plt.legend(loc='best')
# plt.savefig('comparelambdaserror.eps', format="eps")

#%% DBSCAN change eps parameter with svmlinear
# Determine amount of contamination that minimizes error

dista = np.linspace(3,5,1001)
dista_used = []
cv_svmlinear_k5_dbs = []
for i in range(len(dista)):
    dbs = DBSCAN(eps=dista[i], min_samples=2)
    mask = dbs.fit_predict(x_train_2)
    isin = mask != 0
    x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
    if len(x_train_2_dbs)>=90:
        cv_svmlinear_k5_dbs = cv_svmlinear_k5_dbs + [cross_val(x_train_2_dbs,y_train_2_dbs,5,'svmlinear',dista[i])]  
        dista_used += [dista[i]]
        print(str(dista[i]))
        
    
np.save('Data/cv_svmlinear_k5_dbs.npy',cv_svmlinear_k5_dbs)
np.save('Data/dista_used.npy',dista_used)


#%%
cv_svmlinear_k5_dbs = np.load('Data/cv_svmlinear_k5_dbs.npy')
dista_used = np.load('Data/dista_used.npy')
dist_svmlinear_dbs = dista_used[np.where(cv_svmlinear_k5_dbs==np.min(cv_svmlinear_k5_dbs))[0][0]]

plt.figure()
plt.scatter(dist_svmlinear_dbs,np.min(cv_svmlinear_k5_dbs),marker='x',color='k',zorder=3)
plt.plot(dista_used,cv_svmlinear_k5_dbs,label='DBScan')
plt.title('dbscan & svmlinear') #for SVMLinear predictor and Envelope for outliers
plt.xlabel('\u03BB')
plt.ylabel('Error')
# plt.xlim((3, 5))
plt.tight_layout()
plt.legend(loc='best')
# plt.savefig('comparelambdaserror.eps', format="eps")

# For the obtained distance value, calculate error with other predictors

# But first, lambda for ridge and lasso
l = np.logspace(-6, 3, 1000)
cv_ridge_k5_dbs =[]
cv_lasso_k5_dbs =[]
dbs = DBSCAN(eps=dist_svmlinear_dbs, min_samples=2)
mask = dbs.fit_predict(x_train_2)
isin = mask != 0
x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
for i in range(len(l)):
    cv_ridge_k5_dbs += [cross_val(x_train_2_dbs,y_train_2_dbs,5,'ridge',l[i])]  
    cv_lasso_k5_dbs += [cross_val(x_train_2_dbs,y_train_2_dbs,5,'lasso',l[i])]

l_lasso_dbs=l[np.where(cv_lasso_k5_dbs==np.min(cv_lasso_k5_dbs))[0][0]]
l_ridge_dbs=l[np.where(cv_ridge_k5_dbs==np.min(cv_ridge_k5_dbs))[0][0]]


cv_lr_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'lr')  
print('\n')
cv_lasso_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'lasso',l_lasso_dbs)  
print('\n')
cv_ridge_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'ridge',l_ridge_dbs)  
print('\n')
cv_svmlinear_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'svmlinear')  # 2ndbest
print('\n')
cv_sgd_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'sgd')  # best
print('\n')
cv_gauss_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'gauss')  
print('\n')
cv_en_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'en')  
print('\n')
cv_omp_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'omp')  
print('\n')
cv_lars_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'lars')  
print('\n')
cv_larslasso_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'larslasso')  
print('\n')
cv_bayesridge_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'bayesridge')  

# Since the predictor SGD yielded the best result, now we'll perform the same analysis using this.

#%% DBSCAN change eps parameter with sgd
# Determine amount of contamination that minimizes error

dista = np.linspace(3,5,1001)
dista_used = []
cv_sgd_k5_dbs = []
for i in range(len(dista)):
    dbs = DBSCAN(eps=dista[i], min_samples=2)
    mask = dbs.fit_predict(x_train_2)
    isin = mask != 0
    x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
    if len(x_train_2_dbs)>=90:
        cv_sgd_k5_dbs += [cross_val(x_train_2_dbs,y_train_2_dbs,5,'sgd',dista[i])]  
        dista_used += [dista[i]]
        print(str(dista[i]))
        
    
dist_sgd_dbs = dista_used[np.where(cv_sgd_k5_dbs==np.min(cv_sgd_k5_dbs))[0][0]]

plt.figure()
plt.scatter(dist_sgd_dbs,np.min(cv_sgd_k5_dbs),marker='x',color='k',zorder=3)
plt.plot(dista_used,cv_sgd_k5_dbs,label='DBScan')
plt.title('dbscan & sgd') #for SVMLinear predictor and Envelope for outliers
plt.xlabel('\u03BB')
plt.ylabel('Error')
# plt.xlim((3, 5))
plt.tight_layout()
plt.legend(loc='best')
# plt.savefig('comparelambdaserror.eps', format="eps")

# For the obtained distance value, calculate error with other predictors

# But first, lambda for ridge and lasso
# l = np.logspace(-6, 3, 1000)
# cv_ridge_k5_dbs =[]
# cv_lasso_k5_dbs =[]
# dbs = DBSCAN(eps=dist_sgd_dbs, min_samples=2)
# mask = dbs.fit_predict(x_train_2)
# isin = mask != 0
# x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
# for i in range(len(l)):
#     cv_ridge_k5_dbs += [cross_val(x_train_2_dbs,y_train_2_dbs,5,'ridge',l[i])]  
#     cv_lasso_k5_dbs += [cross_val(x_train_2_dbs,y_train_2_dbs,5,'lasso',l[i])]

# l_lasso_dbs=l[np.where(cv_lasso_k5_dbs==np.min(cv_lasso_k5_dbs))[0][0]]
# l_ridge_dbs=l[np.where(cv_ridge_k5_dbs==np.min(cv_ridge_k5_dbs))[0][0]]


cv_lr_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'lr')  
print('\n')
cv_lasso_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'lasso',l_lasso_dbs)  
print('\n')
cv_ridge_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'ridge',l_ridge_dbs)  
print('\n')
cv_svmlinear_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'svmlinear')  
print('\n')
cv_sgd_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'sgd')  
print('\n')
cv_gauss_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'gauss')  
print('\n')
cv_en_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'en')  
print('\n')
cv_omp_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'omp')  
print('\n')
cv_lars_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'lars')  
print('\n')
cv_larslasso_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'larslasso')  
print('\n')
cv_bayesridge_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'bayesridge')  



########

dbs = DBSCAN(eps=dist_sgd_dbs, min_samples=2)
mask = dbs.fit_predict(x_train_2)
isin = mask != 0
x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
cv_sgd_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'sgd')  
print('\n')

dbs = DBSCAN(eps=dist_svmlinear_dbs, min_samples=2)
mask = dbs.fit_predict(x_train_2)
isin = mask != 0
x_train_2_dbs, y_train_2_dbs = x_train_2[isin, :], y_train_2[isin]
cv_svmlinear_k5_dbs = cross_val(x_train_2_dbs,y_train_2_dbs,5,'svmlinear')  
print('\n')

ocsvm = OneClassSVM(nu=0.0757,kernel='sigmoid')
mask = ocsvm.fit_predict(x_train_2)
isin = mask != -1
x_train_2_ocsvm, y_train_2_ocsvm = x_train_2[isin, :], y_train_2[isin]
cv_svmlinear_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'svmlinear')  
print('\n')
cv_sgd_k5_ocsvm = cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'sgd')  


#%%
num = np.linspace(0.01,1,1000)
cv_sgd_k5_ocsvm = []
cv_svmlinear_k5_ocsvm = []
size_xtrain=[]
nu_used = []
for i in range(len(num)):
    ocsvm = OneClassSVM(nu=num[i],kernel='sigmoid')
    mask = ocsvm.fit_predict(x_train_2)
    isin = mask != -1
    x_train_2_ocsvm, y_train_2_ocsvm = x_train_2[isin, :], y_train_2[isin]
    if len(x_train_2_ocsvm)>=90:
        cv_sgd_k5_ocsvm += [cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'sgd')] 
        cv_svmlinear_k5_ocsvm += [cross_val(x_train_2_ocsvm,y_train_2_ocsvm,5,'svmlinear')]  
        nu_used += [num[i]]
        size_xtrain+=[len(x_train_2_ocsvm)]
        print(str(num[i]))

nu_sgd_ocsvm = nu_used[np.where(cv_sgd_k5_ocsvm==np.min(cv_sgd_k5_ocsvm))[0][0]]
nu_svmlinear_ocsvm = nu_used[np.where(cv_svmlinear_k5_ocsvm==np.min(cv_svmlinear_k5_ocsvm))[0][0]]

plt.figure()
plt.scatter(nu_sgd_ocsvm,np.min(cv_sgd_k5_ocsvm),marker='x',color='k',zorder=3)
plt.scatter(nu_svmlinear_ocsvm,np.min(cv_svmlinear_k5_ocsvm),marker='x',color='k',zorder=3)
plt.plot(nu_used,cv_sgd_k5_ocsvm,label='sgd')
plt.plot(nu_used,cv_svmlinear_k5_ocsvm,label='svmlinear')

plt.title('ocsvm + svmlinear & sgd') #for SVMLinear predictor and Envelope for outliers
plt.xlabel('\u03BB')
plt.ylabel('Error')
# plt.xlim((3, 5))
plt.tight_layout()
plt.legend(loc='best')

plt.figure()
plt.plot(nu_used,size_xtrain,label='size xtrain')
