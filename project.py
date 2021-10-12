#%% Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model


#%% Load testing and training data
cd = os.getcwd()
x_train_1 = np.load(cd+'/Data/Xtrain_Regression_Part1.npy')
x_train_2 = np.load(cd+'/Data/Xtrain_Regression_Part2.npy')
y_train_1 = np.load(cd+'/Data/Ytrain_Regression_Part1.npy')
y_train_2 = np.load(cd+'/Data/Ytrain_Regression_Part2.npy')
x_test_1 = np.load(cd+'/Data/Xtest_Regression_Part1.npy')
x_test_2 = np.load(cd+'/Data/Xtest_Regression_Part2.npy')

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

# PREDICTOR 4: NEAREST NEIGHBORS
def knnpredictor(xt,yt,l,xtest):
    knn = KNeighborsRegressor(n_neighbors=l,weights='distance')
    knn.fit(xt,yt)
    return knn.predict(xtest)

# PREDITOR 5: SUPPORT VECTOR MACHINES
def svmpredictor(xt,yt,xtest):
    sv = svm.SVR()
    sv.fit(xt,yt)
    return sv.predict(xtest)

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
    
    elif func == 'knn':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = knnpredictor(x_train[i,:,:],y_train[i,:],l,x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'svm':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = svmpredictor(x_train[i,:,:],y_train[i,:],x_test[i,:,:]) #outcomes predicted using linear regression model
    
    # compute errors for each set
    errors = np.empty(k)
    for i in range(k):
        errors[i] = sse(y_test[i,:],y_pred[i,:])/fold
    
    print('The mean SSE for '+str(k)+'-folds using predictor '+func+' is '+str(np.mean(errors)))
    return np.mean(errors)       


#%% Test function
cv_lr_k5=cross_val(x_train_1,y_train_1,5,'lr')    
cv_lr_k10=cross_val(x_train_1,y_train_1,10,'lr',3)    
cv_ridge_k5=cross_val(x_train_1,y_train_1,5,'ridge',0.1)  
cv_ridge_k10=cross_val(x_train_1,y_train_1,10,'ridge',0.1)    
cv_lasso_k5=cross_val(x_train_1,y_train_1,5,'lasso',0.1)  
cv_lasso_k10=cross_val(x_train_1,y_train_1,10,'lasso',0.1)  

k5=[cv_lr_k5,cv_ridge_k5,cv_lasso_k5]
k10=[cv_lr_k10,cv_ridge_k10,cv_lasso_k10]
del cv_lr_k5,cv_lr_k5,cv_ridge_k5,cv_ridge_k10,cv_lasso_k5,cv_lasso_k10
#%% Plot bar chart
ind=np.arange(3)
width=0.35
plt.bar(ind, k5, width, label='5-fold')
plt.bar(ind + width, k10, width,label='10-fold')
plt.ylabel('Mean squared error')
plt.title('MSE')
plt.grid(axis='y',linestyle='--', linewidth=0.5)
plt.xticks(ind + width / 2, ('Linear Regressor', 'Ridge', 'Lasso'))
plt.yticks(np.linspace(0, 0.24,13))
plt.legend(loc='best')

#%% Using cross-validation, determine the best lambda for ridge regression
l = [1e-6,1e-4,1e-2,1,10,100] #array of lambda values to test

xt = x_train_1
yt = y_train_1

def best_lambda(xt,yt,l):
    
    l_sse = np.empty(len(l))
    for i in range(len(l)):
        l_sse[i] = cross_val(xt,yt,5,'lasso',l[i])
    return l[np.where(l_sse == l_sse.min())[0][0]]

best_lambda(x_train_1,y_train_1,l)
                    
        