#%% Import Libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import linear_model
import os

#%% Load testing and training data
cd = "/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/MachineLearningProject" #working directory
cd = os.getcwd()
x_test_1=np.load(cd+"/Xtest_Regression_Part1.npy")
x_train_1=np.load(cd+"/Xtrain_Regression_Part1.npy")
y_train_1=np.load(cd+"/Ytrain_Regression_Part1.npy")

x_test_2=np.load(cd+"/Xtest_Regression_Part2.npy")
x_train_2=np.load(cd+"/Xtrain_Regression_Part2.npy")
y_train_2=np.load(cd+"/Ytrain_Regression_Part2.npy")

del cd

#%% Load variables for Inês until I can figure this out
cd = os.getcwd()
x_train1 = np.load(cd+'/Xtrain_Regression_Part1.npy')
x_train2 = np.load(cd+'/Xtrain_Regression_Part2.npy')
y_train1 = np.load(cd+'/Ytest_Regression_Part1.npy')
y_train2 = np.load(cd+'/Ytrain_Regression_Part2.npy')
x_test1 = np.load(cd+'/Xtest_Regression_Part1.npy')
x_test2 = np.load(cd+'/Xtest_Regression_Part2.npy')


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

#%% Linear regression 

def lr_par(xtrain,ytrain):# Calculates the beta parameters of linear regression
    c=len(xtrain) #number of observations
    #f=len(xt[0])
    X=np.append(np.ones((c,1)),xtrain,axis=1)
    Xtrans=np.transpose(X)
    
    beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(Xtrans, X)),Xtrans),ytrain)
    return beta


def lr(xtest,beta): #performs linear regression to xtest with the beta parameters given by the above function
    c=len(xtest)
    X=np.append(np.ones((c,1)),xtest,axis=1)
    ytest=np.matmul(X,beta)
    return ytest

def lrpredictor(x_train,y_train,x_test): # predicts y based on training with x_train and y_train (its the result of the merging of the last two functions)
    ytest=lr(x_test,lr_par(x_train,y_train))
    return ytest

def calcsse(ytest_pred,ytest_true): # Calculates sum of squared errors
    sse=0
    for i in range(len(ytest_pred)):
        sse=sse+float((ytest_pred[i]-ytest_true[i])**2)
    return sse

#%% k Fold classification linear regression

def crossvalidation(k,x_train,y_train):
    c=len(x_train)
     
    if (c%k)!=0:
        print("Cannot compute. Choose a divider of "+str(c))
        return
    elif k==1:
        print("Cannot perform 1-fold classification since there is no test set.")
        return
    
    else:
        sse_vector=[] # vector where the sse of each fold will be stored
        f=len(x_train[0]) #number of features
        fold=int(c/k) #size of each fold
        
        Xpart=np.split(x_train,k) #list of k partitions of x_train
        Ypart=np.split(y_train,k)
        
        for i in range(len(Xpart)):
            
            #build matrices xtrain and ytrain
            xtrain=np.empty((fold,f))
            ytrain=np.empty((fold,1))
            for j in range(len(Xpart)):
                if j==i:
                    xtest=Xpart[j]
                    ytest_true=Ypart[j]
                    
                else:
                    xtrain=np.vstack((xtrain, Xpart[j]))
                    ytrain=np.vstack((ytrain, Ypart[j]))
            
            xtrain=xtrain[fold:,:]
            ytrain=ytrain[fold:,:]

            #test
            ytest_pred= lrpredictor(xtrain,ytrain,xtest)
                    
            # calculate sum of squared errors
            sse=calcsse(ytest_pred,ytest_true)/fold # normalized by the size of each fold
            sse_vector=sse_vector+[sse]
            
    print("The mean SSE for "+str(k)+" folds is "+str(np.mean(sse_vector)))
    return np.mean(sse_vector)                                                

a=crossvalidation(5,x_train_1,y_train_1)               

#%% ANA VÊ ESTA SECÇÃO
    # continuei o que estava a fazer a pouco e vou por aqui as funções que sairam de lá para compararmos e escolhermos uma            

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


# SQUARED ERRORS
def sse(y,yt):
    # calculate the squared erros using the training set yt when compared to a predicted set in y
    # yt: training set
    # y: test/ predicted set
    return np.array((y-yt)**2).sum()

# CROSS VALIDATION
def cross_val(xt,yt,k,func,l):
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
        for i in range(k):
            y_pred[i,:] = ridgepredictor(x_train[i,:,:],y_train[i,:],l,x_test[i,:,:]) #outcomes predicted using linear regression model
    
    elif func == 'lasso':
        y_pred = np.empty((k,fold))
        for i in range(k):
            y_pred[i,:] = lassopredictor(x_train[i,:,:],y_train[i,:],l,x_test[i,:,:]) #outcomes predicted using linear regression model
    
    
    # compute errors for each set
    errors = np.empty(k)
    for i in range(k):
        errors[i] = sse(y_test[i,:],y_pred[i,:])/fold
    
    print("The mean SSE for "+str(k)+"-folds is "+str(np.mean(errors)))
    return np.mean(errors)       
    
cross_val(x_train_1,y_train_1,5,'lr',1)   

#%% Test function
crossvalidation(1,x_train_1,y_train_1)               
crossvalidation(2,x_train_1,y_train_1) 
crossvalidation(10,x_train_1,y_train_1)               
crossvalidation(20,x_train_1,y_train_1)
crossvalidation(100,x_train_1,y_train_1)                    
crossvalidation(15,x_train_1,y_train_1)     
          
#%% Using cross-validation, determine the best lambda for ridge regression
l = np.array([1e-6,1e-4,1e-2,1,10,100]) #array of lambda values to test

def best_lambda(xt,yt,l):
    l_sse = np.empty(len(l))
    for i in range(len(l)):
        l_sse[i] = np.mean(cross_val(x_train1,y_train1,5,'ridge',l[i]))
    return l[np.where(l_sse == l_sse.min())[0][0]]

best_lambda(x_train1,y_train1,l)
                    
        