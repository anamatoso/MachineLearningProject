#%% Import Libraries
import matplotlib.pyplot as plt
import numpy as np


#%% Load testing and training data
cd = "/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/MachineLearningProject" #working directory
x_test_1=np.load(cd+"/Data/Xtest_Regression_Part1.npy")
x_train_1=np.load(cd+"/Data/Xtrain_Regression_Part1.npy")
y_train_1=np.load(cd+"/Data/Ytrain_Regression_Part1.npy")

x_test_2=np.load(cd+"/Data/Xtest_Regression_Part2.npy")
x_train_2=np.load(cd+"/Data/Xtrain_Regression_Part2.npy")
y_train_2=np.load(cd+"/Data/Ytrain_Regression_Part2.npy")

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
            sse=calcsse(ytest_pred,ytest_true)
            sse_vector=sse_vector+[sse]
            
    print("The mean SSE for "+str(k)+" folds is "+str(np.mean(sse_vector)))
    return sse_vector                                                
                    
# Test function
crossvalidation(1,x_train_1,y_train_1)               
crossvalidation(2,x_train_1,y_train_1) 
crossvalidation(5,x_train_1,y_train_1)               
crossvalidation(10,x_train_1,y_train_1)               
crossvalidation(20,x_train_1,y_train_1)
crossvalidation(100,x_train_1,y_train_1)                    
crossvalidation(15,x_train_1,y_train_1)     
          

                    
                    
                    
        