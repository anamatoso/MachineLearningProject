#%% Import Libraries
import matplotlib.pyplot as plt
import numpy as np


#%% Load testing and training data
x_test_1=np.load("/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/Projeto/Xtest_Regression_Part1.npy")
x_train_1=np.load("/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/Projeto/Xtrain_Regression_Part1.npy")
y_train_1=np.load('/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/Projeto/Ytrain_Regression_Part1.npy')

x_test_2=np.load("/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/Projeto/Xtest_Regression_Part2.npy")
x_train_2=np.load("/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/Projeto/Xtrain_Regression_Part2.npy")
y_train_2=np.load("/Users/ana/Documents/Ana/universidade/5 ano 1 semestre/Aprendizagem Automática/Projeto/Ytrain_Regression_Part2.npy")

#Plot each feature vs outcome
for i in range(20):
    x=x_train_1[:,i]
    plt.figure()
    plt.scatter(x,y_train_1)
    plt.title("feature " +str(i)+" vs y")

#%% Plot each feature vs feature-check dependencies

for i in range(20):
    for j in range(i):
        if i!=j:
            x1=x_train_1[:,i]
            x2=x_train_1[:,j]
            plt.figure()
            plt.scatter(x1,x2)
            plt.title("feature " +str(i)+" vs feature " +str(j))
            
# The features seem independent

#%% Linear regression 
def lr_par(xtrain,ytrain):
    c=len(xtrain) #number of observations
    #f=len(xt[0])
    X=np.append(np.ones((c,1)),xtrain,axis=1)
    Xtrans=np.transpose(X)
    
    beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(Xtrans, X)),Xtrans),ytrain)
    return beta


def lr(xtest,beta):
    c=len(xtest)
    X=np.append(np.ones((c,1)),xtest,axis=1)
    ytest=np.matmul(X,beta)
    return ytest


ytest1=lr(x_test_1,lr_par(x_train_1,y_train_1))

#%% k Fold classification linear regression

def crossvalidation(k,x_train,y_train):
    c=len(x_train)
     #size of each fold
    if (c%k)!=0:
        return("Choose a divider of "+str(c))
    else:
        Xpart=np.split(x_train,k)
        Ypart=np.split(y_train,k)
        
        for i in range(len(Xpart)):
            xtrain=[]
            ytrain=[]
            for j in range(len(Xpart)):
                if j==i:
                    xtest=Xpart[j]
                    ytest_true=Ypart[j]
                    
                else:
                    xtrain.append(Xpart[j])
                    ytrain.append(Ypart[j])
                    
                    #test
                    ytest_pred=lr(xtest,lr_par(xtrain,ytrain))
                    
                    # calculate accuracy
                    acc=0
                    for i in range(len(ytest_pred)):
                        if ytest_pred[i]!=ytest_true[i]:
                            acc=acc+1
                            
                    print(acc/(c/k))
                            
                    
                    
                    
                    
                    
                    
        
        # Xpart=np.array([])
        # x=np.array()
        # for i in range(len(x_train)):
        #     if fold%i==0:
        #         Xpart.append(x)
        #         x=np.array()
        #     else:
        #         x.append(x_train[i,:])
        
        
        
        
        
        ytest = lr(x_test,lr_par(x_train,y_train))

