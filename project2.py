###### ML PROJECT PART 2 ######

#%% IMPORT LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras


#%% LOAD TEST AND TRAINING DATA
cd = os.getcwd()

# Part 1
x_train_1 = np.load(cd+'/Data/Xtrain_Classification_Part1.npy')
y_train_1 = np.load(cd+'/Data/Ytrain_Classification_Part1.npy')
x_test_1 = np.load(cd+'/Data/Xtest_Classification_Part1.npy')

# Part 2
# x_train_2 = np.load(cd+'/Data/Xtrain_Classification_Part2.npy')
# y_train_2 = np.load(cd+'/Data/Ytrain_Classification_Part2.npy')
# x_test_2 = np.load(cd+'/Data/Xtest_Classificationn_Part2.npy')

del cd

#%% VISUALIZE DATA

data = x_train_1[1030]
data_forimage = np.reshape(data,(50,50))
plt.imshow(data_forimage,cmap='gray', vmin=0, vmax=255)