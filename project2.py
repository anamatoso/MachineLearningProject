###### ML PROJECT PART 2 ######

#%% IMPORT LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image



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
# for i in range(len(x_train_1)-50):
#     plt.imshow(x_train_1[i:i+50,i:i+50], interpolation='nearest') # 50x50 pixels
#     plt.show()
    
# del i 
# for data in x_train_1:
data = x_train_1[0]

data_forimage = np.reshape(data,(50,50))
print(np.shape(data_forimage))
# for i in range(0,len(data),50):
#     data_forimage.append(data[i:i+50])
# data_forimage = np.array(data_forimage)
img = Image.fromarray(data_forimage,'L')
img.save('my.png')
img.show(img)