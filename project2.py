###### ML PROJECT PART 2 ######

#%% IMPORT LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D


#%% LOAD TEST AND TRAINING DATA
cd = os.getcwd()

# Part 1
x_train_1 = np.load(cd+'/Data/Xtrain_Classification_Part1.npy')
y_train_1 = np.load(cd+'/Data/Ytrain_Classification_Part1.npy')
x_test_1 = np.load(cd+'/Data/Xtest_Classification_Part1.npy')

# xtrainreshape = np.reshape(x_train_1,(6513,50,50))
# xtestreshape = np.reshape(x_test_1,(1134,50,50))

# turn y train into categorical data
y_train_1 = np_utils.to_categorical(y_train_1, 2)
# Part 2
# x_train_2 = np.load(cd+'/Data/Xtrain_Classification_Part2.npy')
# y_train_2 = np.load(cd+'/Data/Ytrain_Classification_Part2.npy')
# x_test_2 = np.load(cd+'/Data/Xtest_Classificationn_Part2.npy')

del cd

def accuracy(y1,y2):
    a=0
    if len(y1)!=len(y2):
        return print("The sizes are different")
    else:
        for i in range(len(y1)):
            if np.all(y1[i]==y2[i]):
                a+=1
                
        return a/len(y1)

#%% VISUALIZE DATA

data = x_train_1[87]
data_forimage = np.reshape(data,(50,50))
plt.imshow(data_forimage,cmap='gray', vmin=0, vmax=255)



#%% Create Layers

# inputs = keras.Input(shape=(50, 50, 1))
# # Center-crop images to 150x150
# x = CenterCrop(height=50, width=50)(inputs)
# # Rescale images to [0, 1]
# x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
# x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
# x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
# num_classes = 2
# outputs = layers.Dense(num_classes, activation="softmax")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()


# building a linear stack of layers with the sequential model
model = Sequential()
# hidden layer
model.add(Dense(100, input_shape=(2500,), activation='relu'))
# output layer
model.add(Dense(2, activation='softmax'))


# processed_data = model(xtrainreshape)
# print(processed_data.shape)
#%%
# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy())
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

batch_size=32

# val_dataset = tf.data.Dataset.from_tensor_slices((xtrainreshape[5210:], y_train_1[5210:])).batch(batch_size)
# dataset = tf.data.Dataset.from_tensor_slices((xtrainreshape[0:5210], y_train_1[0:5210])).batch(batch_size)

# history=model.fit(dataset, epochs=1, validation_data=val_dataset)
history=model.fit(x_train_1[0:5210], y_train_1[0:5210],batch_size=batch_size, epochs=10, validation_data=(x_train_1[5210:], y_train_1[5210:]))
# print(history)

predictions = model.predict(x_train_1[5210:])
print(predictions)
for i in range(len(predictions)):
    if predictions[i][0]>=0.5:
        predictions[i]=[1,0]
    else:
        predictions[i]=[0,1]



print(len(predictions))

print(accuracy(predictions,y_train_1[5210:]))

