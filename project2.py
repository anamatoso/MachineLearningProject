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
from keras.callbacks import EarlyStopping

#%% LOAD TEST AND TRAINING DATA
cd = os.getcwd()

# Part 1
x_train_1 = np.load(cd+'/Data/Xtrain_Classification_Part1.npy')
y_train_1 = np.load(cd+'/Data/Ytrain_Classification_Part1.npy')
x_test_1 = np.load(cd+'/Data/Xtest_Classification_Part1.npy')

xtrainreshape = np.reshape(x_train_1,(6470,50,50))
xtestreshape = np.reshape(x_test_1,(1164,50,50))

# turn y train into categorical data
y_train_1 = np_utils.to_categorical(y_train_1, 2)

division=round(0.8*len(y_train_1))
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

def imshow(img):
    img = img.reshape(50,50)
    plt.imshow(img,cmap='gray')

#%% Create Layers

inputs = keras.Input(shape=(50, 50, 1))

# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(inputs)
#x = layers.Dense(32, activation="relu", name="dense_1")(x) # Fully connected layer
# Apply some convolution and pooling layers
x = layers.Conv2D(filters=16, kernel_size=(2, 2), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Apply global average pooling to get flat feature vectors
#x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="relu", name="dense_2")(x) # Fully connected layer
x= layers.Flatten()(x)

# Add a dense classifier on top
num_classes = 2
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()


#%%
model.compile(optimizer='adam', metrics=['accuracy'],loss='categorical_crossentropy')
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.compile(optimizer='adam',
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # metrics=['accuracy'])

batch_size=32

# val_dataset = tf.data.Dataset.from_tensor_slices((xtrainreshape[5210:], y_train_1[5210:])).batch(batch_size)
# dataset = tf.data.Dataset.from_tensor_slices((xtrainreshape[0:5210], y_train_1[0:5210])).batch(batch_size)
callback=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
# history=model.fit(dataset, epochs=1, validation_data=val_dataset)
history=model.fit(xtrainreshape[0:division], y_train_1[0:division],batch_size=batch_size, epochs=40, validation_data=(xtrainreshape[division:], y_train_1[division:]),callbacks=[callback])
# print(history)

predictions = model.predict(xtrainreshape[5210:])

#CONVERTER PREDICTIONS EM VETOR DE ESCALARES 0 E 1

for i in range(len(predictions)):
    if predictions[i][0]>=0.5:
        predictions[i]=[1,0]
    else:
        predictions[i]=[0,1]

# print(accuracy(predictions,y_train_1[5210:]))

#%% Plot accuracy and loss

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()