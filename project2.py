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

#%% LOAD TEST AND TRAINING DATA
cd = os.getcwd()

# Part 1
x_train_1 = np.load(cd+'/Data/Xtrain_Classification_Part1.npy')
y_train_1 = np.load(cd+'/Data/Ytrain_Classification_Part1.npy')
x_test_1 = np.load(cd+'/Data/Xtest_Classification_Part1.npy')

xtrainreshape = np.reshape(x_train_1,(6513,50,50))
xtestreshape = np.reshape(x_test_1,(1134,50,50))



# Part 2
# x_train_2 = np.load(cd+'/Data/Xtrain_Classification_Part2.npy')
# y_train_2 = np.load(cd+'/Data/Ytrain_Classification_Part2.npy')
# x_test_2 = np.load(cd+'/Data/Xtest_Classificationn_Part2.npy')

del cd

#%% VISUALIZE DATA

data = x_train_1[87]
data_forimage = np.reshape(data,(50,50))
plt.imshow(data_forimage,cmap='gray', vmin=0, vmax=255)



#%% Create Layers

inputs = keras.Input(shape=(50, 50, 1))
# Center-crop images to 150x150
x = CenterCrop(height=50, width=50)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=16, kernel_size=(2, 2), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=16, kernel_size=(2, 2), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=16, kernel_size=(2, 2), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
# Add a dense classifier on top
num_classes = 2
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

processed_data = model(xtrainreshape)
print(processed_data.shape)
#%%
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy())
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

batch_size=32

# val_dataset = tf.data.Dataset.from_tensor_slices((xtrainreshape[5210:], y_train_1[5210:])).batch(batch_size)
# dataset = tf.data.Dataset.from_tensor_slices((xtrainreshape[0:5210], y_train_1[0:5210])).batch(batch_size)

# history=model.fit(dataset, epochs=1, validation_data=val_dataset)
history=model.fit(xtrainreshape[0:5210], y_train_1[0:5210],batch_size=batch_size, epochs=10)
print(history)

predictions = model.predict(xtrainreshape[5210:])
print(predictions)



