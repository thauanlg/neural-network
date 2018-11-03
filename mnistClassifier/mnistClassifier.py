"""
This script creates some classifiers for the mnist dataset using tensorflow cnn. Will be varied two parameters:
    - the number of features maps used in convolutional layer
    - the size of filters in convolution layer
"""
# Library used by this exercise
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D

import numpy as np
import time, json

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# The character we'll use are 0, 1, 2 and 3
idx=np.where(train_labels < 4)[0]
train_labels=train_labels[idx]
train_images=train_images[idx]

idx=np.where(test_labels < 4)[0]
test_labels=test_labels[idx]
test_images=test_images[idx]

# normalizing the data from 0 to 1
train_images = train_images/255.0
test_images = test_images/255.0

# expanding dimension to use in tensorflow
train_images=np.expand_dims(train_images, axis=3)
test_images=np.expand_dims(test_images, axis=3)

# the numbers of features maps used and its size. 
featureSize=np.array([2, 8, 16, 64])
filterSize=np.array([2, 5, 10])

accuracy=np.zeros((featureSize.shape[0], filterSize.shape[0]))
timeElapsed=np.zeros((featureSize.shape[0], filterSize.shape[0]))

# varying the features map size and its size, calculating the accuracy of the fitted model and the time for fitting
for i in range(featureSize.shape[0]):
    for j in range(filterSize.shape[0]):
        print("[{:d}:{:d}][{:d}:{:d}]".format(i, featureSize.shape[0], j, filterSize.shape[0]))
        model = keras.Sequential()

        model.add(Conv2D(featureSize[i], (filterSize[j],filterSize[j]), input_shape=(28,28,1)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(4))
        model.add(Activation("softmax"))

        model.compile(optimizer=tf.train.AdamOptimizer(), 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        start = time.time()
        model.fit(train_images, train_labels, epochs=3)
        end = time.time()
        timeElapsed[i][j] = end - start

        (_, accuracy[i][j]) = model.evaluate(test_images, test_labels)

# saving the values calculated for analysis
json.dump(accuracy.tolist(), open("accuracy.json", "w"))
json.dump(timeElapsed.tolist(), open("timeElapsed.json", "w"))
