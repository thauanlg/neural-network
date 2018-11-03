"""
 This program will load the data saved about accuracy and time eleapsed of models, plotting some two graphics about them:
    - Accuracy variation by the numbers of convolution matrix and the size of each convolution matrix
    - Time eleapsed to training the model by the numbers of convolution matrix and the size of each convolution matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

featureSize=np.array([2, 8, 16, 64])
filterSize=np.array([2, 5, 10])

# Loading the data of accuracy and time eleapsed
accuracy = np.array(json.load(open("accuracy.json")))
timeElapsed = np.array(json.load(open("timeElapsed.json")))

df=pd.DataFrame(accuracy, index=[featureSize], columns=filterSize)

# Plotting the first and second one graphics
fig, axes = plt.subplots(1, figsize=(8,5));
axes.plot(featureSize, accuracy.T[0])
axes.plot(featureSize, accuracy.T[1])
axes.plot(featureSize, accuracy.T[2])
axes.legend(filterSize)
fig.suptitle("Variação da acuracia pelo tamanho da matriz de convolução\n e pelo número de mapas de característica", fontsize=14)
axes.set_xlabel("Número de mapas de características (2, 8, 16 e 64)")
axes.set_ylabel("acurácia")
axes.set_ylim(0.95, 1)
axes.grid(True)

fig, axes = plt.subplots(1, figsize=(8,5));
axes.plot(featureSize, timeElapsed.T[0])
axes.plot(featureSize, timeElapsed.T[1])
axes.plot(featureSize, timeElapsed.T[2])
axes.legend(filterSize)
fig.suptitle("Variação do tempo de treinamento pelo tamanho da matriz de convolução\n e pelo número de mapas de característica", fontsize=14)
axes.set_xlabel("Número de mapas de características (2, 8, 16 e 64)")
axes.set_ylabel("tempo (s)")
axes.set_ylim(0, 90)
axes.grid(True)
