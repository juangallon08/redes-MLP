# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 08:21:34 2023

@author: Usuario
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
array=np.load("orden1.npz")
matriz=array['orden1']

entrada=(matriz[:,0:4])
salida=(matriz[:,4])

training_data = entrada
target_data = salida


model = Sequential()
model.add(Dense(8, input_dim=4, activation='sigmoid'))
#model.add(Dense(16, activation='linear'))
#model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='linear')) #1 capa de salida
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])


model.fit(training_data, target_data, epochs=1000)
salida= model.predict(training_data)
error= np.sqrt(np.sum(np.square(target_data-salida)))/len(target_data)
print("El error es: ", error)
print(salida)

model.save('orden11.h5')