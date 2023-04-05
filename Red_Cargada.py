# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:32:29 2023

@author: jgall
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close()
array=np.load("orden1.npz")
matriz=array['orden1']
tt=array['tt']
t=tt[0:len(tt)-2]

entrada=(matriz[:,0:4])
salida=(matriz[:,4])

training_data = entrada
target_data = salida

# cargar json y crear el modelo
loaded_model = load_model('orden1.h5')
prediccion=loaded_model.predict(training_data)


'''Generamos vectores regresores'''
uk_1 = training_data[:, 0]
uk_2 = training_data[:, 1]
yk_1 = training_data[:, 2]
yk_2 = training_data[:, 3]

'''cargamos vector de salida random'''
y_real = target_data

'''inicializo matriz de salida'''
y_k = []

'''ciclo para entrar los datos de entrada'''
for j in range(len(uk_1)):
        vec = uk_1[j], uk_2[j], yk_1[j], yk_2[j]
        vec = np.array(vec)
        vec = vec[np.newaxis]  #Agregar nueva dimension 
        aux = loaded_model.predict(vec)
        y_k.append(np.array(aux))

#print(y_k)
'''se organiza el vector y_k'''
y_k = np.array(y_k)
y_k = y_k.flatten() # convierte un array N-demensional en un array uni-dimensional
#print(y_k)

'''graficamos'''
plt.plot(y_k, color = 'blue')
plt.plot(y_real, color = 'orange')
plt.legend(['Salida estimada', 'Salida real'])
plt.grid()
plt.show()
