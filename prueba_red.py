# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:01:20 2023

@author: jgallon
"""
from array import array
from turtle import color
from keras.layers.core import Dense
from tensorflow.python.keras.layers.core import Dense
from Funcion_Orden1 import *
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

'''cargamos el modelo a un objeto'''
model = load_model('orden11.h5')
print(model.summary())
print("Modelo Cargado!")

'''Generamos datos aleatorios y cargamos el archivo txt'''
#only_u(1.6222,0.3699, 0.1) #Generar base de datos 
data = np.array(np.loadtxt('en.txt')) #Entradas 
data1 = np.array(np.loadtxt('sa.txt')) #Salidas
#normalizacion(data1, data)
val_nor = np.array(np.loadtxt('UY_min_max.txt')) # valores normalizados
u_min = val_nor[0]
u_max = val_nor[1]
y_min = val_nor[2]
y_max = val_nor[3]
[y_norm, u_norm, len_y, len_u] = normalizacion(data1, data)

'''ciclo para valores en linea'''
u_k = u_norm #Cantodad de datos totales 
yk = []
for i in range(len(u_norm)):
    print('iteracion: ', i)
    if i == 0:                #Primera interaccion todo cero 
        vec = 0, 0, 0, 0
        vec = np.array(vec)
        vec = vec[np.newaxis]
        aux = model.predict(vec)
        yk.append(float(np.array(aux)))
        print(yk)
        u_k.insert(0, 0) # se insertan los cero para cuadrar las dimensiones
    elif i == 1:                  # una entra y una salida anterior pero siguen en ceros 
        vec = u_k[i - 1], 0, yk[i - 1], 0
        vec = np.array(vec)
        vec = vec[np.newaxis]
        aux = model.predict(vec)
        yk.append(float(np.array(aux)))
        u_k.insert(0, u_k[i - 1])
    else:
        vec = u_k[i - 1], u_k[i - 2], yk[i - 1], yk[i - 2] # insertar los instantes anteriores
        vec = np.array(vec)
        vec = vec[np.newaxis]
        aux = model.predict(vec)
        yk.append(float(np.array(aux)))

'''desnormalizacion de la salida estimada por la rnn'''
yk_d = []
u_d = []
for j in range(len(yk)):
    yk_d.append((y_max - y_min) * yk[j] + y_min)
    u_d.append((u_max - u_min) * u_norm[j] + u_min)



'''graficas salidas'''
plt.plot(yk_d, color = 'Orange')
plt.plot(data1, color = 'Blue')
plt.title('Salida')
plt.legend(['Salida estimada', 'Salida real'])
plt.grid()
plt.show()