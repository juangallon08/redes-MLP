# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:45:49 2023

@author: jgall
"""
from array import array
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import random

def only_u(k, tao, delta): 
    #k=1.6222
    #tao=0.3699
    #delta=0.1
    u = 0
    y = 0
    t = 0
    z = []
    tt = []
    uu = []
    
    while t<=1000: 
            if t % 100 == 0:
               u = random.random() * 100
            
            y =((((u*k)-y)*delta)/tao)+y
            #y =((((u*delta*k)/tao)+y)*(1/(1+delta/tao)))
            #print(y)
            uu.append(u)
            z.append(y)
            tt.append(t)
            t=t+1
            print(y)
            
    # '''guardar txt de entradas'''
    inputs = list(uu)
    np.savetxt('en.txt', inputs)
    outputs = list(z)
    np.savetxt('sa.txt', outputs)
    
    # '''Grafica respuesta escalon'''
    tt = np.transpose(tt)
    plt.plot(z, color = "red")
    plt.plot(uu, color = "blue")
    plt.legend()
    plt.show()

def normalizacion(z, uu):
    z_max = max(z)
    z_min = min(z)
    u_max = max(uu)
    u_min = min(uu)
    UY_min_max = list(np.array([u_min, u_max, z_min, z_max]))
    np.savetxt('UY_min_max.txt', UY_min_max)
    '''inicializo los vectores para llenar al normalizar'''
    u_norm = []
    z_norm = []

    '''Normalizo y lleno los vectores con el ciclo'''
    for i in range(len(z)):
        z_norm.append((z[i] - z_min)/(z_max - z_min))
        u_norm.append((uu[i] - u_min)/(u_max -  u_min))
    '''tamaño de las matrices'''
    len_z = len(z_norm)
    len_u = len(u_norm)
    return z_norm, u_norm, len_z, len_u
            