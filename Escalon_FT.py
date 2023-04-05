# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 08:51:12 2023

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Definir la función de transferencia
num = [1.6222]
den = [0.3699, 1]
sys = signal.TransferFunction(num, den)

# Definir el tiempo de muestreo
T = 0.1

# Generar un rango de valores para el eje x
t = np.arange(0, 10, T)

# Crear un vector de entrada que represente un escalón unitario
u = 10*np.ones_like(t)

# Evaluar la función de transferencia utilizando la función "lsim"
t, y, _ = signal.lsim(sys, u, t)

# Graficar la respuesta en el tiempo
plt.plot(t, y)
plt.title('Respuesta Escalón')
plt.xlabel('Tiempo [s]')
plt.ylabel('Salida')
plt.grid(True)
plt.show()




