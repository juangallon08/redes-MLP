import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import cm

array=np.load("matriz.npz")
matriz=array['matriz']
tt=array['tt']
t=tt[0:len(tt)-2]

entrada=(matriz[:,0:4])
salida=(matriz[:,4])

training_data = entrada
target_data = salida

# cargar json y crear el modelo
loaded_model = load_model('modelo1.h5')
prediccion=loaded_model.predict(training_data)

plt.plot(t,salida,t,prediccion)
plt.show()

salida_esp=0
anterior1=0
anterior2=0
u=0
u1=0
u2=0
y1=0
y2=0
tiempo=0
uu=[]
uk_1=[]
uk_2=[]
yk_1=[]
yk_2=[]
yt=[]
uk_1.append(u1)
uk_2.append(u2)
yk_1.append(y1)
yk_2.append(y2)
sistema=[uk_1,uk_2,yk_1,yk_2]
sistema=np.transpose(sistema)
while tiempo<=199:
           
    if tiempo>=20:
        u=0.25
    if tiempo>=40:
        u=0.50   
    if tiempo>=60:
        u=0.75
    if tiempo>=80:
        u=0.100
    if tiempo>=100:
        u=0.75 
    if tiempo>=120:
        u=0.50
    if tiempo>=160:
        u=0.25
    if tiempo>=160:
        u=0
    
    uu.append(u)
    salida_esp=loaded_model.predict(uu[tiempo-1])
    yt.append(salida_esp)
    anterior1=uu[tiempo]
    anterior2=uu[tiempo-1]
    uk_1.append(anterior1)
    uk_2.append(anterior2)
    tiempo=tiempo+1

'''
ytotal=[]

#Pares de vectores para las combinaciones de entradas
vecx=np.arange(0, 100, 10)
vecy=np.arange(0, 1.0, 0.05)

for x2 in range (20):
    yt=[]
    for x1 in range(20):
        vec=vecx[x1],vecy[x2]
        vec=np.array(vec)
        vec= vec[np.newaxis]
        xtotal.append(vec)
        yf=loaded_model.predict(vec)
        yt.append(float(np.array(yf)))
    ytotal.append(np.array(yt))
        

#Se genera grafica 3D
vecx,vecy= np.meshgrid(vecx,vecy)
ytotal=np.array(ytotal)
fig= plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(vecx, vecy, ytotal,cmap=cm.coolwarm,rstride=1, cstride=1)
ax.set_zlim(-2.01,2.01)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

plt.show()
'''