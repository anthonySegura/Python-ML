
import pandas as pd                 #Para manipular datos
import matplotlib.pyplot as plt     #Para graficar
import os

pwd = os.pardir  #Carpeta actual

#Lectura del dataset con pandas

data = pd.read_csv(pwd + "/DataSets/ex2data1.csv", names = ['admitidos', 'rechazados', 'condicion'])

#Graficando los datos

admitidos = [x for x in data.values if x[2] == 1.]
rechazados = [x for x in data.values if x[2] == 0.]

plt.scatter([x[0] for x in admitidos], [y[1] for y in admitidos], s = 60, c = 'blue', marker='+',label = 'admitido')
plt.scatter([x[0] for x in rechazados], [y[1] for y in rechazados], s = 60, c = 'red', marker='o',label = 'mamón')
plt.xlim(30,100)
plt.ylim(30,100)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()
