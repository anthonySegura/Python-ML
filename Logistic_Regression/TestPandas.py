
import pandas as pd                 #Para manipular datos
import numpy as np                  #Para algebra lineal
import matplotlib.pyplot as plt     #Para graficar
import os

from Logistic_Regression import logisticRegression, sigmoid

pwd = os.pardir  #Carpeta actual

#Lectura del dataset con pandas

data = pd.read_csv(pwd + "/DataSets/ex2data1.csv", names = ['test1', 'test2', 'condicion'])

#Graficando los datos
admitidos = [x for x in data.values if x[2] == 1.]
rechazados = [x for x in data.values if x[2] == 0.]

# plt.scatter([x[0] for x in admitidos], [y[1] for y in admitidos], s = 60, c = 'blue', marker='+',label = 'admitido')
# plt.scatter([x[0] for x in rechazados], [y[1] for y in rechazados], s = 60, c = 'red', marker='o',label = 'mamón')
# plt.xlim(30,100)
# plt.ylim(30,100)
# plt.xlabel('Exam 1 Score')
# plt.ylabel('Exam 2 Score')
# plt.legend()
# plt.show()

#Preparando los datos
X = np.ones((100,3))
y = np.ones((100,1))
X[:,1] = data['test1'].values
X[:,2] = data['test2'].values
y[:,0] = data['condicion'].values

initial_theta = np.zeros((3, 1))
alpha = 0.001

optimized_theta = logisticRegression(X, y, initial_theta).x

#Predicciones para los datos de entrada
p = [(1 if sigmoid(X[i,:].dot(optimized_theta)) >= 0.5 else 0) for i in range(100)]

print(p)


