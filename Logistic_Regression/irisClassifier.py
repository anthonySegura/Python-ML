#Manipulacion de datos
import pandas as pd
#Para ignorar las advertencias
import warnings
warnings.filterwarnings("ignore")
#Algebra Lineal
import numpy as np 
#Acceso a las carpetas
import os
#Barra de progreso toa' bonita
from ProgressBar import progress
from math import ceil
#Para el entrenamiento y la prediccion
from Logistic_Regression import logisticRegression, sigmoid

pwd = os.curdir

iris = pd.read_csv(pwd + "/Iris.csv")

Species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#Numero de ejemplos
m, _ = iris.shape
#Numero de clases
k = 3  
#Numero de caracteristicas
n = 4  

#Preparacion de los datos
X = np.zeros((m,n + 1))
y = np.array((m,1))
X[:,1] = iris['PetalLengthCm'].values
X[:,2] = iris['PetalWidthCm'].values
X[:,3] = iris['SepalLengthCm'].values
X[:,4] = iris['SepalWidthCm'].values

y = iris['Species'].values

all_theta = np.zeros((k, n + 1))

#Entrenamiento
#OneVsAll
print("Entrenando")
i = 0
for flor in Species: 
	progress(i + 1, k)
	tmp_y = np.array(y == flor, dtype = int)
	optTheta = logisticRegression(X, tmp_y, np.zeros((n + 1,1))).x	
	all_theta[i] = optTheta
	i += 1

#Predicciones 
s = 0	#Numero de aciertos
for i in range(m):
	s += 1 if Species[int(ceil(np.argmax(sigmoid(X[i,:].dot(all_theta.T)))))] == y[i] else 0

#s = sum([1 if p[i] == y[i] else 0 for i in range(m)])

print("\n\n")
print("Train Accuracy ", (s / m) * 100 , "%")

