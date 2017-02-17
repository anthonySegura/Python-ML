#Manipulacion de datos
import pandas as pd
#Para ignorar las advertencias
import warnings
warnings.filterwarnings("ignore")
#Algebra Lineal
import numpy as np
#Barra de progreso toa' bonita
from ProgressBar import progress
from math import ceil
#Graficas
import matplotlib.pyplot as plt
import seaborn as sb
#Confusion Matrix
from sklearn.metrics import confusion_matrix
#Para el entrenamiento y la prediccion
from Logistic_Regression import logisticRegression, sigmoid


iris = pd.read_csv('./Iris.csv')

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
p = []
for i in range(m):
	s += 1 if Species[int(ceil(np.argmax(sigmoid(X[i,:].dot(all_theta.T)))))] == y[i] else 0
	p.append(Species[int(ceil(np.argmax(sigmoid(X[i,:].dot(all_theta.T)))))])

#s = sum([1 if p[i] == y[i] else 0 for i in range(m)])

print("\n\n")
print("Train Accuracy ", (s / m) * 100 , "%")

#Matrix de confusion
cfm = confusion_matrix(y, p, labels = Species)
#plt.yticks(cfm[:,0], Species, rotation = 'horizontal')
sb.heatmap(cfm, annot = True, xticklabels = Species, yticklabels = Species)

plt.show()
