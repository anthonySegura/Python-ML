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
from sklearn.cross_validation import train_test_split
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
X = np.ones((m,n + 1))
y = np.array((m,1))
X[:,1] = iris['PetalLengthCm'].values
X[:,2] = iris['PetalWidthCm'].values
X[:,3] = iris['SepalLengthCm'].values
X[:,4] = iris['SepalWidthCm'].values

y = iris['Species'].values

#Features Scaling
for j in range(n):
    X[:, j] = (X[:, j] - X[:,j].mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

all_theta = np.zeros((k, n + 1))

#Entrenamiento
#OneVsAll
print("Entrenando")
i = 0
for flor in Species:
	progress(i + 1, k)
	tmp_y = np.array(y_train == flor, dtype = int)
	optTheta = logisticRegression(X_train, tmp_y, np.zeros((n + 1,1))).x
	all_theta[i] = optTheta
	i += 1

#Predicciones

P = sigmoid(X_test.dot(all_theta.T))
p = [Species[np.argmax(P[i, :])] for i in range(X_test.shape[0])]

s = sum(np.array(p == y_test, dtype = int))

print("\n\n")
print("Test Accuracy ", (s / X_test.shape[0]) * 100 , "%")

#Matrix de confusion
cfm = confusion_matrix(y_test, p, labels = Species)
#plt.yticks(cfm[:,0], Species, rotation = 'horizontal')
sb.heatmap(cfm, annot = True, xticklabels = Species, yticklabels = Species)

plt.show()


