
#Algebra Lineal
import numpy as np
#Graficos
import matplotlib.pyplot as plt
#Datasets
from sklearn import datasets
import random
#Progress Bar
from ProgressBar import progress
#Para ignorar advertencias
import warnings
warnings.filterwarnings("ignore")

from Logistic_Regression import logisticRegression, sigmoid

#Mnist Dataset
digits = datasets.load_digits()    
m = len(digits.images)			    #Numero de Ejemplos
X = digits.images.reshape((m, -1))
y = digits.target				    #Etiquetas 
_ , n = X.shape

X = np.column_stack((np.ones((m, 1)), X))	#Agregando columna de 1's

k = 10	#Numero de clases

#Parametros theta optimizados para cada numero
all_theta = np.zeros((k, n + 1))

#Training
print("Entrenando")
for i in range(k):
	progress(i + 1, k)
	tmp_y = np.array(y == i, dtype = int)
	optTheta = logisticRegression(X, tmp_y, np.zeros((n+1,1))).x
	all_theta[i] = optTheta


#Predicciones para cada numero
p = np.zeros((m, 1))
for i in range(m):
	p[i] = np.argmax(sigmoid(X[i,:].dot(all_theta.T)))

s = sum([1 if p[i] == y[i] else 0 for i in range(m)])

print("\n\n")
print("Train Accuracy ", (s / m) * 100 , " %")

plt.ion()		#modo interactivo
plt.gray()
fig = plt.figure()
out = "y"

#Mostrando y prediciendo numeros al azar
while(out == "y"):
	xi = random.randint(0, m)
	x = X[xi, :]
	pred = np.argmax(sigmoid(x.dot(all_theta.T)))
	print("Prediccion : ", pred)
	ax = fig.add_subplot(111)
	ax.matshow(digits.images[xi], cmap=plt.cm.binary)
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))

	out = input("Continuar [y/n]\n")
	plt.clf()





