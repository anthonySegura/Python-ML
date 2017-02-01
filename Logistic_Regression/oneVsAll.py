
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random

from Logistic_Regression import logisticRegression, sigmoid

digits = datasets.load_digits()
m = len(digits.images)
X = digits.images.reshape((m, -1))
y = digits.target
_ , n = X.shape

X = np.column_stack((np.ones((m, 1)), X))	#Agregando columna de unos

k = 10
all_theta = np.zeros((k, n + 1))

#Training
for i in range(k):
	tmp_y = np.array(y == i, dtype = int)
	optTheta = logisticRegression(X, tmp_y, np.zeros((n+1,1))).x
	all_theta[i] = optTheta


#Calculando el 
p = np.zeros((m, 1))

for i in range(m):
	p[i] = np.argmax(sigmoid(X[i,:].dot(all_theta.T)))

s = sum([1 if p[i] == y[i] else 0 for i in range(m)])
print("Train Accuracy ", (s / m) * 100 , " %")

plt.ion()
plt.gray()
fig = plt.figure()
out = "y"

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





