import numpy as np

def gradientDescent(X, y, alpha, iters, activationFunction):
	m, n = X.shape   #Numero de ejemplos y caracteristicas
	theta = np.zeros((n, 1))

	for i in range(iters):
		#Calculo de la hipotesis
		h = activationFunction(np.dot(X, theta))
		error = h - y
		grad = X.T.dot(error) / m

		cost = np.sum(error ** 2) / (2 * m)
		print(cost)

		#Actualizacion de theta
		theta += (-alpha * grad)

	return theta
