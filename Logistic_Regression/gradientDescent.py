import numpy as np 
import matplotlib.pyplot as plt 
from Logistic_Regression import regCostFunction, regGradient, sigmoid

def gradientDescent(X, y, theta, alpha, iters):
	jHistory = np.zeros((iters,1))
	m , n = X.shape
	final_theta = np.ones((n,1))
	for i in range(iters):
		h = X.dot(final_theta)
		error = h - y
		grad = regGradient(final_theta, X, y)
		final_theta = final_theta - (alpha * grad)

		#Calculo del costo
		J = regCostFunction(final_theta, X, y)
		print("iteracion : ", i+1, " | Costo : ", J)

		jHistory[i] = J