
from scipy import optimize as op
import numpy as np


#Funcion Logistica
def sigmoid(z):
    #Calcula la funcion sigmoide para un z = {numero, vector o matrix}
    return 1 / (1 + np.exp(-z))


#Funcion de Costo
def costFunction(theta, X, y):
    #Implementación Vectorizada en Matlab/Octave
    #J = (1 / m) * (-y'*log(h)-(1-y)' * log(1 - h));
    m = len(y)
    h = sigmoid(X.dot(theta))
    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))



#Funcion de Costo regularizada
def regCostFunction(theta, X, y, _lambda = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0
    reg = (_lambda/(2*m)) * np.sum(tmp**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg


#Funcion del Gradiente
def gradient(theta, x, y):
    #Version Vectorizada en Matlab/Octave
    #grad = (1/m)*X'*(h - y);
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(x.dot(theta))
    return ((1 / m) * x.T.dot(h - y))



#Funcion del Gradiente Regularizada
def regGradient(theta, X, y, _lambda = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0
    reg = _lambda*tmp /m

    return ((1 / m) * X.T.dot(h - y)) + reg


#Calculo de los valores optimos de theta
def logisticRegression(X, y, theta):
    result = op.minimize(fun = regCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = regGradient)
    return result


