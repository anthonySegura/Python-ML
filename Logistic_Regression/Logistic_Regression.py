
from scipy import optimize as op
import numpy as np
from sklearn.utils import shuffle

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
    #Al regularizar no se incluye Theta[0]
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

def stochasticGradientDescent(X, y, alpha, theta, iters):
    #Desordenar el Dataset
    X_train, y_train = X, y
    m, n = X_train.shape
    X_transpose = X_train.T
    for iter in range(iters):
        h = sigmoid(np.dot(X_train, theta))
        loss = h - y_train
        cost = np.sum(loss ** 2) / (2 * m)
        print('Iteracion # ', iter, ' | Costo : ', cost)

        #Actualizacion de theta
        theta = theta - alpha * (np.dot(X_transpose, loss) / m)

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


 