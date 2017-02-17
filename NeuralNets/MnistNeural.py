
#Manipulacion de Matrices
import numpy as np
#Manipulacion de datos
import pandas as pd


#Datos de entrenamiento
training = pd.read_csv('./Data/train.csv').as_matrix()
#Datos de prueba
test = pd.read_csv('./Data/test.csv').as_matrix()

#Preparacion de los datos
X = training[:,1:].astype(np.float64)
y = training[:,0]
#Datos de entrenamiento y validacion
train_num , val_num = 41000, 1000
X_train, y_train = X[:train_num], y[:train_num]
X_val, y_val = X[train_num:], y[train_num:]

#Columnas de 1's
X_train = np.column_stack((np.ones((train_num, 1)), X_train))
X_val = np.column_stack((np.ones((val_num, 1)), X_val))

#Preparacion de la red neuronal
Y = np.zeros((train_num, 10))
#Matrix binaria con las vectores de salidas esperados por la red
for i in range(train_num):
    Y[i,:] = range(10)
    for j in range(10):
        Y[i, j] = 1 if Y[i, j] == y_train[i] else 0



#Calculo del costo
def costFunction(self, _lambda, Y, h):
    reg = (_lambda / (2* self.m)) * (np.sum(np.sum(self.Theta1[:,1:]**2,2)) + np.sum(np.sum(self.Theta2[:,1:]**2,2)))
    #Calculo del costo
    return (1 / self.m) * np.sum(np.sum((-Y)*np.log(h) - (1 - Y) * np.log(1 - h), 2)) + reg

#Funcion de activacion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x, theta1, theta2):
    #Feedforward
    #salidas de la capa oculta
    a1 = sigmoid(np.dot(x, theta1.T))
     
