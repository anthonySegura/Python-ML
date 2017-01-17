
import numpy as np
from math import e

#Funcion Logistica
def sigmoid(z):
    #Calcula la funcion sigmoide para un z = numero, vector o matrix.
    return 1 / (1 + np.exp(-z))

#Funcion de Costo



