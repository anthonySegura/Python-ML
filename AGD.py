#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Implementación del Algoritmo Gradiente Descendente en Python

from util import float_cmp


#Funciones de Costo
def J0(theta0, theta1, data):
    m = len(data)
    s = 0
    #Por cada punto del dataset
    for x,y in data:
        s += (theta0 + theta1 * x) - y
    return s * (1 / m)

#Funcion de costo para theta1
def J1(theta0, theta1, data):
    m = len(data)
    s = 0
    for x,y in data:
        s+= ((theta0 + theta1 * x) - y) * x
    return s * (1 / m)


#Gradiente Descendente
def AGD(theta0, theta1, alpha, data, iterations = 1000):
    t0, t1 = theta0, theta1

    while iterations > 0:
        iterations -= 1
        #Actualización simultanea de theta0 y theta1
        t0 = t0 - (alpha * J0(theta0, theta1, data))
        t1 = t1 - (alpha * J1(theta1, theta1, data))
        theta0 = t0
        theta1 = t1
        
    return theta0, theta1

data = [(1,1),(2,2),(3,3)]
result = AGD(0,1,0.1,data)
print(result)
