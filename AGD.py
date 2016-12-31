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
def AGD(theta0, theta1, alpha, data):
    t0, t1, convergencia = theta0, theta1, False
    while not convergencia:
        #Actualización simultanea de theta0 y theta1
        t0 = t0 - (alpha * J0(theta0, theta1, data))
        t1 = t1 - (alpha * J1(theta1, theta1, data))

        #Comprobación de convergencia
        if float_cmp(t0,theta0) and float_cmp(t1,theta1):
            convergencia = True
        else:
            theta0 = t0
            theta1 = t1

    return theta0, theta1

