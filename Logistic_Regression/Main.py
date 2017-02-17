

import pandas as pd                 #Para manipular datos
import numpy as np                  #Para algebra lineal
import matplotlib.pyplot as plt     #Para graficar

from Logistic_Regression import logisticRegression, sigmoid


if __name__ == '__main__':

    #Lectura del dataset con pandas

    data = pd.read_csv("ex2data1.csv", names = ['test1', 'test2', 'condicion'])

    #Preparando los datos
    X = np.ones((100,3))
    y = np.ones((100,1))
    X[:,1] = data['test1'].values
    X[:,2] = data['test2'].values
    y[:,0] = data['condicion'].values
    m = len(y)
    initial_theta = np.zeros((3, 1))

    optimized_theta = logisticRegression(X, y, initial_theta).x
    
    #Predicciones para los datos de entrada
    p = sigmoid(X.dot(optimized_theta)) >= 0.5
    #Numero de predicciones exitosas
    s = sum([1 if p[i] == y[i] else 0 for i in range(m)])

    print("Train Accuracy ", (s / m) * 100 , " %")
    prob_test = sigmoid(np.array([1, 45, 85]).dot(optimized_theta))
    print('For a student with scores 45 and 85, we predict an admission probability of ', prob_test)

    #Graficando los datos
    admitidos = [x for x in data.values if x[2] == 1.]
    rechazados = [x for x in data.values if x[2] == 0.]
    plt.scatter([x[0] for x in admitidos], [y[1] for y in admitidos], s = 60, c = 'blue', marker='+',label = 'admitido')
    plt.scatter([x[0] for x in rechazados], [y[1] for y in rechazados], s = 60, c = 'red', marker='o',label = 'rechazado')
    plt.xlim(30,100)
    plt.ylim(30,100)
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    # Decission Boundary
    plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
    plot_y = (-optimized_theta[0] - optimized_theta[1] * plot_x) / optimized_theta[2]
    plt.plot(plot_x, plot_y, color = 'green', label = 'decission boundary')

    plt.legend()
    plt.show()

