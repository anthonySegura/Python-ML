
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('./Data/train.csv')

#42000 x 785

m, m_train, m_test = 42000, 33600, 8400

X = data.as_matrix()[:,1::]
X = np.column_stack((np.ones((m, 1)), X))
y = data['label']

#Division del Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)



#Funcion de Activacion
def sigmoid(z):
    return 1.0 / (1.0  + np.exp(-z))

#Cost Function
def cost(X, y, h, _lambda, Theta1, Theta2):
    m = X.shape[0]
    #Regularizacion
    reg = (_lambda / (2 * m)) * (np.sum(np.sum(Theta1[:, 1::]**2, 2)) + np.sum(np.sum(Theta2[:, 1::]**2, 2)))

    return (1/m) * np.sum(np.sum((-y) * np.log(h) - (1 - y) * np.log(1 - h), 2)) + reg


#Calculo de los gradientes
def gradient(X, y, h, prevGrad):
    pass




