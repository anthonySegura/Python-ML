
import numpy as np

class TwoLayerNeuralNet():
	"""Red neuronal simple completamente conectada"""
	def __init__(self, data, labels, output_units = 10 ,hidden_units = 28):
		self.X = data
		self.y = labels
		self.m , self.n = data.shape
		self.output_units = output_units
		self.hidden_units = hidden_units
		self.Theta1 = np.random.rand(hidden_units, self.n + 1)
		self.Theta2 = np.random.rand(output_units, hidden_units + 1)


	def costFunction(self, _lambda, Y, h):
		reg = (_lambda / (2* self.m)) * (np.sum(np.sum(self.Theta1[:,1:]**2,2)) + np.sum(np.sum(self.Theta2[:,1:]**2,2)))
		#Calculo del costo
		return (1 / self.m) * np.sum(np.sum((-Y)*np.log(h) - (1 - Y) * np.log(1 - h), 2)) + reg


	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))


	def __sigmoidGrad(self, z):
		return self.__sigmoid(z) * (1 - self.__sigmoid(z))


	def __feedForward(self, index):
		#Agregando columna de 1's a X
		a1 = np.column_stack((np.ones((self.m, 1)), self.X))
		z2 = np.dot(a1[index, :], self.Theta1.T)
		a2 = np.ones((z2.shape[0] + 1))  #np.column_stack((np.ones((z2.shape[0], 1)), self.__sigmoid(z2))) # + bias
		a2[1:] = self.__sigmoid(z2)
		z3 = np.dot(a2, self.Theta2.T)
		a3 = self.__sigmoid(z3)

		return a3


nn = TwoLayerNeuralNet(np.zeros((200,784)),[])
