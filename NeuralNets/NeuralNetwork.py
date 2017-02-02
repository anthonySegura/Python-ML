
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn import datasets


digits = datasets.load_digits()
m = len(digits.images)
training_set = digits.images.reshape((m, -1))

print(training_set.shape)




