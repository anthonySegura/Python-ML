
#Manipulacion de datos
import pandas as pd
#Para ignorar las advertencias de seaborn
import warnings
warnings.filterwarnings("ignore")
#Graficos
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates

import os

sns.set(style = 'white', color_codes = True)
#Carpeta Actual
pwd = os.curdir

iris = pd.read_csv(pwd + "/Iris.csv")

#Visualizaciones del Dataset

parallel_coordinates(iris.drop("Id", axis=1), "Species")

#Respecto al sepalo
sepalPlt = sns.FacetGrid(iris, hue="Species", size=6) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")
plt.legend(loc='upper left')

#Respecto al petalo
petalPlt = sns.FacetGrid(iris, hue="Species", size=6) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")
plt.legend(loc='upper left')


plt.show()


