from PIL import Image
class ConvertImage():
    def __init__(self, percentage_to_reducer) -> None:
        self.percentage_to_reducer = percentage_to_reducer
        self.images = []
        self.labels = []

    def add_image_gray(self, name:  str):
        image = Image.open(name).convert('L')
        width, height = image.size
        resized_dimensions = (int(width * self.percentage_to_reducer),
                              int(height * self.percentage_to_reducer))
        resized = image.resize(resized_dimensions)
        self.images.append(np.array(resized).tolist())

    def add_label_gray(self, name:  str):
        image = Image.open(name).convert('L')
        width, height = image.size
        resized_dimensions = (int(width * self.percentage_to_reducer),
                              int(height * self.percentage_to_reducer))
        resized = image.resize(resized_dimensions)
        self.labels.append(np.array(resized).ravel().tolist())

    def get_images(self):
        return self.images

    def get_labels(self):
        return self.labels


convert_image = ConvertImage(percentage_to_reducer=.25)
convert_image.add_image_gray(name='./training/input/{}_test.tif'.format('01'))
convert_image.add_label_gray(name='./training/target/{}_manual1.tif'.format('01'))
# convert_image.add_image_gray(name='./training/input/{}_test.tif'.format('02'))
# convert_image.add_label_gray(name='./training/target/{}_manual1.tif'.format('02'))

# images = convert_image.get_images()
# labels = convert_image.get_labels()

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Creamos nuestros datos artificiales, donde buscaremos clasificar 
# dos anillos concéntricos de datos. 
X, Y = make_circles(n_samples=500, factor=0.5, noise=0.05)

print(X)

import tensorflow as tf
import tensorflow.keras as kr

from IPython.core.display import display, HTML


lr = 0.01           # learning rate
nn = [2, 16, 8, 1]  # número de neuronas por capa.


# Creamos el objeto que contendrá a nuestra red neuronal, como
# secuencia de capas.
model = kr.Sequential()

# Añadimos la capa 1
l1 = model.add(kr.layers.Dense(nn[1], activation='relu'))

# Añadimos la capa 2
l2 = model.add(kr.layers.Dense(nn[2], activation='relu'))

# Añadimos la capa 3
l3 = model.add(kr.layers.Dense(nn[3], activation='sigmoid'))

# Compilamos el modelo, definiendo la función de coste y el optimizador.
model.compile(loss='mse', optimizer=kr.optimizers.SGD(lr=0.05), metrics=['acc'])

# Y entrenamos al modelo. Los callbacks 
model.fit(X, Y, epochs=100)