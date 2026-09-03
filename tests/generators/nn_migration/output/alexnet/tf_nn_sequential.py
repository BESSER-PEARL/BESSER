"""PyTorch code generated based on BUML."""

from keras.models import Sequential
import tensorflow as tf
from keras import layers

import tensorflow_addons as tfa



# Define the network architecture

features = Sequential([
    layers.ZeroPadding2D(padding=2),
    layers.Conv2D(filters=64, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
    layers.ZeroPadding2D(padding=2),
    layers.Conv2D(filters=192, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
    layers.ZeroPadding2D(padding=1),
    layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    layers.ZeroPadding2D(padding=1),
    layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    layers.ZeroPadding2D(padding=1),
    layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
])
classifier = Sequential([
    layers.Dropout(rate=0.5),
    layers.Dense(units=4096, activation='relu'),
    layers.Dropout(rate=0.5),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=1000),
])

my_model = Sequential([
    features,
    tfa.layers.AdaptiveAveragePooling2D(output_size=(6, 6)),
    layers.Flatten(),
    classifier,
])



