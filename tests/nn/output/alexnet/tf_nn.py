import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa  


# Define the network architecture
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.features = Sequential([
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
        self.p1 = tfa.layers.AdaptiveAveragePooling2D(output_size=(6, 6))
        self.f1 = layers.Flatten()

        self.classifier = Sequential([
            layers.Dropout(rate=0.5),
            layers.Dense(units=4096, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(units=4096, activation='relu'),
            layers.Dense(units=1000, activation=None),
        ])
    
    def call(self, x): 
        x = self.features(x)
        x = self.p1(x)
        x = self.f1(x) 
        x = self.classifier(x)
        return x

