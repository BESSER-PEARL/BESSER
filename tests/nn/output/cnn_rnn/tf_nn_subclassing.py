"""TensorFlow code generated based on BUML."""

import tensorflow as tf
from keras import layers





# Define the network architecture
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Embedding(input_dim=5000, output_dim=50)
        self.l2 = layers.Dropout(rate=0.5)
        self.l3 = layers.Conv1D(filters=200, kernel_size=4, strides=1, padding='valid', activation='relu')
        self.l4 = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.l5 = layers.Conv1D(filters=200, kernel_size=5, strides=1, padding='valid', activation='relu')
        self.l6 = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.l7 = layers.Dropout(rate=0.15)
        self.l8 = layers.GRU(units=100, activation=None, dropout=0.0)
        self.l9 = layers.Dense(units=400, activation='relu')
        self.l10 = layers.Dropout(rate=0.1)
        self.l11 = layers.Dense(units=1, activation='sigmoid')

        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x_1 = self.l3(x)
        x_1 = self.l4(x_1)
        x_2 = self.l5(x)
        x_2 = self.l6(x_2)
        x_2 = tf.concat([x_1, x_2], axis=-1)
        x_2 = self.l7(x_2)
        x_2 = self.l8(x_2)
        x_2 = self.l9(x_2)
        x_2 = self.l10(x_2)
        x_2 = self.l11(x_2)
        return x_2

    

