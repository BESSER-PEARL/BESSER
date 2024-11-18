"""TensorFlow code generated based on BUML."""

import tensorflow as tf
from keras import layers





# Define the network architecture
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Embedding(input_dim=10000, output_dim=326)
        self.l2 = layers.Bidirectional(layers.LSTM(units=40, activation=None, dropout=0.5, return_sequences=True))
        self.l3 = layers.Dropout(rate=0.2)
        self.l4 = layers.LSTM(units=40, activation=None, dropout=0.2)
        self.l5 = layers.Dense(units=40, activation='relu')
        self.l6 = layers.Dense(units=2, activation='softmax')

        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x

    

