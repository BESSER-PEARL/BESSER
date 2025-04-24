"""TensorFlow code generated based on BUML."""

import tensorflow as tf
from keras import layers


from datetime import datetime
from sklearn.metrics import mean_absolute_error 
import pandas as pd



# Define the network architecture
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Dense(units=64, activation='relu')
        self.l2 = layers.Dense(units=128, activation='relu')
        self.l3 = layers.Dropout(rate=0.2)
        self.l4 = layers.Dense(units=1, activation=None)

        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

    


# Dataset preparation
def load_dataset(csv_file):
    # Load data from CSV file
    data = pd.read_csv(csv_file)
    # Extract features and targets
    features = data.iloc[:, :-1].values.astype("float32")
    targets = data.iloc[:, -1].values.astype("float32")
    # Convert to TensorFlow tensors
    features_tensor = tf.convert_to_tensor(features)
    targets_tensor = tf.convert_to_tensor(targets)
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((features_tensor, 
                                                  targets_tensor))
    return dataset

# Load training and test data
train_dataset = load_dataset("dataset/BostonHousingTrain.csv")
test_dataset = load_dataset("dataset/BostonHousingTest.csv")

# Create data loaders
def create_data_loader(dataset, mode):
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(6)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Create data loaders
train_loader = create_data_loader(train_dataset, "train")
test_loader = create_data_loader(test_dataset, "test")


# Define the network, loss function, and optimizer
my_model = NeuralNetwork()
criterion = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Train the neural network
print('##### Training the model')
for epoch in range(40):
    # Initialize the running loss for the current epoch
    running_loss = 0.0
    total_loss = 0.0
    # Iterate over mini-batches of training data
    for i, (inputs, labels) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            outputs = my_model(inputs, training=True)
            # Convert labels to one-hot encoding
            
            labels = tf.expand_dims(labels, axis=1)
            loss = criterion(labels, outputs)
        # Compute gradients and update model parameters
        gradients = tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, my_model.trainable_variables))
        total_loss += loss.numpy()
        running_loss += loss.numpy()
        if i % 200 == 199:  # Print every 200 mini-batches
            print(
                f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}"
            )
            running_loss = 0.0
    print(
        f"[{epoch + 1}] overall loss for epoch: "
        f"{total_loss / len(train_loader):.3f}"
    )
    total_loss = 0.0
print('Training finished')

# Evaluate the neural network
print('##### Evaluating the model')
predicted_labels = []
true_labels = []
test_loss = 0.0

for inputs, labels in test_loader:
    outputs = my_model(inputs, training=False)
    true_labels.extend(labels.numpy())
    predicted = outputs.numpy()
    labels = tf.expand_dims(labels, axis=1)
    predicted_labels.extend(predicted)
    test_loss += criterion(labels, outputs).numpy()


average_loss = test_loss / len(test_loader)
print(f"Test Loss: {average_loss:.3f}")

# Calculate the metrics
metrics = ['mae']
mae = mean_absolute_error(true_labels, predicted_labels)
print("Mean Absolute Error (MAE):", mae)

# Save the neural network
print('##### Saving the model')
my_model.save(f"my_model_{datetime.now}")
print("The model is saved successfully")
