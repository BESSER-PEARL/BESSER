"""TensorFlow code generated based on BUML."""

import tensorflow as tf
from keras import layers


from datetime import datetime
from sklearn.metrics import classification_report 

from besser.generators.nn.utils_nn import compute_mean_std


# Define the network architecture
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.l2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.l3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.l4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.l5 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.l6 = layers.Flatten()
        self.l7 = layers.Dense(units=64, activation='relu')
        self.l8 = layers.Dense(units=10, activation=None)

        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        return x

    


# Dataset preparation
IMAGE_SIZE = (32, 32)

# Function to load and preprocess images
scale, _, _ = compute_mean_std("dataset/cifar10/train", num_samples=100,
                               target_size=IMAGE_SIZE)
def preprocess_image(image, label, to_scale):
    if to_scale:
        image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Load dataset (resizes by default)
def load_dataset(directory, mode, image_size):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        label_mode="int",
        image_size=image_size,
        batch_size=32,
        shuffle=True if mode == 'train' else False,
    )
    # Apply preprocessing
    dataset = dataset.map(
        lambda image, label: preprocess_image(image, label, scale))
    # Prefetch for performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Load datasets
train_loader = load_dataset("dataset/cifar10/train", "train", IMAGE_SIZE)
test_loader = load_dataset("dataset/cifar10/test", "test", IMAGE_SIZE)


# Define the network, loss function, and optimizer
my_model = NeuralNetwork()
criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Train the neural network
print('##### Training the model')
for epoch in range(10):
    # Initialize the running loss for the current epoch
    running_loss = 0.0
    total_loss = 0.0
    # Iterate over mini-batches of training data
    for i, (inputs, labels) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            outputs = my_model(inputs, training=True)
            # Convert labels to one-hot encoding
            if labels.shape.rank > 1 and labels.shape[-1] == 1:
                labels = tf.squeeze(labels, axis=-1)
            labels = tf.cast(labels, dtype=tf.int32)
            labels = tf.one_hot(labels, depth=10
    )
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
    predicted = tf.argmax(outputs, axis=-1).numpy()
    if labels.shape.rank > 1 and labels.shape[-1] == 1:
        labels = tf.squeeze(labels, axis=-1)
    labels = tf.cast(labels, dtype=tf.int32)
    labels = tf.one_hot(labels, depth=10
    )
    predicted_labels.extend(predicted)
    test_loss += criterion(labels, outputs).numpy()


average_loss = test_loss / len(test_loader)
print(f"Test Loss: {average_loss:.3f}")

# Calculate the metrics
metrics = ['f1-score']
report = classification_report(true_labels, predicted_labels,
                               output_dict=True)
for metric in metrics:
    metric_list = []
    for class_label in report.keys():
        if class_label not in ('macro avg', 'weighted avg', 'accuracy'):
            print(f"{metric.capitalize()} for class {class_label}:",
                  report[class_label][metric])
            metric_list.append(report[class_label][metric])
    metric_value = sum(metric_list) / len(metric_list)
    print(f"Average {metric.capitalize()}: {metric_value:.2f}")
    print(f"Accuracy: {report['accuracy']}")


# Save the neural network
print('##### Saving the model')
my_model.save(f"my_model_{datetime.now}")
print("The model is saved successfully")
