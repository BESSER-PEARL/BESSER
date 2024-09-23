import tensorflow as tf
from tensorflow.keras import layers


from sklearn.metrics import classification_report 

from besser.generators.tf.utils import compute_mean_std

# Define the network architecture
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')
        self.l2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.l3 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')
        self.l4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.l5 = layers.Flatten()
        self.l6 = layers.Dense(units=120, activation='relu')
        self.l7 = layers.Dense(units=84, activation='relu')
        self.l8 = layers.Dense(units=1, activation='sigmoid')
    
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
IMAGE_SIZE = (125, 125)

# Function to load and preprocess images
scale, _, _ = compute_mean_std(r"dataset\images_cats_dogs\train", num_samples=100,
                                target_size=IMAGE_SIZE, resize=False)
def preprocess_image(image, label, scale):
    if scale:
        image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Load dataset (resizes by default)
def load_dataset(directory, mode):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        label_mode="int",
        image_size=IMAGE_SIZE, 
        batch_size=16,
        shuffle=True if mode == 'train' else False,
    )
    # Apply preprocessing
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, scale))
    # Prefetch for performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Load datasets
train_loader = load_dataset(r"dataset\images_cats_dogs\train", "train")
test_loader = load_dataset(r"dataset\images_cats_dogs\test", "test")


# Define the network, loss function, and optimiser
my_model = NeuralNetwork()
criterion = tf.keras.losses.BinaryCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Train the neural network
print('##### Training the model')
for epoch in range(2):
    # Initialize the running loss for the current epoch
    running_loss = 0.0
    total_loss = 0.0
    # Iterate over mini-batches of training data
    for i, (inputs, labels) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            outputs = my_model(inputs, training=True)
            # Convert labels to one-hot encoding
            loss = criterion(labels, outputs)
        # Compute gradients and update model parameters
        gradients = tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
        total_loss += loss.numpy()
        running_loss += loss.numpy()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    print('[%d] overall loss for epoch: %.3f' % (epoch + 1, total_loss / len(train_loader)))
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
    predicted = (outputs.numpy() > 0.5).astype(int)
    predicted_labels.extend(predicted)
    test_loss += criterion(labels, outputs).numpy()


average_loss = test_loss / len(test_loader)
print(f"Test Loss: {average_loss:.3f}")

# Calculate the metrics
metrics = ['f1-score']
report = classification_report(true_labels, predicted_labels, output_dict=True)
for metric in metrics:
    print(f"{metric.capitalize()}:", report['1'][metric])