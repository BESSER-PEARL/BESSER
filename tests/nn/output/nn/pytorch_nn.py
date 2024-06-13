import torch
import torch.nn as nn
from torchvision import datasets 
from sklearn.metrics import mean_absolute_error
import pandas as pd


import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a specific seed
set_seed(42)




# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=13, out_features=64)
        self.l2 = nn.Linear(in_features=64, out_features=128)
        self.l3 = nn.Dropout(p=0.2)
        self.l4 = nn.Linear(in_features=128, out_features=1)
        self.relu_activ = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu_activ(x)
        x = self.l2(x)
        x = self.relu_activ(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
        
# Dataset preparation
def load_data(csv_file):
    # Load data from CSV file
    data = pd.read_csv(csv_file)
    # Extract features and targets
    features = data.iloc[:, :-1].values.astype("float32")
    targets = data.iloc[:, -1].values.astype("float32")
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features)
    targets_tensor = torch.tensor(targets)
    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(features_tensor, targets_tensor)
    return dataset

# Loading data
train_dataset = load_data(r"dataset\BostonHousingTrain.csv")
test_dataset = load_data(r"dataset\BostonHousingTest.csv")

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=6, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=6, shuffle=False)

# Define the network, loss function, and optimizer
my_model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01, weight_decay=1e-05)

# Train the neural network
print('##### Training the model')
for epoch in range(20):
    # Initialize the running loss for the current epoch
    running_loss = 0.0
    # Iterate over mini-batches of training data
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Zero the gradients to prepare for backward pass
        optimizer.zero_grad()
        outputs = my_model(inputs)
        # Compute the loss
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        # Update model parameters based on computed gradients
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Training finished')

# Evaluate the neural network
correct = 0
total = 0
print('##### Evaluating the model')
# Disable gradient calculation during inference
with torch.no_grad():
    # Initialize lists to store predicted and true labels
    predicted_labels = []
    true_labels = []
    for data in test_loader:
        # Extract inputs and labels from the data batch
        inputs, labels = data
        # Forward pass
        outputs = my_model(inputs)
        predicted = outputs.numpy()
        predicted_labels.extend(predicted)
        true_labels.extend(labels)
        
# Calculate the metrics
metrics = ['mae']
mae = mean_absolute_error(true_labels, predicted_labels)
print("Mean Absolute Error (MAE):", mae)