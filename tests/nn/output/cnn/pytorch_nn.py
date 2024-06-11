import torch
import torch.nn as nn
from torchvision import datasets , transforms
from sklearn.metrics import classification_report


# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.l1_activ = nn.ReLU()
        self.l2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l3 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.l3_activ = nn.ReLU()
        self.l4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l5 = nn.Flatten()
        self.l6 =  nn.Linear(in_features=12544, out_features=120)
        self.l6_activ = nn.ReLU()
        self.l7 =  nn.Linear(in_features=120, out_features=84)
        self.l7_activ = nn.ReLU()
        self.l8 =  nn.Linear(in_features=84, out_features=2)
        self.l8_activ = nn.Softmax()

    def forward(self, x):
        x = self.l1(x)
        x = self.l1_activ(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l3_activ(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l6_activ(x)
        x = self.l7(x)
        x = self.l7_activ(x)
        x = self.l8(x)
        x = self.l8_activ(x)
        return x
        
# Dataset preparation
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((125, 125)),  # Resize images
    transforms.ToTensor()          # Convert images to tensors
    ])

# Load the training dataset
# Directory structure: root/class1/img1.jpg, root/class1/img2.jpg, root/class2/img1.jpg, ...
train_dataset = datasets.ImageFolder(root=r"dataset\images\train", transform=transform)

# Load the testing dataset that is in a similar directory structure
test_dataset = datasets.ImageFolder(root=r"dataset\images\test", transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=6, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=6, shuffle=False)

# Define the network, loss function, and optimizer
my_model = NeuralNetwork()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01, weight_decay=1e-05)

# Train the neural network
print('##### Training the model')
for epoch in range(10):
    # Initialize the running loss for the current epoch
    running_loss = 0.0
    # Iterate over mini-batches of training data
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Zero the gradients to prepare for backward pass
        optimizer.zero_grad()
        outputs = my_model(inputs)
        # Compute the loss
        labels = nn.functional.one_hot(labels, num_classes=2).float() 
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
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted)
        true_labels.extend(labels)
        
# Calculate the metrics
metrics = ['f1-score']
report = classification_report(true_labels, predicted_labels, output_dict=True)
for metric in metrics:
    print(f"{metric.capitalize()}:", report['1'][metric])