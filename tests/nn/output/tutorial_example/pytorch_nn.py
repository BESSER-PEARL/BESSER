import torch.nn as nn
import torch
from torchvision import datasets, transforms 
from besser.generators.pytorch.utils import compute_mean_std
from sklearn.metrics import classification_report 


# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.relu_activ = nn.ReLU()
        self.l2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.l4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.l6 = nn.Flatten(start_dim=1, end_dim=-1)
        self.l7 = nn.Linear(in_features=1024, out_features=64)
        self.l8 = nn.Linear(in_features=64, out_features=10)
    
    def forward(self, x): 
        x = self.l1(x)
        x = self.relu_activ(x) 
        x = self.l2(x) 
        x = self.l3(x)
        x = self.relu_activ(x) 
        x = self.l4(x) 
        x = self.l5(x)
        x = self.relu_activ(x) 
        x = self.l6(x) 
        x = self.l7(x)
        x = self.relu_activ(x) 
        x = self.l8(x)
        return x


# Dataset preparation
IMAGE_SIZE = (32, 32)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), 
	transforms.ToTensor()
    ])


# Load the training dataset
# Directory structure: root/class1/img1.jpg, root/class1/img2.jpg, root/class2/img1.jpg, ...
train_dataset = datasets.ImageFolder(root=r"dataset\cifar10\train", transform=transform)

# Load the testing dataset that is in a similar directory structure
test_dataset = datasets.ImageFolder(root=r"dataset\cifar10\test", transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Define the network, loss function, and optimiser
my_model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(my_model.parameters(), lr=0.001)

# Train the neural network
print('##### Training the model')
for epoch in range(10):
    # Initialize the running loss for the current epoch
    running_loss = 0.0
    total_loss = 0.0
    # Iterate over mini-batches of training data
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Zero the gradients to prepare for backward pass
        optimiser.zero_grad()
        outputs = my_model(inputs)
        # Compute the loss
        loss = criterion(outputs, labels)
        loss.backward()
        # Update model parameters based on computed gradients
        optimiser.step()
        running_loss += loss.item()
        total_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    print('[%d] overall loss for epoch: %.3f' % (epoch + 1, total_loss / len(train_loader)))
    
print('Training finished')

# Evaluate the neural network
print('##### Evaluating the model')
# Disable gradient calculation during inference
with torch.no_grad():
    # Initialize lists to store predicted and true labels
    predicted_labels = []
    true_labels = []
    test_loss = 0.0
    for data in test_loader:
        # Extract inputs and labels from the data batch
        inputs, labels = data
        true_labels.extend(labels)
        # Forward pass
        outputs = my_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted)
        test_loss += criterion(outputs, labels).item()

average_loss = test_loss / len(test_loader)
print(f"Test Loss: {average_loss:.3f}")

# Calculate the metrics
metrics = ['f1-score']
report = classification_report(true_labels, predicted_labels, output_dict=True)
for metric in metrics:
    metric_list = []
    for class_label in report.keys():
        if class_label != 'macro avg' and class_label != 'weighted avg' and class_label != 'accuracy':
            print(f"{metric.capitalize()} for class {class_label}:", report[class_label][metric])
            metric_list.append(report[class_label][metric])
    print(f"Average {metric.capitalize()}: {(sum(metric_list) / len(metric_list)):.2f}")
    print(f"Accuracy: {report['accuracy']}")