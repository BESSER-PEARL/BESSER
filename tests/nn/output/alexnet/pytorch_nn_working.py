import torch.nn as nn


# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
        )
        self.p1 = nn.AdaptiveAvgPool2d((6, 6))
        self.f1 = nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )
    
    def forward(self, x): 
        x = self.features(x) 
        x = self.p1(x) 
        x = self.f1(x) 
        x = self.classifier(x)
        return x