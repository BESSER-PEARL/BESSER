import torch.nn as nn
import torch

# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Embedding(num_embeddings=5000, embedding_dim=50)
        self.l2 = nn.Dropout(p=0.5)
        self.l3 = nn.Conv1d(50, 200, kernel_size=4, stride=1, padding=0)
        self.relu_activ = nn.ReLU()
        self.l4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.l5 = nn.Conv1d(50, 200, kernel_size=5, stride=1, padding=0)
        self.l6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.l7 = nn.Dropout(p=0.15)
        self.l8 = nn.GRU(400, 100, bidirectional=False, dropout=0.0, batch_first=True)
        self.l9 = nn.Linear(in_features=100, out_features=400)
        self.l10 = nn.Dropout(p=0.1)
        self.l11 = nn.Linear(in_features=400, out_features=1)
        self.sigmoid_activ = nn.Sigmoid()
    
    def forward(self, x): 
        x = self.l1(x) 
        x = self.l2(x)
        x = x.permute(0, 2, 1) 
        x_1 = self.l3(x)
        x_1 = self.relu_activ(x_1) 
        x_1 = self.l4(x_1) 
        x_2 = self.l5(x)
        x_2 = self.relu_activ(x_2) 
        x_2 = self.l6(x_2)
        x_2 = torch.cat((x_1, x_2), dim=1) 
        x_2 = self.l7(x_2)
        x_2 = x_2.permute(0, 2, 1)
        x_2, _ = self.l8(x_2)
        x_2 = x_2[:, -1, :] 
        x_2 = self.l9(x_2)
        x_2 = self.relu_activ(x_2) 
        x_2 = self.l10(x_2) 
        x_2 = self.l11(x_2)
        x_2 = self.sigmoid_activ(x_2)
        return x_2

