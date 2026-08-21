"""PyTorch code generated based on BUML."""
import torch

from torch import nn
import torch.nn.functional as F

 

# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Embedding(num_embeddings=5000, embedding_dim=50)
        self.l2 = nn.Dropout(p=0.5)
        self.l3 = nn.Conv1d(in_channels=50, out_channels=200, kernel_size=4, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.actv_func_relu = nn.ReLU()
        self.l4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.l5 = nn.Conv1d(in_channels=50, out_channels=200, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.l6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.l7 = nn.Dropout(p=0.15)
        self.l8 = nn.GRU(input_size=400, hidden_size=100, bidirectional=False, dropout=0.0, batch_first=True, bias=True)
        self.l9 = nn.Linear(in_features=100, out_features=400, bias=True)
        self.l10 = nn.Dropout(p=0.1)
        self.l11 = nn.Linear(in_features=400, out_features=1, bias=True)
        self.actv_func_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x_1 = self.l3(x)
        x_1 = self.actv_func_relu(x_1)
        x_1 = self.l4(x_1)
        x_2 = self.l5(x)
        x_2 = self.actv_func_relu(x_2)
        x_2 = self.l6(x_2)
        x_3 = torch.cat((x_1, x_2), dim=-1)
        x_3 = self.l7(x_3)
        x_3, _ = self.l8(x_3)
        x_3 = x_3[:, -1, :]
        x_3 = self.l9(x_3)
        x_3 = self.actv_func_relu(x_3)
        x_3 = self.l10(x_3)
        x_3 = self.l11(x_3)
        x_3 = self.actv_func_sigmoid(x_3)
        return x_3
