"""PyTorch code generated based on BUML."""



from torch import nn

 


# Define the network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Embedding(num_embeddings=10000, embedding_dim=326)
        self.l2 = nn.LSTM(input_size=326, hidden_size=40, bidirectional=True, dropout=0.5, batch_first=True)
        self.l3 = nn.Dropout(p=0.2)
        self.l4 = nn.LSTM(input_size=80, hidden_size=40, bidirectional=False, dropout=0.2, batch_first=True)
        self.l5 = nn.Linear(in_features=40, out_features=40)
        self.actv_func_relu = nn.ReLU()
        self.l6 = nn.Linear(in_features=40, out_features=2)
        self.actv_func_softmax = nn.Softmax()


    def forward(self, x):
        x = self.l1(x)
        x, _ = self.l2(x)
        x = self.l3(x)
        x, _ = self.l4(x)
        x = x[:, -1, :]
        x = self.l5(x)
        x = self.actv_func_relu(x)
        x = self.l6(x)
        x = self.actv_func_softmax(x)
        return x

