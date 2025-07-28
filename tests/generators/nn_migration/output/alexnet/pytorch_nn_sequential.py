"""PyTorch code generated based on BUML."""


from besser.generators.nn.utils_nn import Permute

from torch import nn

 


# Define the network architecture

features = nn.Sequential(
    Permute(dims=[0, 3, 1, 2]),
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=(4, 4), padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=(1, 1), padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
    nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
    Permute(dims=[0, 2, 3, 1]),
)
classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features=9216, out_features=4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=4096, out_features=4096),
    nn.ReLU(),
    nn.Linear(in_features=4096, out_features=1000),
)


my_model = nn.Sequential(
    features,
    Permute(dims=[0, 3, 1, 2]),
    nn.AdaptiveAvgPool2d(output_size=(6, 6)),
    Permute(dims=[0, 2, 3, 1]),
    nn.Flatten(start_dim=1, end_dim=-1),
    classifier,
)




