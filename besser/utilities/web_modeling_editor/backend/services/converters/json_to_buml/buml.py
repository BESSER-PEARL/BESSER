from besser.BUML.metamodel.nn import (
    NN, Configuration, TensorOp,
    Conv1D, Conv2D, Conv3D, PoolingLayer,
    SimpleRNNLayer, LSTMLayer, GRULayer,
    LinearLayer, FlattenLayer, EmbeddingLayer,
    DropoutLayer, LayerNormLayer, BatchNormLayer,
)

# Sub-Network: Neural_Network1
neural_network1 = NN(name='Neural_Network1')
neural_network1_layer_0 = Conv1D(name='conv1d_layer', kernel_dim=[3], out_channels=16, stride_dim=[1])
neural_network1.add_layer(neural_network1_layer_0)
neural_network1_layer_1 = PoolingLayer(name='Pooling_layer', pooling_type='max', dimension='2D', kernel_dim=[2, 2], stride_dim=[2, 2])
neural_network1.add_layer(neural_network1_layer_1)
neural_network1_layer_2 = Conv1D(name='conv1d_layer', kernel_dim=[3], out_channels=16, stride_dim=[1])
neural_network1.add_layer(neural_network1_layer_2)
neural_network1_layer_3 = PoolingLayer(name='Pooling_layer', pooling_type='max', dimension='2D', kernel_dim=[2, 2], stride_dim=[2, 2])
neural_network1.add_layer(neural_network1_layer_3)

# Neural Network: Neural_Network
neural_network = NN(name='Neural_Network')
neural_network.add_sub_nn(neural_network1)

# Layers
layer_0 = LinearLayer(name='linear_layer', out_features=128)
neural_network.add_layer(layer_0)

layer_1 = LinearLayer(name='linear_layer', out_features=128)
neural_network.add_layer(layer_1)

layer_2 = FlattenLayer(name='Flatten_layer')
neural_network.add_layer(layer_2)

# Configuration
config = Configuration(batch_size=32, epochs=10, learning_rate=0.001, optimizer='adam', loss_function='crossentropy', metrics=['accuracy'])
neural_network.add_configuration(config)



######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="New project")
project = Project(
    name="test",
    models=[neural_network],
    owner="User",
    metadata=metadata
)
