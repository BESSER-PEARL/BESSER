"""
BUML code for a NN using CNNs and the CIFAR10 dataset 
available in this link:
https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders
"""
from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    FlattenLayer, LinearLayer, Configuration, Image, Dataset
from besser.generators.nn.pytorch.pytorch_code_generator import (
    PytorchGenerator
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator


nn_model: NN = NN(name="my_model")
nn_model.add_layer(Conv2D(name="l1", actv_func="relu", in_channels=3,
                          out_channels=32, kernel_dim=[3, 3]))
nn_model.add_layer(PoolingLayer(name="l2", pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2]))

nn_model.add_layer(Conv2D(name="l3", actv_func="relu", in_channels=32,
                          out_channels=64, kernel_dim=[3, 3]))
nn_model.add_layer(PoolingLayer(name="l4", pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2]))
nn_model.add_layer(Conv2D(name="l5", actv_func="relu", in_channels=64,
                          out_channels=64, kernel_dim=[3, 3]))
nn_model.add_layer(FlattenLayer(name="l6"))
nn_model.add_layer(LinearLayer(name="l7", actv_func="relu", in_features=1024,
                               out_features=64))
nn_model.add_layer(LinearLayer(name="l8", in_features=64,
                               out_features=10))

configuration: Configuration = Configuration(batch_size=32, epochs=10,
                                    learning_rate=0.001, optimizer="adam",
                                    metrics=["f1-score"],
                                    loss_function="crossentropy")

nn_model.add_configuration(configuration)


image = Image(shape=[32, 32, 3], normalize=False)

train_data = Dataset(name="train_data",
                     path_data="dataset/cifar10/train",
                     task_type="multi_class", input_format="images",
                     image=image)
test_data = Dataset(name="test_data",
                    path_data="dataset/cifar10/test")

nn_model.add_train_data(train_data)
nn_model.add_test_data(test_data)

pytorch_model = PytorchGenerator(model=nn_model,
                                 output_dir="output/tutorial_example")
pytorch_model.generate()
tf_model = TFGenerator(model=nn_model,
                       output_dir="output/tutorial_example")
tf_model.generate()
