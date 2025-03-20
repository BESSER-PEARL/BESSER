"""
BUML code for a regression prediction problem.
The task is to predict house prices using the BostonHousing dataset.
"""

from besser.BUML.metamodel.nn import NN, LinearLayer, DropoutLayer, \
    Configuration, Dataset
from besser.generators.nn.pytorch.pytorch_code_generator import (
    PytorchGenerator
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator

nn_model: NN = NN(name="my_model")
nn_model.add_layer(LinearLayer(name="l1", actv_func="relu", in_features=13,
                               out_features=64))
nn_model.add_layer(LinearLayer(name="l2", actv_func="relu", in_features=64,
                               out_features=128))
nn_model.add_layer(DropoutLayer(name="l3", rate=0.2))
nn_model.add_layer(LinearLayer(name="l4", in_features=128, out_features=1))

parameters: Configuration = Configuration(batch_size=6, epochs=40,
                                    learning_rate=0.001, optimizer="adam",
                                    loss_function="mse", metrics=["mae"])

nn_model.add_configuration(parameters)

train_data = Dataset(name="train_data",
                     path_data="dataset/BostonHousingTrain.csv",
                     task_type="regression", input_format="csv")
test_data = Dataset(name="test_data",
                    path_data="dataset/BostonHousingTest.csv")

nn_model.add_train_data(train_data)
nn_model.add_test_data(test_data)

pytorch_model = PytorchGenerator(model=nn_model,
                                 output_dir="output/regression")
pytorch_model.generate()

tf_model = TFGenerator(model=nn_model, output_dir="output/regression")
tf_model.generate()
