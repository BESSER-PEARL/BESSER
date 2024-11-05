from besser.BUML.metamodel.nn import NN, LinearLayer, DropoutLayer, \
    Parameters, TrainingDataset, TestDataset
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.tf.tf_code_generator import TFGenerator

nn_model: NN = NN(name="my_model")
nn_model.add_layer(LinearLayer(name="l1", actv_func="relu", in_features=13, 
                               out_features=64))
nn_model.add_layer(LinearLayer(name="l2", actv_func="relu", in_features=64, 
                               out_features=128))
nn_model.add_layer(DropoutLayer(name="l3", rate=0.2))
nn_model.add_layer(LinearLayer(name="l4", in_features=128, out_features=1))

parameters: Parameters = Parameters(batch_size=6, epochs=40, 
                                    learning_rate=0.001, optimiser="adam", 
                                    loss_function="mse", metrics=["mae"])

nn_model.add_parameters(parameters)

train_data = TrainingDataset(name="train_data", 
                             path_data=r"dataset\BostonHousingTrain.csv", 
                             task_type="regression", input_format="csv")
test_data = TestDataset(name="test_data", 
                        path_data=r"dataset\BostonHousingTest.csv")

pytorch_model = PytorchGenerator(model=nn_model, train_data=train_data, 
                                 test_data=test_data, output_dir="output/regression")
pytorch_model.generate()

tf_model = TFGenerator(model=nn_model, train_data=train_data, 
                       test_data=test_data, output_dir="output/regression")
tf_model.generate()