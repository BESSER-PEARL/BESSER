from besser.BUML.metamodel.nn import NN, LinearLayer, DropoutLayer, \
    Parameters, TrainingDataset, TestDataset
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator

nn_model: NN = NN(name="my_model")
nn_model.add_layer(LinearLayer(name="l1", actv_func="relu",  in_features=13, out_features=64))
nn_model.add_layer(LinearLayer(name="l2", actv_func="relu",  in_features=64, out_features=128))
nn_model.add_layer(DropoutLayer(name="l3", rate=0.2))
nn_model.add_layer(LinearLayer(name="l4", actv_func=None, in_features=128, out_features=1))

parameters: Parameters = Parameters(batch_size=6, epochs=20, learning_rate=0.01, optimizer="adam", loss_function="mse", 
                                    metrics=["mae"], regularization="l2", weight_decay=0.00001)

nn_model.add_parameters(parameters)

train_data = TrainingDataset(name="train_data", path_data=r"dataset\BostonHousingTrain.csv", 
                             task_type="regression", has_images=False)
test_data = TestDataset(name="test_data", path_data=r"dataset\BostonHousingTest.csv")

pytorch_model = PytorchGenerator(model=nn_model, train_data=train_data, test_data=test_data, output_dir="output/nn")
pytorch_model.generate()