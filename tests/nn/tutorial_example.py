from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    FlattenLayer, LinearLayer, Parameters, Label, Image, TrainingDataset, \
    TestDataset
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.tf.tf_code_generator import TFGenerator

"https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders"

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

parameters: Parameters = Parameters(batch_size=32, epochs=10, 
                                    learning_rate=0.001, optimiser="adam", 
                                    metrics=["f1-score"], 
                                    loss_function="crossentropy")

nn_model.add_parameters(parameters)


image = Image(shape=[32, 32, 3], normalize=False)

train_data = TrainingDataset(name="train_data", 
                             path_data=r"dataset\cifar10\train", 
                             task_type="multi_class", input_format="images", 
                             image=image)
test_data = TestDataset(name="test_data", 
                        path_data=r"dataset\cifar10\test")

pytorch_model = PytorchGenerator(model=nn_model, train_data=train_data, 
                                 test_data=test_data, output_dir="output/tutorial_example")
pytorch_model.generate() 
tf_model = TFGenerator(model=nn_model, train_data=train_data, 
                            test_data=test_data, output_dir="output/tutorial_example")
tf_model.generate() 