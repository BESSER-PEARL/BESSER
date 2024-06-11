from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    FlattenLayer, LinearLayer, Parameters, Label, Image, TrainingDataset, TestDataset
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator

nn_model: NN = NN(name="my_model")
nn_model.add_layer(Conv2D(name="l1", actv_func="relu", in_channels=3, out_channels=6, kernel_height=5, 
                          kernel_width=5, stride_height=1, stride_width=1, padding_amount=0))
nn_model.add_layer(PoolingLayer(name="l2", actv_func=None, pooling_type="average", dimension="2D", 
                                kernel_height=2, kernel_width=2, stride_height=2, stride_width=2))

nn_model.add_layer(Conv2D(name="l3", actv_func="relu", in_channels=6, out_channels=16, kernel_height=5, 
                          kernel_width=5, stride_height=1, stride_width=1, padding_amount=0))
nn_model.add_layer(PoolingLayer(name="l4", actv_func=None, pooling_type="average", dimension="2D", 
                                kernel_height=2, kernel_width=2, stride_height=2, stride_width=2))

nn_model.add_layer(FlattenLayer(name="l5", actv_func=None))
nn_model.add_layer(LinearLayer(name="l6", actv_func="relu",  in_features=12544, out_features=120))
nn_model.add_layer(LinearLayer(name="l7", actv_func="relu", in_features=120, out_features=84))
nn_model.add_layer(LinearLayer(name="l8", actv_func="softmax", in_features=84, out_features=2))

parameters: Parameters = Parameters(batch_size=6, epochs=10, learning_rate=0.01, optimizer="adam", metrics=["f1-score"], 
                                    loss_function="binary_crossentropy", regularization="l2", weight_decay=0.00001)

nn_model.add_parameters(parameters)

label1 = Label(col_name="target", label_name="lion")
label2 = Label(col_name="target", label_name="cheetah")

image = Image(height=125, width=125, channels=3)

train_data = TrainingDataset(name="train_data", path_data=r"dataset\images\train", 
                             task_type="binary", has_images=True, features={image}, labels={label1, label2})
test_data = TestDataset(name="test_data", path_data=r"dataset\images\test")

pytorch_model = PytorchGenerator(model=nn_model, train_data=train_data, test_data=test_data, output_dir="output/cnn")
pytorch_model.generate() 
