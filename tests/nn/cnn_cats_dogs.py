from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    FlattenLayer, LinearLayer, Parameters, Label, Image, TrainingDataset, \
    TestDataset
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.tf.tf_code_generator import TFGenerator

nn_model: NN = NN(name="my_model")
nn_model.add_layer(Conv2D(name="l1", actv_func="relu", in_channels=3, 
                          out_channels=6, kernel_dim=[5, 5], 
                          stride_dim=[1, 1], padding_amount=0))
nn_model.add_layer(PoolingLayer(name="l2", pooling_type="average", 
                                dimension="2D", kernel_dim=[2, 2], 
                                stride_dim=[2, 2]))

nn_model.add_layer(Conv2D(name="l3", actv_func="relu", in_channels=6,
                          out_channels=16, kernel_dim=[5, 5],
                          stride_dim=[1, 1], padding_amount=0))
nn_model.add_layer(PoolingLayer(name="l4", pooling_type="average", 
                                dimension="2D", kernel_dim=[2, 2], 
                                stride_dim=[2, 2]))

nn_model.add_layer(FlattenLayer(name="l5"))
nn_model.add_layer(LinearLayer(name="l6", actv_func="relu", in_features=12544,
                               out_features=120))
nn_model.add_layer(LinearLayer(name="l7", actv_func="relu", in_features=120, 
                               out_features=84))
nn_model.add_layer(LinearLayer(name="l8", actv_func="sigmoid", in_features=84, 
                               out_features=1))

parameters: Parameters = Parameters(batch_size=16, epochs=2, 
                                    learning_rate=0.01, optimiser="adam", 
                                    metrics=["f1-score"], 
                                    loss_function="binary_crossentropy")

nn_model.add_parameters(parameters)

label1 = Label(col_name="target", label_name="cat")
label2 = Label(col_name="target", label_name="dog")

image = Image(shape=[125, 125, 3])

train_data = TrainingDataset(name="train_data", 
                             path_data=r"dataset\images_cats_dogs\train", 
                             task_type="binary", input_format="images", 
                             image=image, labels={label1, label2})
test_data = TestDataset(name="test_data", 
                        path_data=r"dataset\images_cats_dogs\test")

pytorch_model = PytorchGenerator(model=nn_model, train_data=train_data, 
                                 test_data=test_data, output_dir="output/cnn_cats")
pytorch_model.generate() 

tf_model = TFGenerator(model=nn_model, train_data=train_data, 
                       test_data=test_data, output_dir="output/cnn_cats")
tf_model.generate() 
