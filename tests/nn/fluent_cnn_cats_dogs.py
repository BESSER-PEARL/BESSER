from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    FlattenLayer, LinearLayer, Parameters, Label, Image, TrainingDataset, TestDataset
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator

nn_model: NN = NN(name="my_model") \
               .add_layer(Conv2D(name="l1", actv_func="relu", in_channels=3, out_channels=6, kernel_height=5, 
                                 kernel_width=5, stride_height=1, stride_width=1, padding_amount=0)) \
               .add_layer(PoolingLayer(name="l2", actv_func=None, pooling_type="average", dimension="2D", 
                                       kernel_height=2, kernel_width=2, stride_height=2, stride_width=2)) \
               .add_layer(Conv2D(name="l3", actv_func="relu", in_channels=6, out_channels=16, kernel_height=5, 
                                 kernel_width=5, stride_height=1, stride_width=1, padding_amount=0)) \
               .add_layer(PoolingLayer(name="l4", actv_func=None, pooling_type="average", dimension="2D", 
                                       kernel_height=2, kernel_width=2, stride_height=2, stride_width=2)) \
               .add_layer(FlattenLayer(name="l5", actv_func=None)) \
               .add_layer(LinearLayer(name="l6", actv_func="relu",  in_features=12544, out_features=120)) \
               .add_layer(LinearLayer(name="l7", actv_func="relu", in_features=120, out_features=84)) \
               .add_layer(LinearLayer(name="l8", actv_func="softmax", in_features=84, out_features=2)) \
               .add_parameters(Parameters(batch_size=16, epochs=10, learning_rate=0.01, optimizer="adam", metrics=["f1-score"], 
                                          loss_function="binary_crossentropy", regularization="l2", weight_decay=0.00001))

train_data = TrainingDataset(name="train_data", path_data=r"dataset\images_cats_dogs\train", 
                             task_type="binary", has_images=True) \
                             .add_label(Label(col_name="target", label_name="cat")) \
                             .add_label(Label(col_name="target", label_name="dog")) \
                             .add_image(Image(height=125, width=125, channels=3))
test_data = TestDataset(name="test_data", path_data=r"dataset\images_cats_dogs\test") \

pytorch_model = PytorchGenerator(model=nn_model, train_data=train_data, test_data=test_data, output_dir="output/cnn")
pytorch_model.generate() 
