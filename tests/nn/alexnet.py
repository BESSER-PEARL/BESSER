"""
BUML code for Alexnet.
Paper: A survey of the recent architectures of deep convolutional 
neural networks.
https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html
"""

from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    FlattenLayer, LinearLayer, DropoutLayer
from besser.generators.nn.pytorch.pytorch_code_generator import (
    PytorchGenerator
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator


features: NN = NN(name="features")
features.add_layer(Conv2D(name="f1", actv_func="relu", in_channels=3,
                          out_channels=64, kernel_dim=[11, 11],
                          stride_dim=[4, 4], padding_amount=2))
features.add_layer(PoolingLayer(name="f2", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[3, 3],
                                stride_dim=[2, 2]))
features.add_layer(Conv2D(name="f3", actv_func="relu", in_channels=64,
                          out_channels=192, kernel_dim=[5, 5],
                          padding_amount=2))
features.add_layer(PoolingLayer(name="f4", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[3, 3],
                                stride_dim=[2, 2]))
features.add_layer(Conv2D(name="f5", actv_func="relu", in_channels=192,
                          out_channels=384, kernel_dim=[3, 3],
                          padding_amount=1))
features.add_layer(Conv2D(name="f6", actv_func="relu", in_channels=384,
                          out_channels=256, kernel_dim=[3, 3],
                          padding_amount=1))
features.add_layer(Conv2D(name="f7", actv_func="relu", in_channels=256,
                          out_channels=256, kernel_dim=[3, 3],
                          padding_amount=1))
features.add_layer(PoolingLayer(name="f8", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[3, 3],
                                stride_dim=[2, 2]))



classifier: NN = NN(name="classifier")
classifier.add_layer(DropoutLayer(name="c1", rate=0.5))
classifier.add_layer(LinearLayer(name="c2", actv_func="relu",
                                 in_features=256*6*6, out_features=4096))
classifier.add_layer(DropoutLayer(name="c3", rate=0.5))
classifier.add_layer(LinearLayer(name="c4", actv_func="relu",
                                 in_features=4096, out_features=4096))
classifier.add_layer(LinearLayer(name="c5", actv_func=None, in_features=4096,
                                 out_features=1000))


alexnet: NN = NN(name="alexnet")
alexnet.add_sub_nn(features)
alexnet.add_layer(PoolingLayer(name="p1", actv_func=None,
                               pooling_type="adaptive_average",
                               dimension="2D", output_dim=[6, 6]))
alexnet.add_layer(FlattenLayer(name="f1", actv_func=None, start_dim=1))
alexnet.add_sub_nn(classifier)



pytorch_model = PytorchGenerator(model=alexnet, output_dir="output/alexnet")
pytorch_model.generate()
tf_model = TFGenerator(model=alexnet, output_dir="output/alexnet")
tf_model.generate()
