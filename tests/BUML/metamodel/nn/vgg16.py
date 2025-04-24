"""
BUML code for VGG16 from the literature.
"""
from besser.BUML.metamodel.nn import NN, Conv2D, PoolingLayer, \
    LinearLayer, DropoutLayer, FlattenLayer
from besser.generators.nn.pytorch.pytorch_code_generator import (
    PytorchGenerator
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator



features: NN = NN(name="features")
features.add_layer(Conv2D(name="f1", actv_func="relu", in_channels=3,
                          out_channels=64, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f2", actv_func="relu", in_channels=64,
                          out_channels=64, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(PoolingLayer(name="f3", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2],
                                stride_dim=[2, 2]))
features.add_layer(Conv2D(name="f4", actv_func="relu", in_channels=64,
                          out_channels=128, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f5", actv_func="relu", in_channels=128,
                          out_channels=128, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(PoolingLayer(name="f6", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2],
                                stride_dim=[2, 2]))
features.add_layer(Conv2D(name="f7", actv_func="relu", in_channels=128,
                          out_channels=256, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f8", actv_func="relu", in_channels=256,
                          out_channels=256, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f9", actv_func="relu", in_channels=256,
                          out_channels=256, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(PoolingLayer(name="f10", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2],
                                stride_dim=[2, 2]))
features.add_layer(Conv2D(name="f11", actv_func="relu", in_channels=256,
                          out_channels=512, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f12", actv_func="relu", in_channels=512,
                          out_channels=512, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f13", actv_func="relu", in_channels=512,
                          out_channels=512, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(PoolingLayer(name="f14", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2],
                                stride_dim=[2, 2]))
features.add_layer(Conv2D(name="f15", actv_func="relu", in_channels=512,
                          out_channels=512, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f16", actv_func="relu", in_channels=512,
                          out_channels=512, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(Conv2D(name="f17", actv_func="relu", in_channels=512,
                          out_channels=512, kernel_dim=[3, 3],
                          stride_dim=[1, 1], padding_amount=1))
features.add_layer(PoolingLayer(name="f18", actv_func=None, pooling_type="max",
                                dimension="2D", kernel_dim=[2, 2],
                                stride_dim=[2, 2]))

classifier: NN = NN(name="classifier")
classifier.add_layer(LinearLayer(name="c1", actv_func="relu",
                                 in_features=25088, out_features=4096))
classifier.add_layer(DropoutLayer(name="c2", rate=0.5))
classifier.add_layer(LinearLayer(name="c3", actv_func="relu",
                                 in_features=4096, out_features=4096))
classifier.add_layer(DropoutLayer(name="c4", rate=0.5))
classifier.add_layer(LinearLayer(name="c5", actv_func=None,
                                 in_features=4096, out_features=1000))


vgg16: NN = NN(name="vgg16")
vgg16.add_sub_nn(features)
vgg16.add_layer(PoolingLayer(name="p1", actv_func=None,
                             pooling_type="adaptive_average", dimension="2D",
                             output_dim=[7, 7]))
vgg16.add_layer(FlattenLayer(name="f1", actv_func=None))
vgg16.add_sub_nn(classifier)

pytorch_model = PytorchGenerator(model=vgg16, output_dir="output/vgg16")
pytorch_model.generate()
tf_model = TFGenerator(model=vgg16, output_dir="output/vgg16")
tf_model.generate()
