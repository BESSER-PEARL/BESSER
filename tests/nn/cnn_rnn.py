"""
BUML code for cnn_rnn model.
https://github.com/ultimate010/crnn/blob/master/mr_cnn_rnn.py
"""
from besser.BUML.metamodel.nn import NN, Conv1D, PoolingLayer, \
    GRULayer, LinearLayer, EmbeddingLayer, TensorOp, DropoutLayer
from besser.generators.nn.pytorch.pytorch_code_generator import (
    PytorchGenerator
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator


nn_model: NN = NN(name="my_model")
nn_model.add_layer(EmbeddingLayer(name="l1", actv_func=None,
                                  num_embeddings=5000, embedding_dim=50))
nn_model.add_layer(DropoutLayer(name="l2", rate=0.5))
nn_model.add_layer(Conv1D(name="l3", actv_func="relu", in_channels=50,
                          out_channels=200, kernel_dim=[4],
                          input_reused=True, name_module_input="l2"))
nn_model.add_layer(PoolingLayer(name="l4", actv_func=None, pooling_type="max",
                                dimension="1D", kernel_dim=[2]))
nn_model.add_layer(Conv1D(name="l5", actv_func="relu", in_channels=50,
                          out_channels=200, kernel_dim=[5], input_reused=True,
                          name_module_input="l2"))
nn_model.add_layer(PoolingLayer(name="l6", actv_func=None, pooling_type="max",
                                dimension="1D", kernel_dim=[2]))
nn_model.add_tensor_op(TensorOp(name="op1", tns_type="concatenate",
                                layers_of_tensors=["l4", "l6"],
                                concatenate_dim=-1))
nn_model.add_layer(DropoutLayer(name="l7", rate=0.15))
nn_model.add_layer(GRULayer(name="l8", actv_func=None, input_size=400,
                            hidden_size=100, batch_first=True,
                            return_type="last"))
nn_model.add_layer(LinearLayer(name="l9", actv_func="relu",
                               in_features=100, out_features=400))
nn_model.add_layer(DropoutLayer(name="l10", rate=0.10))
nn_model.add_layer(LinearLayer(name="l11", actv_func="sigmoid",
                               in_features=400, out_features=1))



pytorch_model = PytorchGenerator(model=nn_model, output_dir="output/cnn_rnn")
pytorch_model.generate()
tf_model = TFGenerator(model=nn_model, output_dir="output/cnn_rnn")
tf_model.generate()
