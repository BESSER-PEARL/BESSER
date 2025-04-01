"""
Module to test the NN metamodel
"""

import pytest

from besser.BUML.metamodel.nn import TensorOp, LinearLayer, EmbeddingLayer, \
    FlattenLayer, SimpleRNNLayer, Conv1D, Conv2D, Conv3D, PoolingLayer, \
    DropoutLayer, BatchNormLayer, LayerNormLayer, Dataset, Label, Image, \
    Structured, Configuration, NN


def run_tests():
    """Run tests"""
    pytest.main([__file__])

# Test tensorops
def test_tensorop():
    """Test tensorop"""
    l1: LinearLayer = LinearLayer(name="l1", actv_func="relu",
                                  in_features=10, out_features=20)
    l2: LinearLayer = LinearLayer(name="l2", actv_func="relu",
                                  in_features=10, out_features=20)
    reshape: TensorOp = TensorOp(name="reshape", tns_type="reshape",
                                 reshape_dim=[2, 2])
    permute: TensorOp = TensorOp(name="permute", tns_type="permute",
                                 permute_dim=[0, 2, 1])
    concatenate: TensorOp = TensorOp(name="concatenate",
                                     tns_type="concatenate",
                                     layers_of_tensors=[l1, l2],
                                     concatenate_dim=1)
    transpose: TensorOp = TensorOp(name="transpose", tns_type="transpose",
                                   transpose_dim=[0, 1])

    assert reshape.tns_type == 'reshape'
    assert reshape.reshape_dim == [2, 2]
    assert permute.permute_dim == [0, 2, 1]
    assert concatenate.layers_of_tensors == [l1, l2]
    assert transpose.transpose_dim == [0, 1]


# Test general layers
def test_general_layer():
    """Test general layer"""
    linear_layer: LinearLayer = LinearLayer(name="linear_layer",
                                            actv_func="relu",
                                            in_features=10, out_features=20,
                                            input_reused=True)
    flatten_layer: FlattenLayer = FlattenLayer(name="flatten_layer",
                                               actv_func="relu",
                                               start_dim=1, end_dim=-1)
    embedding_layer: EmbeddingLayer = EmbeddingLayer(
        name="embedding_layer", actv_func="relu", num_embeddings=10,
        embedding_dim=20, name_module_input="linear_layer"
    )

    assert linear_layer.name == "linear_layer"
    assert linear_layer.in_features == 10
    assert linear_layer.input_reused is True
    assert flatten_layer.end_dim == -1
    assert embedding_layer.num_embeddings == 10
    assert embedding_layer.actv_func == "relu"
    assert embedding_layer.name_module_input == "linear_layer"

# Test RNN layers
def test_rnn_layer():
    """Test rnn layer"""
    simple_rnn_layer: SimpleRNNLayer = SimpleRNNLayer(
        name="simple_rnn_layer", input_size=64, hidden_size=32,
        bidirectional=False, dropout=True, batch_first=True,
        return_type="last", input_reused=False
    )

    assert simple_rnn_layer.name == "simple_rnn_layer"
    assert simple_rnn_layer.input_size == 64
    assert simple_rnn_layer.hidden_size == 32
    assert simple_rnn_layer.return_type == "last"
    assert simple_rnn_layer.bidirectional is False
    assert simple_rnn_layer.dropout is True
    assert simple_rnn_layer.input_reused is False

#Test conv layers
def test_conv_layer():
    """Test conv layer"""
    conv1_layer: Conv1D = Conv1D(
        name="conv1_layer", actv_func="relu", in_channels=3, out_channels=6,
        kernel_dim=[3], stride_dim=[2]
    )
    conv2_layer: Conv2D = Conv2D(
        name="conv2_layer", actv_func="relu", in_channels=3, out_channels=6,
        kernel_dim=[3, 4], stride_dim=[2, 3]
    )
    conv3_layer: Conv3D = Conv3D(
        name="conv3_layer", actv_func="relu", in_channels=3, out_channels=6,
        kernel_dim=[3, 4, 5], stride_dim=[2, 3, 4]
    )

    assert conv1_layer.name == "conv1_layer"
    assert conv1_layer.in_channels == 3
    assert conv1_layer.kernel_dim[0] == 3
    assert conv1_layer.stride_dim[0] == 2
    assert conv2_layer.kernel_dim[1] == 4
    assert conv3_layer.stride_dim[2] == 4


#Test pooling layers
def test_pooling_layer():
    """Test pooling layer"""
    pool_avg1_layer: PoolingLayer = PoolingLayer(
        name="pool_avg1_layer", pooling_type="average", padding_type="same",
        stride_dim=[2], actv_func="relu", dimension="1D", kernel_dim=[3]
    )
    pool_adapt_avg_layer: PoolingLayer = PoolingLayer(
        name="pool_adapt_avg_layer", actv_func="relu", dimension="2D",
        pooling_type="adaptive_max", output_dim=[5, 7]
    )
    assert pool_avg1_layer.name == "pool_avg1_layer"
    assert pool_avg1_layer.dimension == "1D"
    assert pool_avg1_layer.kernel_dim[0] == 3
    assert pool_adapt_avg_layer.output_dim[1] == 7

#Test cnn layers padding
def test_cnn_padding():
    """Test cnn layers padding"""
    with pytest.raises(ValueError) as excinfo:
        conv1_layer: Conv1D = Conv1D(
            name="conv1_layer", actv_func="relu", padding_type="my_padding",
            out_channels=6, kernel_dim=[3], stride_dim=[2], in_channels=3
        )
    assert "Invalid padding type" in str(excinfo.value)

#Test pooling type
def test_pooling_type():
    """Test pooling type"""
    with pytest.raises(ValueError) as excinfo:
        pool_avg1_layer: PoolingLayer = PoolingLayer(
            name="pool_avg1_layer", actv_func="relu", kernel_dim=[3],
            stride_dim=[2], pooling_type="my_type", dimension="1D"
        )
    assert "Invalid pooling type" in str(excinfo.value)

#Test pooling dimension
def test_pooling_dimension():
    """Test pooling dimension"""
    with pytest.raises(ValueError) as excinfo:
        pool_avg1_layer: PoolingLayer = PoolingLayer(
            name="pool_avg1_layer", actv_func="relu", kernel_dim=[3],
            stride_dim=[2], pooling_type="average", dimension="my_dimension"
        )
    assert "Invalid pooling dimensionality" in str(excinfo.value)


#Test dropout layer
def test_dropout_layer():
    """Test dropout layer"""
    dropout_layer: DropoutLayer = DropoutLayer(name="dropout_layer", rate=0.5)
    assert dropout_layer.rate == 0.5


#Test normalization layer
def test_normalization_layer():
    """Test normalization layer"""
    batch_norm_layer: BatchNormLayer = BatchNormLayer(
        name="batch_norm_layer", actv_func="relu", num_features=3,
        dimension="3D"
    )
    layer_norm_layer: LayerNormLayer = LayerNormLayer(
        name="layer_norm_layer", actv_func="relu", normalized_shape=[3, 2, 4]
    )
    assert batch_norm_layer.num_features == 3
    assert layer_norm_layer.normalized_shape[0] == 3
    assert layer_norm_layer.normalized_shape[1] == 2
    assert layer_norm_layer.normalized_shape[2] == 4



#Test batch norm dimension
def test_batch_norm_dimension():
    """Test batch norm dimension"""
    with pytest.raises(ValueError) as excinfo:
        batch_norm_layer: BatchNormLayer = BatchNormLayer(
            name="batch_norm_layer", actv_func="relu",
            num_features=3, dimension="my_dimension"
        )
    assert "Invalid data dimensionality" in str(excinfo.value)


#Test activation function
def test_activation_function():
    """Test activation function"""
    conv1_layer: Conv1D = Conv1D(
        name="conv1_layer", actv_func="relu", in_channels=3,
        out_channels=6, kernel_dim=[3], stride_dim=[2]
    )
    assert conv1_layer.actv_func == "relu"


#Test dataset
def test_dataset():
    """Test dataset"""
    my_image = Image(shape=[32, 32, 32], normalize=True)
    training_data: Dataset = Dataset(
        name="training_data", path_data="c/my/path1", image=my_image,
        task_type="binary", input_format="csv"
    )
    test_data: Dataset = Dataset(name="test_data", path_data="c/my/path2")
    assert training_data.name == "training_data"
    assert training_data.path_data == "c/my/path1"
    assert training_data.input_format == "csv"
    assert training_data.task_type == "binary"
    assert training_data.image == my_image
    assert test_data.name == "test_data"
    assert test_data.path_data == "c/my/path2"


#Test data task type
def test_data_task_type():
    """Test data task type"""
    with pytest.raises(ValueError) as excinfo:
        training_data: Dataset = Dataset(
            name="training_data", path_data="c/my/path1",
            task_type="my_task", input_format="csv"
        )
    assert "Invalid value of task_type" in str(excinfo.value)

#Test label
def test_label():
    """Test label"""
    my_label: Label = Label(col_name="target", label_name="dog")
    assert my_label.col_name == "target"
    assert my_label.label_name == "dog"

#Test image feature
def test_image():
    """Test image feature"""
    my_image: Image = Image(shape=[125, 125, 125])
    assert my_image.shape[0] == 125
    assert my_image.shape[1] == 125
    assert my_image.shape[2] == 125

#Test structured feature
def test_structured():
    """Test structured feature"""
    my_feature: Structured = Structured(name="my_feature")
    assert my_feature.name == "my_feature"

#Test parameters
def test_configuration():
    """Test parameters"""
    my_configuration: Configuration = Configuration(
        batch_size=16, epochs=20, learning_rate=0.01, optimizer="sgd",
        loss_function="crossentropy", metrics=["f1-score", "accuracy"],
        weight_decay=0.001, momentum=0.1
    )

    assert my_configuration.batch_size == 16
    assert my_configuration.epochs == 20
    assert my_configuration.learning_rate == 0.01
    assert my_configuration.weight_decay == 0.001
    assert my_configuration.optimizer == "sgd"
    assert my_configuration.loss_function == "crossentropy"
    assert my_configuration.metrics[0] == "f1-score"
    assert my_configuration.momentum == 0.1
    assert len(my_configuration.metrics) == 2


#Test optimizer type
def test_optimizer_type():
    """Test optimizer type"""
    with pytest.raises(ValueError) as excinfo:
        my_configuration: Configuration = Configuration(
            batch_size=16, epochs=20, learning_rate=0.01,
            optimizer="my_optimizer", loss_function="crossentropy",
            metrics=["f1-score"], weight_decay=0.001
        )
    assert "Invalid value of optimizer" in str(excinfo.value)


#Test loss function type
def test_loss_function_type():
    """Test loss function type"""
    with pytest.raises(ValueError) as excinfo:
        my_configuration: Configuration = Configuration(
            batch_size=16, epochs=20, learning_rate=0.01, optimizer="sgd",
            loss_function="my_loss_function", metrics=["f1-score"],
            weight_decay=0.001
        )
    assert "Invalid value of loss_function" in str(excinfo.value)


#Test metrics
def test_metrics():
    """Test metrics"""
    with pytest.raises(ValueError) as excinfo:
        my_configuration: Configuration = Configuration(
            batch_size=16, epochs=20, learning_rate=0.01, optimizer="sgd",
            loss_function="crossentropy", metrics=["my_metric"],
            weight_decay=0.001
        )
    assert "Invalid metric(s) provided" in str(excinfo.value)



# Test nn model
def test_nn():
    """Test nn model"""
    nn_model: NN = NN(name="my_model")
    conv1_layer: Conv1D = Conv1D(
        name="conv1_layer", actv_func="relu", in_channels=3,
        out_channels=6, kernel_dim=[3], stride_dim=[2]
    )
    pool_avg1_layer: PoolingLayer = PoolingLayer(
        name="pool_avg1_layer", actv_func="relu", kernel_dim=[3],
        stride_dim=[2], pooling_type="average", dimension="1D",
        padding_type="same"
    )
    linear_layer: LinearLayer = LinearLayer(
        name="linear_layer", actv_func="relu", in_features=10, out_features=20
    )

    my_configuration: Configuration = Configuration(
        batch_size=16, epochs=20, learning_rate=0.01, optimizer="sgd",
        loss_function="crossentropy", metrics=["f1-score"],
        weight_decay=0.001
    )

    training_data: Dataset = Dataset(
        name="training_data", path_data="c/my/path1",
        task_type="binary", input_format="csv"
    )
    test_data: Dataset = Dataset(name="test_data", path_data="c/my/path2")

    nn_model.add_layer(conv1_layer)
    nn_model.add_layer(pool_avg1_layer)
    nn_model.add_layer(linear_layer)

    nn_model.add_configuration(my_configuration)
    nn_model.add_train_data(training_data)
    nn_model.add_test_data(test_data)


    assert nn_model.name == "my_model"
    assert len(nn_model.layers) == 3
    assert nn_model.layers[0].name == "conv1_layer"
    assert nn_model.configuration.epochs == 20
    assert nn_model.train_data == training_data
    assert nn_model.test_data == test_data

if __name__ == "__main__":
    run_tests()
