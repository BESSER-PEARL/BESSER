import pytest

from besser.BUML.metamodel.nn import InputLayer, LinearLayer, EmbeddingLayer, \
    SimpleRNNLayer, Conv1D, Conv2D, Conv3D, PoolingLayer, DropoutLayer, BatchNormLayer, \
    LayerNormLayer, TrainingDataset, TestDataset, Label, Image, Structured, Parameters, NN



# Test general layers
def test_general_layer():
    input_layer: InputLayer = InputLayer(name="input_layer", activation_function="relu", 
                                         input_dim1=125, input_dim2=125, input_dim3=125)
    linear_layer: LinearLayer = LinearLayer(name="linear_layer", activation_function="relu", 
                                            in_features=10, out_features=20)
    embedding_layer: EmbeddingLayer = EmbeddingLayer(name="embedding_layer", activation_function="relu", 
                                                     num_embeddings=10, embedding_dim=20)
    assert input_layer.name == "input_layer"
    assert input_layer.input_dim1 == 125
    assert linear_layer.in_features == 10
    assert embedding_layer.num_embeddings == 10 
    assert embedding_layer.activation_function == "relu"

# Test RNN layers
def test_rnn_layer():
    simple_rnn_layer: SimpleRNNLayer = SimpleRNNLayer(name="simple_rnn_layer", activation_function="relu", 
                                                      input_size=64, hidden_size=32)
    assert simple_rnn_layer.name == "simple_rnn_layer"
    assert simple_rnn_layer.input_size == 64
    assert simple_rnn_layer.hidden_size == 32

#Test conv layers
def test_conv_layer():
    conv1_layer: Conv1D = Conv1D(name="conv1_layer", activation_function="relu", in_channels=3, out_channels=6, 
                                 kernel_height=3, stride_height=2)
    conv2_layer: Conv2D = Conv2D(name="conv2_layer", activation_function="relu", in_channels=3, out_channels=6, 
                                 kernel_height=3, kernel_width=4, stride_height=2, stride_width=3)
    conv3_layer: Conv3D = Conv3D(name="conv3_layer", activation_function="relu", in_channels=3, out_channels=6, 
                                 kernel_height=3, kernel_width=4, kernel_depth=5, stride_height=2, stride_width=3, 
                                 stride_depth=4)
    
    assert conv1_layer.name == "conv1_layer"
    assert conv1_layer.in_channels == 3
    assert conv1_layer.kernel_height == 3
    assert conv1_layer.stride_height == 2
    assert conv2_layer.kernel_width == 4
    assert conv3_layer.stride_depth == 4


#Test pooling layers
def test_pooling_layer():
    pool_avg1_layer: PoolingLayer = PoolingLayer(name="pool_avg1_layer", activation_function="relu",  kernel_height=3, 
                                                 stride_height=2, pooling_type="average", dimension="1D", padding_type="same")
    assert pool_avg1_layer.name == "pool_avg1_layer"
    assert pool_avg1_layer.stride_height == 2


#Test cnn layers padding
def test_cnn_padding():
    with pytest.raises(ValueError) as excinfo:
        conv1_layer: Conv1D = Conv1D(name="conv1_layer", activation_function="relu", in_channels=3, out_channels=6, 
                                     kernel_height=3, stride_height=2, padding_type="my_padding")
    assert ("Invalid padding type" in str(excinfo.value))
 

#Test pooling type
def test_pooling_type():
    with pytest.raises(ValueError) as excinfo:
        pool_avg1_layer: PoolingLayer = PoolingLayer(name="pool_avg1_layer", activation_function="relu",  kernel_height=3, 
                                                     stride_height=2, pooling_type="my_type", dimension="1D")
    assert ("Invalid pooling type" in str(excinfo.value))

#Test pooling dimension
def test_pooling_dimension():
    with pytest.raises(ValueError) as excinfo:
        pool_avg1_layer: PoolingLayer = PoolingLayer(name="pool_avg1_layer", activation_function="relu",  kernel_height=3, 
                                                     stride_height=2, pooling_type="average", dimension="my_dimension")
    assert ("Invalid pooling dimensionality" in str(excinfo.value))


#Test dropout layer
def test_dropout_layer():
    dropout_layer: DropoutLayer = DropoutLayer(name="dropout_layer", rate=0.5)
    assert dropout_layer.rate == 0.5


#Test normalization layer
def test_normalization_layer():
    batch_norm_layer: BatchNormLayer = BatchNormLayer(name="batch_norm_layer", activation_function="relu", 
                                                      num_features=3, dimension="3D")
    layer_norm_layer: LayerNormLayer = LayerNormLayer(name="layer_norm_layer", activation_function="relu", norm_channels=3,
                                                      norm_height=2, norm_width=4)
    assert batch_norm_layer.num_features == 3
    assert layer_norm_layer.norm_channels == 3
    assert layer_norm_layer.norm_height == 2 
    assert layer_norm_layer.norm_width == 4



#Test batch norm dimension
def test_batch_norm_dimension():
    with pytest.raises(ValueError) as excinfo:
        batch_norm_layer: BatchNormLayer = BatchNormLayer(name="batch_norm_layer", activation_function="relu", 
                                                          num_features=3, dimension="my_dimension")
    assert ("Invalid data dimensionality" in str(excinfo.value))


#Test activation function
def test_activation_function():
    with pytest.raises(ValueError) as excinfo:
        conv1_layer: Conv1D = Conv1D(name="conv1_layer", activation_function="my_af", in_channels=3, out_channels=6, 
                                     kernel_height=3, stride_height=2)
    assert ("Invalid value of activation_function" in str(excinfo.value))


#Test dataset
def test_data():
    training_data: TrainingDataset = TrainingDataset(name="training_data", path_data="c/my/path1", 
                                                     task_type="binary", has_images=True)
    test_data: TestDataset = TestDataset(name="test_data", path_data="c/my/path2")
    assert training_data.name == "training_data"
    assert training_data.path_data == "c/my/path1"
    assert training_data.has_images == True 
    assert training_data.task_type == "binary"
    assert test_data.name == "test_data"
    assert test_data.path_data == "c/my/path2"


#Test data task type
def test_data_task_type():
    with pytest.raises(ValueError) as excinfo:
        training_data: TrainingDataset = TrainingDataset(name="training_data", path_data="c/my/path1", 
                                                         task_type="my_task", has_images=True)
    assert ("Invalid value of task_type" in str(excinfo.value))

#Test label 
def test_label():
    my_label: Label = Label(col_name="target", label_name="dog")
    assert my_label.col_name == "target"
    assert my_label.label_name == "dog"

#Test image feature
def test_image():
    my_image: Image = Image(height=125, width=125, channels=125)
    assert my_image.height == 125
    assert my_image.width == 125
    assert my_image.channels == 125

#Test structured feature
def test_structured():
    my_feature: Structured = Structured(name="my_feature")
    assert my_feature.name == "my_feature"

#Test parameters
def test_parameters():
    my_parameters: Parameters = Parameters(batch_size=16, epochs=20, learning_rate=0.01, 
                                           optimizer="sgd", loss_function="crossentropy", 
                                           metrics=["f1-score", "accuracy"], regularization="l2", weight_decay=0.001)
    
    assert my_parameters.batch_size == 16
    assert my_parameters.epochs == 20
    assert my_parameters.learning_rate == 0.01
    assert my_parameters.weight_decay == 0.001
    assert my_parameters.optimizer == "sgd"
    assert my_parameters.loss_function == "crossentropy"
    assert my_parameters.metrics[0] == "f1-score"
    assert my_parameters.regularization == "l2"
    assert len(my_parameters.metrics) == 2


#Test optimizer type
def test_optimizer_type():
    with pytest.raises(ValueError) as excinfo:
        my_parameters: Parameters = Parameters(batch_size=16, epochs=20, learning_rate=0.01, 
                                               optimizer="my_optimizer", loss_function="crossentropy", 
                                               metrics=["f1-score"], regularization="l2", weight_decay=0.001)
    assert ("Invalid value of optimizer" in str(excinfo.value))


#Test loss function type
def test_loss_function_type():
    with pytest.raises(ValueError) as excinfo:
        my_parameters: Parameters = Parameters(batch_size=16, epochs=20, learning_rate=0.01, 
                                               optimizer="sgd", loss_function="my_loss_function", 
                                               metrics=["f1-score"], regularization="l2", weight_decay=0.001)
    assert ("Invalid value of loss_function" in str(excinfo.value))


#Test metrics
def test_metrics():
    with pytest.raises(ValueError) as excinfo:
        my_parameters: Parameters = Parameters(batch_size=16, epochs=20, learning_rate=0.01, 
                                               optimizer="sgd", loss_function="crossentropy", 
                                               metrics=["my_metric"], regularization="l2", weight_decay=0.001)
    assert ("Invalid metric(s) provided" in str(excinfo.value))



#Test regularization
def test_regularization():
    with pytest.raises(ValueError) as excinfo:
        my_parameters: Parameters = Parameters(batch_size=16, epochs=20, learning_rate=0.01, 
                                               optimizer="sgd", loss_function="crossentropy", 
                                               metrics=["f1-score"], regularization="my_reg", weight_decay=0.001)
    assert ("Invalid value of regularization" in str(excinfo.value))



# Test nn model
def test_nn():
    nn_model: NN = NN(name="my_model")
    conv1_layer: Conv1D = Conv1D(name="conv1_layer", activation_function="relu", in_channels=3, out_channels=6, 
                                 kernel_height=3, stride_height=2)
    pool_avg1_layer: PoolingLayer = PoolingLayer(name="pool_avg1_layer", activation_function="relu",  kernel_height=3, 
                                                 stride_height=2, pooling_type="average", dimension="1D", padding_type="same")
    linear_layer: LinearLayer = LinearLayer(name="linear_layer", activation_function="relu", 
                                            in_features=10, out_features=20)
    
    my_parameters: Parameters = Parameters(batch_size=16, epochs=20, learning_rate=0.01, 
                                               optimizer="sgd", loss_function="crossentropy", 
                                               metrics=["f1-score"], regularization="l2", weight_decay=0.001)
    nn_model.add_layer(conv1_layer)
    nn_model.add_layer(pool_avg1_layer)
    nn_model.add_layer(linear_layer)
    nn_model.add_parameters(my_parameters)

    assert nn_model.name == "my_model"
    assert len(nn_model.layers) == 3
    assert nn_model.layers[0].name == "conv1_layer"
    assert nn_model.parameters.epochs == 20
