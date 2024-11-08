lookup_layers = {
    "Conv1D": "Conv1D", "Conv2D": "Conv2D", "Conv3D": "Conv3D",           
    "MaxPool1D": "PoolingLayer", "MaxPool2D": "PoolingLayer", 
    "MaxPool3D": "PoolingLayer", "AveragePooling1D": "PoolingLayer", 
    "AveragePooling2D": "PoolingLayer", "AveragePooling3D": "PoolingLayer",
    "AdaptiveAveragePooling1D": "PoolingLayer", "AdaptiveAveragePooling2D": "PoolingLayer", 
    "AdaptiveAveragePooling3D": "PoolingLayer", "AdaptiveMaxPooling1D": "PoolingLayer", 
    "AdaptiveMaxPooling2D": "PoolingLayer", "AdaptiveMaxPooling3D": "PoolingLayer", 
    "Flatten": "FlattenLayer", "Dense": "LinearLayer", 
    "Embedding": "EmbeddingLayer", "BatchNormalization": "BatchNormLayer", 
    "LayerNormalization": "LayerNormLayer", "Dropout": "DropoutLayer", 
    "RNN": "SimpleRNNLayer", "LSTM": "LSTMLayer", "GRU": "GRULayer",
}

lookup_layers_params = {
    "filters": "out_channels", "kernel_size": "kernel_dim", 
    "strides": "stride_dim", "padding": "padding_type", 
    "pool_size": "kernel_dim", "output_size": "output_dim", 
    "dropout": "dropout", "return_sequences": "return_sequences",
    "return_state": "return_state", "axis": "normalized_shape", 
    "rate": "rate", "input_dim": "num_embeddings", 
    "output_dim": "embedding_dim", "permute_dim": "permute_dim",
    "padding_amount": "padding_amount", "bidirectional": "bidirectional"
}

layers_specific_params = {
    "SimpleRNNLayer": {"units": "hidden_size"},
    "LSTMLayer": {"units": "hidden_size"},
    "GRULayer": {"units": "hidden_size"},
    "LinearLayer": {"units": "out_features"}
}

layers_fixed_params = {
    "MaxPool1D": {"pooling_type": "max", "dimension": "1D"},
    "MaxPool2D": {"pooling_type": "max", "dimension": "2D"},
    "MaxPool3D": {"pooling_type": "max", "dimension": "3D"},
    "AveragePooling1D": {"pooling_type": "avg", "dimension": "1D"},
    "AveragePooling2D": {"pooling_type": "avg", "dimension": "2D"},
    "AveragePooling3D": {"pooling_type": "avg", "dimension": "3D"},
    "AdaptiveAveragePooling1D": {"pooling_type": "adaptive_average", "dimension": "1D"},
    "AdaptiveAveragePooling2D": {"pooling_type": "adaptive_average", "dimension": "2D"},
    "AdaptiveAveragePooling3D": {"pooling_type": "adaptive_average", "dimension": "3D"},
    "AdaptiveMaxPooling1D": {"pooling_type": "adaptive_max", "dimension": "1D"},
    "AdaptiveMaxPooling2D": {"pooling_type": "adaptive_max", "dimension": "2D"},
    "AdaptiveMaxPooling3D": {"pooling_type": "adaptive_max", "dimension": "3D"},
}

rnn_cnn_layers = ["LSTM", "GRU", "SimpleRNN", 
                  "Conv1D", "Conv2D", "Conv3D"]

config_list = ["batch_size", "epochs", "learning_rate", 
               "optimizer", "metrics", "loss_function"]

train_param_list = ["name", "path_data", "task_type", "input_format"]
test_param_list = ["name", "path_data"]

lookup_loss_functions = {"CategoricalCrossentropy": "crossentropy",
                         "BinaryCrossentropy": "binary_crossentropy",
                         "MeanSquaredError": "mse"}