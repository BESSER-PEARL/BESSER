"""
Module containing lookup dictionaries for transforming TensorFlow code
to BUML code.
"""

lookup_layers = {
    "Conv1D": "Conv1D", "Conv2D": "Conv2D", "Conv3D": "Conv3D",      
    "MaxPool1D": "PoolingLayer", "MaxPool2D": "PoolingLayer",
    "MaxPool3D": "PoolingLayer", "AveragePooling1D": "PoolingLayer",
    "AveragePooling2D": "PoolingLayer", "AveragePooling3D": "PoolingLayer",
    "AdaptiveAveragePooling1D": "PoolingLayer",
    "GlobalAveragePooling1D": "PoolingLayer",
    "GlobalAveragePooling2D": "PoolingLayer",
    "GlobalAveragePooling3D": "PoolingLayer",
    "AdaptiveAveragePooling2D": "PoolingLayer",
    "AdaptiveAveragePooling3D": "PoolingLayer",
    "AdaptiveMaxPooling1D": "PoolingLayer",
    "AdaptiveMaxPooling2D": "PoolingLayer",
    "AdaptiveMaxPooling3D": "PoolingLayer",
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
    "output_dim": "embedding_dim", "permute_in": "permute_in",
    "permute_out": "permute_out", "padding_amount": "padding_amount",
    "in_channels": "in_channels", "in_features": "in_features",
    "input_size": "input_size",  "bidirectional": "bidirectional"
}

layers_specific_params = {
    "RNN": {"units": "hidden_size"},
    "LSTM": {"units": "hidden_size"},
    "GRU": {"units": "hidden_size"},
    "Dense": {"units": "out_features"}
}

layers_fixed_params = {
    "MaxPool1D": {"pooling_type": "max", "dimension": "1D"},
    "MaxPool2D": {"pooling_type": "max", "dimension": "2D"},
    "MaxPool3D": {"pooling_type": "max", "dimension": "3D"},
    "AveragePooling1D": {"pooling_type": "avg", "dimension": "1D"},
    "AveragePooling2D": {"pooling_type": "avg", "dimension": "2D"},
    "AveragePooling3D": {"pooling_type": "avg", "dimension": "3D"},
    "AdaptiveAveragePooling1D": {"pooling_type": "adaptive_average", 
                                 "dimension": "1D"},
    "AdaptiveAveragePooling2D": {"pooling_type": "adaptive_average", 
                                 "dimension": "2D"},
    "AdaptiveAveragePooling3D": {"pooling_type": "adaptive_average", 
                                 "dimension": "3D"},
    "AdaptiveMaxPooling1D": {"pooling_type": "adaptive_max", 
                             "dimension": "1D"},
    "AdaptiveMaxPooling2D": {"pooling_type": "adaptive_max", 
                             "dimension": "2D"},
    "AdaptiveMaxPooling3D": {"pooling_type": "adaptive_max", 
                             "dimension": "3D"},
    "GlobalAveragePooling1D": {"pooling_type": "global_average", 
                               "dimension": "1D"},
    "GlobalAveragePooling2D": {"pooling_type": "global_average", 
                               "dimension": "2D"},
    "GlobalAveragePooling3D": {"pooling_type": "global_average", 
                               "dimension": "3D"},
}

rnn_layers = ["LSTM", "GRU", "SimpleRNN",
                  "Conv1D", "Conv2D", "Conv3D"]

config_list = ["batch_size", "epochs", "learning_rate",
               "optimizer", "metrics", "loss_function"]

train_param_list = ["name", "path_data", "task_type", "input_format"]
test_param_list = ["name", "path_data"]

lookup_loss_func = {"CategoricalCrossentropy": "crossentropy",
                    "BinaryCrossentropy": "binary_crossentropy",
                    "MeanSquaredError": "mse"}


pos_params = {"Dense": ["units", "activation"],
              "Embedding": ["input_dim", "output_dim"],
              "SimpleRNN": ["units", "activation"],
              "LSTM": ["units", "activation"],
              "GRU": ["units", "activation"],
              "Conv": ["filters", "kernel_size", "strides", "padding"],
              "AveragePooling": ["pool_size", "strides", "padding"],
              "MaxPool": ["pool_size", "strides", "padding"],
              "AdaptiveAveragePooling": ["output_size"],
              "AdaptiveMaxPooling": ["output_size"],
              "Dropout ": ["rate"],
              "LayerNormalization": ["axis"]}
