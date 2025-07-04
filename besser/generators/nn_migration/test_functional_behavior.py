"""Module to test PyTorch and TensorFlow models equivalence."""

import importlib.util
import os
import warnings
import pytest
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf


# Suppress TensorFlow Addons warning
warnings.filterwarnings("ignore", message="TensorFlow Addons.*")

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def load_nn_from_file(filepath: str):
    """
    Loads the NN model from the provided path.

    Parameters:
        filepath (str): The path of the NN model file.

    Returns:
        The NN model.
    """
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "NeuralNetwork"):
        model = getattr(module, "NeuralNetwork")
    elif hasattr(module, "my_model"):
        model = getattr(module, "my_model")
    else:
        raise AttributeError("Neither 'NeuralNetwork' nor 'my_model' \
                             found in the module.")

    if isinstance(model, type):
        if issubclass(model, (nn.Module, tf.keras.Model)):
            return model()
        else:
            raise TypeError("Found class is not a subclass of \
                            nn.Module or tf.keras.Model.")
    elif isinstance(model, (nn.Sequential, tf.keras.Sequential)):
        return model
    else:
        raise TypeError("Model is neither a subclassed class \
                        nor a Sequential instance.")



@pytest.fixture(scope="session")
def nn_config(pytestconfig):
    """
    Retrieves the path of NN files and the datashape
    from command-line options, loads the TensorFlow and PyTorch models,
    prepares them for inference, and generates input data accordingly.

    Parameters:
        pytestconfig (Config): Pytest's configuration object used to access
                               command-line options --ftensorflow, --fpytorch,
                               and --datashape.

    Returns:
        tf_model: The loaded and built TensorFlow model.
        pt_model: The loaded PyTorch model set to evaluation mode.
        x (torch.Tensor): The generated input data in PyTorch format.
    """
    ftensorflow = pytestconfig.getoption("--ftensorflow")
    fpytorch = pytestconfig.getoption("--fpytorch")
    datashape = pytestconfig.getoption("--datashape")

    tf_model = load_nn_from_file(ftensorflow)
    pt_model = load_nn_from_file(fpytorch)

    tf_model.build(input_shape=(None, *datashape[1:]))
    pt_model.eval()

    x = get_input_data(pt_model, datashape)
    return tf_model, pt_model, x



def get_input_data(pt_model: torch.nn.Module, shape: tuple):
    """
    Generates input data for the given PyTorch model based on 
    its first submodule. If the first submodule is an Embedding 
    layer, integer input data is generated. Otherwise, random 
    floating-point input data is used.

    Parameters:
        pt_model (torch.nn.Module): The PyTorch model to inspect.
        shape (tuple): The shape of the input data to generate.

    Returns:
        torch.Tensor: A tensor containing the generated input data.
    """
    for name, module in pt_model.named_modules():
        if name == "":
            continue
        first_layer = module
        break
    else:
        raise ValueError("No submodules found in the model.")

    if isinstance(first_layer, nn.Embedding):
        x = torch.randint(0, first_layer.num_embeddings, shape,
                          dtype=torch.long)
    else:
        x = torch.rand(shape)
    return x


def test_output_shape(nn_config):
    """
    Tests whether the TensorFlow and PyTorch models produce outputs 
    with the same shape when given the same input data. This helps 
    validate that the two models are structurally consistent after 
    the migration.

    Parameters:
        nn_config (fixture): A pytest fixture that provides
            the TensorFlow model, PyTorch model, and input tensor.

    Raises:
        AssertionError: If the output shapes from the two models do not match.
    """
    tf_model, pt_model, x_torch = nn_config

    x_np = x_torch.numpy()

    if x_torch.dtype == torch.long:
        tf_input = tf.convert_to_tensor(x_np, dtype=tf.int32)
    else:
        tf_input = tf.convert_to_tensor(x_np, dtype=tf.float32)

    tf_out = tf_model(tf_input).numpy()
    pt_out = pt_model(x_torch).detach().numpy()

    assert tf_out.shape == pt_out.shape


def copy_gru(pt_layer, tf_layer, bidirectional=False):
    """
    Copies weights from a PyTorch GRU layer to a corresponding 
    TensorFlow GRU layer. It handles both unidirectional and 
    bidirectional GRUs, converting weights and biases to match 
    TensorFlow's expected format.

    Parameters:
        pt_layer (torch.nn.GRU): The source PyTorch GRU layer.
        tf_layer (tf.keras.layers.GRU or tf.keras.layers.Bidirectional): 
            The target TensorFlow GRU layer.
        bidirectional (bool): Whether the GRU layer is bidirectional.

    Raises:
        ValueError: If the converted weights do not match the expected shapes 
                    of the TensorFlow layer.
    """
    def reorder_gru_weights(w):
        z, r, n = np.split(w, 3, axis=1)
        return np.concatenate([r, z, n], axis=1)

    def reorder_gru_bias(b):
        b_z, b_r, b_n = np.split(b, 3)
        return np.concatenate([b_r, b_z, b_n])

    def extract_and_convert(prefix, tf_sub_layer):
        w_ih = getattr(pt_layer, f"{prefix}weight_ih_l0").detach().numpy().T
        w_hh = getattr(pt_layer, f"{prefix}weight_hh_l0").detach().numpy().T
        b_ih = getattr(pt_layer, f"{prefix}bias_ih_l0").detach().numpy()
        b_hh = getattr(pt_layer, f"{prefix}bias_hh_l0").detach().numpy()

        w_ih = reorder_gru_weights(w_ih)
        w_hh = reorder_gru_weights(w_hh)
        b_ih = reorder_gru_bias(b_ih)
        b_hh = reorder_gru_bias(b_hh)

        reset_after = getattr(tf_sub_layer, 'reset_after', False)
        bias = np.stack([b_ih, b_hh], axis=0) if reset_after else b_ih + b_hh

        new_weights = [w_ih, w_hh, bias]
        tf_shapes = [w.shape for w in tf_sub_layer.get_weights()]
        for i, (target, actual) in enumerate(
            zip(tf_shapes, [w.shape for w in new_weights])
        ):
            if target != actual:
                raise ValueError(f"Mismatch at weight {i}: expected \
                                 {target}, got {actual}")

        return new_weights

    if bidirectional:
        tf_layer.forward_layer.set_weights(
            extract_and_convert("", tf_layer.forward_layer)
        )
        tf_layer.backward_layer.set_weights(
            extract_and_convert("reverse_", tf_layer.backward_layer)
        )
    else:
        tf_layer.set_weights(extract_and_convert("", tf_layer))

    return



def copy_lstm_rnn(pt_layer, tf_layer, bidirectional=False):
    """
    Copies weights from a PyTorch LSTM layer to a corresponding 
    TensorFlow LSTM layer. It handles both unidirectional and 
    bidirectional LSTMs.

    Parameters:
        pt_layer (torch.nn.LSTM): The PyTorch LSTM layer.
        tf_layer (tf.keras.layers.LSTM or tf.keras.layers.Bidirectional): 
            The target TensorFlow LSTM layer.
        bidirectional (bool): Indicates whether the LSTM is bidirectional.
    Returns:
        None.
    """
    def extract_weights(prefix=""):
        w_ih = getattr(pt_layer, f"weight_ih_l0{prefix}").detach().numpy().T
        w_hh = getattr(pt_layer, f"weight_hh_l0{prefix}").detach().numpy().T
        b_ih = getattr(pt_layer, f"bias_ih_l0{prefix}").detach().numpy()
        b_hh = getattr(pt_layer, f"bias_hh_l0{prefix}").detach().numpy()
        return [w_ih, w_hh, b_ih + b_hh]

    if bidirectional:
        tf_layer.forward_layer.set_weights(extract_weights())
        tf_layer.backward_layer.set_weights(extract_weights("_reverse"))
    else:
        tf_layer.set_weights(extract_weights())

    return


def copy_linear(pt_layer, tf_layer):
    """
    Copies weights from a PyTorch Linear layer to a TensorFlow Dense layer.

    Parameters:
        pt_layer (torch.nn.Linear): The PyTorch linear layer.
        tf_layer (tf.keras.layers.Dense): The corresponding TensorFlow 
            dense layer.
    """
    w = pt_layer.weight.detach().numpy().T
    b = pt_layer.bias.detach().numpy() if pt_layer.bias is not None else None
    tf_layer.set_weights([w, b] if b is not None else [w])
    return

def copy_conv(pt_layer, tf_layer, conv_type="conv1d"):
    """
    Copies weights from a PyTorch convolution layer to a TensorFlow
    Conv layer.

    Parameters:
        pt_layer (torch.nn.ConvNd): The PyTorch convolution layer.
        tf_layer (tf.keras.layers.ConvNd): The TensorFlow convolution layer.
        conv_type (str): Type of convolution: 'conv1d', 'conv2d', or 'conv3d'.
    """
    if conv_type == "conv1d":
        w = pt_layer.weight.detach().numpy().transpose(2,1,0)
        b = pt_layer.bias.detach().numpy()
        tf_layer.set_weights([w, b])
    elif conv_type == "conv2d":
        w = pt_layer.weight.detach().numpy().transpose(2,3,1,0)
        b = pt_layer.bias.detach().numpy()
        tf_layer.set_weights([w, b])
    else:
        w = pt_layer.weight.detach().numpy().transpose(2,3,4,1,0)
        b = pt_layer.bias.detach().numpy()
        tf_layer.set_weights([w, b])
    return


def copy_embedding(pt_layer, tf_layer):
    """
    Copies weights from a PyTorch Embedding layer to a TensorFlow
    Embedding layer.

    Parameters:
        pt_layer (torch.nn.Embedding): The PyTorch embedding layer.
        tf_layer (tf.keras.layers.Embedding): The TensorFlow embedding layer.
    """
    w = pt_layer.weight.detach().numpy()
    tf_layer.set_weights([w])
    return


def copy_batch_norm(pt_layer, tf_layer):
    """
    Copies weights from a PyTorch BatchNorm layer
    to a TensorFlow BatchNormalization layer.

    Parameters:
        pt_layer (torch.nn.BatchNormNd): The PyTorch batch normalization
            layer.
        tf_layer (tf.keras.layers.BatchNormalization): The corresponding 
            TensorFlow layer.
    """
    gamma = pt_layer.weight.detach().numpy()
    beta  = pt_layer.bias.detach().numpy()
    mean  = pt_layer.running_mean.detach().numpy()
    var   = pt_layer.running_var.detach().numpy()
    tf_layer.set_weights([gamma, beta, mean, var])
    return

def copy_layernorm(pt_layer, tf_layer):
    """
    Copies weights from a PyTorch LayerNorm to a TensorFlow 
    LayerNormalization layer.

    Parameters:
        pt_layer (torch.nn.LayerNorm): The PyTorch layer normalization layer.
        tf_layer (tf.keras.layers.LayerNormalization): The corresponding 
            TensorFlow layer.
    """
    gamma = pt_layer.weight.detach().numpy()
    beta  = pt_layer.bias.detach().numpy()
    tf_layer.set_weights([gamma, beta])
    return


def copy_weights(pt_layer, tf_layer):
    """
    Dispatches the appropriate weight-copying function based on the type
    of the PyTorch and TensorFlow layers.
    
    Parameters:
        pt_layer (torch.nn.LayerNorm): The PyTorch layer normalization layer.
        tf_layer (tf.keras.layers.LayerNormalization): The corresponding 
            TensorFlow layer.
    Raises:
        NotImplementedError: If the layer mapping is not supported.
    """
    is_bidir = getattr(pt_layer, "bidirectional", False)

    if isinstance(pt_layer, (nn.RNN, nn.LSTM)):
        if is_bidir and isinstance(tf_layer, tf.keras.layers.Bidirectional):
            copy_lstm_rnn(pt_layer, tf_layer, bidirectional=True)
        elif isinstance(tf_layer, (tf.keras.layers.SimpleRNN,
                                   tf.keras.layers.LSTM)):
            copy_lstm_rnn(pt_layer, tf_layer)

    elif isinstance(pt_layer, nn.GRU):
        if is_bidir and isinstance(tf_layer, tf.keras.layers.Bidirectional):
            copy_gru(pt_layer, tf_layer, bidirectional=True)
        elif isinstance(tf_layer, tf.keras.layers.GRU):
            copy_gru(pt_layer, tf_layer)

    elif isinstance(pt_layer, nn.Linear) and \
        isinstance(tf_layer, tf.keras.layers.Dense):
        copy_linear(pt_layer, tf_layer)

    elif isinstance(pt_layer, nn.Conv1d) and \
        isinstance(tf_layer, tf.keras.layers.Conv1D):
        copy_conv(pt_layer, tf_layer, "conv1d")

    elif isinstance(pt_layer, nn.Conv2d) and \
        isinstance(tf_layer, tf.keras.layers.Conv2D):
        copy_conv(pt_layer, tf_layer, "conv2d")

    elif isinstance(pt_layer, nn.Conv3d) and \
        isinstance(tf_layer, tf.keras.layers.Conv3D):
        copy_conv(pt_layer, tf_layer, "conv3d")

    elif isinstance(pt_layer, nn.Embedding) and \
        isinstance(tf_layer, tf.keras.layers.Embedding):
        copy_embedding(pt_layer, tf_layer)

    elif isinstance(pt_layer, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.BatchNorm3d)) and \
         isinstance(tf_layer, tf.keras.layers.BatchNormalization):
        copy_batch_norm(pt_layer, tf_layer)

    elif isinstance(pt_layer, nn.LayerNorm) and \
        isinstance(tf_layer, tf.keras.layers.LayerNormalization):
        copy_layernorm(pt_layer, tf_layer)

    elif isinstance(pt_layer, (nn.Dropout, nn.Flatten)):
        return

    else:
        raise NotImplementedError(
            f"No weight-copy logic for {type(pt_layer).__name__} → \
                {type(tf_layer).__name__}"
        )


def copy_model_weights(tf_model, pt_model, x_torch=False, subnn=False):
    """
    Copies weights from a PyTorch model to a TensorFlow model.
    This function copies weights layer-by-layer and recursively 
    handles PyTorch nn.Sequential submodules.

    Parameters:
        tf_model: TensorFlow model instance to receive weights.
        pt_model: PyTorch model instance providing weights.
        x_torch (torch.Tensor or False): Example input tensor for TF
        model initialization; required if subnn=False.
        subnn (bool): Flag indicating if the function is called recursively
        on a submodule (default False).

    Raises:
        ValueError: If the number of layers with weights in TensorFlow
        and PyTorch models do not match.
    """

    if not subnn:
        dummy_input = tf.convert_to_tensor(x_torch.numpy(), dtype=tf.float32)
        tf_model(dummy_input)

    tf_layers = [l for l in tf_model.layers if l.weights]

    pt_layers = []
    seen_seqs = set()

    for module in pt_model.modules():
        if module is pt_model:
            continue  # Skip the top-level model

        # Skip if it's part of a Sequential that's already been added
        if any(isinstance(p, nn.Sequential) and \
            module in p.children() for p in seen_seqs):
            continue

        if isinstance(module, nn.Sequential):
            if any(p.requires_grad for p in module.parameters()):
                pt_layers.append(module)
                seen_seqs.add(module)
        elif any(p.requires_grad for p in module.parameters()):
            pt_layers.append(module)

    if len(tf_layers) != len(pt_layers):
        #print("tf", tf_layers, "pt", pt_layers)
        raise ValueError(f"Layer count mismatch: TF={len(tf_layers)}, \
                         PT={len(pt_layers)}")

    for tf_layer, pt_layer in zip(tf_layers, pt_layers):
        if isinstance(pt_layer, nn.Sequential):
            copy_model_weights(tf_layer, pt_layer, subnn=True)

        else:
            copy_weights(pt_layer, tf_layer)


def test_outputs_are_close(nn_config):
    """
    Test that outputs from TensorFlow and PyTorch models are numerically
    close after weight copying. It runs both models on the same input,
    and asserts their outputs are close within specified tolerances.

    Parameters:
        nn_config: pytest fixture providing (tf_model, pt_model, input_tensor).

    Raises:
        AssertionError: if outputs differ beyond given tolerances.
    """
    tf_model, pt_model, x_torch = nn_config

    pt_model.eval()
    tf_model.training = False

    copy_model_weights(tf_model, pt_model, x_torch)
    x_np = x_torch.numpy()

    # Convert to proper TensorFlow dtype
    if x_torch.dtype == torch.long:
        tf_input = tf.convert_to_tensor(x_np, dtype=tf.int32)
    else:
        tf_input = tf.convert_to_tensor(x_np, dtype=tf.float32)


    tf_out = tf_model(tf_input, training=False).numpy()
    with torch.no_grad():
        pt_out = pt_model(x_torch).detach().numpy()

    #print("TensorFlow output:", tf_out)
    #print("PyTorch output:", pt_out)
    #print("Difference:", tf_out - pt_out)
    print("Max absolute difference:", np.max(np.abs(tf_out - pt_out)))

    # Assert close outputs
    np.testing.assert_allclose(tf_out, pt_out, rtol=0.000001, atol=0.000001)
