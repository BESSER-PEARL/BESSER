import os
import pytest

from besser.BUML.metamodel.nn import LinearLayer, FlattenLayer, Conv2D, \
    PoolingLayer, Dataset, Image, Configuration, NN

from besser.generators.nn.pytorch.pytorch_code_generator import (
    PytorchGenerator
)

def run_tests():
    """Run tests"""
    pytest.main([__file__])


def nn_buml():
    nn_model: NN = NN(name="my_model")
    nn_model.add_layer(Conv2D(
        name="l1", actv_func="relu", in_channels=3, out_channels=32,
        kernel_dim=[3, 3], permute_in=True
    ))
    nn_model.add_layer(PoolingLayer(
        name="l2", pooling_type="max", dimension="2D", kernel_dim=[2, 2]
    ))

    nn_model.add_layer(Conv2D(
        name="l3", actv_func="relu", in_channels=32, out_channels=64,
        kernel_dim=[3, 3]
    ))
    nn_model.add_layer(PoolingLayer(
        name="l4", pooling_type="max", dimension="2D", kernel_dim=[2, 2]
    ))
    nn_model.add_layer(Conv2D(
        name="l5", actv_func="relu", in_channels=64, out_channels=64,
        kernel_dim=[3, 3], permute_out=True
    ))
    nn_model.add_layer(FlattenLayer(name="l6"))
    nn_model.add_layer(LinearLayer(
        name="l7", actv_func="relu", in_features=1024, out_features=64
    ))
    nn_model.add_layer(LinearLayer(
        name="l8", in_features=64, out_features=10
    ))

    configuration: Configuration = Configuration(
        batch_size=32, epochs=10, learning_rate=0.001, optimizer="adam",
        metrics=["f1-score"], loss_function="crossentropy"
    )

    nn_model.add_configuration(configuration)

    image = Image(shape=[32, 32, 3], normalize=False)

    train_data = Dataset(
        name="train_data", path_data="C:/Users/dataset/cifar10/train",
        task_type="multi_class", input_format="images", image=image
    )
    test_data = Dataset(
        name="test_data", path_data="C:/Users/dataset/cifar10/test"
    )

    nn_model.add_train_data(train_data)
    nn_model.add_test_data(test_data)
    return nn_model


# Test nn subclassing
def test_nn_subclassing(tmpdir):
    """Test nn subclassing"""
    
    nn_model = nn_buml()

    # Using tmpdir to create a temporary output directory
    output_dir = tmpdir.mkdir("example")

    pytorch_model = PytorchGenerator(
        model=nn_model, output_dir=str(output_dir), generation_type="subclassing"
    )
    pytorch_model.generate()

    output_file = os.path.join(str(output_dir), 'pytorch_nn_subclassing.py')
    assert os.path.exists(output_file), "The file was not created."

    with open(output_file, 'r', encoding="utf-8") as file:
        content = file.read()

    # Check for expected lines in the file
    assert f"class NeuralNetwork(nn.Module):" in content, "Missing NN class definition in the generated file."
    assert f"{nn_model.name} = NeuralNetwork()" in content, "Missing NN class instantiation in the generated file."
    assert f"root=\"{nn_model.train_data.path_data}\"" in content, "Training data path is incorrect."
    assert f"IMAGE_SIZE = ({nn_model.train_data.image.shape[0]}, {nn_model.train_data.image.shape[1]})" in content, "Image shape is incorrect."
    assert f"batch_size={nn_model.configuration.batch_size}" in content, "Batch size is incorrect."
    assert f"lr={nn_model.configuration.learning_rate}" in content, "Learning rate is incorrect."
    assert f"for epoch in range({nn_model.configuration.epochs})" in content, "Epochs is incorrect."
    assert f"metrics = {nn_model.configuration.metrics}" in content, "Metrics is incorrect."


# Test nn sequential
def test_nn_sequential(tmpdir):
    """Test nn sequential"""
    
    nn_model = nn_buml()

    # Using tmpdir to create a temporary output directory
    output_dir = tmpdir.mkdir("example")

    pytorch_model = PytorchGenerator(
        model=nn_model, output_dir=str(output_dir), generation_type="sequential"
    )
    pytorch_model.generate()

    output_file = os.path.join(str(output_dir), 'pytorch_nn_sequential.py')
    assert os.path.exists(output_file), "The file was not created."

    with open(output_file, 'r', encoding="utf-8") as file:
        content = file.read()

    # Check for expected lines in the file
    assert f"{nn_model.name} = nn.Sequential(" in content, "Missing NN sequential module in the generated file."


if __name__ == "__main__":
    run_tests()
