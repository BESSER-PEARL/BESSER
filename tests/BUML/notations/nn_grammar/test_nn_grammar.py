import os
import pytest
from besser.BUML.metamodel.nn import NN, Label
from besser.BUML.notations.nn import buml_neural_network

@pytest.fixture
def nn_model_data(tmpdir):
    """Fixture to generate NN model using a temporary directory."""
    model_path = os.path.join(os.path.dirname(__file__), "nn.txt")
    output_dir = tmpdir.mkdir("buml_output")  # Create temp output directory
    return buml_neural_network(nn_path=model_path, output_dir=str(output_dir), buml_model_file_name="nn_test")

def test_nn_model(nn_model_data):
    """Test NN model structure."""
    nn_model, _, _ = nn_model_data
    assert len(nn_model.modules) == 4
    assert len(nn_model.layers) == 2
    assert len(nn_model.sub_nns) == 2

def test_sub_nn(nn_model_data):
    """Test NN sub-network structure."""
    nn_model, _, _ = nn_model_data
    for sub_nn in nn_model.sub_nns:
        assert sub_nn.name in ["classifier", "features"]
        if sub_nn.name == "classifier":
            assert len(sub_nn.layers) == 5
        else:
            assert len(sub_nn.layers) == 8

def test_training_ds(nn_model_data):
    """Test training dataset properties."""
    _, train_data, _ = nn_model_data
    assert train_data.name == "train_data"
    assert train_data.image.shape == [224, 224]
    for label in train_data.labels:
        assert isinstance(label, Label)
