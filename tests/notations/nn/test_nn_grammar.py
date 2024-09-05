import os
import shutil
from besser.BUML.metamodel.nn import NN, Label
from besser.BUML.notations.nn import buml_neural_network

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "nn.txt")
nn_model, train_data, test_data = buml_neural_network(nn_path=model_path)
shutil.rmtree("buml")

# Test the classes of the BUML output model
def test_nn_model():
    assert len(nn_model.modules) == 5
    assert nn_model.parameters
    assert len(nn_model.layers) == 2
    assert len(nn_model.modules) == 5
    assert len(nn_model.sub_nns) == 2

def test_parameters():
    assert nn_model.parameters.batch_size == 16
    assert nn_model.parameters.optimiser == "adam"
    assert nn_model.parameters.metrics == ["f1-score"]

def test_sub_nn():
    for sub_nn in nn_model.sub_nns:
        assert sub_nn.name in ["classifier", "features"]
        if sub_nn.name == "classifier":
            assert len(sub_nn.layers) == 2
        else:
            assert len(sub_nn.layers) == 3

def test_trainig_ds():
    assert train_data.name == "train_data"
    assert train_data.image.shape == [125, 125, 3]
    for label in train_data.labels:
        assert isinstance(label, Label)
