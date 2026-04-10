"""
Tests for the NN diagram processor (web editor JSON to B-UML NN metamodel).
"""

import json
import pathlib
import pytest

from besser.BUML.metamodel.nn import NN, Conv1D, Conv2D, PoolingLayer, FlattenLayer, LinearLayer, DropoutLayer, EmbeddingLayer, GRULayer, LSTMLayer
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
    process_nn_diagram,
    topological_sort,
)


def run_tests():
    """Run tests"""
    pytest.main([__file__])


FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name):
    """Load a JSON fixture file exported from the web editor."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# topological_sort tests
# ---------------------------------------------------------------------------

def test_topological_sort_linear_chain():
    """A -> B -> C must yield [A, B, C]."""
    result = topological_sort({"A", "B", "C"}, {"A": ["B"], "B": ["C"]})
    assert result == ["A", "B", "C"]


def test_topological_sort_deterministic():
    """Multiple zero-in-degree nodes are always sorted alphabetically."""
    result1 = topological_sort({"Z", "A", "M"}, {})
    result2 = topological_sort({"Z", "A", "M"}, {})
    assert result1 == result2
    assert result1 == sorted(result1)


def test_topological_sort_single_node():
    """Single node returns a list with that node."""
    assert topological_sort({"only"}, {}) == ["only"]


def test_topological_sort_includes_all_nodes():
    """All nodes appear in the result even when some are disconnected."""
    result = topological_sort({"A", "B", "C"}, {"A": ["B"]})
    assert set(result) == {"A", "B", "C"}
    assert result.index("A") < result.index("B")

# ---------------------------------------------------------------------------
# AlexNet fixture-based tests
# ---------------------------------------------------------------------------

def _alexnet():
    """Helper: load and process the AlexNet fixture."""
    return process_nn_diagram(load_fixture("alexnet.json"))


def test_alexnet_returns_nn():
    """process_nn_diagram returns an NN instance for the AlexNet export."""
    assert isinstance(_alexnet(), NN)


def test_alexnet_name():
    """Main NN is named alexnet."""
    assert _alexnet().name == "alexnet"


def test_alexnet_two_sub_nns():
    """AlexNet has exactly two sub_nns: features and classifier."""
    model = _alexnet()
    assert len(model.sub_nns) == 2
    names = {s.name for s in model.sub_nns}
    assert names == {"features", "classifier"}


def test_alexnet_features_module_count():
    """features sub_nn has 8 modules (5×Conv2D + 3×Pooling)."""
    model = _alexnet()
    features = next(s for s in model.sub_nns if s.name == "features")
    assert len(features.modules) == 8


def test_alexnet_features_layer_types():
    """features contains Conv2D and PoolingLayer modules."""
    model = _alexnet()
    features = next(s for s in model.sub_nns if s.name == "features")
    types = [type(m) for m in features.modules]
    assert types.count(Conv2D) == 5
    assert types.count(PoolingLayer) == 3


def test_alexnet_classifier_module_count():
    """classifier sub_nn has 5 modules (2×Dropout + 3×Linear)."""
    model = _alexnet()
    classifier = next(s for s in model.sub_nns if s.name == "classifier")
    assert len(classifier.modules) == 5


def test_alexnet_classifier_layer_types():
    """classifier contains DropoutLayer and LinearLayer modules."""
    model = _alexnet()
    classifier = next(s for s in model.sub_nns if s.name == "classifier")
    types = [type(m) for m in classifier.modules]
    assert types.count(DropoutLayer) == 2
    assert types.count(LinearLayer) == 3


def test_alexnet_main_nn_modules():
    """Main NN has 4 modules: sub_nn ref, pooling, flatten, sub_nn ref."""
    model = _alexnet()
    assert len(model.modules) == 4
    type_names = [type(m).__name__ for m in model.modules]
    assert "FlattenLayer" in type_names
    assert "PoolingLayer" in type_names
    assert type_names.count("NN") == 2


def test_alexnet_no_configuration():
    """AlexNet fixture has no training configuration block."""
    assert _alexnet().configuration is None


# ---------------------------------------------------------------------------
# CNN-RNN fixture-based tests
# ---------------------------------------------------------------------------

def _cnn_rnn():
    """Helper: load and process the CNN-RNN fixture."""
    return process_nn_diagram(load_fixture("cnn_rnn.json"))


def test_cnn_rnn_returns_nn():
    """process_nn_diagram returns an NN instance for the CNN-RNN export."""
    assert isinstance(_cnn_rnn(), NN)


def test_cnn_rnn_name():
    """Main NN is named cnn_rnn."""
    assert _cnn_rnn().name == "cnn_rnn"


def test_cnn_rnn_no_sub_nns():
    """CNN-RNN has no sub_nns (flat architecture)."""
    assert len(_cnn_rnn().sub_nns) == 0


def test_cnn_rnn_module_count():
    """CNN-RNN has 12 modules."""
    assert len(_cnn_rnn().modules) == 12


def test_cnn_rnn_layer_types():
    """CNN-RNN contains the expected layer types."""
    types = [type(m).__name__ for m in _cnn_rnn().modules]
    assert types.count("EmbeddingLayer") == 1
    assert types.count("Conv1D") == 2
    assert types.count("PoolingLayer") == 2
    assert types.count("GRULayer") == 1
    assert types.count("LinearLayer") == 2
    assert types.count("DropoutLayer") == 3
    assert types.count("TensorOp") == 1


def test_cnn_rnn_no_configuration():
    """CNN-RNN fixture has no training configuration block."""
    assert _cnn_rnn().configuration is None


# ---------------------------------------------------------------------------
# LSTM fixture-based tests
# ---------------------------------------------------------------------------

def _lstm():
    """Helper: load and process the LSTM fixture."""
    return process_nn_diagram(load_fixture("lstm.json"))


def test_lstm_returns_nn():
    """process_nn_diagram returns an NN instance for the LSTM export."""
    assert isinstance(_lstm(), NN)


def test_lstm_name():
    """Main NN is named lstm."""
    assert _lstm().name == "lstm"


def test_lstm_no_sub_nns():
    """LSTM has no sub_nns (flat architecture)."""
    assert len(_lstm().sub_nns) == 0


def test_lstm_module_count():
    """LSTM has 6 modules."""
    assert len(_lstm().modules) == 6


def test_lstm_layer_types():
    """LSTM contains the expected layer types."""
    types = [type(m).__name__ for m in _lstm().modules]
    assert types.count("EmbeddingLayer") == 1
    assert types.count("LSTMLayer") == 2
    assert types.count("DropoutLayer") == 1
    assert types.count("LinearLayer") == 2


def test_lstm_layer_order():
    """LSTM layers follow the correct topological order."""
    names = [m.name for m in _lstm().modules]
    assert names == ["l1", "l2", "l3", "l4", "l5", "l6"]


def test_lstm_configuration():
    """Configuration is parsed and attached to the LSTM model."""
    model = _lstm()
    assert model.configuration is not None
    assert model.configuration.batch_size == 32
    assert model.configuration.epochs == 10
    assert model.configuration.learning_rate == 0.001
    assert model.configuration.optimizer == "adam"
