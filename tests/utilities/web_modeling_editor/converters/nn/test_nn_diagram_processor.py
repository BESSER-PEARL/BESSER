"""
Tests for the NN diagram processor (web editor JSON to B-UML NN metamodel).
"""

import json
import pathlib
import pytest

from besser.BUML.metamodel.nn import NN, Conv1D, Conv2D, PoolingLayer, FlattenLayer, LinearLayer, DropoutLayer, EmbeddingLayer, GRULayer, LSTMLayer, Dataset, Image
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
    process_nn_diagram,
    topological_sort,
    create_dataset,
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


# ---------------------------------------------------------------------------
# Tutorial example fixture-based tests (with TrainingDataset / TestDataset)
# ---------------------------------------------------------------------------

def _tutorial():
    """Helper: load and process the tutorial_example fixture."""
    return process_nn_diagram(load_fixture("tutorial_example.json"))


def test_tutorial_returns_nn():
    """process_nn_diagram returns an NN instance for the tutorial export."""
    assert isinstance(_tutorial(), NN)


def test_tutorial_name():
    """Main NN is named NeuralNetwork."""
    assert _tutorial().name == "NeuralNetwork"


def test_tutorial_no_sub_nns():
    """Tutorial example has no sub_nns (flat architecture)."""
    assert len(_tutorial().sub_nns) == 0


def test_tutorial_module_count():
    """Tutorial example has 8 modules."""
    assert len(_tutorial().modules) == 8


def test_tutorial_layer_types():
    """Tutorial example contains the expected layer types and counts."""
    types = [type(m).__name__ for m in _tutorial().modules]
    assert types.count("Conv2D") == 3
    assert types.count("PoolingLayer") == 2
    assert types.count("FlattenLayer") == 1
    assert types.count("LinearLayer") == 2


def test_tutorial_configuration():
    """Configuration is parsed and attached to the tutorial model."""
    model = _tutorial()
    assert model.configuration is not None
    assert model.configuration.batch_size == 32
    assert model.configuration.epochs == 10
    assert model.configuration.learning_rate == 0.001
    assert model.configuration.optimizer == "adam"


def test_tutorial_train_dataset():
    """TrainingDataset is parsed with all attributes and attached via add_train_data."""
    train = _tutorial().train_data
    assert isinstance(train, Dataset)
    assert train.name == "train_data"
    assert train.path_data == "dataset/cifar10/train"
    assert train.task_type == "multi_class"
    assert train.input_format == "images"


def test_tutorial_train_dataset_image():
    """Training dataset with input_format=images has an Image with shape and normalize."""
    image = _tutorial().train_data.image
    assert isinstance(image, Image)
    assert image.shape == [32, 32, 3]
    assert image.normalize is False


def test_tutorial_test_dataset():
    """TestDataset with only mandatory attributes is parsed and attached via add_test_data."""
    test = _tutorial().test_data
    assert isinstance(test, Dataset)
    assert test.name == "test_data"
    assert test.path_data == "dataset/cifar10/test"
    assert test.image is None


# ---------------------------------------------------------------------------
# Error-case tests for dataset parsing
# ---------------------------------------------------------------------------

def _dataset_element(attrs):
    """Build a minimal TrainingDataset element + attribute children dict."""
    elements = {}
    attr_ids = []
    for field, value in attrs.items():
        aid = f"attr_{field}"
        elements[aid] = {"attributeName": field, "value": value}
        attr_ids.append(aid)
    elements["ds1"] = {"type": "TrainingDataset", "attributes": attr_ids}
    return elements["ds1"], elements


def test_dataset_missing_name_raises():
    """create_dataset raises ValueError when name attribute is missing."""
    element, elements = _dataset_element({"path_data": "/data"})
    with pytest.raises(ValueError, match="name"):
        create_dataset(element, elements)


def test_dataset_empty_name_raises():
    """create_dataset raises ValueError when name is empty string."""
    element, elements = _dataset_element({"name": "", "path_data": "/data"})
    with pytest.raises(ValueError, match="name"):
        create_dataset(element, elements)


def test_dataset_missing_path_data_raises():
    """create_dataset raises ValueError when path_data is missing."""
    element, elements = _dataset_element({"name": "train"})
    with pytest.raises(ValueError, match="path_data"):
        create_dataset(element, elements)


def test_dataset_empty_path_data_raises():
    """create_dataset raises ValueError when path_data is empty string."""
    element, elements = _dataset_element({"name": "train", "path_data": ""})
    with pytest.raises(ValueError, match="path_data"):
        create_dataset(element, elements)


def test_dataset_invalid_shape_raises():
    """Image shape parsing raises ValueError when the literal is malformed.

    Silently falling back to a default hid user-entry mistakes; the
    processor now surfaces them so the editor can show a validation error.
    """
    element, elements = _dataset_element({
        "name": "train", "path_data": "/d",
        "input_format": "images", "shape": "not-a-list",
    })
    with pytest.raises(ValueError, match="malformed 'shape'"):
        create_dataset(element, elements)


def test_dataset_default_shape_when_empty():
    """Empty/missing shape still falls back to [256, 256] (unchanged)."""
    element, elements = _dataset_element({
        "name": "train", "path_data": "/d",
        "input_format": "images", "shape": "",
    })
    ds = create_dataset(element, elements)
    assert ds.image.shape == [256, 256]


def test_dataset_without_input_format_has_no_image():
    """Dataset with no input_format (or non-images) has no image attached."""
    element, elements = _dataset_element({
        "name": "train", "path_data": "/d", "input_format": "csv",
    })
    ds = create_dataset(element, elements)
    assert ds.image is None


# ---------------------------------------------------------------------------
# Regression tests for PR review fixes
# ---------------------------------------------------------------------------

def _minimal_container_json(extra_elements=None, extra_relationships=None):
    """Build a minimal NNDiagram JSON with one container + one Linear layer."""
    elements = {
        "c1": {
            "id": "c1", "type": "NNContainer", "name": "Net", "owner": None,
            "bounds": {"x": 0, "y": 0, "width": 200, "height": 100},
        },
        "l1": {
            "id": "l1", "type": "LinearLayer", "name": "LinearLayer", "owner": "c1",
            "bounds": {"x": 0, "y": 0, "width": 100, "height": 100},
            "attributes": ["l1a1", "l1a2"], "methods": [],
        },
        "l1a1": {
            "id": "l1a1", "type": "NameAttributeLinear",
            "attributeName": "name", "value": "lin", "owner": "l1",
        },
        "l1a2": {
            "id": "l1a2", "type": "OutFeaturesAttributeLinear",
            "attributeName": "out_features", "value": "10", "owner": "l1",
        },
    }
    if extra_elements:
        elements.update(extra_elements)
    return {
        "title": "Net",
        "model": {
            "type": "NNDiagram",
            "elements": elements,
            "relationships": extra_relationships or {},
        },
    }


def test_project_format_dict_payload_accepted():
    """Legacy single-diagram dict under 'NNDiagram' must be accepted (regression for review #1)."""
    diagram = _minimal_container_json()
    project_payload = {"project": {"diagrams": {"NNDiagram": diagram}}}
    nn = process_nn_diagram(project_payload)
    assert nn.name == "Net"


def test_project_format_list_payload_accepted():
    """List shape under 'NNDiagram' (new format) must be accepted."""
    diagram = _minimal_container_json()
    project_payload = {"project": {"diagrams": {"NNDiagram": [diagram]}}}
    nn = process_nn_diagram(project_payload)
    assert nn.name == "Net"


def test_project_format_empty_list_raises():
    """Empty list surfaces a clear error instead of IndexError."""
    project_payload = {"project": {"diagrams": {"NNDiagram": []}}}
    with pytest.raises(ValueError, match="no NNDiagram entries"):
        process_nn_diagram(project_payload)


def test_unresolved_nnreference_raises():
    """An NNReference pointing to a missing container must raise (regression for review #3)."""
    extra = {
        "ref1": {
            "id": "ref1", "type": "NNReference", "name": "ghost",
            "referencedNN": "ghost", "owner": "c1",
            "bounds": {"x": 0, "y": 0, "width": 100, "height": 100},
        },
    }
    diagram = _minimal_container_json(extra_elements=extra)
    with pytest.raises(ValueError, match="NNReference 'ghost'"):
        process_nn_diagram(diagram)


def test_nnnext_cycle_raises():
    """A cycle in NNNext relationships must raise (regression for review #6)."""
    extra = {
        "l2": {
            "id": "l2", "type": "LinearLayer", "name": "LinearLayer", "owner": "c1",
            "bounds": {"x": 0, "y": 0, "width": 100, "height": 100},
            "attributes": ["l2a1", "l2a2"], "methods": [],
        },
        "l2a1": {
            "id": "l2a1", "type": "NameAttributeLinear",
            "attributeName": "name", "value": "lin2", "owner": "l2",
        },
        "l2a2": {
            "id": "l2a2", "type": "OutFeaturesAttributeLinear",
            "attributeName": "out_features", "value": "5", "owner": "l2",
        },
    }
    rels = {
        "r1": {"id": "r1", "type": "NNNext", "source": {"element": "l1"}, "target": {"element": "l2"}, "name": "next"},
        "r2": {"id": "r2", "type": "NNNext", "source": {"element": "l2"}, "target": {"element": "l1"}, "name": "next"},
    }
    diagram = _minimal_container_json(extra_elements=extra, extra_relationships=rels)
    with pytest.raises(ValueError, match="cycle"):
        process_nn_diagram(diagram)


def test_multiple_training_datasets_raises():
    """Two TrainingDataset elements must raise rather than silently overwrite (review #8)."""
    extra = {
        "ds1": {"id": "ds1", "type": "TrainingDataset", "owner": None, "attributes": ["ds1a1", "ds1a2"]},
        "ds1a1": {"id": "ds1a1", "attributeName": "name", "value": "t1", "owner": "ds1"},
        "ds1a2": {"id": "ds1a2", "attributeName": "path_data", "value": "/a", "owner": "ds1"},
        "ds2": {"id": "ds2", "type": "TrainingDataset", "owner": None, "attributes": ["ds2a1", "ds2a2"]},
        "ds2a1": {"id": "ds2a1", "attributeName": "name", "value": "t2", "owner": "ds2"},
        "ds2a2": {"id": "ds2a2", "attributeName": "path_data", "value": "/b", "owner": "ds2"},
    }
    diagram = _minimal_container_json(extra_elements=extra)
    with pytest.raises(ValueError, match="TrainingDataset"):
        process_nn_diagram(diagram)


def test_multiple_top_level_containers_raises():
    """Two NNContainers that aren't referenced by each other must raise (review #19)."""
    extra = {
        "c2": {
            "id": "c2", "type": "NNContainer", "name": "Other", "owner": None,
            "bounds": {"x": 400, "y": 0, "width": 200, "height": 100},
        },
    }
    diagram = _minimal_container_json(extra_elements=extra)
    with pytest.raises(ValueError, match="top-level NNContainers"):
        process_nn_diagram(diagram)


def test_attribute_prefix_match_no_collision():
    """'NameAttribute' must not match 'NameModuleInputAttributeX' (review #5)."""
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
        get_element_attribute,
    )
    # Intentionally only provide NameModuleInput (a longer-prefix attribute);
    # the shorter 'NameAttribute' key must NOT mistake it for a match.
    element = {"attributes": ["a1"]}
    elements = {
        "a1": {
            "id": "a1", "type": "NameModuleInputAttributeConv2D",
            "attributeName": "name_module_input", "value": "prev_layer",
        },
    }
    assert get_element_attribute(element, 'NameAttribute', elements) is None
    assert (
        get_element_attribute(element, 'NameModuleInputAttribute', elements)
        == 'prev_layer'
    )
