"""v4 wire-shape coverage for the extended NN attributes.

Layers gained ``bias`` / ``dilation`` / ``groups`` / ``is_layer_call`` /
``input_var`` / ``output_var`` (plus per-kind extras such as ``eps``,
``affine``, ``hx_source``), TensorOps gained the reduce / split / pad /
interpolate / subscript / dropout families, and the NN container carries its
forward signature (``input_var`` / ``return_vars``). In v4 every layer
attribute lives on ``node.data.attributes`` and the container signature on
``node.data``; these tests pin both directions of the converter pair.
"""

import json

from besser.BUML.metamodel.nn import (
    NN,
    BatchNormLayer,
    Conv2D,
    LSTMLayer,
    TensorOp,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
    nn_model_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
    process_nn_diagram,
)


def _node(node_id, node_type, attributes, parent_id="c1"):
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "width": 110,
        "height": 110,
        "parentId": parent_id,
        "data": {"name": attributes.get("name", node_type), "attributes": attributes},
    }


def _next(edge_id, source, target):
    return {"id": edge_id, "type": "NNNext", "source": source, "target": target,
            "data": {"name": "next", "points": []}}


def _diagram(nodes, edges, container_data=None):
    container = {
        "id": "c1", "type": "NNContainer", "position": {"x": 0, "y": 0},
        "width": 800, "height": 200,
        "data": {"name": "net", **(container_data or {})},
    }
    return {"title": "net", "model": {"version": "4.0.0", "type": "NNDiagram",
                                      "nodes": [container, *nodes], "edges": edges}}


def _layers_by_name(nn):
    return {m.name: m for m in nn.modules}


def test_layer_extended_attributes_are_read_from_data_attributes():
    nodes = [
        _node("n1", "Conv2DLayer", {
            "name": "conv", "kernel_dim": "[3, 3]", "out_channels": "8",
            "dilation": "[2, 2]", "groups": "1", "bias": "false",
            "is_layer_call": "true", "output_var": "feat",
        }),
        _node("n2", "LSTMLayer", {
            "name": "rnn", "hidden_size": "16", "bias": "false",
            "hx_source": "feat", "hidden_state_var": "h", "cell_state_var": "c",
            "cell_unused": "true", "input_var": "feat", "output_var": "seq",
        }),
        _node("n3", "BatchNormalizationLayer", {
            "name": "bn", "num_features": "16",
            "batch_normalization.dimension": "1D", "eps": "0.001",
            "momentum": "0.2", "affine": "false", "track_running_stats": "false",
            "input_var": "seq",
        }),
    ]
    nn = process_nn_diagram(_diagram(nodes, [_next("e1", "n1", "n2"), _next("e2", "n2", "n3")]))
    layers = _layers_by_name(nn)

    conv = layers["conv"]
    assert conv.dilation == [2, 2]
    assert conv.bias is False
    assert conv.is_layer_call is True
    assert conv.output_var == "feat"

    lstm = layers["rnn"]
    assert lstm.bias is False
    assert lstm.hx_source == "feat"
    assert lstm.hidden_state_var == "h"
    assert lstm.cell_state_var == "c"
    assert lstm.cell_unused is True
    assert lstm.input_var == "feat"

    bn = layers["bn"]
    assert bn.dimension == "1D"
    assert bn.eps == 0.001
    assert bn.momentum == 0.2
    assert bn.affine is False
    assert bn.track_running_stats is False


def test_container_signature_is_read_from_container_data():
    nodes = [_node("n1", "LinearLayer", {"name": "fc", "out_features": "4"})]
    # return_vars accepted both as the list the backend emits and as the
    # comma-separated string a user may have typed.
    for return_vars in (["y", "z"], "y, z"):
        nn = process_nn_diagram(_diagram(
            nodes, [], container_data={"input_var": "x", "return_vars": return_vars},
        ))
        assert nn.input_var == "x"
        assert nn.return_vars == "y, z"


def test_reference_target_resolves_container_id():
    """The editor stores the referenced container's node id in ``referenceTarget``."""
    sub = {"id": "sub", "type": "NNContainer", "position": {"x": 0, "y": 0},
           "data": {"name": "encoder"}}
    sub_layer = _node("s1", "LinearLayer", {"name": "enc_fc", "out_features": "4"}, parent_id="sub")
    ref = {"id": "r1", "type": "NNReference", "parentId": "c1", "position": {"x": 0, "y": 0},
           "data": {"name": "encoder", "referenceTarget": "sub"}}
    head = _node("n1", "LinearLayer", {"name": "head", "out_features": "2"})
    payload = _diagram([sub, sub_layer, ref, head], [_next("e1", "r1", "n1")])
    nn = process_nn_diagram(payload)
    assert [m.name for m in nn.modules] == ["encoder", "head"]
    assert nn.sub_nns[0].name == "encoder"


def test_tensor_op_extended_families():
    nodes = [
        _node("n1", "LinearLayer", {"name": "fc", "out_features": "4"}),
        _node("n2", "TensorOp", {
            "name": "chunks", "tns_type": "split", "split_dim": "1",
            "split_sizes": "2", "output_vars": "[a, b]", "input_var": "x",
        }),
        _node("n3", "TensorOp", {
            "name": "padded", "tns_type": "pad", "pad_amount": "[[1, 1], [2, 2]]",
            "pad_mode": "constant", "pad_value": "0.5", "input_var": "a",
        }),
        _node("n4", "TensorOp", {
            "name": "picked", "tns_type": "subscript",
            "subscript_indices": json.dumps([{"type": "index", "value": 0},
                                             {"type": "slice", "start": 1}]),
            "layers_of_tensors": "['padded']",
        }),
        _node("n5", "TensorOp", {
            "name": "summed", "tns_type": "binop_add",
            "layers_of_tensors": "['picked', 1.5]",
        }),
    ]
    edges = [_next("e1", "n1", "n2"), _next("e2", "n2", "n3"),
             _next("e3", "n3", "n4"), _next("e4", "n4", "n5")]
    nn = process_nn_diagram(_diagram(nodes, edges))
    ops = _layers_by_name(nn)
    assert ops["chunks"].split_dim == 1
    assert ops["chunks"].split_sizes == 2
    assert ops["chunks"].output_vars == ["a", "b"]
    assert ops["padded"].pad_amount == [[1, 1], [2, 2]]
    assert ops["padded"].pad_value == 0.5
    assert ops["picked"].subscript_indices[0] == {"type": "index", "value": 0}
    assert ops["summed"].layers_of_tensors == ["picked", 1.5]


def test_extended_attributes_round_trip_through_v4():
    nn = NN(name="net", input_var="x", return_vars="out, h")
    nn.add_layer(Conv2D(name="conv", kernel_dim=[3, 3], out_channels=8,
                        dilation=[2, 2], bias=False, output_var="feat"))
    nn.add_layer(LSTMLayer(name="rnn", hidden_size=16, hx_source="feat",
                           input_var="feat", output_var="out", hidden_state_var="h"))
    nn.add_layer(BatchNormLayer(name="bn", num_features=16, dimension="1D",
                                eps=0.001, input_var="out"))
    nn.add_tensor_op(TensorOp(name="flat", tns_type="reshape", reshape_dim=[-1],
                              input_var="out"))

    model = nn_model_to_json(nn)
    assert model["version"] == "4.0.0"
    container = next(n for n in model["nodes"] if n["type"] == "NNContainer")
    assert container["data"]["input_var"] == "x"
    assert container["data"]["return_vars"] == ["out", "h"]
    conv_attrs = next(n for n in model["nodes"] if n["type"] == "Conv2DLayer")["data"]["attributes"]
    assert conv_attrs["dilation"] == "[2, 2]"
    assert conv_attrs["bias"] == "false"
    assert conv_attrs["output_var"] == "feat"
    bn_attrs = next(n for n in model["nodes"] if n["type"] == "BatchNormalizationLayer")["data"]["attributes"]
    assert bn_attrs["batch_normalization.dimension"] == "1D"
    assert bn_attrs["eps"] == "0.001"

    back = process_nn_diagram({"title": "net", "model": model})
    assert back.input_var == "x"
    assert back.return_vars == "out, h"
    layers = _layers_by_name(back)
    assert layers["conv"].dilation == [2, 2]
    assert layers["conv"].bias is False
    assert layers["rnn"].hx_source == "feat"
    assert layers["rnn"].hidden_state_var == "h"
    assert layers["bn"].eps == 0.001
    assert layers["flat"].input_var == "out"
