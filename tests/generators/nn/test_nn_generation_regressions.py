"""Generation-time regression tests for the NN generators.

Everything here exercises code *generation* only, so no torch/tensorflow
import is needed — these run everywhere, not just in CI.
"""

import os

import pytest

from besser.BUML.metamodel.nn import (
    NN,
    FlattenLayer,
    LinearLayer,
    PoolingLayer,
    TensorOp,
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator
from besser.generators.nn.tf.utils_tf import get_tensorop_syntax


def _flatten_pool_model(name: str) -> NN:
    """A model whose PoolingLayer activation forces the shared-activation
    path in the TF generator."""
    nn_model: NN = NN(name=name)
    nn_model.add_layer(FlattenLayer(name="f1"))
    nn_model.add_layer(PoolingLayer(
        name="p1", pooling_type="max", dimension="2D", kernel_dim=[2, 2],
        actv_func="relu",
    ))
    nn_model.add_layer(LinearLayer(name="l1", in_features=8, out_features=2))
    return nn_model


def _generate_tf(model: NN, out_dir: str) -> str:
    TFGenerator(model=model, output_dir=out_dir).generate()
    with open(os.path.join(out_dir, "tf_nn_subclassing.py"),
              encoding="utf-8") as f:
        return f.read()


def test_tf_generation_is_stable_across_runs_in_one_process(tmp_path):
    """The shared-activation registry must be per-generation-run state.

    A process-global registry made the FIRST generation emit the shared
    ``layers.Activation`` definition and every LATER generation in the same
    process (e.g. the long-lived web editor backend) skip it while still
    emitting the call — an AttributeError at model construction.
    """
    first = _generate_tf(_flatten_pool_model("run_one"),
                         str(tmp_path / "a"))
    second = _generate_tf(_flatten_pool_model("run_two"),
                          str(tmp_path / "b"))

    assert "self.activation_relu = layers.Activation('relu')" in first
    assert "self.activation_relu = layers.Activation('relu')" in second
    assert (first.replace("run_one", "MODEL")
            == second.replace("run_two", "MODEL"))


def test_tf_sequential_wraps_tensorops_in_keras_lambda(tmp_path):
    """Keras ``Sequential`` rejects bare Python lambdas — tensor-op
    expressions must be wrapped in ``layers.Lambda`` (the TF counterpart of
    the PyTorch ``Lambda(nn.Module)`` wrapper)."""
    nn_model: NN = NN(name="seq_lambda")
    nn_model.add_layer(LinearLayer(name="l1", in_features=4, out_features=8))
    nn_model.add_tensor_op(TensorOp(
        name="r1", tns_type="reshape", reshape_dim=[-1, 8],
        input_var="x",
    ))
    nn_model.add_layer(LinearLayer(name="l2", in_features=8, out_features=2))

    out_dir = str(tmp_path / "seq")
    TFGenerator(model=nn_model, output_dir=out_dir,
                generation_type="sequential").generate()
    with open(os.path.join(out_dir, "tf_nn_sequential.py"),
              encoding="utf-8") as f:
        code = f.read()

    assert "layers.Lambda(lambda x:" in code
    for line in code.splitlines():
        assert not line.strip().startswith("lambda x:"), (
            f"bare lambda emitted into Sequential: {line.strip()}"
        )


def test_tf_reduce_max_honors_keepdims():
    op = TensorOp(name="m1", tns_type="max", reduce_dim=1,
                  reduce_keepdims=True, input_var="x")
    syntax = get_tensorop_syntax(op, {}, in_var="x")
    assert "keepdims=True" in syntax

    op_no = TensorOp(name="m2", tns_type="max", reduce_dim=1, input_var="x")
    assert "keepdims" not in get_tensorop_syntax(op_no, {}, in_var="x")


def test_tf_interpolate_scale_casts_shape_arithmetic():
    """tf.shape() is int32; multiplying it by a float scale needs tf.cast."""
    op = TensorOp(name="i1", tns_type="interpolate", interpolate_scale=2.0,
                  interpolate_mode="bilinear", input_var="x")
    syntax = get_tensorop_syntax(op, {}, in_var="x")
    assert "tf.cast(" in syntax
    assert "tf.int32" in syntax


def test_tf_pad_rejects_untranslatable_shapes():
    """A 1-D pad_amount used to silently emit an all-zero (no-op) paddings
    tensor; now it must fail loudly instead of changing the model."""
    op = TensorOp(name="p1", tns_type="pad", pad_amount=[[2, 3]],
                  input_var="x")
    with pytest.raises(ValueError, match="pad_amount"):
        get_tensorop_syntax(op, {}, in_var="x")
