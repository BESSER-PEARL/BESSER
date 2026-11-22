"""Tests for methods implemented by a neural network in BackendGenerator output.

A method with ``implementation_type=NEURAL_NETWORK`` and a linked ``NN`` model
becomes a class-level endpoint that calls ``nn_runtime.run_network``; the
backend ships the PyTorch module, a weights folder and the runtime.
"""

import ast
import importlib
import os
import sys

import pytest

from besser.BUML.metamodel.nn import NN, LinearLayer
from besser.BUML.metamodel.structural import (
    AnyType, Class, DomainModel, FloatType, Method, MethodImplementationType, Parameter,
)
from besser.generators.backend import BackendGenerator


def _scorer_network(name="Scorer"):
    network = NN(name=name)
    network.add_layer(LinearLayer(name="l1", actv_func="relu", in_features=2, out_features=4))
    network.add_layer(LinearLayer(name="l2", actv_func=None, in_features=4, out_features=1))
    return network


def _model(network, with_link=True):
    score = Method(
        name="score",
        parameters=[Parameter(name="height", type=FloatType), Parameter(name="weight", type=FloatType)],
        type=AnyType,
        implementation_type=MethodImplementationType.NEURAL_NETWORK,
        neural_network=network if with_link else None,
    )
    patient = Class(name="Patient", methods={score})
    return DomainModel(name="Clinic", types={patient})


def _generate(tmp_path, model):
    out = tmp_path / "backend"
    generator = BackendGenerator(model, output_dir=str(out))
    generator.generate()
    return out, generator


def test_nn_method_generates_network_module_runtime_and_endpoint(tmp_path):
    out, generator = _generate(tmp_path, _model(_scorer_network()))

    assert generator.has_nn_methods is True
    assert (out / "nn_runtime.py").is_file()
    assert (out / "neural_networks" / "__init__.py").is_file()
    assert (out / "neural_networks" / "scorer.py").is_file()
    assert (out / "neural_networks" / "weights" / "README.md").is_file()
    assert "class NeuralNetwork" in (out / "neural_networks" / "scorer.py").read_text(encoding="utf-8")
    assert "torch" in (out / "requirements.txt").read_text(encoding="utf-8")

    router = (out / "routers" / "patient_methods.py").read_text(encoding="utf-8")
    ast.parse(router)
    ast.parse((out / "nn_runtime.py").read_text(encoding="utf-8"))
    # Class-level endpoint (no entity id): the network is stateless.
    assert '@router.post("/patient/methods/score/"' in router
    assert 'run_network("scorer", [height, weight])' in router
    assert "has no implementation" not in router


def test_nn_method_without_linked_network_stays_501(tmp_path):
    out, generator = _generate(tmp_path, _model(_scorer_network(), with_link=False))

    assert generator.has_nn_methods is False
    assert not (out / "nn_runtime.py").exists()
    assert not (out / "neural_networks").exists()
    assert "torch" not in (out / "requirements.txt").read_text(encoding="utf-8")
    router = (out / "routers" / "patient_methods.py").read_text(encoding="utf-8")
    assert "has no implementation" in router


def test_distinct_networks_with_the_same_name_get_distinct_modules(tmp_path):
    first = Method(name="a", parameters=[Parameter(name="x", type=AnyType)],
                   neural_network=_scorer_network("Net"))
    second = Method(name="b", parameters=[Parameter(name="x", type=AnyType)],
                    neural_network=_scorer_network("Net"))
    model = DomainModel(name="Two", types={Class(name="Holder", methods={first, second})})
    out, _ = _generate(tmp_path, model)

    modules = sorted(p.name for p in (out / "neural_networks").glob("*.py") if p.name != "__init__.py")
    assert modules == ["net.py", "net_2.py"]


def test_docker_files_include_the_network_runtime(tmp_path):
    out = tmp_path / "backend"
    BackendGenerator(_model(_scorer_network()), output_dir=str(out), docker_image=True).generate()

    dockerfile = (out / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY nn_runtime.py ./" in dockerfile
    assert "COPY neural_networks/ ./neural_networks/" in dockerfile
    assert "torch" in (out / "requirements.txt").read_text(encoding="utf-8")


def test_generated_runtime_runs_the_network(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    out, _ = _generate(tmp_path, _model(_scorer_network()))
    monkeypatch.syspath_prepend(str(out))
    for name in [m for m in sys.modules if m == "nn_runtime" or m.startswith("neural_networks")]:
        monkeypatch.delitem(sys.modules, name)

    runtime = importlib.import_module("nn_runtime")
    from fastapi import HTTPException

    # No weights yet: 503, and nothing cached, so adding them later works.
    with pytest.raises(HTTPException) as untrained:
        runtime.run_network("scorer", [1.7, 70.0])
    assert untrained.value.status_code == 503

    network = importlib.import_module("neural_networks.scorer").NeuralNetwork()
    torch.save(network.state_dict(), os.path.join(out, "neural_networks", "weights", "scorer.pt"))

    result = runtime.run_network("scorer", [1.7, 70.0])
    expected = network.eval()(torch.tensor([[1.7, 70.0]])).squeeze(0).tolist()
    assert result == {"network": "scorer", "output": pytest.approx(expected)}

    with pytest.raises(HTTPException) as bad_input:
        runtime.run_network("scorer", ["tall", None])
    assert bad_input.value.status_code == 422
