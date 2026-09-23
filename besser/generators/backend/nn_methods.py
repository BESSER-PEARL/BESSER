"""Support for methods implemented by a neural network in the generated backend.

A B-UML ``Method`` whose ``implementation_type`` is ``NEURAL_NETWORK`` and
whose ``neural_network`` points to an ``NN`` model becomes a class-level
endpoint that runs that network. For every linked network the backend gets:

- ``neural_networks/<module>.py``: the PyTorch ``NeuralNetwork`` class, produced
  by :class:`~besser.generators.nn.pytorch.pytorch_code_generator.PytorchGenerator`;
- ``neural_networks/weights/<module>.pt``: where the user drops the trained
  ``state_dict`` (the endpoint answers ``503`` until it exists);
- ``nn_runtime.py``: loads a network once and runs it on the method's
  parameters.

The method body itself is plain generated Python calling
``nn_runtime.run_network``, so it goes through the same endpoint rendering as
``CODE`` methods (parameter parsing, error mapping, response shape).
"""

import os
import re
import shutil
import tempfile
from typing import Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import DomainModel, Method, MethodImplementationType

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

NN_PACKAGE = "neural_networks"
TORCH_REQUIREMENT = "torch>=2.0.0"


def _module_base_name(name: str) -> str:
    base = re.sub(r"\W", "_", str(name or "")).strip("_").lower() or "network"
    return f"nn_{base}" if base[0].isdigit() else base


def _nn_methods(model: DomainModel) -> List[Method]:
    """NN-implemented methods that are linked to a network, in a stable order."""
    methods = []
    for cls in model.classes_sorted_by_inheritance():
        for method in sorted(cls.methods, key=lambda m: m.name):
            if (method.implementation_type == MethodImplementationType.NEURAL_NETWORK
                    and getattr(method, "neural_network", None) is not None):
                methods.append(method)
    return methods


def collect_nn_modules(model: DomainModel) -> Dict[int, Tuple[str, object]]:
    """Map ``id(nn)`` to ``(module_name, nn)`` for every network linked to a method.

    A network shared by several methods gets one module; two distinct networks
    with the same name get suffixed module names.
    """
    modules: Dict[int, Tuple[str, object]] = {}
    used = set()
    for method in _nn_methods(model):
        network = method.neural_network
        if id(network) in modules:
            continue
        base = _module_base_name(network.name)
        module_name, suffix = base, 2
        while module_name in used:
            module_name, suffix = f"{base}_{suffix}", suffix + 1
        used.add(module_name)
        modules[id(network)] = (module_name, network)
    return modules


def nn_method_code(method: Method, modules: Dict[int, Tuple[str, object]]) -> str:
    """Python body that runs the method's network, or ``""`` if it has none."""
    network = getattr(method, "neural_network", None)
    if network is None or id(network) not in modules:
        return ""
    module_name = modules[id(network)][0]
    name = str(method.name).split("(")[0].strip()
    params = [p.name for p in sorted(method.parameters, key=lambda p: p.timestamp)]
    return (
        f"def {name}({', '.join(params)}):\n"
        f"    from nn_runtime import run_network\n"
        f"    return run_network(\"{module_name}\", [{', '.join(params)}])\n"
    )


def generate_nn_methods(model: DomainModel, output_dir: str) -> bool:
    """Emit the network modules, weights folder and runtime into ``output_dir``.

    Returns ``True`` when at least one method is implemented by a network.
    """
    modules = collect_nn_modules(model)
    if not modules:
        return False

    # Imported lazily: only projects that actually link a network pay for it.
    from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator

    package_dir = os.path.join(output_dir, NN_PACKAGE)
    weights_dir = os.path.join(package_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(package_dir, "__init__.py"), mode="w", encoding="utf-8") as f:
        f.write("")

    for module_name, network in modules.values():
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                generator = PytorchGenerator(network, output_dir=tmp_dir, generation_type="subclassing")
                generator.generate()
            except ValueError as exc:
                raise ValueError(
                    f"Neural network '{network.name}' linked to a method cannot be generated: {exc}"
                ) from exc
            generated = os.path.join(tmp_dir, "pytorch_nn_subclassing.py")
            shutil.copyfile(generated, os.path.join(package_dir, f"{module_name}.py"))

    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), trim_blocks=True, lstrip_blocks=True)
    module_names = sorted(name for name, _ in modules.values())
    with open(os.path.join(output_dir, "nn_runtime.py"), mode="w", encoding="utf-8") as f:
        f.write(env.get_template("nn_runtime.py.j2").render(package=NN_PACKAGE))
    with open(os.path.join(weights_dir, "README.md"), mode="w", encoding="utf-8") as f:
        f.write(env.get_template("nn_weights_readme.md.j2").render(package=NN_PACKAGE, modules=module_names))

    requirements = os.path.join(output_dir, "requirements.txt")
    if os.path.isfile(requirements):
        with open(requirements, mode="r", encoding="utf-8") as f:
            content = f.read()
        if TORCH_REQUIREMENT not in content:
            with open(requirements, mode="w", encoding="utf-8") as f:
                f.write(content.rstrip("\n") + "\n" + TORCH_REQUIREMENT)
    return True
