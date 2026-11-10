"""One buildable generator per regenerable generator id.

Run as a script, ``regenerable_builders.py <folder>`` regenerates each generator
twice into ``<folder>/<id>`` and prints the second run's report as JSON. A second
run with the same model must change nothing.
"""
import dataclasses
import importlib.util
import json
import logging
import os
import sys

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    Method,
    Property,
    StringType,
)
from tests.conftest import library_book_author


def library() -> DomainModel:
    """The shared library model, plus more for the templates to loop over: enumerations, a subclass and methods."""
    model = library_book_author()
    book = next(cls for cls in model.get_classes() if cls.name == "Book")
    genre = Enumeration(name="Genre", literals={EnumerationLiteral(name="novel"), EnumerationLiteral(name="essay")})
    size = Enumeration(name="Size", literals={EnumerationLiteral(name="small"), EnumerationLiteral(name="large")})
    book.attributes = book.attributes | {Property(name="genre", type=genre), Property(name="size", type=size)}
    book.methods = {Method(name="summary", type=StringType), Method(name="blurb", type=StringType)}
    prize_book = Class(name="PrizeBook", attributes={Property(name="prize", type=StringType)})
    model.types = model.types | {genre, size, prize_book}
    model.generalizations = {Generalization(general=book, specific=prize_book)}
    return model


def _editor_models():
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        process_class_diagram,
        process_gui_diagram,
    )
    from tests.generators.react.conftest import class_diagram, gui_json

    classes = {"Guest": [("full_name", "str"), ("age", "int")], "Room": [("number", "int")]}

    def form(name, attributes):
        inputs = [{"tagName": "input", "type": "input", "attributes": {"type": "text", "name": a}}
                  for a, _ in attributes]
        return {"type": "gui-form", "tagName": "form", "components": inputs,
                "attributes": {"id": f"{name}-form", "data-gui-type": "Form", "data-source": f"cls-{name}",
                               "method": "POST"}}

    class_json, gui = class_diagram(classes), gui_json({name: [form(name, a)] for name, a in classes.items()})
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    return domain, process_gui_diagram(gui, class_json, domain)


def library_gui(model: DomainModel):
    """A GUI with several modules, screens and fields: Django loops over each of these sets."""
    from besser.BUML.metamodel.gui import DataList, DataSourceElement, GUIModel, Module, Screen
    from besser.utilities import sort_by_timestamp

    def screen(cls):
        source = DataSourceElement(name=f"{cls.name}Source", dataSourceClass=cls, fields=set(cls.attributes))
        return Screen(name=f"{cls.name}Screen", description="", x_dpi="", y_dpi="", screen_size="Small",
                      view_elements={DataList(name=f"{cls.name}List", description="", list_sources={source})})

    home = Screen(name="Home", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements=set(),
                  is_main_page=True)
    classes = sort_by_timestamp(model.get_classes())
    modules = {Module(name="Main", screens={home, *(screen(cls) for cls in classes[1:])}),
               Module(name="Other", screens={screen(classes[0])})}
    return GUIModel(name="Shelf", package="p", versionCode="1", versionName="1", description="",
                    screenCompatibility=True, modules=modules)


def _django(out):
    from besser.generators.django import DjangoGenerator

    model = library()
    return DjangoGenerator(model, project_name="shelfsite", app_name="shelfapp", gui_model=library_gui(model),
                           output_dir=out, containerization=False)


def _agent():
    from besser.BUML.metamodel.state_machine.agent import Agent

    agent = Agent("helper")
    greeting = agent.new_intent("greeting", ["hello", "good morning"])
    goodbye = agent.new_intent("goodbye", ["bye", "see you"])
    start = agent.new_state("start", initial=True)
    greeted = agent.new_state("greeted")
    start.when_intent_matched(greeting).go_to(greeted)
    greeted.when_intent_matched(goodbye).go_to(start)
    return agent


def _web_app(out):
    from besser.generators.web_app import WebAppGenerator

    return WebAppGenerator(*_editor_models(), output_dir=out, agent_model=_agent())


def _react(out):
    from besser.generators.react import ReactGenerator

    return ReactGenerator(*_editor_models(), output_dir=out)


def _jsonobject(out):
    from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
    from besser.generators.json import JSONObjectGenerator

    name = Property(name="name", type=StringType)
    person = Class(name="Person", attributes={name})
    DomainModel(name="People", types={person})
    alice = Object(name="alice", classifier=person,
                   slots=[AttributeLink(attribute=name, value=DataValue(classifier=StringType, value="Alice"))])
    return JSONObjectGenerator(ObjectModel(name="Objects", objects={alice}), output_dir=out)


def _qiskit(out):
    from besser.BUML.metamodel.quantum import ClassicalRegister, HadamardGate, Measurement, QuantumCircuit
    from besser.generators.qiskit import QiskitGenerator

    circuit = QuantumCircuit("Circuit", qubits=1)
    circuit.add_creg(ClassicalRegister("c", 1))
    circuit.add_operation(HadamardGate(target_qubit=0))
    circuit.add_operation(Measurement(target_qubit=0, output_bit=0))
    return QiskitGenerator(model=circuit, output_dir=out)


def _bpmn(out):
    from besser.generators.bpmn import BPMNGenerator
    from tests.bpmn_models import _gateway_model

    return BPMNGenerator(_gateway_model(), output_dir=out)


def _network():
    from besser.BUML.metamodel.nn import NN, LinearLayer

    network = NN(name="net")
    network.add_layer(LinearLayer(name="hidden", in_features=3, out_features=4))
    network.add_layer(LinearLayer(name="out", in_features=4, out_features=2))
    return network


def _pytorch(out):
    from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator

    return PytorchGenerator(_network(), output_dir=out)


def _tensorflow(out):
    from besser.generators.nn.tf.tf_code_generator import TFGenerator

    return TFGenerator(_network(), output_dir=out)


def _forecaster(out):
    from besser.generators.nn.forecaster import ForecastConfig, TimeSeriesForecasterGenerator
    from tests.forecaster_networks import HORIZON, WINDOW, mlp

    return TimeSeriesForecasterGenerator(mlp(), ForecastConfig(window=WINDOW, horizon=HORIZON, seeds=(0,)),
                                         output_dir=out)


def _on_library(path, name):
    def build(out):
        module = __import__(path, fromlist=[name])
        return getattr(module, name)(library(), output_dir=out)

    return build


BUILDERS = {
    "python": _on_library("besser.generators.python_classes", "PythonGenerator"),
    "java": _on_library("besser.generators.java_classes", "JavaGenerator"),
    "pydantic": _on_library("besser.generators.pydantic_classes", "PydanticGenerator"),
    "test_case": _on_library("besser.generators.testgen", "TestCaseGenerator"),
    "spring": _on_library("besser.generators.spring.spring_backend_generator", "SpringBackendGenerator"),
    "backend": _on_library("besser.generators.backend", "BackendGenerator"),
    "sqlalchemy": _on_library("besser.generators.sql_alchemy", "SQLAlchemyGenerator"),
    "sql": _on_library("besser.generators.sql", "SQLGenerator"),
    "jsonschema": _on_library("besser.generators.json", "JSONSchemaGenerator"),
    "rdf": _on_library("besser.generators.rdf", "RDFGenerator"),
    "rest_api": _on_library("besser.generators.rest_api", "RESTAPIGenerator"),
    "web_app": _web_app,
    "react": _react,
    "jsonobject": _jsonobject,
    "qiskit": _qiskit,
    "bpmn": _bpmn,
    "pytorch": _pytorch,
    "tensorflow": _tensorflow,
    "forecaster": _forecaster,
}
if importlib.util.find_spec("django"):  # DjangoGenerator runs `python -m django startproject`.
    BUILDERS["django"] = _django


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    top = sys.argv[1]
    reports = {}
    for generator_id, build in BUILDERS.items():
        folder = os.path.join(top, generator_id, "out")
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        build(folder).regenerate()
        reports[generator_id] = dataclasses.asdict(build(folder).regenerate())
    print(json.dumps(reports))
