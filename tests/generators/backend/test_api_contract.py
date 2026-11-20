"""``api_contract.json`` states what the generated FastAPI app really serves.

The contract (``besser/generators/backend/api_contract.py``) restates the router templates'
rules; each model below is generated, imported, and its routes, ``<Class>Create`` fields and
mapped columns compared with the contract.
"""

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, BooleanType, Class, DateTimeType, DomainModel, Enumeration,
    EnumerationLiteral, FloatType, Generalization, IntegerType, Method, MethodImplementationType,
    Multiplicity, Parameter, PrimitiveDataType, Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.backend.api_contract import build_api_contract

_DUMP = """
import pydantic_classes
import sql_alchemy
from sqlalchemy.orm import configure_mappers

configure_mappers()
routes = sorted(
    [method.upper(), path] for path, operations in main_api.app.openapi()["paths"].items()
    for method in operations
)
names = {names}
create = {{name: {{field: info.is_required() for field, info in getattr(pydantic_classes, name + "Create").model_fields.items()}}
          for name in names}}
columns = {{name: sorted(attr.key for attr in getattr(sql_alchemy, name).__mapper__.column_attrs) for name in names}}
print(json.dumps({{"routes": routes, "create": create, "columns": columns}}))
"""


def _task_model():
    """Project 1 - * Task * - * Tag, a 1:1 Task - Note, enum Priority, a self 1:N on Task."""
    priority = Enumeration(name="Priority", literals={EnumerationLiteral(name="LOW"), EnumerationLiteral(name="HIGH")})
    project = Class(name="Project", attributes={Property(name="name", type=StringType)})
    task = Class(name="Task", attributes={
        Property(name="title", type=StringType),
        Property(name="done", type=BooleanType),
        Property(name="priority", type=priority, is_optional=True),
        Property(name="due", type=DateTimeType, is_optional=True),
    })
    tag = Class(name="Tag", attributes={Property(name="label", type=StringType)})
    note = Class(name="Note", attributes={Property(name="text", type=StringType)})
    associations = {
        BinaryAssociation(name="project_tasks", ends={
            Property(name="project", type=project, multiplicity=Multiplicity(1, 1)),
            Property(name="tasks", type=task, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="task_tags", ends={
            Property(name="tasks_tagged", type=task, multiplicity=Multiplicity(0, "*")),
            Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="task_note", ends={
            Property(name="task", type=task, multiplicity=Multiplicity(1, 1)),
            Property(name="note", type=note, multiplicity=Multiplicity(0, 1))}),
        BinaryAssociation(name="subtasks", ends={
            Property(name="parent", type=task, multiplicity=Multiplicity(0, 1)),
            Property(name="children", type=task, multiplicity=Multiplicity(0, "*"))}),
    }
    return DomainModel(name="Tasks", types={priority, project, task, tag, note}, associations=associations)


def _enrollment_model():
    """Student * - * Course through the association class Enrollment, Course keyed by a string code."""
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={
        Property(name="code", type=StringType, is_id=True), Property(name="title", type=StringType)})
    takes = BinaryAssociation(name="takes", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(1, "*"))})
    enrollment = AssociationClass(name="Enrollment", association=takes, attributes={
        Property(name="grade", type=FloatType, is_optional=True), Property(name="term", type=StringType)})
    return DomainModel(name="School", types={student, course, enrollment}, associations={takes})


def _methods_model():
    """Order with an instance method (no body), a class-level CODE method and a read-only Rate."""
    status = Enumeration(name="Status", literals={EnumerationLiteral(name="OPEN"), EnumerationLiteral(name="SHIPPED")})
    customer = Class(name="Customer", attributes={Property(name="name", type=StringType)})
    order = Class(name="Order", attributes={
        Property(name="reference", type=StringType), Property(name="total", type=FloatType),
        Property(name="status", type=status)})
    order.methods = {
        Method(name="ship", parameters={Parameter(name="carrier", type=StringType),
                                        Parameter(name="express", type=BooleanType, default_value=False)}),
        Method(name="count_open", code="def count_open(database):\n    return 0\n",
               implementation_type=MethodImplementationType.CODE),
        Method(name="archive", code="def archive(self):\n    return None\n",
               implementation_type=MethodImplementationType.CODE),
    }
    rate = Class(name="Rate", attributes={Property(name="value", type=IntegerType)}, is_read_only=True)
    places = BinaryAssociation(name="places", ends={
        Property(name="customer", type=customer, multiplicity=Multiplicity(0, 1)),
        Property(name="orders", type=order, multiplicity=Multiplicity(0, "*"))})
    return DomainModel(name="Shop", types={status, customer, order, rate}, associations={places})


def _abstract_parent_model():
    """An abstract Vehicle with an association, extended by Car and Truck."""
    vehicle = Class(name="Vehicle", is_abstract=True, attributes={Property(name="plate", type=StringType)})
    car = Class(name="Car", attributes={Property(name="seats", type=IntegerType)})
    truck = Class(name="Truck", attributes={Property(name="load", type=FloatType)})
    garage = Class(name="Garage", attributes={Property(name="city", type=StringType)})
    parks = BinaryAssociation(name="parks", ends={
        Property(name="garage", type=garage, multiplicity=Multiplicity(0, 1)),
        Property(name="vehicles", type=vehicle, multiplicity=Multiplicity(0, "*"))})
    return DomainModel(name="Fleet", types={vehicle, car, truck, garage}, associations={parks},
                       generalizations={Generalization(general=vehicle, specific=car),
                                        Generalization(general=vehicle, specific=truck)})


def _concrete_root_model():
    """An abstract Shape with no association, so mapped without a table of its own."""
    shape = Class(name="Shape", is_abstract=True, attributes={Property(name="label", type=StringType)})
    circle = Class(name="Circle", attributes={Property(name="radius", type=FloatType)})
    square = Class(name="Square", attributes={Property(name="side", type=FloatType)})
    return DomainModel(name="Shapes", types={shape, circle, square},
                       generalizations={Generalization(general=shape, specific=circle),
                                        Generalization(general=shape, specific=square)})


_MODELS = {
    "tasks": _task_model,
    "enrollment": _enrollment_model,
    "methods": _methods_model,
    "abstract_parent": _abstract_parent_model,
    "concrete_root": _concrete_root_model,
}
_FIXTURES = ("library_book_author_model", "employee_self_assoc_model", "library_model_with_enum",
             "library_model_with_inheritance", "custom_role_name_model", "non_navigable_assoc_model",
             "self_assoc_non_navigable_model", "no_attrib_list_assoc_model")


def _model(request, name):
    return _MODELS[name]() if name in _MODELS else request.getfixturevalue(name)


@pytest.mark.parametrize("name", list(_MODELS) + list(_FIXTURES))
def test_contract_matches_the_generated_app(request, run_generated_backend, tmp_path, name):
    model = _model(request, name)
    class_names = sorted(cls.name for cls in model.get_classes())
    served = run_generated_backend(model, _DUMP.format(names=class_names))

    with open(os.path.join(str(tmp_path), "api_contract.json"), encoding="utf-8") as handle:
        contract = json.load(handle)
    assert contract == build_api_contract(model)

    assert sorted([route["method"], route["path"]] for route in contract["routes"]) == served["routes"]
    for entity in contract["entities"]:
        name = entity["name"]
        assert {field["name"]: field["required"] for field in entity["write"]} == served["create"][name], name
        assert sorted(field["name"] for field in entity["read"]) == served["columns"][name], name


_TASK_RESPONSES = """
project = client.post("/project/", json={"name": "P"}).json()
tag = client.post("/tag/", json={"label": "t"}).json()
created = client.post("/task/", json={"title": "a", "done": False, "project": project["project"]["id"],
                                      "tags": [tag["tag"]["id"]], "priority": "LOW"}).json()
task_id = created["task"]["id"]
print(json.dumps({
    "create": sorted(created), "create_record": sorted(created["task"]),
    "get": sorted(client.get(f"/task/{task_id}/").json()),
    "list_record": sorted(client.get("/task/").json()[0]),
    "detailed_record": sorted(client.get("/task/?detailed=true").json()[0]),
    "tags": sorted(client.get(f"/task/{task_id}/tags/").json()),
    "page": sorted(client.get("/task/paginated/").json()),
    "update": sorted(client.put(f"/task/{task_id}/", json={"title": "b", "done": True,
                                                        "project": project["project"]["id"]}).json()),
    "not_found": client.get("/task/999/").json(),
    "invalid": client.post("/task/", json={"done": "maybe"}).json(),
    "delete": sorted(client.delete(f"/task/{task_id}/").json()),
}))
"""


def test_responses_have_the_shapes_the_contract_states(run_generated_backend):
    model = _task_model()
    served = run_generated_backend(model, _TASK_RESPONSES)
    task = next(entity for entity in build_api_contract(model)["entities"] if entity["name"] == "Task")
    record = sorted(field["name"] for field in task["read"])
    wrapper = sorted(["task"] + [field["name"] for field in task["record_ids"]])

    assert served["create"] == served["get"] == served["update"] == wrapper
    assert served["create_record"] == served["list_record"] == served["delete"] == record
    assert served["detailed_record"] == sorted(record + [field["name"] for field in task["detailed"]])
    assert served["tags"] == ["tags", "tags_count", "task_id"]
    assert served["page"] == ["data", "limit", "skip", "total"]
    # The two error shapes the client turns into one ApiError.
    assert served["not_found"] == {"error": "Task not found", "message": "Task not found", "detail": "Task not found"}
    assert {item["loc"][-1] for item in served["invalid"]["detail"]} >= {"title", "project", "done"}


def test_api_md_lists_every_route(tmp_path):
    model = _task_model()
    BackendGenerator(model=model, output_dir=str(tmp_path)).generate()
    with open(os.path.join(str(tmp_path), "API.md"), encoding="utf-8") as handle:
        text = handle.read()
    for route in build_api_contract(model)["routes"]:
        assert f"`{route['method']} {route['path']}`" in text, route
    # What a caller gets wrong without it: the create key and the read key of one relation differ.
    assert "`project` (int, required)" in text
    assert "`project_id`" in text
    assert '"LOW" | "HIGH"' in text


def test_method_parameters_are_described_as_the_route_reads_them():
    """Every parameter is optional (the route reads them with ``params.get``); an enumeration and a
    duration are values, a class is sent as the id of one of its records."""
    level = Enumeration(name="Level", literals={EnumerationLiteral(name="LOW")})
    owner = Class(name="Owner", attributes={Property(name="name", type=StringType)})
    item = Class(name="Item", attributes={Property(name="label", type=StringType)})
    item.methods = {Method(name="plan", parameters={
        Parameter(name="to", type=level), Parameter(name="after", type=PrimitiveDataType("timedelta")),
        Parameter(name="by", type=owner)})}
    contract = build_api_contract(DomainModel(name="Plans", types={level, owner, item}))
    [route] = [route for route in contract["routes"] if route["group"] == "method"]
    params = {param["name"]: param for param in route["parameters"]}
    assert not any(param["required"] for param in params.values())
    assert params["to"] == {"name": "to", "type": "Level", "required": False, "enum": True}
    assert params["after"] == {"name": "after", "type": "timedelta", "required": False}
    assert params["by"] == {"name": "by", "type": "int", "required": False, "record": "Owner"}
