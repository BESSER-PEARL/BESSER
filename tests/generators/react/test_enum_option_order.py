"""Enumeration options keep the order of the model.

The table's columns, its Add/Edit dialog, association class fields and method
parameters listed the literals alphabetically, so LOW/MEDIUM/HIGH was offered
as HIGH/LOW/MEDIUM.
"""
import json

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import FieldColumn, Table
from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, Multiplicity,
    Property, StringType,
)
from besser.generators.react import ReactGenerator
from besser.generators.react.serialization import GuiSerializationMixin

DECLARED = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _priority():
    # One literal at a time: each gets the next creation timestamp, in declaration order.
    return Enumeration(name="Priority", literals={EnumerationLiteral(name=name) for name in DECLARED})


def _table_chart(generator):
    payload = json.loads(generator._build_generation_context()["components_json"])

    def walk(nodes):
        for node in nodes:
            if "formColumns" in (node.get("chart") or {}):
                return node["chart"]
            found = walk(node.get("children") or [])
            if found is not None:
                return found
        return None

    for page in payload.get("pages", []):
        found = walk(page.get("components") or [])
        if found is not None:
            return found
    raise AssertionError("no table in the serialized GUI")


def test_table_columns_and_dialog_list_literals_in_model_order():
    priority = _priority()
    level = Property(name="priority", type=priority)
    task = Class(name="Task", attributes={Property(name="title", type=StringType), level})
    table = Table(name="TaskTable", title="Tasks", action_buttons=True,
                  columns=[FieldColumn(label="Priority", field=level)],
                  data_binding=DataBinding(name="task_binding", domain_concept=task))
    screen = Screen(name="Tasks", description="Tasks", view_elements={table}, is_main_page=True)
    gui_model = GUIModel(name="TasksApp", package="", versionCode="1", versionName="1.0",
                         modules={Module(name="M", screens={screen})}, description="Tasks")
    generator = ReactGenerator(model=DomainModel(name="Tasks", types={task, priority}), gui_model=gui_model)

    chart = _table_chart(generator)
    assert chart["columns"][0]["options"] == DECLARED
    assert next(col for col in chart["formColumns"] if col["field"] == "priority")["options"] == DECLARED


def test_association_class_fields_list_literals_in_model_order():
    priority = _priority()
    task = Class(name="Task", attributes={Property(name="title", type=StringType)})
    member = Class(name="Member", attributes={Property(name="name", type=StringType)})
    association = BinaryAssociation(name="assigned", ends={
        Property(name="tasks", type=task, multiplicity=Multiplicity(0, "*")),
        Property(name="members", type=member, multiplicity=Multiplicity(0, "*")),
    })
    assignment = AssociationClass(name="Assignment", attributes={Property(name="priority", type=priority)},
                                  association=association)

    fields = GuiSerializationMixin._association_class_fields(assignment)
    assert fields[0]["options"] == DECLARED
