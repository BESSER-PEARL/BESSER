"""A method parameter typed by an enumeration or a duration reaches the method body.

Any parameter type other than a plain one was read as the id of a record and loaded with
``routers.<type>.get_<type>``, a module an enumeration or a duration does not have, so the
route answered 500.
"""
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Method, MethodImplementationType, Parameter,
    PrimitiveDataType, Property, StringType,
)

_SCRIPT = """
item = client.post("/item/", json={"label": "a"}).json()
url = f"/item/{item['id']}/methods/"
print(json.dumps({
    "bump": client.post(url + "bump/", json={"params": {"to": "HIGH"}}).json(),
    "bump_default": client.post(url + "bump/", json={"params": {}}).json(),
    "bump_unknown": client.post(url + "bump/", json={"params": {"to": "URGENT"}}).status_code,
    "wait_seconds": client.post(url + "wait/", json={"params": {"after": 90}}).json(),
    "wait_iso": client.post(url + "wait/", json={"params": {"after": "PT2M"}}).json(),
    "wait_bad": client.post(url + "wait/", json={"params": {"after": "soon"}}).status_code,
    "levels": client.post("/item/methods/levels/", json={"params": {"start": "LOW"}}).json(),
}))
"""


def _model():
    level = Enumeration(name="Level", literals={EnumerationLiteral(name="LOW"), EnumerationLiteral(name="HIGH")})
    item = Class(name="Item", attributes={Property(name="label", type=StringType)})
    item.methods = {
        Method(name="bump", parameters={Parameter(name="to", type=level, default_value="LOW")},
               code="def bump(self, to):\n    return to.name\n", implementation_type=MethodImplementationType.CODE),
        Method(name="wait", parameters={Parameter(name="after", type=PrimitiveDataType("timedelta"))},
               code="def wait(self, after):\n    return after.total_seconds()\n",
               implementation_type=MethodImplementationType.CODE),
        Method(name="levels", parameters={Parameter(name="start", type=level)},
               code="def levels(database, start):\n    return [start.name]\n",
               implementation_type=MethodImplementationType.CODE),
    }
    return DomainModel(name="Items", types={level, item})


def test_enumeration_and_duration_parameters_are_accepted_and_checked(run_generated_backend):
    served = run_generated_backend(_model(), _SCRIPT)
    assert served["bump"]["result"] == "HIGH"
    assert served["bump_default"]["result"] == "LOW"
    assert served["bump_unknown"] == 422
    assert served["wait_seconds"]["result"] == "90.0"
    assert served["wait_iso"]["result"] == "120.0"
    assert served["wait_bad"] == 422
    assert served["levels"]["result"] == ["LOW"]


_RECORD_SCRIPT = """
owner = client.post("/owner/", json={"name": "ann"}).json()
item = client.post("/item/", json={"label": "a"}).json()
url = f"/item/{item['id']}/methods/"
assigned = client.post(url + "assign/", json={"params": {"owner": owner["id"]}})
unknown = client.post(url + "assign/", json={"params": {"owner": 999}})
print(json.dumps({
    "assigned": [assigned.status_code, assigned.json().get("result")],
    "unknown": [unknown.status_code, unknown.json().get("detail")],
    "no_body": client.post(url + "assign/").status_code,
    "no_body_enum": client.post(url + "bump/").json().get("result"),
}))
"""


def test_a_class_parameter_is_the_record_and_a_missing_body_is_no_parameters(run_generated_backend):
    """A class-typed parameter is sent as the record's id and reaches the method as the record; an
    instance method with parameters called with no body gets none (the class-level route did already)."""
    model = _model()
    level = next(enumeration for enumeration in model.get_enumerations())
    owner = Class(name="Owner", attributes={Property(name="name", type=StringType)})
    item = next(cls for cls in model.get_classes() if cls.name == "Item")
    item.methods = {
        Method(name="assign", parameters={Parameter(name="owner", type=owner)},
               code="def assign(self, owner):\n    return owner.name if owner is not None else 'nobody'\n",
               implementation_type=MethodImplementationType.CODE),
        Method(name="bump", parameters={Parameter(name="to", type=level, default_value="LOW")},
               code="def bump(self, to):\n    return to.name\n", implementation_type=MethodImplementationType.CODE),
    }
    served = run_generated_backend(DomainModel(name="Items", types={level, owner, item}), _RECORD_SCRIPT)
    assert served["assigned"] == [200, "ann"]
    assert served["unknown"] == [404, "Owner not found"]
    assert served["no_body"] == 200
    assert served["no_body_enum"] == "LOW"
