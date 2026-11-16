"""Which attributes the generated search endpoint offers as filters."""
from besser.BUML.metamodel.structural import (
    BooleanType, Class, DomainModel, IntegerType, Property, StringType,
)

SCRIPT = """
    client.post("/task/", json={"title": "Write report", "estimate": 3})
    parameters = client.get("/openapi.json").json()["paths"]["/task/search/"]["get"]["parameters"]
    print(json.dumps({
        "filters": sorted(parameter["name"] for parameter in parameters),
        "by_title": len(client.get("/task/search/", params={"title": "report"}).json()),
        "stored": client.get("/task/").json()[0],
    }))
"""


def _model() -> DomainModel:
    task = Class(name="Task", attributes={
        Property(name="title", type=StringType),
        Property(name="estimate", type=IntegerType),
        Property(name="overdue", type=BooleanType, is_derived=True),
        Property(name="daysLeft", type=IntegerType, is_derived=True),
    })
    return DomainModel(name="Planner", types={task})


def test_a_derived_attribute_is_not_a_search_filter(run_generated_backend):
    """Its column is a stored default no generated code updates, so a filter on it answers wrongly."""
    result = run_generated_backend(_model(), SCRIPT)
    assert result["filters"] == ["estimate", "title"]
    assert result["by_title"] == 1
    # The value is still returned: the column exists for code that maintains it.
    assert result["stored"]["overdue"] is False and result["stored"]["daysLeft"] == 0
