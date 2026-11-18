"""Class and association names that equal a variable of the generated list routes.

A many-to-many or association-class target called ``Item`` named its records
``item_dict``, the dictionary of the entity being listed, and an association
called ``Order_Item`` has the table ``order_item``, the name of the loop
variable over the orders: ``GET /<class>/?detailed=true`` answered 500.
"""
from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)

SCRIPT = """
    client.post("/item/", json={"label": "pen"})
    client.post("/row/", json={"label": "first"})
    client.post("/order/", json={"title": "o", "items": [1]})
    client.post("/sheet/", json={"title": "s", "lines": [{"target": 1, "quantity": 2}],
                                 "rows": [{"target": 1, "quantity": 3}]})
    answers = {}
    for name in ("order", "sheet", "item", "row"):
        for url in (f"/{name}/?detailed=true", f"/{name}/paginated/?detailed=true"):
            response = client.get(url)
            answers[url] = [response.status_code, response.json() if response.status_code == 200 else None]
    print(json.dumps(answers))
"""


def _model() -> DomainModel:
    many = Multiplicity(0, "*")
    item = Class(name="Item", attributes={Property(name="label", type=StringType)})
    row = Class(name="Row", attributes={Property(name="label", type=StringType)})
    order = Class(name="Order", attributes={Property(name="title", type=StringType)})
    sheet = Class(name="Sheet", attributes={Property(name="title", type=StringType)})
    order_item = BinaryAssociation(name="Order_Item", ends={
        Property(name="orders", type=order, multiplicity=many), Property(name="items", type=item, multiplicity=many)})
    sheet_item = BinaryAssociation(name="Sheet_Item", ends={
        Property(name="sheets", type=sheet, multiplicity=many), Property(name="lines", type=item, multiplicity=many)})
    sheet_row = BinaryAssociation(name="Sheet_Row", ends={
        Property(name="row_sheets", type=sheet, multiplicity=many), Property(name="rows", type=row, multiplicity=many)})
    entry = AssociationClass(name="Entry", attributes={Property(name="quantity", type=IntegerType)},
                             association=sheet_item)
    cell = AssociationClass(name="Cell", attributes={Property(name="quantity", type=IntegerType)},
                            association=sheet_row)
    return DomainModel(name="Names", types={item, row, order, sheet, entry, cell},
                       associations={order_item, sheet_item, sheet_row})


def test_related_records_never_take_the_name_of_a_route_variable(run_generated_backend):
    answers = run_generated_backend(_model(), SCRIPT)

    assert {url: answer[0] for url, answer in answers.items()} == dict.fromkeys(answers, 200)
    order = answers["/order/?detailed=true"][1][0]
    assert order["title"] == "o" and [item["label"] for item in order["items"]] == ["pen"]
    sheet = answers["/sheet/?detailed=true"][1][0]
    assert sheet["title"] == "s"
    assert [item["label"] for item in sheet["lines"]] == ["pen"]
    assert [row["label"] for row in sheet["rows"]] == ["first"]
    assert [link["quantity"] for link in sheet["lines_links"]] == [2]
    page = answers["/order/paginated/?detailed=true"][1]["data"][0]
    assert page["order"]["title"] == "o" and page["item_ids"] == [1]
