"""The routes ``besser info`` lists for each class are those the backend generator writes."""
import re

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, FloatType, Generalization, IntegerType, Method,
    Multiplicity, Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.backend.endpoints import class_endpoints

ROUTE = re.compile(r'@router\.(get|post|put|patch|delete)\("([^"]+)"')
# A relationship route, /x/{x_id}/<end>/...: class_endpoints does not list them.
RELATIONSHIP = re.compile(r"_id}/(?!methods/)[^{]")


def _generated(model, tmp_path):
    """Every route of the generated routers but the relationship ones."""
    BackendGenerator(model=model, output_dir=str(tmp_path)).generate()
    routes = {
        f"{verb.upper()} {path}"
        for router in (tmp_path / "routers").glob("*.py")
        for verb, path in ROUTE.findall(router.read_text(encoding="utf-8"))
    }
    return {route for route in routes if not RELATIONSHIP.search(route)}


def test_every_listed_route_is_generated(tmp_path):
    book = Class(name="Book", attributes={Property(name="title", type=StringType)}, methods={
        Method(name="archive", code="def archive(self):\n    return 'ok'\n"),
        Method(name="count_all", code="def count_all():\n    return 0\n"),
    })
    model = DomainModel(name="Shelf", types={book})

    listed = class_endpoints(book, model)
    assert set(listed) == _generated(model, tmp_path)
    assert "POST /book/{book_id}/methods/archive/" in listed and "POST /book/methods/count_all/" in listed


def _shapes_model():
    """An association class (one end keyed by a string id), a read-only class, a class without attributes
    and a subclass that only inherits its attributes."""
    cat = Class(name="Cat", attributes={Property(name="code", type=StringType, is_id=True)})
    item = Class(name="Item", attributes={Property(name="label", type=StringType)})
    stock = BinaryAssociation(name="stock", ends={
        Property(name="cats", type=cat, multiplicity=Multiplicity(0, "*")),
        Property(name="items", type=item, multiplicity=Multiplicity(0, "*")),
    })
    listing = AssociationClass(name="Listing", attributes={Property(name="price", type=FloatType)}, association=stock)
    prod = Class(name="Prod", is_read_only=True, attributes={Property(name="n", type=IntegerType)})
    empty = Class(name="Empty")
    child = Class(name="Child")
    return DomainModel(name="Shapes", types={cat, item, listing, prod, empty, child}, associations={stock},
                       generalizations={Generalization(general=item, specific=child)})


def test_routes_follow_association_classes_read_only_classes_and_searchable_attributes(tmp_path):
    model = _shapes_model()
    classes = {cls.name: cls for cls in model.get_classes()}

    listed = {route for cls in classes.values() for route in class_endpoints(cls, model)}
    assert listed == _generated(model, tmp_path)
    assert class_endpoints(classes["Listing"], model) == [
        "POST /listing/", "GET /listing/", "GET /listing/{cats_id}/{items_id:int}/",
        "PUT /listing/{cats_id}/{items_id:int}/", "DELETE /listing/{cats_id}/{items_id:int}/",
        "GET /listing/count/", "GET /listing/paginated/", "GET /listing/search/",
    ]
    assert class_endpoints(classes["Prod"], model) == [
        "POST /prod/", "GET /prod/", "GET /prod/{prod_id}/", "POST /prod/bulk/", "DELETE /prod/bulk/",
        "GET /prod/count/", "GET /prod/paginated/", "GET /prod/search/",
    ]
    assert "GET /empty/search/" not in class_endpoints(classes["Empty"], model)
    assert "GET /child/search/" in class_endpoints(classes["Child"], model)
