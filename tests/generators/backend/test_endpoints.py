"""The routes ``besser info`` lists for each class are those the backend generator writes."""
import re

from besser.BUML.metamodel.structural import Class, DomainModel, Method, Property, StringType
from besser.generators.backend import BackendGenerator
from besser.generators.backend.endpoints import EXTRA_ROUTES, class_endpoints

ROUTE = re.compile(r'@router\.(get|post|put|delete)\("([^"]+)"')


def test_every_listed_route_is_generated(tmp_path):
    book = Class(name="Book", attributes={Property(name="title", type=StringType)}, methods={
        Method(name="archive", code="def archive(self):\n    return 'ok'\n"),
        Method(name="count_all", code="def count_all():\n    return 0\n"),
    })
    BackendGenerator(model=DomainModel(name="Shelf", types={book}), output_dir=str(tmp_path)).generate()
    generated = {
        f"{verb.upper()} {path}"
        for router in (tmp_path / "routers").glob("*.py")
        for verb, path in ROUTE.findall(router.read_text(encoding="utf-8"))
    }

    listed = class_endpoints(book)
    assert set(listed) <= generated
    assert "POST /book/{book_id}/methods/archive/" in listed and "POST /book/methods/count_all/" in listed
    assert all(f"GET /book/{route}/" in generated or f"POST /book/{route}/" in generated for route in EXTRA_ROUTES)
