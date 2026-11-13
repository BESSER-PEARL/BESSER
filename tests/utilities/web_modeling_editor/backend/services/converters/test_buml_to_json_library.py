import datetime

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, StringType, IntegerType,
    FloatType, DateType, Enumeration, EnumerationLiteral,
)
from besser.utilities.buml_code_builder.domain_model_builder import object_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)
from tests.utilities.web_modeling_editor.backend.services.converters.utils_buml_to_json_tests import (
    object_attributes, object_links, object_names,
)


@pytest.fixture()
def model() -> DomainModel:
    """``Library``, ``Book`` and ``Author``; libraries and authors both reach their books through
    a many-to-many association whose end is named ``books``."""
    genre = Enumeration("Genre", literals={EnumerationLiteral("Poetry"), EnumerationLiteral("Fantasy")})
    author = Class("Author", attributes={
        Property("name", StringType),
        Property("birth", DateType),
    })
    book = Class("Book", attributes={
        Property("title", StringType),
        Property("pages", IntegerType),
        Property("stock", IntegerType),
        Property("price", FloatType),
        Property("release", DateType),
        Property("genre", genre),
    })
    library = Class("Library", attributes={
        Property("name", StringType),
        Property("web_page", StringType),
        Property("address", StringType),
        Property("telephone", StringType),
    })
    library_books = BinaryAssociation("books", ends={
        Property("library", library, multiplicity=Multiplicity(1, "*")),
        Property("books", book, multiplicity=Multiplicity(0, "*")),
    })
    author_books = BinaryAssociation("books_1", ends={
        Property("authors", author, multiplicity=Multiplicity(1, "*")),
        Property("books", book, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Library", types={author, book, library, genre},
                       associations={library_books, author_books})


@pytest.fixture()
def object(model) -> ObjectModel:
    """``central`` holds ``ficciones`` and ``antologia``; ``antologia`` has two authors and
    ``ficciones`` one."""
    author = model.get_class_by_name("Author")
    book = model.get_class_by_name("Book")
    library = model.get_class_by_name("Library")
    genre = model.get_type_by_name("Genre")
    borges = author("borges").attributes(name="Borges", birth=datetime.date(1899, 8, 24)).build()
    casares = author("casares").attributes(name="Casares", birth=datetime.date(1914, 9, 15)).build()
    ficciones = book("ficciones").attributes(title="Ficciones", pages=200, stock=3, price=12.5,
                                              release=datetime.date(1944, 1, 1), genre=genre.Fantasy).build()
    antologia = book("antologia").attributes(title="Antologia", pages=400, stock=0, price=20.0,
                                              release=datetime.date(1940, 1, 1), genre=genre.Poetry).build()
    central = library("central").attributes(name="Central", web_page="central.org",
                                            address="Main St 1", telephone="123").build()
    central.books = {ficciones, antologia}
    antologia.authors = {borges, casares}
    ficciones.authors = {borges}
    return ObjectModel("Library", objects={borges, casares, ficciones, antologia, central})


def test_to_json_keeps_information(object, model, tmp_path):
    code_file = tmp_path / "library_objects.py"
    object_model_to_code(object, str(code_file))
    result = object_buml_to_json(code_file.read_text(), class_buml_to_json(model))

    names = object_names(result).values()
    attributes = object_attributes(result)
    links = object_links(result)
    # The importer used to drop everything below
    assert "borges.birth = 1899-08-24" in attributes
    assert "casares.birth = 1914-09-15" in attributes
    assert "ficciones.release = 1944-01-01" in attributes
    assert "antologia.release = 1940-01-01" in attributes
    assert "ficciones.genre = Fantasy" in attributes
    assert "antologia.genre = Poetry" in attributes
    assert "central -books-> ficciones" in links
    assert "central -books-> antologia" in links
    assert "antologia -authors-> borges" in links
    assert "antologia -authors-> casares" in links
    # The importer already kept everything below
    assert "borges" in names
    assert "casares" in names
    assert "ficciones" in names
    assert "antologia" in names
    assert "central" in names
    assert "borges.name = Borges" in attributes
    assert "casares.name = Casares" in attributes
    assert "ficciones.title = Ficciones" in attributes
    assert "ficciones.pages = 200" in attributes
    assert "ficciones.stock = 3" in attributes
    assert "ficciones.price = 12.5" in attributes
    assert "antologia.title = Antologia" in attributes
    assert "antologia.pages = 400" in attributes
    assert "antologia.stock = 0" in attributes
    assert "antologia.price = 20.0" in attributes
    assert "central.name = Central" in attributes
    assert "central.web_page = central.org" in attributes
    assert "central.address = Main St 1" in attributes
    assert "central.telephone = 123" in attributes
    assert "ficciones -authors-> borges" in links


@pytest.mark.parametrize("reverse", [False, True])
def test_links_get_their_own_association_whatever_the_order(object, model, tmp_path, reverse):
    # Both associations have a "books" end; a substring match on the role picked
    # whichever association came first in the class diagram.
    code_file = tmp_path / "library_objects.py"
    object_model_to_code(object, str(code_file))
    domain_json = class_buml_to_json(model)
    if reverse:
        domain_json["edges"] = list(reversed(domain_json["edges"]))
    result = object_buml_to_json(code_file.read_text(), domain_json)

    names = object_names(result)
    assoc_name = {edge["id"]: edge["data"].get("name") for edge in domain_json["edges"]}
    by_link = {
        f"{names[e['source']]} -{e['data']['name']}-> {names[e['target']]}":
            assoc_name.get(e["data"]["associationId"])
        for e in result["edges"] if e["type"] == "ObjectLink"
    }
    assert by_link["central -books-> ficciones"] == "books"
    assert by_link["antologia -authors-> casares"] == "books_1"
