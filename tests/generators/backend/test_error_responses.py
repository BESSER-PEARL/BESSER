"""What the generated backend tells the client, and its log, when a request fails.

A constraint violation must not reach the client as SQL or as the database
driver's text when the cause is the request (a duplicate value, an unknown
related id), and a refused request (4xx) is not a server failure.
"""
import json

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
)

SCRIPT = """
    shelf = client.post("/shelf/", json={"label": "A"}).json()["shelf"]["id"]
    first = client.post("/book/", json={"isbn": "111", "title": "Maps", "shelf": shelf})
    duplicate = client.post("/book/", json={"isbn": "111", "title": "Charts", "shelf": shelf})
    bulk = client.post("/book/bulk/", json=[
        {"isbn": "222", "title": "Atlas", "shelf": shelf},
        {"isbn": "333", "title": "Globe", "shelf": 999},
    ])
    bulk_duplicate = client.post("/book/bulk/", json=[{"isbn": "111", "title": "Again", "shelf": shelf}])
    after_bulk = sorted(book["isbn"] for book in client.get("/book/").json())

    log.lines.clear()
    missing = client.get("/book/unknown/").status_code
    refused = client.post("/book/", json={"isbn": "444", "title": "Lost", "shelf": 999}).status_code
    invalid = client.post("/book/", json={"isbn": "555"}).status_code
    client_error_log = list(log.lines)
    log.lines.clear()
    client.post("/book/", json={"isbn": "111", "title": "Charts", "shelf": shelf})
    failure_log = list(log.lines)
    print(json.dumps({
        "first": first.status_code, "duplicate": [duplicate.status_code, duplicate.json()],
        "bulk": [bulk.status_code, bulk.json()], "bulk_duplicate": [bulk_duplicate.status_code, bulk_duplicate.json()],
        "after_bulk": after_bulk, "missing": missing, "refused": refused, "invalid": invalid,
        "client_error_log": client_error_log, "failure_log": failure_log,
    }))
"""


def _model() -> DomainModel:
    book = Class(name="Book", attributes={
        Property(name="isbn", type=StringType, is_id=True),
        Property(name="title", type=StringType),
    })
    shelf = Class(name="Shelf", attributes={Property(name="label", type=StringType)})
    shelving = BinaryAssociation(name="Shelving", ends={
        Property(name="shelf", type=shelf, multiplicity=Multiplicity(1, 1)),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
    })
    edition = Class(name="UniqueEdition", attributes={Property(name="label", type=StringType)})
    return DomainModel(name="Library", types={book, shelf, edition}, associations={shelving})


def _texts(body) -> str:
    return str(body)


def test_a_duplicate_value_is_named_without_the_driver_text(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)
    assert result["first"] == 200
    status, body = result["duplicate"]
    assert status == 409
    assert body["detail"] == "Book with this isbn already exists"
    assert "constraint" not in _texts(body)


def test_bulk_create_reports_a_failed_item_without_sql(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)
    status, body = result["bulk"]
    assert status == 400
    assert body["detail"]["errors"] == [{
        "index": 1,
        "error": "The record refers to a record that does not exist, or another record still refers to it",
    }]
    for text in (_texts(body), _texts(result["bulk_duplicate"][1])):
        assert "SQL" not in text and "INSERT" not in text and "sqlalche.me" not in text
    assert result["bulk_duplicate"][1]["detail"]["errors"] == [
        {"index": 0, "error": "Book with this isbn already exists"}]
    # A refused bulk leaves nothing behind.
    assert result["after_bulk"] == ["111"]


def test_a_refused_request_is_not_logged_as_a_database_failure(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)
    assert (result["missing"], result["refused"], result["invalid"]) == (404, 400, 422)
    assert not [line for line in result["client_error_log"] if line.startswith("ERROR")]
    # A constraint violation is a failure: it stays an error, with the driver's text.
    assert any(line.startswith("ERROR:database:") for line in result["failure_log"])
    assert any("UNIQUE constraint failed" in line for line in result["failure_log"])


DRIVER_TEXTS = {
    # A table whose name contains "unique" is not a unique violation.
    "NOT NULL constraint failed: uniqueedition.label": "NOT NULL constraint failed: uniqueedition.label",
    "UNIQUE constraint failed: book.isbn": "Book with this isbn already exists",
    'duplicate key value violates unique constraint "uniqueedition_label_key"': (
        "UniqueEdition with this label already exists"),
    "Duplicate entry '111' for key 'book.isbn'": "Book with this isbn already exists",
    "UNIQUE constraint failed: ebook.isbn": "A record with the same values already exists",
    "FOREIGN KEY constraint failed": (
        "The record refers to a record that does not exist, or another record still refers to it"),
    # PostgreSQL appends the rejected row, which can hold any stored value.
    'null value in column "title" of relation "book" violates not-null constraint\n'
    'DETAIL:  Failing row contains (111, null, 4).': (
        'null value in column "title" of relation "book" violates not-null constraint'),
}


def test_the_driver_text_is_classified_by_phrase(run_generated_backend):
    script = """
    from database import integrity_error_detail

    class Violation(Exception):
        def __init__(self, text):
            self.orig = text

    texts = json.loads(%r)
    print(json.dumps({text: integrity_error_detail(Violation(text)) for text in texts}))
    """ % json.dumps(list(DRIVER_TEXTS))
    assert run_generated_backend(_model(), script) == DRIVER_TEXTS
