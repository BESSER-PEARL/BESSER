"""An attribute that stores a credential is written through the API and never read from it.

The search endpoint offered every text attribute as a partial-match filter and
every response carried every column, so a stored password hash could be read
directly, or recovered one character at a time through the filter.
"""
import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.structural_utils import is_credential_attribute

SCRIPT = """
    created = client.post("/account/", json={"email": "ann@example.org", "passwordHash": "s3cr3t-digest",
                                             "apiToken": "tok-123", "tokensUsed": 7})
    account = created.json()["account"]["id"]
    session = client.post("/session/", json={"device": "laptop", "account": account}).json()
    updated = client.put(f"/account/{account}/", json={"email": "ann@example.org", "passwordHash": "n3w-digest",
                                                       "apiToken": "tok-456", "tokensUsed": 8})
    responses = {
        "created": created.json(), "updated": updated.json(),
        "one": client.get(f"/account/{account}/").json(),
        "list": client.get("/account/").json(),
        "detailed": client.get("/account/", params={"detailed": True}).json(),
        "page": client.get("/account/paginated/").json(),
        "related": client.get("/session/", params={"detailed": True}).json(),
        "sessions": client.get(f"/account/{account}/sessions/").json(),
    }
    by_hash = client.get("/account/search/", params={"passwordHash": "zzz"}).json()
    by_email = client.get("/account/search/", params={"email": "ann"}).json()
    responses["search"] = by_email
    import sql_alchemy
    from database import SessionLocal
    with SessionLocal() as db:
        stored = db.query(sql_alchemy.Account).first()
        stored = [stored.passwordHash, stored.apiToken]
    responses["deleted"] = [client.delete(f"/session/{session['id']}/").json(),
                            client.delete(f"/account/{account}/").json()]
    print(json.dumps({
        "responses": responses, "by_hash": len(by_hash), "by_email": len(by_email), "stored": stored,
        "search_params": [p["name"] for p in client.get("/openapi.json").json()
                          ["paths"]["/account/search/"]["get"].get("parameters", [])],
    }))
"""


def _model() -> DomainModel:
    account = Class(name="Account", attributes={
        Property(name="email", type=StringType),
        Property(name="passwordHash", type=StringType),
        Property(name="apiToken", type=StringType, is_optional=True),
        Property(name="tokensUsed", type=IntegerType),
    })
    session = Class(name="Session", attributes={Property(name="device", type=StringType)})
    sessions = BinaryAssociation(name="Account_Sessions", ends={
        Property(name="account", type=account, multiplicity=Multiplicity(1, 1)),
        Property(name="sessions", type=session, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Portal", types={account, session}, associations={sessions})


@pytest.mark.parametrize("name, expected", [
    ("password", True), ("passwordHash", True), ("hashed_password", True), ("Passwd", True),
    ("api_key", True), ("apiKey", True), ("clientSecret", True), ("refresh_token", True),
    ("email", False), ("title", False), ("keyword", False), ("hashtag", False),
])
def test_credentials_are_recognised_by_name_whatever_the_separators(name, expected):
    assert is_credential_attribute(Property(name=name, type=StringType)) is expected


def test_only_text_attributes_are_credentials():
    assert not is_credential_attribute(Property(name="tokensUsed", type=IntegerType))


def test_a_credential_is_stored_but_never_returned_or_filtered(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)

    assert result["stored"] == ["n3w-digest", "tok-456"]
    returned = str(result["responses"])
    for leaked in ("passwordHash", "apiToken", "digest", "tok-"):
        assert leaked not in returned
    assert "tokensUsed" in returned and "ann@example.org" in returned

    assert result["search_params"] == ["email", "tokensUsed"]
    # An unknown query parameter filters nothing, so the probe value changes no result.
    assert result["by_hash"] == result["by_email"] == 1
