"""An attribute that stores a credential is written through the API and never read from it.

The search endpoint offered every text attribute as a partial-match filter and
every response carried every column, so a stored password hash could be read
directly, or recovered one character at a time through the filter.
"""
import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
    is_credential_attribute,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator

CREDENTIALS = [
    "password", "Passwd", "pwd", "passphrase", "passcode", "passwordHash", "hashed_password", "password_digest",
    "secret", "clientSecret", "secret_key", "token", "apiToken", "access_token", "refresh_token", "authToken",
    "resetToken", "sessionToken", "api_key", "apiKey", "APIKey", "apikey", "privateKey", "accessKey", "authKey",
    "signingKey", "salt", "passwordSalt", "otp", "pin", "userPin", "PIN", "cvv", "credential", "credentials",
]
ORDINARY = [
    "email", "title", "keyword", "hashtag", "hash", "key", "publicKey", "secretary", "secretaryName", "tokens",
    "tokenized_text", "tokenSymbol", "tokenName", "tokenType", "tokenCount", "tokenBudget", "passwordHint",
    "passwordChangedAt", "secretQuestion", "secretSanta", "isSecret", "hasPassword", "spinner", "pinCode",
    "saltLevel", "description",
]

SCRIPT = """
    created = client.post("/account/", json={"email": "ann@example.org", "passwordHash": "s3cr3t-digest",
                                             "apiToken": "tok-123", "tokensUsed": 7})
    account = created.json()["account"]["id"]
    session = client.post("/session/", json={"device": "laptop", "account": account}).json()
    fetched = client.get(f"/account/{account}/").json()
    # A form sends back what it was given: the credentials it never received are absent or empty.
    echoed = client.put(f"/account/{account}/", json=dict(fetched["account"], email="ann@example.net"))
    emptied = client.put(f"/account/{account}/", json={"email": "ann@example.net", "tokensUsed": 9,
                                                       "passwordHash": "", "apiToken": None})
    import sql_alchemy
    from database import SessionLocal

    def stored():
        with SessionLocal() as db:
            record = db.query(sql_alchemy.Account).first()
        # Read after the session is closed: a credential is an ordinary loaded column.
        return [record.email, record.passwordHash, record.apiToken, record.tokensUsed]

    kept = stored()
    changed = client.put(f"/account/{account}/", json={"email": "ann@example.net", "tokensUsed": 9,
                                                       "passwordHash": "n3w-digest", "apiToken": "tok-456"})
    responses = {
        "created": created.json(), "echoed": echoed.json(), "changed": changed.json(),
        "one": client.get(f"/account/{account}/").json(),
        "list": client.get("/account/").json(),
        "detailed": client.get("/account/", params={"detailed": True}).json(),
        "page": client.get("/account/paginated/").json(),
        "related": client.get("/session/", params={"detailed": True}).json(),
        "sessions": client.get(f"/account/{account}/sessions/").json(),
        "search": client.get("/account/search/", params={"email": "ann"}).json(),
    }
    by_hash = len(client.get("/account/search/", params={"passwordHash": "zzz"}).json())
    without = client.post("/account/", json={"email": "bob@example.org", "tokensUsed": 0})
    bulk_without = client.post("/account/bulk/", json=[{"email": "cy@example.org", "tokensUsed": 0}])

    # A route written by hand that reads the credential and then returns the record.
    from fastapi import Depends
    from database import get_db

    @main_api.app.get("/login-check/")
    def login_check(database=Depends(get_db)):
        record = database.query(sql_alchemy.Account).first()
        return {"matches": record.passwordHash == "n3w-digest", "account": record}

    responses["custom"] = client.get("/login-check/").json()
    responses["deleted"] = [client.delete(f"/session/{session['id']}/").json(),
                            client.delete(f"/account/{account}/").json()]
    print(json.dumps({
        "responses": responses, "kept": kept,
        "statuses": [echoed.status_code, emptied.status_code, changed.status_code],
        "without": [without.status_code, without.json().get("detail")],
        "bulk_without": [bulk_without.status_code, bulk_without.json()["detail"]["errors"]],
        "by_hash": by_hash,
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


@pytest.mark.parametrize("name", CREDENTIALS)
def test_a_credential_is_recognised_by_the_words_of_its_name(name):
    assert is_credential_attribute(Property(name=name, type=StringType))


@pytest.mark.parametrize("name", ORDINARY)
def test_ordinary_data_that_mentions_a_credential_word_is_left_alone(name):
    assert not is_credential_attribute(Property(name=name, type=StringType))


def test_only_text_attributes_are_credentials():
    assert not is_credential_attribute(Property(name="pin", type=IntegerType))
    assert not is_credential_attribute(Property(name="token", type=StringType, is_id=True))


def test_the_model_validation_names_the_credentials():
    warnings = [w for w in _model().validate()["warnings"] if w.startswith("Credential attribute:")]
    assert len(warnings) == 2
    assert "attribute 'apiToken' of class 'Account'" in warnings[0]
    assert "attribute 'passwordHash' of class 'Account'" in warnings[1]


@pytest.fixture
def result(run_generated_backend):
    return run_generated_backend(_model(), SCRIPT)


def test_a_credential_is_never_returned_or_filtered(result):
    returned = str(result["responses"])
    for leaked in ("passwordHash", "apiToken", "digest", "tok-"):
        assert leaked not in returned
    assert "tokensUsed" in returned and "ann@example.net" in returned
    # A hand-written route that has read the credential still does not send it.
    assert result["responses"]["custom"] == {
        "matches": True, "account": {"id": 1, "email": "ann@example.net", "tokensUsed": 9}}

    assert result["search_params"] == ["email", "tokensUsed"]
    # An unknown query parameter filters nothing, so the probe value changes no result.
    assert result["by_hash"] == 1


def test_an_update_without_the_credential_keeps_the_stored_value(result):
    assert result["statuses"] == [200, 200, 200]
    assert result["kept"] == ["ann@example.net", "s3cr3t-digest", "tok-123", 9]


def test_a_required_credential_is_still_required_on_create(result):
    assert result["without"] == [422, "passwordHash is required"]
    status, errors = result["bulk_without"]
    assert status == 400 and errors == [{"index": 0, "error": "passwordHash is required"}]


def test_a_record_is_a_plain_mapping_without_its_credentials(tmp_path):
    """What ``SQLAlchemyGenerator`` alone gives: the attribute works, ``dict(record)`` leaves it out."""
    import importlib.util

    SQLAlchemyGenerator(model=_model(), output_dir=str(tmp_path)).generate()
    spec = importlib.util.spec_from_file_location("credential_models", tmp_path / "sql_alchemy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    account = module.Account(email="ann@example.org", passwordHash="digest", apiToken="tok", tokensUsed=1)
    assert account.passwordHash == "digest"
    assert module.Account.__credentials__ == ("apiToken", "passwordHash")
    assert dict(account) == {"email": "ann@example.org", "tokensUsed": 1}
    assert dict(module.Session(device="laptop")) == {"device": "laptop"}
