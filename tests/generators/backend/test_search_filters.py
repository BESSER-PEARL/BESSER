"""Which attributes the generated search endpoint offers as filters."""
import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DateTimeType, DomainModel, IntegerType, Multiplicity, Property,
    StringType,
)
from besser.generators.structural_utils import is_secret_attribute

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


# (attribute name, type, whether the search endpoint keeps it as a filter)
SECRET_NAMES = [
    ("password", StringType, False), ("Password", StringType, False), ("userPassword", StringType, False),
    ("passwordHash", StringType, False), ("hashed_password", StringType, False), ("pwd", StringType, False),
    ("passphrase", StringType, False), ("passcode", StringType, False), ("secret", StringType, False),
    ("clientSecret", StringType, False), ("api_secret", StringType, False), ("salt", StringType, False),
    ("seaSalt", StringType, False), ("token", StringType, False), ("accessToken", StringType, False),
    ("refreshTokenHash", StringType, False), ("resetToken", StringType, False), ("sessionToken", StringType, False),
    ("authToken", StringType, False), ("APIToken", StringType, False),
    # Hints, questions, counts, dates, flags and tokens a client reads back stay filters.
    ("passwordHint", StringType, True), ("secret_question_answer", StringType, True),
    ("secretSanta", StringType, True), ("passwordChangedAt", StringType, True),
    ("password_expires_at", StringType, True), ("password_expires_at", DateTimeType, True),
    ("passwordResetRequired", BooleanType, True), ("passwordHashAlgorithm", StringType, True),
    ("passwordHash2", StringType, True), ("password2", StringType, True), ("hasPassword", StringType, True),
    ("isSecret", StringType, True), ("tokenCount", StringType, True), ("tokenBudget", StringType, True),
    ("tokenExpiry", StringType, True), ("tokenExpiry", DateTimeType, True), ("tokenizer", StringType, True),
    ("token", IntegerType, True), ("csrfToken", StringType, True), ("deviceToken", StringType, True),
    ("inviteToken", StringType, True), ("shareToken", StringType, True), ("verificationToken", StringType, True),
    ("idToken", StringType, True), ("publicToken", StringType, True), ("saltLevel", StringType, True),
    ("salt_rounds", StringType, True), ("secretary", StringType, True), ("secretKey", StringType, True),
    ("apiKey", StringType, True), ("api_key", StringType, True), ("privateKey", StringType, True),
    ("publicKey", StringType, True), ("pin", StringType, True), ("pinHash", StringType, True),
    ("otp", StringType, True), ("cvv", StringType, True), ("credentials", StringType, True),
    ("keyword", StringType, True), ("hashtag", StringType, True), ("hash", StringType, True),
    ("digest", StringType, True), ("signature", StringType, True), ("resetCode", StringType, True),
    ("email", StringType, True),
]


@pytest.mark.parametrize("name, type_, stays_a_filter", SECRET_NAMES)
def test_a_stored_secret_is_recognised_by_the_words_of_its_name(name, type_, stays_a_filter):
    assert is_secret_attribute(Property(name=name, type=type_)) is not stays_a_filter


SECRET_SCRIPT = """
    def keys(body):
        return {key: sorted(value) if isinstance(value, dict) else value for key, value in body.items()}

    created = client.post("/account/", json={"email": "ann@example.org", "passwordHash": "digest-1",
                                             "apiToken": "tok-1", "tokensUsed": 7})
    account = created.json()["account"]["id"]
    session = client.post("/session/", json={"device": "laptop", "account": account})
    updated = client.put(f"/account/{account}/", json={"email": "ann@example.net", "passwordHash": "digest-2",
                                                       "apiToken": None, "tokensUsed": 8})
    one = client.get(f"/account/{account}/").json()
    listed = client.get("/account/").json()
    detailed = client.get("/session/", params={"detailed": True}).json()
    parameters = client.get("/openapi.json").json()["paths"]["/account/search/"]["get"]["parameters"]
    print(json.dumps({
        "statuses": [created.status_code, session.status_code, updated.status_code],
        "created": keys(created.json()), "updated": keys(updated.json()), "one": keys(one),
        "listed": sorted(listed[0]), "stored": [listed[0]["passwordHash"], listed[0]["apiToken"]],
        "related": sorted(detailed[0]["account"]),
        "filters": sorted(parameter["name"] for parameter in parameters),
        # An unknown query parameter is ignored, so it narrows nothing.
        "by_hash": len(client.get("/account/search/", params={"passwordHash": "zzz"}).json()),
        "by_email": len(client.get("/account/search/", params={"email": "ann"}).json()),
    }))
"""


def _account_model() -> DomainModel:
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


def test_a_stored_secret_is_not_a_search_filter_and_is_otherwise_unchanged(run_generated_backend):
    """A partial match on a password hash would reveal it one character at a time; nothing else changes."""
    result = run_generated_backend(_account_model(), SECRET_SCRIPT)

    assert result["filters"] == ["email", "tokensUsed"]
    assert result["by_hash"] == result["by_email"] == 1

    columns = ["apiToken", "email", "id", "passwordHash", "tokensUsed"]
    assert result["statuses"] == [200, 200, 200]
    assert result["created"] == {"account": columns, "sessions_ids": []}
    assert result["updated"] == {"account": columns, "sessions_ids": [1]}
    assert result["one"] == {"account": columns, "sessions_ids": [1]}
    assert result["listed"] == columns
    assert result["related"] == columns
    # The update stores what it is sent, including clearing an optional value.
    assert result["stored"] == ["digest-2", None]
