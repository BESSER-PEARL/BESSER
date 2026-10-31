"""Credential formats the scrub must catch, and code it must leave alone.

The samples are synthetic: the right shape for each provider, not real keys.
"""

import ast
import hashlib
import os
import re
import time
import tomllib
import warnings

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.web_app import WebAppGenerator
from besser.utilities.web_modeling_editor.backend.services.spec_driven import secret_redaction
from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import (
    REDACTED_SECRET,
    redact_text,
    scrub_secret_files,
)

# Assembled from parts so no sample is a literal token that push protection flags.
A = "A1b2C3d4E5f6G7h8I9j0" + "K1l2M3n4O5p6Q7r8S9t0"

# 13 of these 18 passed through unredacted before the gitleaks rules.
CREDENTIALS = {
    "openai": "sk-proj-" + A + A,
    "anthropic": "sk-ant-api03-" + A,
    "github classic": "ghp_" + A[:36],
    "github fine-grained": "github_pat_11ABCDEFG0" + A[:12] + "_" + A + "abcdefghijklmnopqrs",
    "aws access key": "AKIA" + "IOSFODNN7EXAMPLQ",
    "aws temporary key": "ASIA" + "IOSFODNN7EXAMPLQ",
    "stripe live": "sk_live_" + A[:24],
    "stripe restricted": "rk_live_" + A[:24],
    "google api key": "AIzaSyD" + A[:32],
    "slack bot": "xoxb-123456789012-1234567890123-" + A[:24],
    "slack webhook": "https://hooks.slack.com/services/T00000000/B00000000/" + A[:24],
    "gitlab pat": "glpat-" + A[:20],
    "hugging face": "hf_" + "qWeRtYuIoPaSdFgHj" + "KlZxCvBnMqWeRtYuI",
    "npm": "npm_" + A[:36],
    "sendgrid": "SG." + A[:22] + "." + A[:43].ljust(43, "x"),
    "pem private key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA" + A + "\n-----END RSA PRIVATE KEY-----",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0." + A[:43],
    "database url password": "S3cr3tPassw0rd",
}


def _in_source(name: str, secret: str) -> str:
    if name == "database url password":
        return f'DB = "postgresql://admin:{secret}@db.internal:5432/app"\n'
    return f'client = Client("""{secret}""")\n'


@pytest.mark.parametrize("name", CREDENTIALS)
def test_common_credential_formats_are_redacted_and_the_source_still_parses(name):
    out, findings = redact_text(_in_source(name, CREDENTIALS[name]), env_style=False)

    assert CREDENTIALS[name] not in out
    assert findings >= 1
    assert REDACTED_SECRET in out
    ast.parse(out)


def test_every_vendored_rule_compiles_in_python_re():
    with open(secret_redaction._GITLEAKS_RULES_PATH, "rb") as handle:
        config = tomllib.load(handle)
    patterns = list(config["allowlist"]["regexes"])
    for rule in config["rules"]:
        patterns.append(rule["regex"])
        for allowlist in rule.get("allowlists", ()):
            patterns.extend(allowlist.get("regexes", ()))

    assert config["rules"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. "Possible nested set" from a Go [[:alnum:]]
        for pattern in patterns:
            re.compile(pattern, re.ASCII)


def test_prefixless_rules_are_not_vendored():
    """``generic-api-key`` matches on a variable name and entropy alone;
    rewriting source on that basis breaks generated apps."""
    ids = {rule.id for rule in secret_redaction._TOKEN_RULES}

    assert {"generic-api-key", "adafruit-api-key", "kubernetes-secret-yaml"}.isdisjoint(ids)
    assert {"github-fine-grained-pat", "private-key", "jwt", "sendgrid-api-token"} <= ids


def test_a_long_base64_line_is_scanned_in_linear_time():
    """gitleaks writes for RE2, which is linear; under Python's backtracking
    ``re`` the nested ``[\\w.-]{0,50}?`` of the okta rule took 38 s on one
    607 KB inlined-wasm bundle whose base64 happened to contain "okta"."""
    blob = ("Zm9vYmFyYmF6cXV4" * 25 + "okta" + "SUMOLOGIC") * 1000  # ~410 KB, one line
    started = time.monotonic()

    redact_text(blob, env_style=False)

    assert time.monotonic() - started < 5  # ~0.1 s now; ~66 s with the nested prefix


def test_a_placeholder_shaped_like_a_token_is_left_alone():
    src = 'STRIPE_KEY = "sk_live_yourstripekeyhere12"\n'

    assert redact_text(src, env_style=False) == (src, 0)


@pytest.mark.parametrize("url", [
    "postgresql://postgres:postgres@db:5432/app",
    "postgresql+psycopg2://username:password@localhost:5432/db_name",
    "mysql://app:s0me-l0cal-pass@mysql/app",
    "postgresql://app:${DB_PASSWORD}@db.example.org/app",
    "postgresql://app:Local_pass_42@127.0.0.1:5432/app",
    "postgresql://app:Local_pass_42@10.0.0.7:5432/app",
    "postgresql://app:Local_pass_42@host.docker.internal:5432/app",
    "postgresql://app:Local_pass_42@db.localhost:5432/app",
    "postgresql://app:Local_pass_42@postgres.local:5432/app",
])
def test_a_database_url_for_a_local_host_or_a_placeholder_is_left_alone(url):
    """A generated docker-compose gives the database container the same
    password; redacting it in the URL would break the app."""
    src = f'DATABASE_URL = "{url}"\n'

    assert redact_text(src, env_style=False) == (src, 0)


@pytest.mark.parametrize("src", [
    'url = f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:5432/x"\n',
    'url = f"postgresql://{settings.db_user}:{settings.db_password}@{settings.db_host}/{settings.db_name}"\n',
    'url = "mysql+pymysql://%s:%s@%s/%s" % (user, password, host, name)\n',
    'url = "postgresql://%(user)s:%(password)s@%(host)s.example.org/x" % config\n',
    'url = "postgresql://{}:{}@{}.{}".format(user, password, host, domain)\n',
    'url = "postgresql://app:$DB_PASSWORD@db.example.org/app"\n',
])
def test_code_that_builds_a_database_url_is_not_rewritten(src):
    """``{settings.db_password}`` became ``[REDACTED]``: the app then failed
    database authentication. A template is not a secret."""
    assert redact_text(src, env_style=False) == (src, 0)


def test_an_env_holding_only_a_local_database_url_is_kept(tmp_path):
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://app:Local_pass_42@127.0.0.1:5432/app\n",
                                   encoding="utf-8")

    assert scrub_secret_files(str(tmp_path)).removed_files == ()


@pytest.mark.parametrize("url", [
    "mongodb+srv://svc:Zq9!xT2%23pLm@cluster0.ab1cd.mongodb.net/app",
    "redis://:Zq9xT2pLmQ@cache.example.net:6379/0",
])
def test_a_database_url_password_for_a_remote_host_is_redacted_alone(url):
    password = url.split("@")[0].rsplit(":", 1)[1]
    out, findings = redact_text(f'url = "{url}"\n', env_style=False)

    assert out == f'url = "{url.replace(password, REDACTED_SECRET)}"\n'
    assert findings == 1


PEM_BODY = "MIIEowIBAAKCAQEA" + A * 3


@pytest.mark.parametrize("src", [
    f"KEY = '''-----BEGIN RSA PRIVATE KEY-----\n{PEM_BODY}\n'''\n",  # cut before END
    '{"type": "service_account", "private_key": "-----BEGIN PRIVATE KEY-----\\n' + PEM_BODY
    + '\\n-----END PRIVATE KEY-----\\n"}',
])
def test_a_private_key_is_redacted_with_or_without_its_end_line(src):
    out, findings = redact_text(src, env_style=False)

    assert A not in out
    assert findings == 1


def test_a_marker_only_rule_redacts_the_token_after_the_marker():
    """Upstream's short-lived Bedrock rule is just its constant prefix:
    redacting that alone left the signed URL in place."""
    import base64
    signed = ("bedrock.amazonaws.com/?Action=CallWithBearerToken&X-Amz-Algorithm=AWS4-HMAC-SHA256"
              "&X-Amz-Signature=" + "9f" * 32 + "&Version=1")
    token = "bedrock-api-key-" + base64.b64encode(signed.encode()).decode()
    out, _ = redact_text(f'AWS_BEARER_TOKEN_BEDROCK = "{token}"\n', env_style=False)

    assert out == f'AWS_BEARER_TOKEN_BEDROCK = "{REDACTED_SECRET}"\n'


def test_a_base64_jwt_is_redacted_whole_not_just_its_header_label():
    """jwt-base64 labels its header with named groups; as the "first non-empty
    group" only that constant label would have been redacted."""
    import base64
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0." + A[:43]
    token = base64.b64encode(jwt.encode()).decode()
    out, _ = redact_text(f'TOKEN = "{token}"\n', env_style=False)

    assert out == f'TOKEN = "{REDACTED_SECRET}"\n'


def test_entropy_at_the_threshold_is_not_a_secret_as_in_gitleaks():
    """gitleaks skips ``entropy <= threshold`` (detect.go); this alibaba-shaped
    value has entropy exactly 2.0, the rule's threshold."""
    src = 'K = "' + "LTAI" * 6 + '"\n'

    assert redact_text(src, env_style=False) == (src, 0)


def test_the_one_pass_keyword_scan_finds_exactly_the_keywords_present():
    """A rule runs only when one of its keywords is found, so a missed keyword is a missed secret."""
    assert all(rule.keywords for rule in secret_redaction._TOKEN_RULES)
    text = (open(secret_redaction.__file__, encoding="utf-8").read()
            + " ".join(CREDENTIALS.values()) + " xoxb- xoxe- glpat- sk_live sk-ant-admin01-").lower()

    expected = {keyword for keyword in secret_redaction._KEYWORDS if keyword in text}
    assert secret_redaction._keywords_in(text) == expected


def test_no_value_rule_can_match_a_string_shorter_than_the_skip_length():
    from re import _parser as sre_parse

    patterns = [rule.regex.pattern for rule in secret_redaction._TOKEN_RULES] + [
        secret_redaction._SECRET_VALUE_TOKEN_RE.pattern, secret_redaction._PEM_PRIVATE_KEY_RE.pattern,
        secret_redaction._DB_URL_PASSWORD_RE.pattern]

    shortest = min(sre_parse.parse(pattern, re.ASCII).getwidth()[0] for pattern in patterns)
    assert shortest >= secret_redaction._MIN_VALUE_SECRET_LENGTH


def _tree_digest(root: str) -> dict:
    digests = {}
    for directory, _dirs, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            with open(path, "rb") as handle:
                digests[os.path.relpath(path, root)] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def test_redaction_leaves_generated_apps_unchanged(tmp_path, library_book_author_model):
    """Golden: BESSER's own generators write no credential, so the scrub must
    not rewrite a single byte of their output."""
    screen = Screen(name="Home", description="Home", view_elements={Text(name="t", content="Hi")},
                    is_main_page=True)
    gui = GUIModel(name="LibraryApp", package="org.library", versionCode="1", versionName="1.0",
                   modules={Module(name="Main", screens={screen})}, description="Library")
    WebAppGenerator(model=library_book_author_model, gui_model=gui,
                    output_dir=str(tmp_path / "web_app")).generate()
    for dbms in ("sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"):
        SQLAlchemyGenerator(model=library_book_author_model,
                            output_dir=str(tmp_path / f"sqlalchemy_{dbms}")).generate(dbms=dbms)
    before = _tree_digest(str(tmp_path))
    assert len(before) > 20

    result = scrub_secret_files(str(tmp_path))

    assert result.findings == 0, result
    assert _tree_digest(str(tmp_path)) == before
