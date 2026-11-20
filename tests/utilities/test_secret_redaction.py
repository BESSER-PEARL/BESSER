"""Security-boundary tests for generated artifacts and SSE-safe text."""

import os
import threading

import pytest

from besser.utilities.secret_redaction import (
    REDACTED_SECRET,
    looks_like_secret_env,
    redact_record_text,
    redact_text,
    scrub_secret_files,
)


def test_env_placeholders_are_not_treated_as_secrets():
    content = (
        "OPENAI_API_KEY=your-key-here\n"
        "DATABASE_URL=${DATABASE_URL}\n"
        "PASSWORD=<set-me>\n"
    )
    assert looks_like_secret_env(content) is False
    assert redact_text(content) == (content, 0)


def test_populated_generic_secret_assignment_is_redacted():
    safe, findings = redact_text("DATABASE_PASSWORD=correct-horse-battery-staple\n")
    assert safe == f"DATABASE_PASSWORD={REDACTED_SECRET}\n"
    assert findings == 1


def test_scrubber_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / "outside.txt"
    token = "ghp_0123456789abcdefghijklmnopqrstuv"
    outside.write_text(token, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    link = workspace / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        return

    result = scrub_secret_files(str(workspace))
    assert result.findings == 0
    assert outside.read_text(encoding="utf-8") == token


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_scrubber_skips_a_fifo_instead_of_blocking_on_it(tmp_path):
    """A run's ``mkfifo`` left in the workspace hung packaging forever:
    ``open()`` on a FIFO blocks until a writer appears."""
    (tmp_path / "app.py").write_text("KEY = 'ghp_0123456789abcdefghijklmnopqrstuvwxyz'\n", encoding="utf-8")
    os.mkfifo(tmp_path / "pipe")
    results = []
    worker = threading.Thread(target=lambda: results.append(scrub_secret_files(str(tmp_path))), daemon=True)
    worker.start()
    worker.join(5)

    assert results, "scrub_secret_files blocked on the FIFO"
    assert results[0].redacted_files == ("app.py",)

# ----------------------------------------------------------------------
# Redaction must never corrupt source code
#
# Regression: generated files came out unable to import. The name-based .env
# heuristic was being
# applied to Python, where it matches on the variable NAME and replaces the
# whole value with a bare [REDACTED] - legal in a .env file, a list containing
# an undefined name in Python:
#
#     SECRET_KEY = [REDACTED]                    NameError at import
#     ACCESS_TOKEN_EXPIRE_MINUTES = [REDACTED]   an INTEGER, not a secret
#     token_data = [REDACTED]                    a local variable
#
# The last two are not credentials at all; their names merely contain "TOKEN".
# ----------------------------------------------------------------------


def test_python_source_stays_parseable():
    import ast
    src = (
        'SECRET_KEY = "dev-secret"\n'
        "ACCESS_TOKEN_EXPIRE_MINUTES = 30\n"
        'API_TOKEN_HEADER = "X-Token"\n'
    )
    out, _ = redact_text(src, env_style=False)
    ast.parse(out)          # the whole point: it must still be Python
    assert "[REDACTED]" not in out


def test_non_secrets_whose_NAME_contains_token_are_untouched():
    src = "ACCESS_TOKEN_EXPIRE_MINUTES = 30\ntoken_data = payload.get('sub')\n"
    out, findings = redact_text(src, env_style=False)
    assert out == src
    assert findings == 0


def test_a_real_provider_key_in_source_is_still_redacted_inside_its_quotes():
    """The value-token pattern matches the token, not the quotes, so the file
    keeps parsing while the credential goes."""
    import ast
    src = 'KEY = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA"\n'
    out, findings = redact_text(src, env_style=False)
    assert findings == 1
    assert "sk-ant-api03" not in out
    assert out.strip() == 'KEY = "[REDACTED]"'
    ast.parse(out)


def test_env_style_redaction_is_unchanged():
    """The .env behaviour this protects must not regress."""
    out, findings = redact_text(
        "API_KEY=sk-live-abcdefghijklmnop\n", env_style=True
    )
    assert "sk-live" not in out
    assert findings >= 1


def test_env_style_defaults_to_true():
    """Other callers (SSE payload redaction) rely on the default."""
    out, _ = redact_text("SECRET=hunter2\n")
    assert "hunter2" not in out


@pytest.mark.parametrize("text, secret", [
    # curl -v output and the headers of its command line
    ("> X-API-Key: k9f8e7d6c5b4a3f2e1d0\r\n", "k9f8e7d6c5b4a3f2e1d0"),
    ("> Cookie: session=abcdef0123456789abcdef\r\n", "abcdef0123456789abcdef"),
    ("< Set-Cookie: sid=zyxwvu9876543210zyxw; HttpOnly\r\n", "zyxwvu9876543210zyxw"),
    ("curl -H 'X-Auth-Token: t0k3nv4lu3abcdef' https://api.example.com/x", "t0k3nv4lu3abcdef"),
    ('curl --header "Cookie: session=abcdef0123456789abcdef" https://api.example.com/x',
     "abcdef0123456789abcdef"),
    # a URL's basic credentials, as pip config and git remotes print them
    ("global.index-url='https://deploy:Pa55w0rdHere99@pypi.corp.example.com/simple'", "Pa55w0rdHere99"),
    ("origin\thttps://builder:Pa55w0rdHere99@git.example.com/team/app.git (fetch)", "Pa55w0rdHere99"),
])
def test_a_record_redacts_header_and_url_credentials(text, secret):
    safe = redact_record_text(text)
    assert secret not in safe and REDACTED_SECRET in safe


@pytest.mark.parametrize("text", [
    "headers: {'X-API-Key': apiKey, 'Cookie': cookieHeader}",
    "curl -H 'X-API-Key: $API_KEY' https://api.example.com/x",
    "BASE_URL = 'http://admin:admin@localhost:8000'",
    "a request with an X-API-Key header",
])
def test_the_records_header_and_url_rules_leave_code_and_templates_alone(text):
    assert redact_record_text(text) == text


# Text users receive: delivered files and the event stream keep these lines as they were.
DELIVERED = {
    "README.md": 'Try it:\n\ncurl -H "X-API-Key: dev-key-123456" http://localhost:8000/books\n'
                 'curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/me\n'
                 'curl -H "X-API-Key: <your key>" http://localhost:8000/books\n'
                 'Clone: git clone https://user:password@example.com/repo.git\n',
    "scripts/smoke_test.sh": "#!/bin/sh\ncurl -s -H 'X-Api-Key: local-dev-key-42' http://localhost:8000/health\n"
                             "curl -s --header \"Cookie: session=test-session-0001\" http://localhost:8000/me\n",
    "api.http": "GET http://localhost:8000/items\nX-API-Key: devkey-abcdef12\n\n###\n"
                "GET http://localhost:8000/me\nCookie: sid=abc123def456\n",
    "app/main.py": 'headers = {\n    "X-API-Key": api_key,\n}\nresponse.headers["Set-Cookie"] = f"sid={sid}"\n',
    "docker-compose.yml": "services:\n  web:\n    environment:\n"
                          "      UPSTREAM: https://svc:svcpass123@api.partner.example.org/v1\n",
    "openapi.yaml": "      example: Set-Cookie: sessionid=38afes7a8; HttpOnly\n",
    "message.txt": "Test it with `curl -H 'X-API-Key: dev-key-123456' http://localhost:8000/books`.",
}


@pytest.mark.parametrize("name", DELIVERED)
def test_delivered_files_and_streamed_text_keep_headers_and_web_urls(tmp_path, name):
    text = DELIVERED[name]
    assert redact_text(text, env_style=False) == (text, 0)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))

    result = scrub_secret_files(str(tmp_path))

    assert path.read_bytes() == text.encode("utf-8") and result.findings == 0


@pytest.mark.parametrize("text", [
    "Cookie: a" + " " * 40_000 + "b",
    " " * 40_000 + "x",
    "> " + " " * 40_000 + "Cookie: a",
    "X-" + "a" * 40_000,
    "-H 'Cookie: " + "a" * 40_000,
    "-H '" * 10_000,
    '"' * 40_000,
    "https://" + "a" * 20_000 + ":" + "b" * 20_000,
    "https://" + ":" * 40_000,
    'apiKey="' + "a" * 40_000,
    'apiKey="aaaaaaaa' * 4_000,
], ids=["spaced-value", "spaces", "spaced-prefix", "header-name", "unterminated-arg", "repeated-arg",
        "quotes", "url-no-at", "url-colons", "unterminated-literal", "repeated-literal"])
def test_the_record_rules_run_in_linear_time(text):
    import time

    started = time.perf_counter()
    redact_record_text(text)
    assert time.perf_counter() - started < 2


def test_a_long_run_of_letters_is_redacted_in_linear_time():
    """The quoted-literal rule tried every start inside a word: minutes on a minified line or base64."""
    import time

    from besser.utilities.secret_redaction import redact_shared_file

    started = time.perf_counter()
    assert redact_shared_file("app.js", "y" * 100_000)[1] == 0
    assert time.perf_counter() - started < 5
    assert redact_shared_file("app.js", 'const apiKey = "wk_4f8a9b2c7d1e6f3a";')[0] == 'const apiKey = "[REDACTED]";'


@pytest.mark.parametrize("text", [
    "\n" * 20_000,
    "\r\n" * 10_000,
    (" " * 50 + "\n") * 400,
    "TOKEN" * 4_000,
    "API_KEY=" + "\n" * 20_000,
    "TOKEN=" + " " * 20_000 + "\r\n",
], ids=["newlines", "crlf-lines", "blank-lines", "repeated-name", "empty-value-then-newlines",
        "spaces-then-crlf"])
def test_env_assignments_are_redacted_in_linear_time(text):
    """Blank lines before a line tried each line start against all the whitespace after it."""
    import time

    started = time.perf_counter()
    redact_text(text)
    assert time.perf_counter() - started < 2


# The .env assignment rule before it was made linear: the reference for its matches.
_ENV_RULE_BEFORE = (
    r"(?im)^(\s*[A-Za-z0-9_]*"
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)"
    r"[A-Za-z0-9_]*\s*=\s*)([^\r\n]+)$"
)

ENV_EDGE_CASES = [
    "API_KEY=abc123\n",
    "  API_KEY=abc123\n",
    "\n\n\nAPI_KEY=abc123\n",
    "\n  \n\t\nAPI_KEY=abc123",
    "API_KEY=abc123\r\nSECRET=def456\r\n",
    "API_KEY=abc123\r\nSECRET=def456",
    "\tSECRET_TOKEN\t=\tvalue with spaces  \n",
    "export API_KEY=abc123\nexport DB_PASSWORD='p@ss w0rd'\n",
    "MY_API-KEY_X=1\nAPI_KEY-X=2\nTOKEN-foo=3\nXAPI-KEYZ=4\nPRIVATE-KEY=5\n",
    "A_TOKEN=\nB_TOKEN=x\n",
    "A_TOKEN=   \n\n",
    "A_TOKEN=   \r\n  \n",
    "A_TOKEN= \t",
    "A_TOKEN=\n   \nnext\n",
    "API_KEY\n=split\n",
    "API_KEY =\n\n  value\n",
    "config:\n  api_key: abc\n  secret_key = xyz\n  token=\n",
    "TOKENTOKEN=1\nsecretsecret = 2\n",
    "API_KEY=changeme\nSECRET=${SECRET}\nTOKEN=your-key-here\n",
    "\r\rTOKEN=a\r\n",
    " API_KEY=abc\n SECRET=x\n",
    "",
    "TOKEN=",
    "=TOKEN\n",
]


def _env_rule_before(text):
    import re

    def replace(match):
        return match.group(0) if match.group(2).strip().lower() in ("", "changeme") else match.group(1) + "[R]"
    return re.compile(_ENV_RULE_BEFORE).sub(replace, text)


def _env_rule_now(text):
    from besser.utilities.secret_redaction import _SECRET_ENV_ASSIGNMENT_RE as rule

    def replace(match):
        value = match.group("value") if match.group("value") is not None else match.group("blank")
        prefix = match.string[match.start():match.start("value" if match.group("value") is not None else "blank")]
        return match.group(0) if value.strip().lower() in ("", "changeme") else prefix + "[R]"
    return rule.sub(replace, text)


@pytest.mark.parametrize("text", ENV_EDGE_CASES)
def test_the_env_rule_matches_as_before(text):
    assert _env_rule_now(text) == _env_rule_before(text)


def test_the_env_rule_matches_as_before_on_random_text():
    import random

    pieces = [" ", "\t", "\r", "\n", "\r\n", "=", " = ", "A", "x9", "_", "-", "#", "export ", " ",
              "TOKEN", "api_key", "API-KEY", "private-key", "Secret", "passwd", "value", "'v'", ":"]
    rng = random.Random(20261120)
    for _ in range(20_000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 14)))
        assert _env_rule_now(text) == _env_rule_before(text), repr(text)
