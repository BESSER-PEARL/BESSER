"""Security-boundary tests for generated artifacts and SSE-safe text."""

from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import (
    REDACTED_SECRET,
    looks_like_secret_env,
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

# ----------------------------------------------------------------------
# Redaction must never corrupt source code
#
# Live regression, 2026-09-11: across a batch of 10 generated apps, 4 files in
# 2 apps came out unable to import. The name-based .env heuristic was being
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
