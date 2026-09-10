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
