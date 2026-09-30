"""The scrub must not rewrite ordinary kebab-case identifiers.

``-`` is a body character in the token pattern, so an unanchored ``sk-``
matched inside any identifier long enough to satisfy the {16,} bound:
``./task-list-item-component`` became ``./ta[REDACTED]``. scrub_secret_files
rewrites files IN PLACE before the download zip and before a GitHub push,
and after Phase 3 has validated the tree - so a task app, the most common
demo prompt, shipped with a broken import and destroyed class names while
the run reported zero blockers.
"""

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import (
    _SECRET_VALUE_TOKEN_RE,
)


CORRUPTED_BEFORE_THE_FIX = [
    'import TaskListItem from "./task-list-item-component";',
    '<ul className="task-list-item-container-row">',
    ".task-list-item-detail-panel { color: red; }",
    'const helpdesk = "helpdesk-ticket-priority-high";',
    'export const KEYS = ["task-list-item-selected-id"];',
]

REAL_CREDENTIALS = [
    "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA",
    "sk-AAAAAAAAAAAAAAAAAAAAAAAA",
    "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AKIAIOSFODNN7EXAMPLE",
    "xoxb-AAAAAAAAAAAA-BBBBBBBBBBBB",
]


@pytest.mark.parametrize("line", CORRUPTED_BEFORE_THE_FIX)
def test_a_kebab_identifier_survives(line):
    assert _SECRET_VALUE_TOKEN_RE.sub("[REDACTED]", line) == line


@pytest.mark.parametrize("secret", REAL_CREDENTIALS)
def test_a_real_credential_is_still_redacted(secret):
    assert _SECRET_VALUE_TOKEN_RE.sub("[REDACTED]", secret) == "[REDACTED]"


@pytest.mark.parametrize("secret", REAL_CREDENTIALS)
def test_a_credential_in_context_is_still_redacted(secret):
    line = f'OPENAI_API_KEY = "{secret}"'

    assert secret not in _SECRET_VALUE_TOKEN_RE.sub("[REDACTED]", line)


def test_a_credential_after_a_word_boundary_is_still_caught():
    """The lookbehind must not exempt a key that follows ordinary punctuation."""
    for line in ('key="sk-AAAAAAAAAAAAAAAAAAAAAAAA"',
                 "key: sk-AAAAAAAAAAAAAAAAAAAAAAAA",
                 "(sk-AAAAAAAAAAAAAAAAAAAAAAAA)"):
        assert "[REDACTED]" in _SECRET_VALUE_TOKEN_RE.sub("[REDACTED]", line)
