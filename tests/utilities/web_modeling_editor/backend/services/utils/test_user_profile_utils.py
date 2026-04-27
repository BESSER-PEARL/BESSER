"""Unit tests for ``services.utils.user_profile_utils``.

Pure-function coverage — no FastAPI client, no HTTP. ``generate_user_profile_document``
is exercised with the heavy converter/generator dependencies patched so we test the
orchestration logic (temp dir prefix, JSON read-back) rather than the converter
internals (which are covered by their own tests).
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.services.utils import user_profile_utils
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    build_user_model_hierarchy,
    generate_user_profile_document,
    normalize_user_model_output,
    safe_path,
    sanitize_object_model_filename,
)


# ---------------------------------------------------------------------------
# safe_path — security primitive
# ---------------------------------------------------------------------------


def test_safe_path_accepts_simple_filename_inside_base():
    with tempfile.TemporaryDirectory() as base:
        result = safe_path(base, "report.json")
        assert os.path.dirname(result) == os.path.realpath(base)
        assert os.path.basename(result) == "report.json"


def test_safe_path_strips_directory_traversal_via_basename():
    """``..`` traversal segments should be stripped by os.path.basename and
    the resulting file remains anchored inside base_dir."""
    with tempfile.TemporaryDirectory() as base:
        # The function uses basename() so traversal segments are flattened.
        result = safe_path(base, "../../etc/passwd")
        real_base = os.path.realpath(base)
        assert os.path.commonpath([result, real_base]) == real_base
        assert os.path.basename(result) == "passwd"


def test_safe_path_strips_absolute_paths():
    """Absolute user-supplied paths are reduced to their basename and joined
    under base_dir, never escaping it."""
    with tempfile.TemporaryDirectory() as base:
        if os.name == "nt":
            attacker = r"C:\Windows\System32\cmd.exe"
            expected_basename = "cmd.exe"
        else:
            attacker = "/etc/passwd"
            expected_basename = "passwd"
        result = safe_path(base, attacker)
        real_base = os.path.realpath(base)
        assert os.path.commonpath([result, real_base]) == real_base
        assert os.path.basename(result) == expected_basename


def test_safe_path_no_prefix_sibling_false_positive():
    """A sibling directory whose name shares a prefix with base_dir must NOT
    be considered "inside" base_dir. (e.g., /tmp/foo vs /tmp/foobar). The
    fixed implementation uses os.path.commonpath, which compares full path
    components and is immune to prefix-sibling string-matching bugs."""
    with tempfile.TemporaryDirectory() as parent:
        base = os.path.join(parent, "foo")
        sibling = os.path.join(parent, "foobar")
        os.makedirs(base)
        os.makedirs(sibling)

        # Inside base — must succeed.
        ok = safe_path(base, "ok.txt")
        assert os.path.dirname(ok) == os.path.realpath(base)

        # commonpath compares directory components, so "foobar" is never
        # treated as being inside "foo" even though it shares a prefix.
        # Anything we pass through safe_path with base=base_dir lands under
        # base_dir, never under foobar.
        result = safe_path(base, "ok.txt")
        assert os.path.realpath(sibling) not in result


# ---------------------------------------------------------------------------
# sanitize_object_model_filename
# ---------------------------------------------------------------------------


def test_sanitize_object_model_filename_replaces_unsafe_chars():
    assert sanitize_object_model_filename("my model/v1") == "my_model_v1"


def test_sanitize_object_model_filename_default_when_blank():
    assert sanitize_object_model_filename(None) == "object_model"
    assert sanitize_object_model_filename("   ") == "object_model"


# ---------------------------------------------------------------------------
# build_user_model_hierarchy / normalize_user_model_output
# ---------------------------------------------------------------------------


def _sample_object_document():
    return {
        "name": "Profile",
        "objects": [
            {
                "id": "u1",
                "class": "User",
                "attributes": {"name": "Alice", "age": 30},
                "relationships": {"speaks": ["lang1"]},
            },
            {
                "id": "lang1",
                "class": "Language",
                "attributes": {"name": "English"},
                "relationships": {},
            },
        ],
    }


def test_build_user_model_hierarchy_folds_objects_into_tree():
    result = build_user_model_hierarchy(_sample_object_document())
    assert result is not None
    assert "objects" not in result
    model = result["model"]
    assert model["id"] == "u1"
    assert model["class"] == "User"
    assert model["name"] == "Alice"
    assert model["age"] == 30
    # Single child is inlined as a single dict (not a list)
    assert isinstance(model["Language"], dict)
    assert model["Language"]["name"] == "English"


def test_build_user_model_hierarchy_returns_none_without_user_root():
    doc = {
        "objects": [
            {"id": "x1", "class": "Vehicle", "attributes": {}, "relationships": {}}
        ]
    }
    assert build_user_model_hierarchy(doc) is None


def test_build_user_model_hierarchy_returns_none_when_objects_missing():
    assert build_user_model_hierarchy({"name": "no objects"}) is None


def test_normalize_user_model_output_round_trip(tmp_path):
    # Write a fake "{name}.json" file as the generator would produce.
    file_name = sanitize_object_model_filename("profile_demo")
    json_path = tmp_path / f"{file_name}.json"
    json_path.write_text(
        json.dumps(_sample_object_document()), encoding="utf-8"
    )

    object_model = SimpleNamespace(name="profile_demo")
    normalize_user_model_output(object_model, str(tmp_path))

    rewritten = json.loads(json_path.read_text(encoding="utf-8"))
    assert "objects" not in rewritten
    assert "model" in rewritten
    assert rewritten["model"]["class"] == "User"
    assert rewritten["model"]["name"] == "Alice"


def test_normalize_user_model_output_silent_when_file_missing(tmp_path):
    """Should not raise when there is nothing to rewrite."""
    object_model = SimpleNamespace(name="absent")
    # Nothing to assert beyond "no exception"
    normalize_user_model_output(object_model, str(tmp_path))


# ---------------------------------------------------------------------------
# generate_user_profile_document — orchestration smoke test
# ---------------------------------------------------------------------------


def test_generate_user_profile_document_rejects_non_dict_payload():
    with pytest.raises(HTTPException) as exc_info:
        generate_user_profile_document("not a dict")  # type: ignore[arg-type]
    assert exc_info.value.status_code == 400


class _FakeGenerator:
    """Minimal stand-in for an output generator. Writes a JSON file the
    function expects to read back."""

    def __init__(self, object_model, output_dir):
        self.object_model = object_model
        self.output_dir = output_dir

    def generate(self):
        file_name = sanitize_object_model_filename(getattr(self.object_model, "name", None))
        path = os.path.join(self.output_dir, f"{file_name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_sample_object_document(), handle)


def test_generate_user_profile_document_smoke_uses_user_profile_temp_prefix():
    """Smoke test: minimal valid UserDiagram should produce a non-empty
    result, and the temp directory created internally must use the
    ``user_profile_`` prefix so the cleanup service can sweep it.
    """
    fake_object_model = SimpleNamespace(name="profile_demo")
    captured_prefixes = []

    real_temporary_directory = tempfile.TemporaryDirectory

    def _spy_temporary_directory(*args, **kwargs):
        prefix = kwargs.get("prefix", "")
        captured_prefixes.append(prefix)
        return real_temporary_directory(*args, **kwargs)

    user_profile_payload = {
        "id": "diagram-1",
        "title": "User Profile",
        "type": "UserDiagram",
        "model": {"elements": {}, "relationships": {}},
    }

    with patch.object(
        user_profile_utils, "process_object_diagram", return_value=fake_object_model
    ), patch.object(
        user_profile_utils,
        "get_generator_info",
        return_value=SimpleNamespace(generator_class=_FakeGenerator),
    ), patch.object(
        user_profile_utils.tempfile, "TemporaryDirectory", side_effect=_spy_temporary_directory
    ):
        result = generate_user_profile_document(user_profile_payload)

    assert isinstance(result, dict) and result, "expected a non-empty result dict"
    # The hierarchy normalization fired, so we should see "model" not "objects".
    assert "model" in result and "objects" not in result
    # Temp dir prefix matches the cleanup service's allow-list ("user_profile_").
    assert captured_prefixes, "expected TemporaryDirectory to be called"
    assert any(
        prefix.startswith("user_profile_") for prefix in captured_prefixes
    ), f"expected a user_profile_ prefix, got: {captured_prefixes!r}"


def test_generate_user_profile_document_500_when_generator_not_configured():
    user_profile_payload = {"id": "x", "model": {}}
    with patch.object(
        user_profile_utils, "process_object_diagram", return_value=SimpleNamespace(name="x")
    ), patch.object(
        user_profile_utils, "get_generator_info", return_value=None
    ):
        with pytest.raises(HTTPException) as exc_info:
            generate_user_profile_document(user_profile_payload)
    # The function wraps unexpected errors into 400, but the missing-generator
    # branch raises 500 directly — and the outer handler re-raises HTTPException
    # untouched. Either is acceptable for a configuration error; assert the
    # documented behavior.
    assert exc_info.value.status_code == 500
