"""Tests for the multi-language agent generation helper."""

import asyncio
import io
import os
import zipfile
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.constants.constants import AGENT_TEMP_DIR_PREFIX
from besser.utilities.web_modeling_editor.backend.services.utils import agent_generation_utils as utils


class _FakeAgentGenerator:
    """Stand-in for BAFGenerator: records its arguments and writes a few files."""

    instances = []

    def __init__(self, model, output_dir=None, config=None, openai_api_key=None, config_yaml=None, **_):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.openai_api_key = openai_api_key
        self.config_yaml = config_yaml
        _FakeAgentGenerator.instances.append(self)

    def generate(self):
        # Mimic BAFGenerator's default when output_dir is None: <cwd>/output.
        out = self.output_dir or os.path.join(os.getcwd(), "output")
        os.makedirs(os.path.join(out, "skills"), exist_ok=True)
        with open(os.path.join(out, "agent.py"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# language={self.config.get('language')}\n")
        with open(os.path.join(out, "skills", "skill.py"), "w", encoding="utf-8") as fh:
            fh.write("pass\n")


def _fake_process_agent_diagram(json_data):
    return SimpleNamespace(language=(json_data.get("config") or {}).get("language"))


def _fake_agent_model_to_code(agent_model, file_path):
    # No ``agent`` attribute, so the helper falls back to the in-memory model.
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(f"LANGUAGE = {agent_model.language!r}\n")


def _patch(monkeypatch, tmp_path):
    _FakeAgentGenerator.instances = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "process_agent_diagram", _fake_process_agent_diagram)
    monkeypatch.setattr(utils, "agent_model_to_code", _fake_agent_model_to_code)
    monkeypatch.setattr(
        utils, "get_generator_info", lambda name: SimpleNamespace(generator_class=_FakeAgentGenerator)
    )


def _run(json_data, config, languages):
    return asyncio.run(
        utils.handle_multi_language_generation(json_data, config, languages, generate_agent_files_fn=None)
    )


def test_multi_language_generation_isolates_variants(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    config = {
        "languages": {"source": "english", "target": ["french", "german"]},
        "openai_api_key": "sk-test",
        "style": "formal",
    }
    json_data = {"model": {}, "config": config, "configYaml": "yaml: true"}

    buf, name = _run(json_data, config, config["languages"])

    assert name == "agents_multi_lang.zip"
    # Nothing is written to the process working directory.
    assert not (tmp_path / "output").exists()
    assert list(tmp_path.iterdir()) == []

    # Each variant gets its own config and its own output directory.
    gens = _FakeAgentGenerator.instances
    assert len(gens) == 3
    default, french, german = gens
    assert "language" not in default.config
    assert "languages" not in default.config
    assert default.config["style"] == "formal"
    assert french.config["language"] == "french"
    assert german.config["language"] == "german"
    assert french.config["source_language"] == "english"
    assert all(g.openai_api_key == "sk-test" for g in gens)
    assert all(g.config_yaml == "yaml: true" for g in gens)
    output_dirs = {g.output_dir for g in gens}
    assert len(output_dirs) == 3
    for out in output_dirs:
        assert os.path.isabs(out)
        assert os.path.basename(os.path.dirname(os.path.dirname(out))).startswith(AGENT_TEMP_DIR_PREFIX)
        # The per-request temp dir is cleaned up.
        assert not os.path.exists(out)

    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        names = set(zf.namelist())
        assert names == {
            f"{lang}/{path}"
            for lang in ("default", "french", "german")
            for path in (f"agent_model_{lang}.py", "agent.py", "skills/skill.py")
        }
        # Files from one language never leak into another.
        assert zf.read("french/agent.py").decode() == "# language=french\n"
        assert zf.read("german/agent.py").decode() == "# language=german\n"
        assert zf.read("default/agent.py").decode() == "# language=None\n"


def test_multi_language_generation_without_targets(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    config = {"languages": {"source": "english"}}

    buf, _ = _run({"model": {}, "config": config}, config, config["languages"])

    assert len(_FakeAgentGenerator.instances) == 1
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        assert {n.split("/")[0] for n in zf.namelist()} == {"default"}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [
    "../../../tmp/pwned",
    "..",
    "fr/../../x",
    "fr\\..\\x",
    "/etc/passwd",
    "fr.py",
    "",
    " french",
    "default",          # would clobber the default variant's folder
    123,
    None,
    "a" * 200,
])
def test_multi_language_generation_rejects_unsafe_language_codes(monkeypatch, tmp_path, bad):
    """Target languages become file and ZIP entry names; anything that is not
    a plain language name or code must be refused before a file is written."""
    _patch(monkeypatch, tmp_path)
    config = {"languages": {"source": "english", "target": ["french", bad]}}

    with pytest.raises(HTTPException) as excinfo:
        _run({"model": {}, "config": config}, config, config["languages"])

    assert excinfo.value.status_code == 400
    assert "language" in str(excinfo.value.detail).lower()
    assert _FakeAgentGenerator.instances == []
    assert list(tmp_path.iterdir()) == []


def test_multi_language_generation_rejects_duplicate_targets(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    config = {"languages": {"target": ["french", "French"]}}
    with pytest.raises(HTTPException) as excinfo:
        _run({"model": {}, "config": config}, config, config["languages"])
    assert excinfo.value.status_code == 400


def test_multi_language_generation_rejects_non_list_targets(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    config = {"languages": {"target": "../x"}}
    with pytest.raises(HTTPException) as excinfo:
        _run({"model": {}, "config": config}, config, config["languages"])
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("good", ["fr", "pt-BR", "zh_Hant", "luxembourgish", "en-GB-oxendict"])
def test_multi_language_generation_accepts_language_names_and_codes(monkeypatch, tmp_path, good):
    _patch(monkeypatch, tmp_path)
    config = {"languages": {"target": [good]}}
    buf, _ = _run({"model": {}, "config": config}, config, config["languages"])
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        assert f"{good}/agent_model_{good}.py" in zf.namelist()
