"""Tool results reach the model with non-ASCII text literal, not ``\\uXXXX``."""

import json

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import dumps_for_model
from besser.spec_driven_agent.providers.llm_client import _openai_messages_to_api

SAMPLES = ["Créer un élément", "图书管理", "Saved \U0001F680", "✓ built in 3s"]


@pytest.mark.parametrize("text", SAMPLES)
def test_non_ascii_stays_literal_and_round_trips(text):
    encoded = dumps_for_model({"content": text})
    assert text in encoded
    assert "\\u" not in encoded
    assert json.loads(encoded) == {"content": text}


def test_lone_surrogate_does_not_raise_and_stays_escaped():
    value = {"stdout": "bad byte \udcff here é"}
    encoded = dumps_for_model(value)
    encoded.encode("utf-8")  # a request body must be valid UTF-8
    assert "\\udcff" in encoded and "é" in encoded
    assert json.loads(encoded) == value


def test_read_file_result_keeps_accents(tmp_path):
    (tmp_path / "labels.ts").write_text('export const t = "Libellé à créer";\n', encoding="utf-8")
    result = ToolExecutor(str(tmp_path)).execute("read_file", {"path": "labels.ts"})
    assert "Libellé à créer" in result
    assert json.loads(result)["content"].splitlines()[0].endswith('"Libellé à créer";')


def test_openai_tool_call_arguments_keep_non_ascii():
    block = {"type": "tool_use", "id": "c1", "name": "write_file",
             "input": {"path": "a.ts", "content": "const é = '图'"}}
    api = _openai_messages_to_api("sys", [{"role": "assistant", "content": [block]}])
    arguments = api[-1]["tool_calls"][0]["function"]["arguments"]
    assert "const é = '图'" in arguments
    assert json.loads(arguments) == block["input"]
