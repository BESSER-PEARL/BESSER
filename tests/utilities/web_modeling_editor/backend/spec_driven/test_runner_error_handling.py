"""Runner error mapping, packaging exclusions and slot release (review 2026-09-28).

* any ValueError from the worker became INVALID_KEY, and a provider failure
  after files were written deleted the paid-for workspace;
* ``*.log`` / ``*.db`` are skipped by the secret scrub but the zip walker
  filtered only directory names, so those files shipped unscrubbed;
* releasing a run slot on a plain Semaphore grew capacity past the cap.
"""

import asyncio
import os
import zipfile

import pytest

from besser.spec_driven_agent.errors import CheckpointMismatchError, InvalidApiKeyError, UpstreamLLMError
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runtime import RunSlots
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _FakeClient,
    _FakeOrchestrator,
    _build_request,
    _collect_frames,
    _parse_frame,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime

TOKEN = "sk-ant-REALSECRET0123456789abcdef"


class _WroteThenProviderFailed(_FakeOrchestrator):
    def run(self, instructions: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "main.py"), "w", encoding="utf-8") as fh:
            fh.write("print('partial')\n")
        raise UpstreamLLMError("OpenAI API call failed: Request timed out.")


class _BareValueError(_FakeOrchestrator):
    def run(self, instructions: str) -> str:
        raise ValueError("could not parse the generator selection")


def _run(monkeypatch, orchestrator_cls):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", orchestrator_cls)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    parsed = [_parse_frame(f) for f in asyncio.run(_collect_frames(runner))]
    return runner, parsed


def test_provider_failure_after_output_keeps_the_workspace(monkeypatch):
    runner, parsed = _run(monkeypatch, _WroteThenProviderFailed)
    done = [p for p in parsed if p["event"] == "done"]
    codes = [p["code"] for p in parsed if p["event"] == "error"]
    assert len(done) == 1 and done[0]["incomplete"] is True
    assert "INCOMPLETE" in codes and "UPSTREAM_LLM" not in codes
    assert runner.temp_dir and os.path.isdir(runner.temp_dir)


def test_a_non_auth_value_error_is_not_invalid_key(monkeypatch):
    _runner, parsed = _run(monkeypatch, _BareValueError)
    codes = [p["code"] for p in parsed if p["event"] == "error"]
    assert "INVALID_KEY" not in codes
    assert codes[-1] == "INTERNAL"


def test_scrub_skipped_files_never_reach_the_zip(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    (result / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (result / "README.md").write_text("# app\n", encoding="utf-8")
    (result / "server.log").write_text(f"auth with {TOKEN}\n", encoding="utf-8")
    (result / "app.db").write_text(TOKEN, encoding="utf-8")

    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    runner.temp_dir = str(tmp_path)
    done, entry = runner._package_result(str(result))

    with zipfile.ZipFile(entry.file_path) as archive:
        names = set(archive.namelist())
        assert names == {
            "main.py", "README.md", "BESSER_GENERATION.md", f".agent-trace/{done.runId}.json",
            ".besser_attribution.json",
        }
        assert all(TOKEN not in archive.read(n).decode() for n in names)


def test_over_release_does_not_raise_the_cap():
    slots = RunSlots(1)

    async def _exercise():
        assert slots.try_acquire() is True
        slots.release()
        slots.release()  # double release
        assert slots.try_acquire() is True
        assert slots.try_acquire() is False

    asyncio.run(_exercise())


# A Mistral/Nebius-style key: no known prefix, so the token pattern misses it.
PLAIN_KEY = "Zq8mV3xK9pLr2Tw7Yb4Nc6Hd1Fg5Js0A"


def test_own_key_without_a_prefix_is_redacted_everywhere():
    from besser.utilities.secret_redaction import (
        redact_text,
    )

    assert redact_text(f"key={PLAIN_KEY}")[0] == f"key={PLAIN_KEY}"  # the gap
    out, findings = redact_text(f"auth {PLAIN_KEY} / {PLAIN_KEY}", secrets=(PLAIN_KEY,))
    assert PLAIN_KEY not in out and findings == 2


def test_short_values_are_not_treated_as_secrets():
    from besser.utilities.secret_redaction import (
        redact_text,
    )

    assert redact_text("the test passed", secrets=("test", ""))[0] == "the test passed"


class _EchoesTheKey(_FakeOrchestrator):
    def run(self, instructions: str) -> str:
        if self.on_text:
            self.on_text(f"provider said: bad key {PLAIN_KEY}")
        os.makedirs(self.output_dir, exist_ok=True)
        for name in ("main.py", "config.py"):
            with open(os.path.join(self.output_dir, name), "w", encoding="utf-8") as fh:
                fh.write(f'KEY = "{PLAIN_KEY}"\n')
        return self.output_dir


def _interrupted_workspace(root, run_id: str) -> str:
    """What an interrupted run leaves behind: its files and a checkpoint."""
    workspace = root / f"besser_llm_{run_id}_interrupted"
    (workspace / "app").mkdir(parents=True)
    (workspace / "app" / "main.py").write_text("# 40 paid turns of work\n", encoding="utf-8")
    (workspace / ".besser_checkpoint.json").write_text("{}", encoding="utf-8")
    return str(workspace)


def _fails_on_resume(exc: Exception):
    class _Orchestrator(_FakeOrchestrator):
        def resume(self, instructions: str) -> str:
            raise exc

    return _Orchestrator


def _rejects_the_key(**_):
    raise ValueError("Invalid API key format")


def _rejects_the_project(*_):
    raise ValueError("The class diagram does not convert")


@pytest.mark.parametrize(
    ("orchestrator", "client_factory", "assemble", "code"),
    [
        (_fails_on_resume(CheckpointMismatchError("Checkpoint fingerprint does not match")),
         None, None, "BAD_REQUEST"),
        (_fails_on_resume(InvalidApiKeyError("The API rejected the key")), None, None, "INVALID_KEY"),
        (_fails_on_resume(FileNotFoundError("No checkpoint")), None, None, "BAD_REQUEST"),
        (_fails_on_resume(UpstreamLLMError("Request timed out")), None, None, "INCOMPLETE"),
        (_FakeOrchestrator, _rejects_the_key, None, "INVALID_KEY"),
        (_FakeOrchestrator, None, _rejects_the_project, "BAD_REQUEST"),
    ],
    ids=["fingerprint mismatch", "key rejected by the API", "no checkpoint", "provider failure",
         "malformed key", "malformed project"],
)
def test_a_failed_resume_keeps_the_interrupted_workspace(
    monkeypatch, tmp_path, orchestrator, client_factory, assemble, code,
):
    """Review E4: a resume that failed on bad input deleted the earlier
    attempt's workspace, so the retry with the right input got a 404."""
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path))
    run_id = "c" * 32
    workspace = _interrupted_workspace(tmp_path, run_id)
    monkeypatch.setattr(runner_module, "LLMOrchestrator", orchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", client_factory or (lambda **_: _FakeClient()))
    if assemble is not None:
        monkeypatch.setattr(runner_module, "assemble_models_from_project", assemble)

    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime(), resume_run_id=run_id)
    parsed = [_parse_frame(f) for f in asyncio.run(_collect_frames(runner))]

    assert code in [p["code"] for p in parsed if p["event"] == "error"]
    assert os.path.isfile(os.path.join(workspace, "app", "main.py"))
    assert runner_module._locate_run_temp_dir(run_id) == workspace


def test_run_key_never_leaves_in_frames_or_the_zip(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _EchoesTheKey)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runner = SmartGenerationRunner(_build_request(provider="mistral", api_key=PLAIN_KEY,
                                                  llm_model="mistral-large-latest"), runtime=current_runtime())
    frames = asyncio.run(_collect_frames(runner))
    assert frames and all(PLAIN_KEY.encode() not in f for f in frames)
    entry = asyncio.run(current_runtime().registry.pop(runner.run_id))
    with zipfile.ZipFile(entry.file_path) as archive:
        assert all(PLAIN_KEY not in archive.read(n).decode() for n in archive.namelist())
