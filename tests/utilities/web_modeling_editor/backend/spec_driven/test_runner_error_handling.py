"""Runner error mapping, packaging exclusions and slot release (review 2026-09-28).

* any ValueError from the worker became INVALID_KEY, and a provider failure
  after files were written deleted the paid-for workspace;
* ``*.log`` / ``*.db`` are skipped by the secret scrub but the zip walker
  filtered only directory names, so those files shipped unscrubbed;
* ``release_run_slot`` on a plain Semaphore grew capacity past the cap.
"""

import asyncio
import os
import zipfile

from besser.spec_driven_agent.errors import UpstreamLLMError
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _FakeClient,
    _FakeOrchestrator,
    _build_request,
    _collect_frames,
    _parse_frame,
)

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
    runner = SmartGenerationRunner(_build_request())
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

    runner = SmartGenerationRunner(_build_request())
    runner.temp_dir = str(tmp_path)
    _done, entry = runner._package_result(str(result))

    with zipfile.ZipFile(entry.file_path) as archive:
        names = set(archive.namelist())
        assert names == {"main.py", "README.md", "BESSER_GENERATION.md"}
        assert all(TOKEN not in archive.read(n).decode() for n in names)


def test_over_release_does_not_raise_the_cap(monkeypatch):
    from besser.utilities.web_modeling_editor.backend.constants import constants as C

    monkeypatch.setattr(C, "LLM_MAX_CONCURRENT_RUNS", 1)
    runner_module._reset_concurrency_semaphore_for_tests()

    async def _exercise():
        assert runner_module.try_acquire_run_slot() is True
        runner_module.release_run_slot()
        runner_module.release_run_slot()  # double release
        assert runner_module.try_acquire_run_slot() is True
        assert runner_module.try_acquire_run_slot() is False

    try:
        asyncio.run(_exercise())
    finally:
        runner_module._reset_concurrency_semaphore_for_tests()


# A Mistral/Nebius-style key: no known prefix, so the token pattern misses it.
PLAIN_KEY = "Zq8mV3xK9pLr2Tw7Yb4Nc6Hd1Fg5Js0A"


def test_own_key_without_a_prefix_is_redacted_everywhere():
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import (
        redact_text,
    )

    assert redact_text(f"key={PLAIN_KEY}")[0] == f"key={PLAIN_KEY}"  # the gap
    out, findings = redact_text(f"auth {PLAIN_KEY} / {PLAIN_KEY}", secrets=(PLAIN_KEY,))
    assert PLAIN_KEY not in out and findings == 2


def test_short_values_are_not_treated_as_secrets():
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import (
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


def test_run_key_never_leaves_in_frames_or_the_zip(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _EchoesTheKey)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runner = SmartGenerationRunner(_build_request(provider="mistral", api_key=PLAIN_KEY,
                                                  llm_model="mistral-large-latest"))
    frames = asyncio.run(_collect_frames(runner))
    assert frames and all(PLAIN_KEY.encode() not in f for f in frames)
    entry = asyncio.run(runner_module.SMART_RUN_REGISTRY.pop(runner.run_id))
    with zipfile.ZipFile(entry.file_path) as archive:
        assert all(PLAIN_KEY not in archive.read(n).decode() for n in archive.namelist())
