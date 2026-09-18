"""Adaptive per-model compaction threshold.

Resolution order: measured override -> provider catalog (exact id) ->
known hosted windows (family substring) -> the flat default. The catalog
is the keyless tier's ``GET /models``; every test injects it, none fetch.
"""

import contextlib
import importlib
import json
import logging

import pytest

import besser.generators.llm.compaction as c
from besser.generators.llm.compaction import (
    COMPACT_RESERVE_TOKENS,
    COMPACT_TOKEN_THRESHOLD,
    _MIN_WORKABLE_THRESHOLD,
    effective_threshold,
    maybe_compact,
)

# Real ids and context_length values from the Command Code catalog, 2026-09-17.
COMMAND_CODE = {
    "gpt-5.6-luna": 1_050_000,
    "moonshotai/Kimi-K2.7-Code": 256_000,
    "inclusionai/ling-3.0-flash-sante:free": 262_144,
}
RESERVE = 32_768   # the from-scratch / modify output budget
# One whole-file read (tool_executor.MAX_FILE_READ at the measured 4.83
# chars/token) - the unit compaction.py sizes its own floor in.
ONE_FILE_READ = 4_100


def _install_catalog(monkeypatch, catalog):
    """Inject the catalog fetcher; returns the list of fetch calls made."""
    calls: list = []

    def fetch():
        calls.append(1)
        if isinstance(catalog, Exception):
            raise catalog
        return dict(catalog)

    monkeypatch.setattr(c, "_CATALOG", None)
    monkeypatch.setattr(c, "_CATALOG_LOADED", False)
    monkeypatch.setattr(c, "_fetch_catalog", fetch)
    return calls


@contextlib.contextmanager
def _reloaded_with_env(monkeypatch, **env):
    """Re-import compaction with ``env`` applied, then with it undone.

    The env-derived constants are read at import time, like the rest of the
    module's config. ``undo`` restores the pre-test environment (and every
    other patch) before the second reload, so the module ends up exactly as
    it started whatever the host shell exports.
    """
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    importlib.reload(c)
    try:
        yield
    finally:
        monkeypatch.undo()
        importlib.reload(c)


def _expected_adaptive(window: int, reserve: int) -> int:
    return min(c.MAX_COMPACT_THRESHOLD, int((window - reserve) * c._USABLE_WINDOW_FRACTION))


def _bulk(target_tokens: int) -> str:
    """Code-like text whose ESTIMATED size exceeds ``target_tokens``."""
    lines: list[str] = []
    while True:
        i = len(lines)
        lines.append(
            f"    value_{i} = compute_total(items_{i}, offset={i})  "
            f"# step {i} of the pipeline, validated against schema_{i}"
        )
        if i % 256 == 0 and c._estimate_tokens(
            [{"role": "user", "content": "\n".join(lines)}]
        ) > target_tokens:
            return "\n".join(lines)


class TestCatalogLayer:

    def test_advertised_window_raises_the_threshold_above_the_flat_default(self, monkeypatch):
        """A 256k model was getting the same 80k as a 1M one - ~30% of its window."""
        _install_catalog(monkeypatch, COMMAND_CODE)
        threshold = effective_threshold("moonshotai/Kimi-K2.7-Code", reserve=RESERVE)
        assert threshold > COMPACT_TOKEN_THRESHOLD
        # Never the whole usable window: the count is tiktoken-approximate
        # and the system prompt plus tool schemas ride outside ``messages``.
        assert threshold < (256_000 - RESERVE) * c._USABLE_WINDOW_FRACTION + 1
        assert threshold + RESERVE < 256_000

    def test_million_token_window_is_capped_by_the_ceiling(self, monkeypatch):
        _install_catalog(monkeypatch, COMMAND_CODE)
        uncapped = (1_050_000 - RESERVE) * c._USABLE_WINDOW_FRACTION
        assert COMPACT_TOKEN_THRESHOLD < c.MAX_COMPACT_THRESHOLD < uncapped
        assert effective_threshold("gpt-5.6-luna", reserve=RESERVE) == c.MAX_COMPACT_THRESHOLD

    def test_ceiling_is_env_configurable(self, monkeypatch):
        with _reloaded_with_env(monkeypatch, BESSER_LLM_MAX_COMPACT_THRESHOLD="120000"):
            _install_catalog(monkeypatch, COMMAND_CODE)
            assert c.effective_threshold("gpt-5.6-luna", reserve=RESERVE) == 120_000

    def test_measured_override_beats_a_larger_advertised_window(self, monkeypatch):
        """The Ollama box advertises 131k and truncates at ~64k (measured, see
        _SMALL_CONTEXT_WINDOWS). A catalog figure must never out-vote that."""
        _install_catalog(monkeypatch, {"qwen3-coder:30b": 1_000_000})
        assert effective_threshold("qwen3-coder:30b", reserve=RESERVE) == 60_000 - RESERVE

    def test_catalog_beats_the_family_table(self, monkeypatch):
        """The served figure outranks a family guess for the same id."""
        _install_catalog(monkeypatch, {"gpt-5.6-luna": 200_000})
        want = _expected_adaptive(200_000, RESERVE)
        assert want < c.MAX_COMPACT_THRESHOLD      # the catalog value, not the family row
        assert effective_threshold("gpt-5.6-luna", reserve=RESERVE) == want

    def test_catalog_failure_falls_through_and_is_not_retried(self, monkeypatch):
        """A dead endpoint costs one attempt per process, then the default."""
        calls = _install_catalog(monkeypatch, RuntimeError("connection refused"))
        for model in ("moonshotai/Kimi-K2.7-Code", "meituan/LongCat-2.0:free", "whatever"):
            assert effective_threshold(model) == COMPACT_TOKEN_THRESHOLD, model
            assert effective_threshold(model, reserve=RESERVE) == COMPACT_TOKEN_THRESHOLD, model
        assert len(calls) == 1

    def test_catalog_is_fetched_once_per_process(self, monkeypatch):
        calls = _install_catalog(monkeypatch, COMMAND_CODE)
        for model in ("gpt-5.6-luna", "moonshotai/Kimi-K2.7-Code", "unknown-model"):
            effective_threshold(model)
            effective_threshold(model, reserve=RESERVE)
        assert len(calls) == 1

    def test_advertised_path_keeps_the_workable_floor_and_warning(self, monkeypatch, caplog):
        _install_catalog(monkeypatch, {"tiny-model": 40_000})
        with caplog.at_level(logging.WARNING, logger="besser.generators.llm.compaction"):
            threshold = effective_threshold("tiny-model", reserve=RESERVE)
        assert threshold == 8_000
        assert threshold < _MIN_WORKABLE_THRESHOLD
        assert "under ~4 whole-file reads" in caplog.text

    def test_fetch_reads_the_free_tier_env_and_parses_context_length(self, monkeypatch):
        """Parses the live GET /models shape; sends the bearer the client uses."""
        monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://api.commandcode.ai/provider/v1/")
        monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "cc-secret")
        body = json.dumps({"object": "list", "data": [
            {"id": "gpt-5.6-luna", "object": "model", "created": 1,
             "owned_by": "command-code", "name": "GPT-5.6 Luna",
             "context_length": 1050000},
            {"id": "no-window", "object": "model"},
        ]}).encode()
        seen: dict = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return body

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            seen["timeout"] = timeout
            return _Response()

        monkeypatch.setattr(c.urllib.request, "urlopen", fake_urlopen)
        assert c._fetch_catalog() == {"gpt-5.6-luna": 1_050_000}
        assert seen["url"] == "https://api.commandcode.ai/provider/v1/models"
        assert seen["auth"] == "Bearer cc-secret"
        assert seen["timeout"] is not None and 0 < seen["timeout"] <= 5

    def test_fetch_is_a_noop_without_either_tier(self, monkeypatch):
        monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("BESSER_SPONSORED_LLM_BASE_URL", raising=False)

        def no_network(*args, **kwargs):
            raise AssertionError("network call attempted")

        monkeypatch.setattr(c.urllib.request, "urlopen", no_network)
        assert c._fetch_catalog() == {}

    def test_fetch_reads_the_sponsored_tier_too(self, monkeypatch):
        """A demo run on the sponsored tier gets real windows as well."""
        monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("BESSER_SPONSORED_LLM_BASE_URL", "https://api.commandcode.ai/provider/v1")
        monkeypatch.setenv("BESSER_SPONSORED_LLM_TOKEN", "cc-sponsored")
        body = json.dumps({"data": [{"id": "gpt-5.6-terra", "context_length": 1050000}]}).encode()
        seen: dict = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return body

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            return _Response()

        monkeypatch.setattr(c.urllib.request, "urlopen", fake_urlopen)
        assert c._fetch_catalog() == {"gpt-5.6-terra": 1_050_000}
        assert seen["url"] == "https://api.commandcode.ai/provider/v1/models"
        assert seen["auth"] == "Bearer cc-sponsored"

    def test_a_shared_endpoint_is_fetched_once_and_one_dead_endpoint_does_not_fail(self, monkeypatch):
        monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://api.commandcode.ai/provider/v1")
        monkeypatch.setenv("BESSER_SPONSORED_LLM_BASE_URL", "https://api.commandcode.ai/provider/v1")
        calls: list = []

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"data": [{"id": "m", "context_length": 256000}]}).encode()

        def fake_urlopen(request, timeout=None):
            calls.append(request.full_url)
            return _Response()

        monkeypatch.setattr(c.urllib.request, "urlopen", fake_urlopen)
        assert c._fetch_catalog() == {"m": 256_000}
        assert len(calls) == 1

        # Different endpoints, one dead: the live one still answers.
        monkeypatch.setenv("BESSER_SPONSORED_LLM_BASE_URL", "https://dead.example/v1")

        def flaky_urlopen(request, timeout=None):
            if "dead.example" in request.full_url:
                raise OSError("connection refused")
            return _Response()

        monkeypatch.setattr(c.urllib.request, "urlopen", flaky_urlopen)
        assert c._fetch_catalog() == {"m": 256_000}


class TestKnownWindows:
    """Hosted models whose catalogs do not expose context_length (OpenAI,
    Anthropic, Mistral) resolve through the family table."""

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6", "claude-opus-4-8", "claude-fable-5-1",
        "gpt-5.6-terra", "gpt-5.5", "gpt-4.1",
        "mistral-large-latest",
    ])
    def test_hosted_frontier_models_get_well_above_the_flat_default(self, model):
        threshold = effective_threshold(model, reserve=RESERVE)
        assert threshold > COMPACT_TOKEN_THRESHOLD, model
        assert threshold <= c.MAX_COMPACT_THRESHOLD, model

    def test_known_windows_follow_the_usable_fraction_and_ceiling(self):
        windows = {
            "claude-sonnet-4-6": 1_000_000,
            "claude-haiku-4-5": 200_000,
            "gpt-5.6-terra": 1_050_000,
            "gpt-5.4-mini": 400_000,
            "gpt-4o-mini": 128_000,
            "mistral-large-latest": 256_000,
        }
        for model, window in windows.items():
            want = _expected_adaptive(window, RESERVE)
            assert effective_threshold(model, reserve=RESERVE) == want, model

    def test_no_known_model_lands_materially_below_the_flat_default(self):
        """The invariant behind the fraction: adaptive thresholds exist to
        REMOVE compaction, so no row in the family table may hand a model
        materially less than today's flat default - only a measured row may
        clamp below it. gpt-4o (the OpenAI default, 128k) is the binding case:
        it was 31k under an earlier whole-window fraction, which would have
        shipped the read/compact/re-read spiral to the default BYOK path.
        "Materially" is one whole-file read.
        """
        for marker, _window in c._KNOWN_CONTEXT_WINDOWS:
            # A bare marker that ALSO matches a measured row is the documented
            # exception above ("only a measured row may clamp below it"):
            # un-namespaced ids look self-hosted, and the measured window for
            # that box is the more trustworthy figure. The cloud form of such
            # a model is vendor-namespaced and is checked below instead.
            if any(m in marker for m, _ in c._SMALL_CONTEXT_WINDOWS):
                continue
            for reserve in (COMPACT_RESERVE_TOKENS, RESERVE):
                threshold = effective_threshold(marker, reserve=reserve)
                assert threshold >= COMPACT_TOKEN_THRESHOLD - ONE_FILE_READ, (marker, reserve)

    def test_namespaced_form_of_a_measured_family_keeps_its_full_window(self):
        """The exception above must not swallow the cloud case: a vendor-
        namespaced id is served by a provider, not by our box."""
        for marker, window in c._KNOWN_CONTEXT_WINDOWS:
            if not any(m in marker for m, _ in c._SMALL_CONTEXT_WINDOWS):
                continue
            namespaced = f"Vendor/{marker}"
            threshold = effective_threshold(namespaced, reserve=COMPACT_RESERVE_TOKENS)
            assert threshold >= COMPACT_TOKEN_THRESHOLD - ONE_FILE_READ, namespaced

    def test_measured_small_mistral_rows_still_win_over_the_family_table(self):
        assert effective_threshold("mistral-small-latest") == 16_000
        assert effective_threshold("mistral-7b-instruct") == 16_000

    def test_unknown_model_id_keeps_exactly_todays_default(self, monkeypatch):
        """The BYOK path for anything we cannot place: byte-identical to today."""
        _install_catalog(monkeypatch, COMMAND_CODE)
        for model in ("gpt-5-mini", "llama-3.3-70b", "meituan/LongCat-2.0:free",
                      "some-byok-model", None):
            assert effective_threshold(model) == COMPACT_TOKEN_THRESHOLD, model
            assert effective_threshold(model, reserve=RESERVE) == COMPACT_TOKEN_THRESHOLD, model
            assert effective_threshold(model, threshold=40_000) == 40_000, model


class TestOperatorThreshold:
    """``BESSER_LLM_COMPACT_THRESHOLD`` set = a hard cap on every result;
    unset = the adaptive value stands (up to the ceiling)."""

    def test_when_set_it_caps_every_adaptive_result(self, monkeypatch):
        with _reloaded_with_env(monkeypatch, BESSER_LLM_COMPACT_THRESHOLD="50000"):
            _install_catalog(monkeypatch, COMMAND_CODE)
            assert c.COMPACT_TOKEN_THRESHOLD == 50_000
            for model in ("gpt-5.6-terra", "mistral-large-latest", "claude-haiku-4-5",
                          "moonshotai/Kimi-K2.7-Code", "some-byok-model"):
                assert c.effective_threshold(model, reserve=RESERVE) == 50_000, model
            # A measured row is still clamped below the cap, not raised to it.
            assert c.effective_threshold("devstral:24b") == 16_000

    def test_when_unset_the_adaptive_value_stands(self, monkeypatch):
        with _reloaded_with_env(monkeypatch, BESSER_LLM_COMPACT_THRESHOLD=None):
            _install_catalog(monkeypatch, COMMAND_CODE)
            assert c._OPERATOR_CAP is None
            assert c.effective_threshold("gpt-5.6-terra", reserve=RESERVE) == c.MAX_COMPACT_THRESHOLD
            assert c.effective_threshold("moonshotai/Kimi-K2.7-Code", reserve=RESERVE) > c.COMPACT_TOKEN_THRESHOLD
            assert c.effective_threshold("some-byok-model") == c.COMPACT_TOKEN_THRESHOLD


class TestEndToEnd:

    def test_wide_window_model_keeps_a_history_the_default_would_compact(
        self, tmp_path, monkeypatch,
    ):
        """~90k tokens: over the 80k default, under a 256k model's threshold."""
        _install_catalog(monkeypatch, COMMAND_CODE)
        chunk = _bulk(22_000)
        messages = [{"role": "user", "content": "build"}]
        for _ in range(4):
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})
            messages.append({"role": "user", "content": "go on"})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
        assert c._estimate_tokens(messages) > COMPACT_TOKEN_THRESHOLD

        _, unknown = maybe_compact(
            messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
            model="some-byok-model",
        )
        assert unknown is True
        _, wide = maybe_compact(
            messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
            model="moonshotai/Kimi-K2.7-Code",
        )
        assert wide is False
