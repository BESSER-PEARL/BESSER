"""Per-model inference settings for the spec-driven agent.

One table, one lookup. Everything here shapes the *request* we send:
sampling parameters, the output-token ceiling, and model-specific request
quirks. Anything not in the table gets the defaults -- an unknown model must
never be changed by adding a row for a different one.

What deliberately does NOT live here:

* **Context windows / compaction thresholds** -- ``compaction.py``. That is
  not a flat per-model value but a four-source precedence chain (measured
  self-hosted deployment > provider catalog > advertised family window >
  default), and the precedence is the part that makes it correct. Flattening
  it into this table would lose the provenance ordering. Edit the tables in
  ``compaction.py``; this module intentionally keeps no copy.
* **Pricing** -- ``llm_client._MODEL_PRICING``. Billing, not request shaping.

Why rows are conservative: a wrong row is worse than a missing one. Every
value below cites the vendor's own ``generation_config.json`` or model card,
or a probe recorded in the comment. Sampling defaults we cannot observe are
left alone rather than guessed at.
"""

from dataclasses import dataclass, field

# Output-token ceiling when neither the caller nor a row below says otherwise.
# Shared by the Claude / OpenAI / Mistral / Nebius providers.
DEFAULT_MAX_OUTPUT_TOKENS = 16384


@dataclass(frozen=True)
class ModelSettings:
    """Per-model request settings. Every field is optional; empty means the
    provider default.

    ``sampling`` holds OpenAI-protocol parameters (``temperature``, ``top_p``)
    that every OpenAI-compatible endpoint accepts by definition.

    ``extra_body`` holds parameters that are NOT in the OpenAI protocol
    (``top_k``, ``min_p``, ``repetition_penalty``). The ``openai`` SDK raises
    ``TypeError`` on these as plain kwargs, so they must ride in ``extra_body``
    -- and only for an endpoint verified to accept them, since an endpoint that
    rejects an unknown body key fails the whole run.
    """

    sampling: dict[str, float] = field(default_factory=dict)
    extra_body: dict[str, float] = field(default_factory=dict)
    max_output_tokens: int | None = None
    # ``reasoning_effort`` value required to make function tools work, or None.
    # Endpoint-gated by the caller -- see ``_needs_reasoning_none_for_tools``.
    reasoning_effort_for_tools: str | None = None


# Qwen3 "-2507" releases publish their tuned sampling values in the model repo's
# generation_config.json, which is what the model was evaluated with.
#
# Both rows below are the NON-THINKING variants: each card states the model
# "supports only non-thinking mode and does not generate <think></think> blocks",
# and Instruct-2507 adds that "specifying enable_thinking=False is no longer
# required" (confirmed against Nebius: no <think> tag, no reasoning_content).
# So there is no thinking knob to expose for these models, and none is added.
_QWEN3_30B_A3B_INSTRUCT_2507 = ModelSettings(
    # generation_config.json: temperature 0.7, top_p 0.8, top_k 20.
    # Model card "Best Practices": Temperature=0.7, TopP=0.8, TopK=20, MinP=0.
    # Aider ships the same values for Qwen3.
    #
    # Sent explicitly: a server that reads generation_config (vLLM, Ollama
    # Modelfile) applies them anyway, but an aggregator may substitute OpenAI
    # protocol defaults, and Nebius's effective default cannot be observed
    # from outside (pre-temperature logprobs, nondeterministic FP8 serving).
    sampling={"temperature": 0.7, "top_p": 0.8},
    # Nebius accepts both in extra_body (HTTP 200).
    extra_body={"top_k": 20, "min_p": 0.0},
)

_QWEN3_CODER_30B_A3B = ModelSettings(
    # generation_config.json: temperature 0.7, top_p 0.8, top_k 20,
    # repetition_penalty 1.05. Note this differs from Instruct-2507 above,
    # which recommends min_p=0 and no repetition_penalty -- the reason this
    # is a table and not one global setting.
    #
    # Only the two OpenAI-protocol values are sent. top_k and
    # repetition_penalty are deliberately withheld: this row serves the
    # keyless tier and self-hosted Ollama, and neither endpoint has
    # been verified to tolerate a non-protocol extra_body key. An endpoint
    # that 400s on one would fail every call.
    sampling={"temperature": 0.7, "top_p": 0.8},
)

_GPT_5_6 = ModelSettings(
    # No sampling parameters: OpenAI's reasoning models reject temperature /
    # top_p on chat/completions. The row records that "send nothing" is a
    # decision, so a future global sampling default cannot silently reach
    # gpt-5.6.
    #
    # gpt-5.6 rejects function tools unless reasoning is explicitly disabled;
    # the API answers "Function tools with reasoning_effort are not supported
    # for <model>" and directs you to set reasoning_effort to 'none'. Only
    # true on api.openai.com -- gateways validate the value against their own
    # enum and 400 on "none" -- so the caller applies the host gate.
    reasoning_effort_for_tools="none",
)


# Ordered longest-marker-first at import so a specific id always beats a
# shorter family prefix. Matched as a lowercased substring, the same way
# _get_pricing and compaction's window tables match.
#
# Markers are chosen not to overlap: "qwen3-30b-a3b-instruct-2507" and
# "qwen3-coder" cannot both match one id.
_REGISTRY: tuple[tuple[str, ModelSettings], ...] = tuple(
    sorted(
        (
            # Nebius Token Factory: "Qwen/Qwen3-30B-A3B-Instruct-2507".
            ("qwen3-30b-a3b-instruct-2507", _QWEN3_30B_A3B_INSTRUCT_2507),
            # Keyless tier + self-hosted Ollama: "qwen3-coder:30b",
            # "Qwen/Qwen3-Coder-30B-A3B-Instruct".
            ("qwen3-coder", _QWEN3_CODER_30B_A3B),
            # sol / terra / luna.
            ("gpt-5.6", _GPT_5_6),
        ),
        key=lambda row: len(row[0]),
        reverse=True,
    )
)

_DEFAULTS = ModelSettings()


def settings_for(model: str | None) -> ModelSettings:
    """Settings for ``model``, or empty defaults when it is not in the table."""
    if not model:
        return _DEFAULTS
    low = model.lower()
    for marker, settings in _REGISTRY:
        if marker in low:
            return settings
    return _DEFAULTS


def sampling_kwargs(model: str | None) -> dict:
    """Request kwargs carrying this model's sampling parameters.

    Empty for an unknown model, so the caller's request is left unchanged.
    """
    settings = settings_for(model)
    kwargs: dict = dict(settings.sampling)
    if settings.extra_body:
        kwargs["extra_body"] = dict(settings.extra_body)
    return kwargs


def max_output_tokens(model: str | None) -> int:
    """Output-token ceiling for ``model``.

    No row currently overrides the default. Instruct-2507's card recommends an
    output length of 16,384 while the from-scratch and modify paths raise the
    live ceiling to 32,768; that is left alone on purpose, because lowering it
    trades a documented soft recommendation for a hard truncation risk on a
    large write_file, and we have no measurement saying the trade pays.
    """
    return settings_for(model).max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS


def reasoning_effort_for_tools(model: str | None) -> str | None:
    """``reasoning_effort`` value this model needs to use tools, or None.

    Model half only. The caller must still apply the endpoint gate: the value
    is valid on the vendor's own API and rejected by gateways that re-serve
    the model.
    """
    return settings_for(model).reasoning_effort_for_tools
