import json
import logging
import os
import re

from besser.BUML.metamodel.state_machine.agent import AgentReply

logger = logging.getLogger(__name__)


OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


def _normalize_setting(value):
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _is_effective_setting(value) -> bool:
    normalized = _normalize_setting(value)
    return isinstance(normalized, str) and normalized not in ("", "none", "original")


def _has_non_empty_texts(texts) -> bool:
    return any(isinstance(text, str) and text.strip() for text in texts or [])


def _resolve_model_name(config: dict | None, default: str = "gpt-5") -> str:
    model_config = config.get("llm") if isinstance(config, dict) else None
    if isinstance(model_config, str) and model_config.strip():
        return model_config.strip()
    if isinstance(model_config, dict):
        model_name = model_config.get("model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return default


def _extract_numbered_results(response_text: str) -> list[str]:
    numbered_matches = re.findall(r"^\s*\d+\.\s*(.+)$", response_text or "", flags=re.MULTILINE)
    if numbered_matches:
        return [entry.strip() for entry in numbered_matches if entry.strip()]
    return [line.strip() for line in (response_text or "").splitlines() if line.strip()]


def _resolve_openai_api_key(config: dict = None, openai_api_key: str = None) -> str | None:
    if isinstance(openai_api_key, str) and openai_api_key.strip():
        return openai_api_key.strip()

    if isinstance(config, dict):
        for candidate_key in ("openai_api_key", "openaiApiKey", OPENAI_API_KEY_ENV_VAR, "apiKey"):
            candidate_value = config.get(candidate_key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                return candidate_value.strip()

        system_section = config.get("system")
        if isinstance(system_section, dict):
            for candidate_key in ("openai_api_key", "openaiApiKey", OPENAI_API_KEY_ENV_VAR, "apiKey"):
                candidate_value = system_section.get(candidate_key)
                if isinstance(candidate_value, str) and candidate_value.strip():
                    return candidate_value.strip()

    env_value = os.getenv(OPENAI_API_KEY_ENV_VAR)
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip()

    return None


def flatten_agent_config_structure(raw_config):
    """Flatten structured agent configuration sections into the legacy flat shape."""
    if not isinstance(raw_config, dict):
        return raw_config

    flattened = dict(raw_config)
    section_field_map = {
        "presentation": {
            "agentLanguage": "agentLanguage",
            "agentStyle": "agentStyle",
            "languageComplexity": "languageComplexity",
            "sentenceLength": "sentenceLength",
            "interfaceStyle": "interfaceStyle",
            "voiceStyle": "voiceStyle",
            "avatar": "avatar",
            "useAbbreviations": "useAbbreviations",
        },
        "modality": {
            "inputModalities": "inputModalities",
            "outputModalities": "outputModalities",
        },
        "behavior": {
            "responseTiming": "responseTiming",
        },
        "content": {
            "adaptContentToUserProfile": "adaptContentToUserProfile",
        },
        "system": {
            "agentPlatform": "agentPlatform",
            "intentRecognitionTechnology": "intentRecognitionTechnology",
            "llm": "llm",
        },
    }

    for section_name, mapping in section_field_map.items():
        section_data = flattened.get(section_name)
        if not isinstance(section_data, dict):
            continue
        for source_key, target_key in mapping.items():
            if source_key in section_data:
                flattened[target_key] = section_data[source_key]
        flattened.pop(section_name, None)

    return flattened


def call_openai_chat(system_prompt, user_prompt, model="gpt-5", openai_api_key=None, config=None):
    """
    Calls OpenAI ChatCompletion with a system prompt and user prompt (openai>=1.0.0).
    Returns the response text only.
    """
    resolved_api_key = _resolve_openai_api_key(config=config, openai_api_key=openai_api_key)
    if not resolved_api_key:
        raise RuntimeError(
            f"OpenAI API key not found. Set '{OPENAI_API_KEY_ENV_VAR}' or pass it via generator config."
        )

    try:
        from openai import OpenAI  # lazy import – optional dependency
    except ImportError as exc:
        raise ImportError(
            "OpenAI personalization requires the 'agents' extra: pip install besser[agents]"
        ) from exc
    client = OpenAI(api_key=resolved_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content.strip()
    logger.debug("OpenAI response: %s", result)
    return result



def translate_text_batch(texts, target_language, model="gpt-5", openai_api_key=None, config=None):
    """
    Translates each text in `texts` to the target language using OpenAI GPT-5.
    Returns a list of translated texts in the same order as input.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings")
    if not _has_non_empty_texts(texts):
        return list(texts)

    # Define the system prompt for translation
    system_prompt = (
        "You are a translation engine. "
        "Translate each numbered text below into {}. "
        "Return only the translated texts as a numbered list, matching the input order, with no additional explanation."
    ).format(target_language)

    # Combine all texts into a single prompt with numbering for clarity
    user_prompt = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))

    # Call the model once with all texts
    response_text = call_openai_chat(
        system_prompt,
        user_prompt,
        model=model,
        openai_api_key=openai_api_key,
        config=config,
    )

    # Try to split the returned text back into a list of outputs
    results = _extract_numbered_results(response_text)
    return results

def translate_text_api(text, target_language):
    """
    Translate text using Google Cloud Translate API instead of an LLM.

    This function prefers the older `google-cloud-translate` v2 interface if available
    (simple Client). If that package is not installed it will try the v3 TranslationServiceClient
    which requires a `project_id` to be provided (or set via environment).

    Returns the translated text as a string.

    Note: Google auth must be configured (for example by setting
    the GOOGLE_APPLICATION_CREDENTIALS environment variable to a service account JSON).
    """

    try:
        # deep_translator uses language codes like 'de' for German. It will auto-detect source by default.
        from deep_translator import GoogleTranslator  # lazy import – optional dependency
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        raise RuntimeError(f"GoogleTranslator failed: {e}") from e



def style_text_batch(texts, style, model="gpt-5", openai_api_key=None, config=None):
    """
    Rewrites each text in `texts` to the requested `style` while preserving meaning and content.
    `style` should be 'formal' or 'informal'.
    Returns a list of rewritten texts in the same order as input.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings")
    if not _has_non_empty_texts(texts):
        return list(texts)

    style = (style or '').lower()
    if style not in ('formal', 'informal'):
        raise ValueError("style must be 'formal' or 'informal'")

    if style == 'formal':
        system_prompt = (
            "you are a professional editor specializing in formal writing. "
            "for each numbered text below, transform it into a formal, polished version while preserving its original meaning and intent. "
            "requirements: - use formal tone and vocabulary suitable for professional or academic contexts "
            "- ensure grammar, punctuation, and syntax are correct and refined "
            "- remove colloquial expressions, contractions, or slang "
            "- keep the length and structure close to the original unless improvement requires rephrasing "
            "- maintain clarity and natural flow. "
            "return the rewritten texts as a numbered list, matching the input order, with no explanations."
        )
    else:  # informal
        system_prompt = (
            "you are a professional writer specializing in natural, conversational language. "
            "for each numbered text below, transform it into an informal, friendly version while keeping its original meaning and intent. "
            "requirements: - use a casual and approachable tone "
            "- you may use contractions, everyday expressions, or light slang if appropriate "
            "- keep grammar correct but natural, not overly strict "
            "- make it sound like something someone would say in conversation "
            "- keep the length and structure close to the original unless rewording improves flow. "
            "return the rewritten texts as a numbered list, matching the input order, with no explanations."
        )

    # Combine all texts into a single prompt with numbering for clarity
    user_prompt = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))

    # Call the model once with all texts
    response_text = call_openai_chat(
        system_prompt,
        user_prompt,
        model=model,
        openai_api_key=openai_api_key,
        config=config,
    )

    # Try to split the returned text back into a list of outputs
    results = [entry.replace("'", "\\'") for entry in _extract_numbered_results(response_text)]
    return results


def configure_agent(agent, config, openai_api_key: str = None):
    """
    Personalize the agent using OpenAI API or other logic.
    This is a placeholder for future agent personalization logic.
    Args:
        agent: The agent model to personalize.
        config: The configuration dictionary for personalization.
    Returns:
        None (modifies agent in place)
    """
    config = flatten_agent_config_structure(config or {})
    resolved_api_key = _resolve_openai_api_key(config=config, openai_api_key=openai_api_key)

    # Example: You could use OpenAI API here to modify agent intents, responses, etc.
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    # ... personalization logic ...
    language_setting = config.get('agentLanguage')
    should_translate_language = _is_effective_setting(language_setting)

    training_sentences = []
    for intent in getattr(agent, 'intents', []):
        for idx, sentence in enumerate(getattr(intent, 'training_sentences', [])):
            logger.debug("Intent: %s, Sentence %d: %s", getattr(intent, 'name', ''), idx, sentence)
            if should_translate_language:
                logger.debug("Translating using API...")
                training_sentences.append(sentence)
    if should_translate_language and training_sentences:
        if not resolved_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        target_language = language_setting
        translated_sentences = translate_text_batch(
            training_sentences,
            target_language,
            openai_api_key=resolved_api_key,
            config=config,
        )
        ti = 0
        for intent in getattr(agent, 'intents', []):
            for idx, sentence in enumerate(getattr(intent, 'training_sentences', [])):
                if should_translate_language:
                    logger.debug("Replacing sentence %d with translated version.", ti)
                    intent.training_sentences[idx] = translated_sentences[ti]
                    ti += 1

    messages = []
    for state in getattr(agent, 'states', []):

        for body_attr in ['body', 'fallback_body']:
            body = getattr(state, body_attr, None)
            if body and body.actions:
                for action in body.actions:
                    if isinstance(action, AgentReply):
                        # process each message individually
                        # action.message = replace_reply(action.message, config)
                        messages.append(action.message)

    if not _has_non_empty_texts(messages):
        return

    personalized_messages = replace_reply_batch(
        messages,
        config,
        openai_api_key=resolved_api_key,
    )

    if len(personalized_messages) != len(messages):
        logger.warning(
            "Personalization returned %d messages but expected %d; falling back to originals.",
            len(personalized_messages), len(messages),
        )
        personalized_messages = list(messages)

    msg_index = 0
    for state in getattr(agent, 'states', []):
        for body_attr in ['body', 'fallback_body']:
            body = getattr(state, body_attr, None)
            if body and body.actions:
                for action in body.actions:
                    if isinstance(action, AgentReply):
                        action.message = personalized_messages[msg_index]
                        action.message = action.message.replace("'", "\\'")
                        msg_index += 1




def replace_reply_batch(messages: list[str], config: dict, openai_api_key: str = None) -> list[str]:
    config = flatten_agent_config_structure(config or {})
    personalized_messages = list(messages)

    if not _has_non_empty_texts(personalized_messages):
        return personalized_messages

    style = _normalize_setting(config.get('agentStyle'))
    complexity = _normalize_setting(config.get('languageComplexity'))
    sentence_length = _normalize_setting(config.get('sentenceLength'))
    target_language = _normalize_setting(config.get('agentLanguage'))
    user_profile = config.get('userProfileModel') if isinstance(config.get('userProfileModel'), dict) else None

    has_transforms = any([
        style in {'formal', 'informal'},
        complexity in {'simple', 'medium', 'complex'},
        sentence_length in {'concise', 'verbose'},
        isinstance(user_profile, dict),
        _is_effective_setting(target_language),
    ])

    if not has_transforms:
        return personalized_messages

    if not openai_api_key:
        raise RuntimeError(
            f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
            "or include 'openaiApiKey'/'openai_api_key' in generator config."
        )

    return rewrite_reply_batch(
        personalized_messages,
        config=config,
        style=style if style in {'formal', 'informal'} else None,
        complexity=complexity if complexity in {'simple', 'medium', 'complex'} else None,
        sentence_length=sentence_length if sentence_length in {'concise', 'verbose'} else None,
        user_profile=user_profile,
        target_language=target_language if _is_effective_setting(target_language) else None,
        openai_api_key=openai_api_key,
    )


def rewrite_reply_batch(
    messages: list[str],
    config: dict,
    style: str | None = None,
    complexity: str | None = None,
    sentence_length: str | None = None,
    user_profile: dict | None = None,
    target_language: str | None = None,
    openai_api_key: str | None = None,
) -> list[str]:
    """Apply all enabled reply transformations in a single LLM request."""
    personalized_messages: list[str] = list(messages)
    valid_entries: list[tuple[int, str]] = []

    for idx, original_text in enumerate(messages):
        if isinstance(original_text, str) and original_text.strip():
            valid_entries.append((idx, original_text))
        else:
            personalized_messages[idx] = original_text

    if not valid_entries:
        return personalized_messages

    instruction_lines = [
        "Rewrite each numbered reply while preserving meaning and intent.",
    ]
    if style:
        instruction_lines.append(f"- Use a {style} style.")
    if complexity:
        instruction_lines.append(f"- Target a {complexity} language complexity.")
    if sentence_length:
        instruction_lines.append(f"- Prefer {sentence_length} sentence length.")
    if user_profile:
        instruction_lines.append(
            "- Adapt to the provided user profile only when needed to avoid contradictions."
        )
    if target_language:
        instruction_lines.append(f"- Translate the final text to {target_language}.")
    instruction_lines.append(
        "Return only a numbered list of rewritten replies in the same order with no explanations."
    )

    profile_context = ""
    if user_profile:
        profile_json = json.dumps(user_profile, ensure_ascii=False)
        max_profile_chars = 6000
        if len(profile_json) > max_profile_chars:
            profile_json = profile_json[:max_profile_chars] + '... (truncated)'
        profile_context = f"User profile (JSON):\n{profile_json}\n\n"

    numbered_replies = "\n".join(
        f"{i + 1}. {text}" for i, (_, text) in enumerate(valid_entries)
    )
    user_prompt = (
        f"{profile_context}"
        f"Instructions:\n{chr(10).join(instruction_lines)}\n\n"
        "Original replies:\n"
        f"{numbered_replies}"
    )

    try:
        model_name = _resolve_model_name(config, default='gpt-5')
        response_text = call_openai_chat(
            "You are a precise rewriting assistant for agent responses.",
            user_prompt,
            model=model_name,
            openai_api_key=openai_api_key,
            config=config,
        )
        rewritten_values = _extract_numbered_results(response_text)

        if len(rewritten_values) != len(valid_entries):
            logger.warning(
                "Reply rewriting returned %d results but expected %d; falling back to originals.",
                len(rewritten_values),
                len(valid_entries),
            )
            rewritten_values = [text for _, text in valid_entries]

        for (msg_idx, original_text), rewritten in zip(valid_entries, rewritten_values):
            personalized_messages[msg_idx] = rewritten if rewritten else original_text
    except Exception as exc:
        logger.error("Reply rewriting failed: %s", exc, exc_info=True)
        for msg_idx, original_text in valid_entries:
            personalized_messages[msg_idx] = original_text

    return personalized_messages


def replace_content_profile_batch(messages: list[str], config: dict, openai_api_key: str = None) -> list[str]:
    """Adapt reply content so it aligns with the supplied user profile model."""
    flattened_config = flatten_agent_config_structure(config or {})
    user_profile = flattened_config.get('userProfileModel')
    if not isinstance(user_profile, dict):
        return messages

    profile_json = json.dumps(user_profile, ensure_ascii=False)
    max_profile_chars = 6000
    if len(profile_json) > max_profile_chars:
        profile_context = profile_json[:max_profile_chars] + '... (truncated)'
    else:
        profile_context = profile_json

    model_name = _resolve_model_name(flattened_config, default='gpt-5')

    system_prompt = (
        "You personalize agent replies for a single user based on a provided profile. "
        "The reply and semantics should only be changed, when original content would contradict profile data. Keep information accurate, and return only "
        "the rewritten reply text and do not use any kind of formatting."
    )

    personalized_messages: list[str] = list(messages)
    valid_entries: list[tuple[int, str]] = []
    for idx, original_text in enumerate(messages):
        if isinstance(original_text, str) and original_text.strip():
            valid_entries.append((idx, original_text))
        else:
            personalized_messages[idx] = original_text

    if not valid_entries:
        return personalized_messages

    numbered_replies = "\n".join(
        f"{i + 1}. {text}" for i, (_, text) in enumerate(valid_entries)
    )
    user_prompt = (
        f"User profile (JSON):\n{profile_context}\n\n"
        "Original agent replies (numbered):\n"
        f"{numbered_replies}\n\n"
        "Rewrite each reply so it aligns with the profile only when necessary.\n"
        "Return the rewritten replies as a numbered list matching the inputs and do not use any formatting."
    )

    try:
        response_text = call_openai_chat(
            system_prompt,
            user_prompt,
            model=model_name,
            openai_api_key=openai_api_key,
            config=flattened_config,
        )
        results = _extract_numbered_results(response_text)

        if len(results) != len(valid_entries):
            logger.warning(
                "Content profile adaptation returned %d results but expected %d; falling back to originals.",
                len(results), len(valid_entries),
            )
            results = [text for _, text in valid_entries]

        for (msg_idx, original_text), rewritten in zip(valid_entries, results):
            final_text = rewritten if rewritten else original_text
            if isinstance(final_text, str):
                final_text = final_text.replace("'", "\\'")
            personalized_messages[msg_idx] = final_text
    except Exception as exc:
        logger.error("Content profile adaptation failed: %s", exc, exc_info=True)
        for msg_idx, original_text in valid_entries:
            fallback_text = (
                original_text.replace("'", "\\'")
                if isinstance(original_text, str)
                else original_text
            )
            personalized_messages[msg_idx] = fallback_text

    return personalized_messages

def append_speech(match):
    text = match.group(1)
    return f"session.reply('{text}')\n    platform.reply_speech(session, '{text}')"

def complexity_text_batch(texts, complexity, model="gpt-5", openai_api_key=None, config=None):
    """
    Adjusts the complexity of each text in `texts` based on the specified `complexity` level.
    Complexity can be "simple", "medium", or "complex".
    Returns a list of texts with the adjusted complexity.
    """
    if complexity not in {"simple", "medium", "complex"}:
        raise ValueError("Invalid complexity level. Choose from 'simple', 'medium', or 'complex'.")

    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings")
    if not _has_non_empty_texts(texts):
        return list(texts)

    complexity_prompts = {
        "simple": (
            "You are a text simplification engine. "
            "Simplify the following texts to make them easier to understand while retaining their meaning. "
            "If a text already fits the A1-A2 complexity level, no changes are required. "
            "Return the simplified texts as a numbered list."
        ),
        "medium": (
            "You are a text adjustment engine. "
            "Adjust the following texts to a medium level of complexity, suitable for a general audience. "
            "If a text already fits the B1-B2 complexity level, no changes are required. "
            "Return the adjusted texts as a numbered list."
        ),
        "complex": (
            "You are a text enhancement engine. "
            "Enhance the following texts to make them more sophisticated and complex while retaining their meaning. "
            "If a text already fits the C1-C2 complexity level, no changes are required. "
            "Return the enhanced texts as a numbered list."
        ),
    }
    system_prompt = complexity_prompts[complexity]

    # Combine all texts into a single prompt with numbering for clarity
    user_prompt = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))

    # Call the model once with all texts
    response_text = call_openai_chat(
        system_prompt,
        user_prompt,
        model=model,
        openai_api_key=openai_api_key,
        config=config,
    )

    # Try to split the returned text back into a list of outputs
    results = _extract_numbered_results(response_text)
    return results


def sentence_length_batch(texts, preference, model="gpt-5", openai_api_key=None, config=None):
    """
    Adjusts each text in `texts` to be more concise or verbose.
    `preference` accepts "concise" or "verbose" (case-insensitive).
    Returns a list of rewritten texts matching the input order.
    """
    normalized_pref = (preference or "").strip().lower()
    if normalized_pref not in {"concise", "verbose"}:
        raise ValueError("Invalid sentence length preference. Choose 'Concise' or 'Verbose'.")

    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings")
    if not _has_non_empty_texts(texts):
        return list(texts)

    if normalized_pref == "concise":
        system_prompt = (
            "You are an editing engine focused on brevity. "
            "Rewrite each numbered text to be concise while keeping the original meaning intact. "
            "Remove redundancy, trim filler, and keep sentences short. "
            "Return the rewritten texts as a numbered list in the same order."
        )
    else:  # verbose
        system_prompt = (
            "You are an editing engine focused on elaboration. "
            "Rewrite each numbered text to be more detailed and verbose while preserving meaning. "
            "You may add clarifying context or additional descriptive phrasing. "
            "Return the rewritten texts as a numbered list in the same order."
        )

    user_prompt = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
    response_text = call_openai_chat(
        system_prompt,
        user_prompt,
        model=model,
        openai_api_key=openai_api_key,
        config=config,
    )
    results = _extract_numbered_results(response_text)
    return results
