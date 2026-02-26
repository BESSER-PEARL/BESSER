import os
import re
import traceback

from besser.BUML.metamodel.state_machine.agent import AgentReply
import json


OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


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

    from openai import OpenAI  # lazy import – optional dependency
    client = OpenAI(api_key=resolved_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    # response.choices[0].message.content is the message content in the new client
    print("OpenAI response:", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()



def translate_text_batch(texts, target_language, model="gpt-5", openai_api_key=None, config=None):
    """
    Translates each text in `texts` to the target language using OpenAI GPT-5.
    Returns a list of translated texts in the same order as input.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings")

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
    results = [
        line.split(". ", 1)[1] if ". " in line else line
        for line in response_text.splitlines()
        if line.strip()
    ]
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
    results = [line.split(". ", 1)[1].replace("'", "\\'") if ". " in line else line for line in response_text.splitlines() if line.strip()]
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
    training_sentences = []
    for intent in getattr(agent, 'intents', []):
        for idx, sentence in enumerate(getattr(intent, 'training_sentences', [])):
            # Here you can personalize each training sentence
            print(f"Intent: {getattr(intent, 'name', '')}, Sentence {idx}: {sentence}")
            # intent.training_sentences[idx] = ... # personalize as needed
            if 'agentLanguage' in config and config['agentLanguage'] != 'none' and False:
                target_language = config['agentLanguage']
                translated_sentence = translate_text(sentence, target_language)
                intent.training_sentences[idx] = translated_sentence
                print(f"Translated Sentence {idx}: {translated_sentence}")
            elif 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
                print("Translating using API...")
                training_sentences.append(sentence)
    if 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
        if not resolved_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        target_language = config['agentLanguage']
        translated_sentences = translate_text_batch(
            training_sentences,
            target_language,
            openai_api_key=resolved_api_key,
            config=config,
        )
        ti = 0
        for intent in getattr(agent, 'intents', []):
            for idx, sentence in enumerate(getattr(intent, 'training_sentences', [])):
                if 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
                    print(f"Replacing sentence {ti} with translated version.")
                    intent.training_sentences[idx] = translated_sentences[ti]
                    ti += 1
    """ code for translation using traditional API (Google Translate)
    for intent in getattr(agent, 'intents', []):
        for idx, sentence in enumerate(getattr(intent, 'training_sentences', [])):
            # Here you can personalize each training sentence
            print(f"Intent: {getattr(intent, 'name', '')}, Sentence {idx}: {sentence}")
            # intent.training_sentences[idx] = ... # personalize as needed
            if 'agentLanguage' in config and config['agentLanguage'] != 'none' and False:
                target_language = config['agentLanguage']
                translated_sentence = translate_text(sentence, target_language)
                intent.training_sentences[idx] = translated_sentence
                print(f"Translated Sentence {idx}: {translated_sentence}")
            elif 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
                print("Translating using API...")
                target_language = config['agentLanguage']
                
                translated_sentence = translate_text_api(sentence, target_language)
                intent.training_sentences[idx] = translated_sentence.replace("'", "\\'")
    """
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

    personalized_messages = replace_reply_batch(
        messages,
        config,
        openai_api_key=resolved_api_key,
    )
    for state in getattr(agent, 'states', []):
        for body_attr in ['body', 'fallback_body']:
            body = getattr(state, body_attr, None)
            if body and body.actions:
                for action in body.actions:
                    if isinstance(action, AgentReply):  
                        # process each message individually
                        action.message = personalized_messages.pop(0)
                        action.message = action.message.replace("'", "\\'")




def replace_reply_batch(messages: list[str], config: dict, openai_api_key: str = None) -> list[str]:
    config = flatten_agent_config_structure(config or {})
    personalized_messages = messages

    if 'agentLanguage' in config and config['agentLanguage'] != 'none' and False:
        target_language = config['agentLanguage']
        # personalized_message = translate_text(personalized_message, target_language)
    elif 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original' and False:
        target_language = config['agentLanguage']
        for i, msg in enumerate(personalized_messages):
            personalized_messages[i] = translate_text_api(msg, target_language)
    if 'agentStyle' in config and config['agentStyle'] != 'original':
        if not openai_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        style = config['agentStyle']
        personalized_messages = style_text_batch(
            personalized_messages,
            style,
            openai_api_key=openai_api_key,
            config=config,
        )
    if 'languageComplexity' in config and config['languageComplexity'] != 'original':
        if not openai_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        complexity = config['languageComplexity']
        personalized_messages = complexity_text_batch(
            personalized_messages,
            complexity,
            openai_api_key=openai_api_key,
            config=config,
        )
    if 'sentenceLength' in config and config['sentenceLength'] != 'original':
        if not openai_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        length_pref = config['sentenceLength']
        personalized_messages = sentence_length_batch(
            personalized_messages,
            length_pref,
            openai_api_key=openai_api_key,
            config=config,
        )
    if isinstance(config.get('userProfileModel'), dict):
        if not openai_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        personalized_messages = replace_content_profile_batch(
            personalized_messages,
            config,
            openai_api_key=openai_api_key,
        )
    if 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
        if not openai_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for agent personalization. Set '{OPENAI_API_KEY_ENV_VAR}' "
                "or include 'openaiApiKey'/'openai_api_key' in generator config."
            )
        target_language = config['agentLanguage']
        personalized_messages = translate_text_batch(
            personalized_messages,
            target_language,
            openai_api_key=openai_api_key,
            config=config,
        )
        
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

    model_name = flattened_config.get('llm')
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = 'gpt-5'
    else:
        model_name = model_name.strip()

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
        results = [
            line.split(". ", 1)[1] if ". " in line else line
            for line in response_text.splitlines()
            if line.strip()
        ]

        if len(results) != len(valid_entries):
            print(
                "Content profile adaptation returned unexpected count; falling back to originals."
            )
            results = [text for _, text in valid_entries]

        for (msg_idx, original_text), rewritten in zip(valid_entries, results):
            final_text = rewritten if rewritten else original_text
            if isinstance(final_text, str):
                final_text = final_text.replace("'", "\\'")
            personalized_messages[msg_idx] = final_text
    except Exception as exc:
        print("Content profile adaptation failed:", exc)
        traceback.print_exc()
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

    # Define the system prompt based on the complexity level
    if complexity == "simple":
        system_prompt = (
            "You are a text simplification engine. "
            "Simplify the following texts to make them easier to understand while retaining their meaning. "
            "If a text already fits the A1-A2 complexity level, no changes are required. "
            "Return the simplified texts as a numbered list."
        )
    elif complexity == "medium":
        system_prompt = (
            "You are a text adjustment engine. "
            "Adjust the following texts to a medium level of complexity, suitable for a general audience. "
            "If a text already fits the B1-B2 complexity level, no changes are required. "
            "Return the adjusted texts as a numbered list."
        )
    elif complexity == "complex":
        system_prompt = (
            "You are a text enhancement engine. "
            "Enhance the following texts to make them more sophisticated and complex while retaining their meaning. "
            "If a text already fits the C1-C2 complexity level, no changes are required. "
            "Return the enhanced texts as a numbered list."
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
    results = [
        line.split(". ", 1)[1] if ". " in line else line
        for line in response_text.splitlines()
        if line.strip()
    ]
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
    results = [
        line.split(". ", 1)[1] if ". " in line else line
        for line in response_text.splitlines()
        if line.strip()
    ]
    return results