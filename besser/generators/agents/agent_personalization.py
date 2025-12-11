import os
import re

from openai import OpenAI
from deep_translator import GoogleTranslator
import traceback

from besser.BUML.metamodel.state_machine.agent import AgentReply
import json


# Initialize OpenAI API key from environment variable or config
OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    raise RuntimeError('OPENAI_API_KEY environment variable not set.')

# Create a client instance for openai>=1.0.0
client = OpenAI(api_key=OPENAI_API_KEY)


def call_openai_chat(system_prompt, user_prompt, model="gpt-5"):
    """
    Calls OpenAI ChatCompletion with a system prompt and user prompt (openai>=1.0.0).
    Returns the response text only.
    """
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




def translate_text(text, target_language, model="gpt-5"):
    """
    Translates the given text to the target language using OpenAI GPT-5.
    Returns only the translated text (no extra output).
    """
    system_prompt = (
        "You are a translation engine. "
        "Translate the user's message into {}. "
        "Return only the translated text, with no additional explanation or formatting."
    ).format(target_language)
    return call_openai_chat(system_prompt, text, model=model)

def translate_text_batch(texts, target_language, model="gpt-5"):
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
    response_text = call_openai_chat(system_prompt, user_prompt, model=model)

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
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        raise RuntimeError(f"GoogleTranslator failed: {e}") from e

def style_text(text, style, model="gpt-5"):
    """
    Rewrites `text` to the requested `style` while preserving meaning and content.
    `style` should be 'formal' or 'informal'. Returns only the rewritten text.
    """
    style = (style or '').lower()
    if style not in ('formal', 'informal'):
        raise ValueError("agentStyle must be 'formal' or 'informal'")
    if style == 'formal':
        system_prompt = "you are a professional editor specializing in formal writing. transform the following text into a formal, polished version while preserving its original meaning and intent. requirements: - use formal tone and vocabulary suitable for professional or academic contexts - ensure grammar, punctuation, and syntax are correct and refined - remove colloquial expressions, contractions, or slang - keep the length and structure close to the original unless improvement requires rephrasing - maintain clarity and natural flow. Return only the rewritten text, with no additional explanation."
    else:  # informal
        system_prompt = "you are a professional writer specializing in natural, conversational language. transform the following text into an informal, friendly version while keeping its original meaning and intent. requirements: - use a casual and approachable tone - you may use contractions, everyday expressions, or light slang if appropriate - keep grammar correct but natural, not overly strict - make it sound like something someone would say in conversation - keep the length and structure close to the original unless rewording improves flow text to make informal. Return only the rewritten text, with no additional explanation."
        
    return call_openai_chat(system_prompt, text, model=model)

def style_text_batch(texts, style, model="gpt-5"):
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
    response_text = call_openai_chat(system_prompt, user_prompt, model=model)

    # Try to split the returned text back into a list of outputs
    results = [line.split(". ", 1)[1].replace("'", "\\'") if ". " in line else line for line in response_text.splitlines() if line.strip()]
    return results


def configure_agent(agent, config):
    """
    Personalize the agent using OpenAI API or other logic.
    This is a placeholder for future agent personalization logic.
    Args:
        agent: The agent model to personalize.
        config: The configuration dictionary for personalization.
    Returns:
        None (modifies agent in place)
    """
    # Example: You could use OpenAI API here to modify agent intents, responses, etc.
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    # ... personalization logic ...
    print("beep boop - personalizing agent with config:", config)
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
        target_language = config['agentLanguage']
        translated_sentences = translate_text_batch(training_sentences, target_language)
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

    personalized_messages = replace_reply_batch(messages, config)
    for state in getattr(agent, 'states', []):
        for body_attr in ['body', 'fallback_body']:
            body = getattr(state, body_attr, None)
            if body and body.actions:
                for action in body.actions:
                    if isinstance(action, AgentReply):  
                        # process each message individually
                        action.message = personalized_messages.pop(0)
                        action.message = action.message.replace("'", "\\'")

def personalize_agent(agent, personalization_rules, personalized_messages):
    
    languages = []
    
    for mapping in personalization_rules:
        for rule in mapping.get('rules', []):
            if rule.get('target') == 'agentLanguage' and rule.get('targetValue') != 'original':
                languages.append(rule.get('targetValue'))
    
    
    
    
    training_sentences = []
    if languages and False:
        for intent in getattr(agent, 'intents', []):
            for idx, sentence in enumerate(getattr(intent, 'training_sentences', [])):
                # Here you can personalize each training sentence
                print(f"Intent: {getattr(intent, 'name', '')}, Sentence {idx}: {sentence}")
                # intent.training_sentences[idx] = ... # personalize as needed
                training_sentences.append(sentence)

        for lang in languages:
            target_language = lang
            translated_sentences = translate_text_batch(training_sentences, target_language)
            translated_sentences = [s.replace("'", "\\'") for s in translated_sentences]
            ti = 0
            for intent in getattr(agent, 'intents', []):
                original_len = len(getattr(intent, 'training_sentences', []))
                out_of_translations = False
                for idx in range(original_len):
                    if ti >= len(translated_sentences):
                        out_of_translations = True
                        break
                    print(f"Appending sentence {ti} with translated version.")
                    intent.training_sentences.append(translated_sentences[ti])
                    ti += 1
                if out_of_translations:
                    break

    # how to handle various replies in the same state for different profiles?
    # 2^n - 1 combinations, thus llm calls
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
    all_combinations = combinations(messages, personalization_rules)
   
    # build a dict keyed by the original message with a dict of its variations



    for idx, orig_msg in enumerate(messages):
        variants = {}
        for variant_key, variant_list in all_combinations.items():
            # skip if lists are shorter than messages
            if not isinstance(variant_list, (list, tuple)) or idx >= len(variant_list):
                continue
            # encode tuple keys as JSON arrays (strings) so they are JSON-serializable
            if isinstance(variant_key, tuple):
                key = json.dumps(list(variant_key), ensure_ascii=False)
            else:
                key = str(variant_key)
            variants[key] = variant_list[idx]
        personalized_messages[orig_msg] = variants

    # personalized_messages now looks like:
    # {"original message text": {"french": "...", "formal": "...", "fr_formal": "..."}, ...}
   
   
def combinations(messages, personalization_rules):
    """
    Generate all combinations of personalized messages based on the given rules.
    Args:
        messages: List of original messages to personalize.
        personalization_rules: List of personalization rules.
    Returns:
        List of lists, where each sublist is a combination of personalized messages.
    """


    # Create a list of lists, where each sublist contains the personalized versions of a message
    personalized_versions = []
    rules = {}
    for mapping in personalization_rules:
        for rule in mapping.get('rules', []):
            if rule.get('target') == 'agentLanguage' and rule.get('targetValue') != 'original':
                target_language = rule.get('targetValue')
                if 'agentLanguage' not in rules:
                    rules['agentLanguage'] = [target_language]
                else:
                    rules['agentLanguage'].append(target_language)
            elif rule.get('target') == 'agentStyle' and rule.get('targetValue') != 'original':
                style = rule.get('targetValue')
                if 'agentStyle' not in rules:
                    rules['agentStyle'] = [style]
                else:
                    rules['agentStyle'].append(style)
            elif rule.get('target') == 'agentLanguageComplexity' and rule.get('targetValue') != 'original':
                complexity = rule.get('targetValue')
                if 'agentLanguageComplexity' not in rules:
                    rules['agentLanguageComplexity'] = [complexity]
                else:
                    rules['agentLanguageComplexity'].append(complexity)

    print("Personalization rules:", rules)
    
    translated_messages = {}
    for lang in rules.get('agentLanguage', []):
        target_language = lang
        translated = translate_text_batch(messages, target_language)
        translated_messages[lang] = translated
    
    styled_messages = {}
    for style in rules.get('agentStyle', []):
        styled = style_text_batch(messages, style)
        styled_messages[style] = styled
        
    complexity_messages = {}
    for complexity in rules.get('agentLanguageComplexity', []):
        adjusted = complexity_text_batch(messages, complexity)
        complexity_messages[complexity] = adjusted
        
    translated_and_styled_messages = {}
    # Translate the already-styled messages using the batch translator.
    # We iterate over the existing styled_messages dict and translate each styled list
    # into each target language from the rules using translate_text_batch.
    for style, styled_list in styled_messages.items():
        for lang in rules.get('agentLanguage', []):
            # use batch translation on the styled texts
            translated_list = translate_text_batch(styled_list, lang)
            # keep behavior consistent with other parts of the module (escape single quotes)
            translated_list = [s.replace("'", "\\'") for s in translated_list]
            translated_and_styled_messages[(lang, style)] = translated_list
    
    styled_and_complexity_messages = {}
    for style, styled_list in styled_messages.items():
        for complexity in rules.get('agentLanguageComplexity', []):
            adjusted_list = complexity_text_batch(styled_list, complexity)
            styled_and_complexity_messages[(style, complexity)] = adjusted_list

    translated_and_complexity_messages = {}
    for lang, translated_list in translated_messages.items():
        for complexity in rules.get('agentLanguageComplexity', []):
            adjusted_list = complexity_text_batch(translated_list, complexity)
            translated_and_complexity_messages[(lang, complexity)] = adjusted_list

    translate_styled_and_complexity_messages = {}
    for style, styled_list in styled_messages.items():
        for lang in rules.get('agentLanguage', []):
            translated_list = translate_text_batch(styled_list, lang)
            for complexity in rules.get('agentLanguageComplexity', []):
                adjusted_list = complexity_text_batch(translated_list, complexity)
                translate_styled_and_complexity_messages[(lang, style, complexity)] = adjusted_list
    print("Translated messages:", translated_messages)
    print("Styled messages:", styled_messages)
    print("Translated and styled messages:", translated_and_styled_messages)
    personalized_versions = {}

    # keep original messages
    #  personalized_versions['original'] = list(messages)

    # add translations keyed by language
    for lang, msgs in translated_messages.items():
        personalized_versions[lang] = [s for s in msgs]

    # add styled versions keyed by style
    for style, msgs in styled_messages.items():
        personalized_versions[style] = [s for s in msgs]

    # add translated+styled combinations keyed by language and style
    for (lang, style), msgs in translated_and_styled_messages.items():
        personalized_versions[(lang, style)] = [s for s in msgs]
        
    # add complexity-only versions
    for complexity, msgs in complexity_messages.items():
        personalized_versions[complexity] = [s for s in msgs]

    # add styled+complexity combinations
    for (style, complexity), msgs in styled_and_complexity_messages.items():
        personalized_versions[(style, complexity)] = [s for s in msgs]

    # add translated+complexity combinations
    for (lang, complexity), msgs in translated_and_complexity_messages.items():
        personalized_versions[(lang, complexity)] = [s for s in msgs]

    # add translated+styled+complexity combinations
    for (lang, style, complexity), msgs in translate_styled_and_complexity_messages.items():
        personalized_versions[(lang, style, complexity)] = [s for s in msgs]

    print("Built personalized_versions with keys:", list(personalized_versions.keys()))
    return personalized_versions



def replace_reply(message: str, config: dict) -> str:
    
    personalized_message = message
    if 'agentStyle' in config and config['agentStyle'] != 'original':
        style = config['agentStyle']
        personalized_message = style_text(personalized_message, style)
    if 'agentLanguage' in config and config['agentLanguage'] != 'none' and False:
        target_language = config['agentLanguage']
        personalized_message = translate_text(personalized_message, target_language)
    elif 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
        target_language = config['agentLanguage']
        personalized_message = translate_text_api(personalized_message, target_language)



    return personalized_message.replace("'", "\\'")

def replace_reply_batch(messages: list[str], config: dict) -> list[str]:
    personalized_messages = messages

    if 'agentLanguage' in config and config['agentLanguage'] != 'none' and False:
        target_language = config['agentLanguage']
        # personalized_message = translate_text(personalized_message, target_language)
    elif 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original' and False:
        target_language = config['agentLanguage']
        for i, msg in enumerate(personalized_messages):
            personalized_messages[i] = translate_text_api(msg, target_language)
    elif 'agentLanguage' in config and config['agentLanguage'] != 'none' and config['agentLanguage'] != 'original':
        target_language = config['agentLanguage']
        personalized_messages = translate_text_batch(personalized_messages, target_language)
    if 'agentStyle' in config and config['agentStyle'] != 'original':
        style = config['agentStyle']
        personalized_messages = style_text_batch(personalized_messages, style)
    if 'languageComplexity' in config and config['languageComplexity'] != 'original':
        complexity = config['languageComplexity']
        personalized_messages = complexity_text_batch(personalized_messages, complexity)

    return personalized_messages

def append_speech(match):
    text = match.group(1)
    return f"session.reply('{text}')\n    platform.reply_speech(session, '{text}')"

def complexity_text_batch(texts, complexity, model="gpt-5"):
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
    response_text = call_openai_chat(system_prompt, user_prompt, model=model)

    # Try to split the returned text back into a list of outputs
    results = [
        line.split(". ", 1)[1] if ". " in line else line
        for line in response_text.splitlines()
        if line.strip()
    ]
    return results