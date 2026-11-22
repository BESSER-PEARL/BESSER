"""On-demand speech for static prose. Preview audio is returned in memory, never cached."""
import base64
import hashlib
import io
import json
import wave
from copy import deepcopy

from .gui_personalization import _ReadingText, _PROTECTED
from .tts import Text2Speech

def reading_text(component):
    if not isinstance(component, dict):
        return None
    tag = str(component.get("tagName", "div")).lower()
    if tag in {"button", "input", "textarea", "select", "table", "script", "style", "code", "pre", "svg", "iframe", "audio"}:
        return None
    if component.get("type") not in (None, "default", "text", "textnode"):
        return None
    attrs = component.get("attributes") or {}
    if any("binding" in key or key in ("data-source", "data-entity") for key in attrs):
        return None
    content = component.get("content", "")
    if not isinstance(content, str) or _PROTECTED.search(content):
        return None
    parsed = _ReadingText(content)
    if not parsed.valid:
        return None
    parts = ["".join(parsed.parts)]
    for child in component.get("components", []) or []:
        text = reading_text(child)
        if text is None:
            return None
        parts.append(text)
    return " ".join(" ".join(parts).split())


def audio_signature(text, language, configuration):
    return hashlib.sha256(json.dumps([text, language, configuration], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _chunks(text, limit=3900):
    while len(text) > limit:
        end = text.rfind(" ", 0, limit + 1)
        if end < 1:
            end = limit
        yield text[:end]
        text = text[end:].lstrip()
    if text:
        yield text


def _combine_wav(clips):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        params = None
        for clip in clips:
            if clip.mime_type != "audio/wav" or not clip.content:
                raise ValueError("GUI TTS providers must return nonempty WAV audio")
            with wave.open(io.BytesIO(clip.content), "rb") as source:
                if source.getnframes() == 0:
                    raise ValueError("GUI TTS provider returned audio without any samples")
                settings = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                if params is None:
                    params = settings
                    target.setnchannels(settings[0])
                    target.setsampwidth(settings[1])
                    target.setframerate(settings[2])
                elif params != settings:
                    raise ValueError("TTS chunks have incompatible audio formats")
                target.writeframes(source.readframes(source.getnframes()))
    return output.getvalue()


def synthesize_audio_text(text, provider: Text2Speech, *, language=None):
    if not isinstance(text, str) or not text.strip() or len(text) > 80000:
        raise ValueError("Audio text must be nonempty and at most 80000 characters")
    return _combine_wav([provider.text2speech(chunk, language) for chunk in _chunks(text)])


def generate_gui_audio(gui_page, provider: Text2Speech, *, language=None):
    if not isinstance(gui_page, dict) or not isinstance(gui_page.get("components"), list):
        raise ValueError("guiPage must contain a components array")
    if language is not None and (not isinstance(language, str) or len(language) > 64):
        raise ValueError("Invalid GUI audio language")
    page = deepcopy(gui_page)
    blocks = []

    def visit(component, path):
        if not isinstance(component, dict):
            return
        attrs = component.get("attributes") or {}
        if any("binding" in key or key in ("data-source", "data-entity") for key in attrs):
            return
        if isinstance(component.get("content"), str) and _PROTECTED.search(component["content"]):
            return
        tag = str(component.get("tagName", "")).lower()
        text = reading_text(component)
        is_block = component.get("type") == "text" or tag in ("p", "blockquote", "article")
        if is_block and tag not in ("span", "a", "h1", "h2", "h3", "h4", "h5", "h6") and text:
            blocks.append((component, path, text))
            return
        if text is None and component.get("type") not in (None, "default", "text"):
            return
        if tag in ("form", "table", "script", "style", "code", "pre", "button", "iframe"):
            return
        for index, child in enumerate(component.get("components", []) or []):
            visit(child, f"{path}.components[{index}]")

    for index, component in enumerate(page["components"]):
        visit(component, f"components[{index}]")
    if len(blocks) > 128 or sum(len(text) for _, _, text in blocks) > 80000:
        raise ValueError("This page is too large for one audio generation request; split it into smaller pages")
    clips = []
    clips_by_signature = {}
    for component, path, text in blocks:
        attrs = component.get("attributes") or {}
        block_language = attrs.get("lang") or language
        signature = audio_signature(text, block_language, provider.configuration)
        if signature not in clips_by_signature:
            audio = synthesize_audio_text(text, provider, language=block_language)
            clips_by_signature[signature] = base64.b64encode(audio).decode("ascii")
        clips.append({"text": text, "path": path, "language": block_language,
            "mimeType": "audio/wav", "audio": clips_by_signature[signature],
            "isSummary": bool(attrs.get("data-besser-gui-summary"))})
    return {"clips": clips, "blocks": len(blocks)}
