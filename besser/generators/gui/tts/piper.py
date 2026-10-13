import io
import wave
import hashlib
import requests
from .base import SpeechAudio, Text2Speech


class PiperText2Speech(Text2Speech):
    """Adapter for the same PCM16 Piper HTTP service used by the agent framework."""
    def __init__(self, *, url: str, model: str, sample_rate: int = 22500, language: str | None = None):
        if not url or not model or sample_rate <= 0:
            raise ValueError("Piper requires a service URL, model name and positive sample rate")
        self.url, self.model, self.sample_rate, self.language = url, model, sample_rate, language

    @property
    def configuration(self):
        return {"provider": "piper", "model": self.model, "sampleRate": self.sample_rate, "language": self.language, "serviceHash": hashlib.sha256(self.url.encode()).hexdigest()}

    def text2speech(self, text, language=None):
        codes = {"english": "en", "french": "fr", "german": "de", "spanish": "es", "portuguese": "pt", "luxembourgish": "lb", "italian": "it"}
        normalize = lambda value: codes.get(value.lower(), value.lower()) if value else None
        if language and (not self.language or normalize(language) != normalize(self.language)):
            raise ValueError("Requested language does not match BESSER_GUI_TTS_LANGUAGE for the deployed Piper voice")
        response = requests.post(self.url, json={"text": text}, timeout=120)
        response.raise_for_status()
        if not response.content:
            raise ValueError("Piper returned empty audio")
        mime = response.headers.get("content-type", "").split(";")[0].lower()
        if mime in ("audio/wav", "audio/x-wav"):
            return SpeechAudio(response.content)
        if mime not in ("audio/l16", "application/octet-stream") or len(response.content) % 2:
            raise ValueError("Piper must return WAV or mono little-endian PCM16 audio")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(response.content)
        return SpeechAudio(output.getvalue())
