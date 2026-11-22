from .base import SpeechAudio, Text2Speech


class OpenAIText2Speech(Text2Speech):
    def __init__(self, *, api_key: str, model: str = "gpt-4o-mini-tts", voice: str = "coral", speed: float = 1.0):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI GUI TTS provider")
        if model not in ("gpt-4o-mini-tts", "tts-1", "tts-1-hd"):
            raise ValueError("Unsupported OpenAI GUI TTS model")
        if not 0.25 <= speed <= 4:
            raise ValueError("GUI TTS speed must be between 0.25 and 4")
        self.api_key, self.model, self.voice, self.speed = api_key, model, voice, speed

    @property
    def configuration(self):
        return {"provider": "openai", "model": self.model, "voice": self.voice, "speed": self.speed}

    def text2speech(self, text, language=None):
        from openai import OpenAI
        options = dict(model=self.model, voice=self.voice, input=text, speed=self.speed, response_format="wav")
        if self.model == "gpt-4o-mini-tts":
            options["instructions"] = "Read clearly and naturally without adding or omitting words." + (f" Speak in {language}." if language else " Keep the text's language.")
        with OpenAI(api_key=self.api_key) as client:
            response = client.audio.speech.create(**options)
            return SpeechAudio(response.content)
