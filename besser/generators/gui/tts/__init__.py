"""Server-controlled provider selection; no model, URL or credentials from the browser."""
import os
from .base import SpeechAudio, Text2Speech


def log_text2speech_environment(logger) -> None:
    """Startup diagnostics without contacting a provider or logging credentials."""
    provider = os.getenv("BESSER_GUI_TTS_PROVIDER", "").strip().lower()
    if not provider:
        # The backend's optional-env check already reports the missing selector.
        return
    if provider == "openai":
        model = os.getenv("BESSER_GUI_TTS_MODEL", "gpt-4o-mini-tts")
        defaults = {"BESSER_GUI_TTS_MODEL": "gpt-4o-mini-tts",
                    "BESSER_GUI_TTS_VOICE": "coral" if model == "gpt-4o-mini-tts" else "alloy",
                    "BESSER_GUI_TTS_SPEED": "1"}
    elif provider == "piper":
        defaults = {"BESSER_GUI_TTS_MODEL": "mbarnig/lb_rhasspy_piper_tts",
                    "BESSER_GUI_TTS_URL": "http://localhost:8000/synthesize",
                    "BESSER_GUI_TTS_SAMPLE_RATE": "22500", "BESSER_GUI_TTS_SPEED": "1"}
        if not os.getenv("BESSER_GUI_TTS_LANGUAGE"):
            logger.warning("Optional environment variable BESSER_GUI_TTS_LANGUAGE is not set. "
                           "Language-specific Piper TTS functionality will be unavailable.")
    else:
        logger.warning("BESSER_GUI_TTS_PROVIDER is invalid. GUI TTS functionality will be unavailable. "
                       "Choose 'openai' or 'piper'.")
        return
    for variable, default in defaults.items():
        if variable not in os.environ:
            logger.info("Optional environment variable %s is not set. Using default %s.", variable, default)
    try:
        create_text2speech()
    except (ValueError, ImportError):
        # Exception strings can contain malformed environment values; omit them.
        logger.warning("GUI TTS configuration is invalid or incomplete. Related functionality will be unavailable. "
                       "Check the selected provider's environment settings.")


def create_text2speech() -> Text2Speech:
    provider = os.getenv("BESSER_GUI_TTS_PROVIDER", "").strip().lower()
    if provider == "openai":
        from .openai import OpenAIText2Speech
        model = os.getenv("BESSER_GUI_TTS_MODEL", "gpt-4o-mini-tts")
        return OpenAIText2Speech(api_key=os.getenv("OPENAI_API_KEY", ""),
            model=model, voice=os.getenv("BESSER_GUI_TTS_VOICE", "coral" if model == "gpt-4o-mini-tts" else "alloy"),
            speed=float(os.getenv("BESSER_GUI_TTS_SPEED", "1")))
    if provider == "piper":
        from .piper import PiperText2Speech
        if float(os.getenv("BESSER_GUI_TTS_SPEED", "1")) != 1:
            raise ValueError("The Piper service does not support GUI TTS speed changes")
        return PiperText2Speech(url=os.getenv("BESSER_GUI_TTS_URL", "http://localhost:8000/synthesize"),
            model=os.getenv("BESSER_GUI_TTS_MODEL", "mbarnig/lb_rhasspy_piper_tts"),
            sample_rate=int(os.getenv("BESSER_GUI_TTS_SAMPLE_RATE", "22500")), language=os.getenv("BESSER_GUI_TTS_LANGUAGE"))
    raise ValueError("Set BESSER_GUI_TTS_PROVIDER to 'openai' or 'piper' before generating GUI audio")
