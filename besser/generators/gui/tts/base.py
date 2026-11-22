"""Small provider contract following the agent framework's Text2Speech pattern."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechAudio:
    content: bytes
    mime_type: str = "audio/wav"


class Text2Speech(ABC):
    @property
    @abstractmethod
    def configuration(self) -> dict:
        """Public synthesis settings used to identify reusable clips; never includes secrets."""

    @abstractmethod
    def text2speech(self, text: str, language: str | None = None) -> SpeechAudio:
        """Return playable audio bytes or raise; providers must not silently swallow errors."""
