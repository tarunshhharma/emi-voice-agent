"""The small interface every streaming STT provider implements, so Sarvam and Deepgram are swappable."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class Transcript:
    text: str
    final: bool  # partial = the provider's best guess so far; final = it won't change


class StreamingSTT(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def start_utterance(self) -> None:
        """Our VAD heard speech begin."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """16 kHz 16-bit mono PCM, any length."""

    @abstractmethod
    async def end_utterance(self) -> None:
        """Our VAD decided the turn is over: the provider should finalise now."""

    @abstractmethod
    def transcripts(self) -> AsyncIterator[Transcript]: ...

    @abstractmethod
    async def close(self) -> None: ...
