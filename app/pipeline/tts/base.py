"""The interface every streaming TTS provider implements: sentences in, audio out as soon as it exists."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class StreamingTTS(ABC):
    @abstractmethod
    def stream(self, sentences: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Yields 16 kHz 16-bit mono PCM chunks. Sentences can still be arriving while audio
        for the first one plays. Stopping iteration early (barge-in) must close the stream."""
