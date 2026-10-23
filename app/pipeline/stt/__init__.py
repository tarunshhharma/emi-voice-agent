import os

from app.pipeline.stt.base import StreamingSTT


def make_stt(provider: str | None = None) -> StreamingSTT:
    """Pick the STT provider by name (or STT_PROVIDER in .env). Session code never imports a provider."""
    provider = provider or os.getenv("STT_PROVIDER") or "sarvam"
    if provider == "sarvam":
        from app.pipeline.stt.sarvam import SarvamSTT

        return SarvamSTT()
    if provider == "deepgram":
        from app.pipeline.stt.deepgram import DeepgramSTT

        return DeepgramSTT()
    raise ValueError(f"unknown STT provider: {provider}")
