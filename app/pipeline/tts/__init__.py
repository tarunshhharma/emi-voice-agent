import os

from app.pipeline.tts.base import StreamingTTS


def make_tts() -> StreamingTTS:
    """The TTS provider for a call: TTS_PROVIDER=sarvam (default) or elevenlabs (fallback)."""
    if (os.getenv("TTS_PROVIDER") or "sarvam").lower() == "elevenlabs":
        from app.pipeline.tts.elevenlabs import ElevenLabsTTS
        return ElevenLabsTTS()
    from app.pipeline.tts.sarvam import SarvamTTS
    return SarvamTTS()
