"""Audio format the whole pipeline uses: 16 kHz, mono, 16-bit PCM, in 20 ms frames."""
import math
import sys
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 16_000  # what speech models expect; phone audio (8 kHz) gets upsampled to this
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = SAMPLES_PER_FRAME * 2  # 16-bit = 2 bytes per sample -> 640
FRAMES_PER_SECOND = 1000 // FRAME_MS  # 50
SILENCE_DB = -100.0


def pcm16_samples(data: bytes) -> array:
    """Bytes -> signed 16-bit samples. The browser sends little-endian, like every PC."""
    samples = array("h", data)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def dbfs(data: bytes) -> float:
    """Loudness in dB relative to full scale: 0 is the loudest possible, about -60 is near silence."""
    samples = pcm16_samples(data)
    if not samples:
        return SILENCE_DB
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    return 20 * math.log10(rms / 32768) if rms > 0 else SILENCE_DB


def save_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)
    return path
