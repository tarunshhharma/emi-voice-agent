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


def resample(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resampling of 16-bit PCM (simple; no anti-aliasing filter)."""
    import numpy as np

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(len(x) * to_rate / from_rate)
    positions = np.arange(n_out) * (from_rate / to_rate)
    return np.interp(positions, np.arange(len(x)), x).astype(np.int16).tobytes()


# G.711 mu-law: the 8-bit telephone codec Twilio uses. Quiet sounds keep more detail
# than loud ones, which suits speech: 8 bits per sample instead of 16.
MULAW_BIAS = 0x84
MULAW_CLIP = 32635


def mulaw_encode(pcm: bytes) -> bytes:
    import numpy as np

    s = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(s), MULAW_CLIP) + MULAW_BIAS
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def mulaw_decode(data: bytes) -> bytes:
    import numpy as np

    b = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    exponent = (b >> 4) & 0x07
    magnitude = (((b & 0x0F) << 3) + MULAW_BIAS) << exponent
    s = np.where(b & 0x80, MULAW_BIAS - magnitude, magnitude - MULAW_BIAS)
    return s.astype(np.int16).tobytes()


def phone_quality(pcm: bytes) -> bytes:
    """What a phone line does to 16 kHz audio: down to 8 kHz, through mu-law, back up to 16 kHz."""
    narrow = resample(pcm, SAMPLE_RATE, 8000)
    return resample(mulaw_decode(mulaw_encode(narrow)), 8000, SAMPLE_RATE)


def save_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)
    return path
