"""Generate the synthetic test clips with ElevenLabs TTS. Usage: python -m evals.make_clips

Writes evals/clips/synthetic/<id>.wav (16 kHz mono 16-bit). ElevenLabs is used rather than
Sarvam's own TTS so neither STT provider gets a home advantage. Two voices alternate.
"""
import json
import logging
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from app.pipeline.audio import save_wav
from evals.wer_clips import CLIPS

load_dotenv()
log = logging.getLogger(__name__)
API = "https://api.elevenlabs.io"
OUT = Path(__file__).resolve().parent / "clips" / "synthetic"
MODEL = "eleven_multilingual_v2"  # the documented default; supports Hindi


def _request(path: str, body: dict | None = None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body else None,
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def pick_voices(n: int = 2) -> list[str]:
    """Prefer voices labelled with an Indian accent; otherwise take the first available."""
    voices = json.loads(_request("/v2/voices?page_size=100"))["voices"]
    indian = [v for v in voices if "indian" in json.dumps(v.get("labels", {})).lower()]
    chosen = (indian + [v for v in voices if v not in indian])[:n]
    for v in chosen:
        log.info("voice: %s (%s)", v["name"], v.get("labels", {}))
    return [v["voice_id"] for v in chosen]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    voices = pick_voices()
    for i, clip in enumerate(CLIPS):
        path = OUT / f"{clip['id']}.wav"
        if path.exists():
            continue  # keep earlier clips: re-running only fills gaps
        pcm = _request(
            f"/v1/text-to-speech/{voices[i % len(voices)]}?output_format=pcm_16000",
            {"text": clip["tts"], "model_id": MODEL},
        )
        save_wav(path, pcm)
        log.info("%s  %.1f s  %s", clip["id"], len(pcm) / 32000, clip["roman"])


if __name__ == "__main__":
    main()
