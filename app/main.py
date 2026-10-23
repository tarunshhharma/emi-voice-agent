import asyncio
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from app.pipeline import audio

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"

app = FastAPI(title="EMI Voice Agent")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def browser_client():
    return FileResponse(ROOT / "client" / "index.html")


@app.websocket("/ws/browser")
async def browser_audio(ws: WebSocket):
    """Receive 20 ms PCM frames from the browser mic. For now: report what arrives and
    save the recording, so we can check the audio before VAD and STT are added."""
    await ws.accept()
    frames: list[bytes] = []
    bad_frames = 0
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            break
        data = message.get("bytes")
        if data is None:
            continue  # text messages are not used yet
        if len(data) != audio.FRAME_BYTES:
            bad_frames += 1
            continue
        frames.append(data)
        if len(frames) % audio.FRAMES_PER_SECOND == 0:  # once a second
            last_second = b"".join(frames[-audio.FRAMES_PER_SECOND:])
            await ws.send_json({
                "type": "stats",
                "seconds": len(frames) / audio.FRAMES_PER_SECOND,
                "level_db": round(audio.dbfs(last_second), 1),
                "bad_frames": bad_frames,
            })

    if frames:
        name = f"browser-{datetime.now():%Y%m%d-%H%M%S}.wav"
        # Writing a file blocks; run it in a thread so other calls on the event loop keep going.
        path = await asyncio.to_thread(audio.save_wav, RECORDINGS / name, b"".join(frames))
        log.info("saved %.1f s of audio to %s", len(frames) / audio.FRAMES_PER_SECOND, path)
