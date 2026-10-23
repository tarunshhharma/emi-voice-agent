import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from app.pipeline import audio, vad
from app.pipeline.session import VoiceSession
from app.pipeline.stt.sarvam import SarvamSTT

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (and on first run download) the VAD model before any call arrives,
    # in a thread so startup doesn't block the event loop.
    await asyncio.to_thread(vad.load_model)
    yield


app = FastAPI(title="EMI Voice Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def browser_client():
    return FileResponse(ROOT / "client" / "index.html")


def _log_task_failure(task: asyncio.Task) -> None:
    """A background task that crashes is otherwise silent until someone awaits it."""
    if not task.cancelled() and task.exception():
        log.warning("transcript task stopped: %r", task.exception())


@app.websocket("/ws/browser")
async def browser_audio(ws: WebSocket):
    """Browser mic -> VAD -> streaming STT -> transcripts back to the browser.
    The audio is also saved, to build the clip set for the WER test."""
    await ws.accept()
    stt = SarvamSTT()
    try:
        await stt.connect()
    except Exception as e:  # bad key, network down: tell the user instead of failing silently
        log.warning("STT connect failed: %s", e)
        await ws.send_json({"type": "error", "message": f"Speech-to-text unavailable: {e}"})
        await ws.close()
        return

    send_lock = asyncio.Lock()

    async def send_json(message: dict) -> None:
        # Two tasks send to this socket (VAD events and transcripts); one at a time.
        async with send_lock:
            await ws.send_json(message)

    session = VoiceSession(vad.SileroVAD(), stt, send_json)
    # Two things happen at once for the whole call: frames come in from the browser,
    # and transcripts come back from the STT provider. Each gets its own task.
    forwarder = asyncio.create_task(session.forward_transcripts())
    forwarder.add_done_callback(_log_task_failure)
    frames: list[bytes] = []
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is None or len(data) != audio.FRAME_BYTES:
                continue  # never trust network input
            frames.append(data)
            await session.on_frame(data)
    finally:
        forwarder.cancel()
        await stt.close()
        if frames:
            name = f"browser-{datetime.now():%Y%m%d-%H%M%S}.wav"
            path = await asyncio.to_thread(audio.save_wav, RECORDINGS / name, b"".join(frames))
            log.info("saved %.1f s of audio to %s", len(frames) / audio.FRAMES_PER_SECOND, path)
