import asyncio
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from app.agent.prompts import opening
from app.pipeline import audio, vad
from app.pipeline.session import VoiceSession
from app.pipeline.stt import make_stt
from app.pipeline.tts.elevenlabs import ElevenLabsTTS

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
        log.warning("%s task stopped: %r", task.get_name(), task.exception())


@app.websocket("/ws/browser")
async def browser_audio(ws: WebSocket):
    """Browser mic -> VAD -> streaming STT -> transcripts back to the browser, and agent
    speech (TTS) back as binary PCM. The mic audio is also saved, for test clips."""
    await ws.accept()
    try:
        stt = make_stt()  # Sarvam by default; STT_PROVIDER=deepgram in .env to switch
        await stt.connect()
    except Exception as e:  # bad key, network down: tell the user instead of failing silently
        log.warning("STT connect failed: %s", e)
        await ws.send_json({"type": "error", "message": f"Speech-to-text unavailable: {e}"})
        await ws.close()
        return

    send_lock = asyncio.Lock()

    async def send_json(message: dict) -> None:
        # Several tasks send to this socket (VAD events, transcripts, agent audio); one at a time.
        async with send_lock:
            await ws.send_json(message)

    async def send_audio(pcm: bytes) -> None:
        async with send_lock:
            await ws.send_bytes(pcm)

    session = VoiceSession(vad.SileroVAD(), stt, send_json, tts=ElevenLabsTTS(), send_audio=send_audio)
    # Things happen at once for the whole call: frames come in from the browser,
    # transcripts come back from STT, and the agent speaks. Each gets its own task.
    forwarder = asyncio.create_task(session.forward_transcripts(), name="transcript")
    forwarder.add_done_callback(_log_task_failure)
    # The disclosure line is fixed in code and spoken first. Task 2 replaces the fixed
    # name with the borrower being called and hands the turns to the agent.
    sentences = re.split(r"(?<=[.?!])\s+", opening("Rahul Sharma"))
    speaker = asyncio.create_task(session.speak(sentences), name="speak")
    speaker.add_done_callback(_log_task_failure)
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
        speaker.cancel()
        await stt.close()
        if frames:
            name = f"browser-{datetime.now():%Y%m%d-%H%M%S}.wav"
            path = await asyncio.to_thread(audio.save_wav, RECORDINGS / name, b"".join(frames))
            log.info("saved %.1f s of audio to %s", len(frames) / audio.FRAMES_PER_SECOND, path)
