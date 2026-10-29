import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from openai import OpenAI

from app import demo_gate
from app.db.models import SessionLocal
from app.pipeline import audio, vad
from app.pipeline.call import CallAgent, CallBlocked
from app.pipeline.session import VoiceSession
from app.pipeline.stt import make_stt
from app.pipeline.tts import make_tts

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"
# Spoken calls running right now. A plain int is safe: one process, one event loop,
# and it only changes between awaits.
active_calls = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (and on first run download) the VAD model before any call arrives,
    # in a thread so startup doesn't block the event loop.
    await asyncio.to_thread(vad.load_model)
    # One OpenAI client for every call: it keeps its connections open, and a new one costs
    # ~400 ms of TLS and setup on the first request (measured).
    app.state.openai = OpenAI()
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
    """A full spoken call: browser mic -> VAD -> STT -> agent -> TTS -> binary PCM back.
    ?loan=<id> picks the borrower (default 1). The mic audio is also saved, for test clips."""
    global active_calls
    await ws.accept()
    db = SessionLocal()
    # A public demo link: refuse before anything is created or any credit is spent.
    why_not = await asyncio.to_thread(demo_gate.check, db, ws.query_params.get("code"))
    # Checked after that await, with no await before the count goes up: two calls can't both slip in.
    why_not = why_not or demo_gate.busy(active_calls, demo_gate.env_int("MAX_ACTIVE_CALLS"))
    if why_not:
        log.info("demo call refused: %s", why_not)
        await ws.send_json({"type": "error", "message": why_not})
        await ws.close()
        db.close()
        return
    active_calls += 1
    try:
        await _run_call(ws, db)
    finally:  # every way out of a call, including errors, frees its slot
        active_calls -= 1


async def _run_call(ws: WebSocket, db) -> None:
    try:
        # Opening the call reads and writes the DB, so it runs in a thread like every agent turn.
        agent = await asyncio.to_thread(
            CallAgent, db, ws.app.state.openai, int(ws.query_params.get("loan", 1)),
            allow_off_hours=os.getenv("ALLOW_OFF_HOURS") == "1",
        )
    except (CallBlocked, ValueError) as e:  # guardrail refused, or a bad loan id
        log.info("call not started: %s", e)
        await ws.send_json({"type": "error", "message": f"Call not started: {e}"})
        await ws.close()
        db.close()
        return
    # Open (or refresh) the connection to OpenAI while the opening line plays, so the
    # caller's first turn doesn't pay for it. A cheap request; its result is unused.
    warm = asyncio.create_task(asyncio.to_thread(ws.app.state.openai.models.list), name="warm-openai")
    warm.add_done_callback(_log_task_failure)
    if agent.off_hours_bypassed:  # say so openly: the guardrail exists, this demo skips it
        await ws.send_json({"type": "notice", "text": "Demo mode: the 8 am–7 pm IST calling-hours rule is bypassed "
                                                     "so you can try it at any time."})
    b = agent.borrower
    # Synthetic data, shown only in the browser so you know what to say (never logged).
    await ws.send_json({"type": "dev", "text": f"{b.name}, DOB {b.dob:%d-%m-%Y}, phone ends {b.phone[-4:]}"})
    try:
        stt = make_stt()  # Sarvam by default; STT_PROVIDER=deepgram in .env to switch
        await stt.connect()
    except Exception as e:  # bad key, network down: tell the user instead of failing silently
        log.warning("STT connect failed: %s", e)
        await ws.send_json({"type": "error", "message": f"Speech-to-text unavailable: {e}"})
        await ws.close()
        await asyncio.to_thread(agent.close)
        return

    send_lock = asyncio.Lock()

    async def send_json(message: dict) -> None:
        # Several tasks send to this socket (VAD events, transcripts, agent audio); one at a time.
        async with send_lock:
            await ws.send_json(message)

    async def send_audio(pcm: bytes) -> None:
        async with send_lock:
            await ws.send_bytes(pcm)

    session = VoiceSession(vad.SileroVAD(), stt, send_json, tts=make_tts(), send_audio=send_audio, agent=agent)
    # Things happen at once for the whole call: frames come in from the browser,
    # transcripts come back from STT, and the agent speaks. Each gets its own task.
    forwarder = asyncio.create_task(session.forward_transcripts(), name="transcript")
    forwarder.add_done_callback(_log_task_failure)
    # The opening (fixed in code: it carries the disclosure), then one agent turn per final transcript.
    conversation = asyncio.create_task(session.converse(), name="conversation")
    conversation.add_done_callback(_log_task_failure)
    frames: list[bytes] = []
    try:
        # STT, LLM and TTS all cost money while the socket is open, so a public demo bounds
        # every call. timeout(None), the default when MAX_CALL_SECONDS isn't set, never fires.
        async with asyncio.timeout(demo_gate.env_int("MAX_CALL_SECONDS")):
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None or len(data) != audio.FRAME_BYTES:
                    continue  # never trust network input
                frames.append(data)
                await session.on_frame(data)
    except TimeoutError:
        log.info("call ended at the demo time limit")
        await send_json({"type": "error", "message": "This demo call reached its time limit. Thanks for trying it!"})
        await ws.close()
    finally:
        forwarder.cancel()
        conversation.cancel()
        await stt.close()
        await asyncio.to_thread(agent.close)
        if frames:
            name = f"browser-{datetime.now():%Y%m%d-%H%M%S}.wav"
            path = await asyncio.to_thread(audio.save_wav, RECORDINGS / name, b"".join(frames))
            log.info("saved %.1f s of audio to %s", len(frames) / audio.FRAMES_PER_SECOND, path)
