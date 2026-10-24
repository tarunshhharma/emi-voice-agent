"""One voice session per call: frames in -> VAD -> STT -> final text -> agent -> TTS -> audio out.

Turn-taking while Priya is busy (thinking, speaking, or her audio still playing):
  300 ms of caller speech pauses her playback. The final transcript then decides:
  - backchannel ("haan", "hmm", "ji") or noise: she resumes where she paused;
  - anything else: barge-in, her reply is cancelled and the queued audio dropped;
  - and if she hadn't said anything real yet, the caller was just pausing mid-sentence:
    the pending reply is rolled back and both parts become one turn (a continuation).
Every turn's stage latencies are timed (metrics.TurnTimer) and saved with the turn.
"""
import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator

from app.pipeline.metrics import TurnTimer
from app.pipeline.stt.base import StreamingSTT
from app.pipeline.tts.base import StreamingTTS

log = logging.getLogger(__name__)
PREROLL_FRAMES = 15  # 300 ms kept from before speech was detected, so the first word isn't clipped
# 300 ms of real speech over Priya counts as an interruption. Lower reacts faster; higher
# ignores coughs, echo and "hmm". Tuned on Day 5 with the recorded "haan" clips.
BARGE_IN_FRAMES = 15
# Words that mean "go on", not "stop". Final transcripts are Roman (translit); partials can
# be Devanagari, so both forms are listed. Up to four of them in a row count as a backchannel.
BACKCHANNEL_WORDS = {
    "haan", "han", "haa", "ha", "haanji", "hmm", "hm", "hmmm", "mm", "umm", "ji", "jee",
    "achha", "acha", "accha", "achchha", "ok", "okay", "theek", "thik", "hai", "right", "yes",
    "हाँ", "हां", "जी", "हम्म", "अच्छा", "ठीक", "है", "ओके",
}
ANSWER_WINDOW_S = 1.0  # a "haan" in her last second of audio is probably the answer to her question
HOLD_TIMEOUT_S = 3.0  # caller stopped but no final transcript came: resume anyway
PCM_BYTES_PER_SECOND = 32_000  # 16 kHz x 2 bytes
# Said the moment the model starts a tool call, to cover the silence of the round trip.
# Synthesized once per server process and replayed, so it starts with no TTS wait.
FILLER_LINE = "Ek second, main check kar rahi hoon."
NO_FILLER_TOOLS = {"end_call"}  # "one second, let me check" before hanging up would be odd
_filler_pcm: bytes | None = None


def _log_late_error(task: asyncio.Future) -> None:
    if not task.cancelled() and task.exception():
        log.warning("interrupted turn failed: %r", task.exception())


def is_backchannel(text: str) -> bool:
    words = [w.strip(".,!?।") for w in text.lower().split()]
    words = [w for w in words if w]
    return 0 < len(words) <= 4 and all(w in BACKCHANNEL_WORDS for w in words)


async def from_list(sentences: list[str]) -> AsyncIterator[str]:
    for sentence in sentences:
        yield sentence


class VoiceSession:
    def __init__(self, vad, stt: StreamingSTT, send_json, tts: StreamingTTS | None = None,
                 send_audio=None, agent=None):
        self.vad = vad
        self.stt = stt
        self.send_json = send_json  # async function that sends a message to the caller's browser
        self.tts = tts
        self.send_audio = send_audio  # async function that sends PCM bytes to the browser for playback
        self.agent = agent  # CallAgent: blocking, so every call into it goes through a thread
        self.preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        # final transcripts waiting for the agent, each with the timer of its utterance
        self.user_turns: asyncio.Queue[tuple[str, TurnTimer]] = asyncio.Queue()
        self.reply_task: asyncio.Task | None = None  # the agent turn being thought or spoken now
        # TTS is faster than real time: the server finishes sending long before the browser
        # finishes playing. This estimates when the caller stops hearing Priya.
        self.playing_until = 0.0
        self.no_barge_in_until = 0.0  # the opening can't be interrupted
        self.voiced_frames = 0  # speech frames in the caller's current utterance
        self.interrupted = False  # tell the agent on its next turn
        # Latency timing
        self.speech_started_at = 0.0  # VAD heard the caller start (for barge-in time)
        self.last_voiced_at: float | None = None  # caller's last voiced frame: "they stopped talking"
        self.ending: TurnTimer | None = None  # utterance whose final transcript is awaited
        self.timer = TurnTimer()  # the reply being produced now
        self.said: list[str] = []  # sentences of that reply sent to TTS (what the caller could hear)
        self.filler_task: asyncio.Task | None = None
        # Pause-then-decide state
        self.held = False  # playback paused while we find out what the caller is saying
        self.paused_at = 0.0
        self.remaining_at_pause = 0.0  # seconds of Priya's audio still to play when paused
        self.hold_ms: int | None = None  # caller started speaking -> pause sent (barge-in reaction)
        self.hold_timeout: asyncio.Task | None = None
        self.reply_text: str | None = None  # caller text the current reply answers
        self.continuation_text: str | None = None  # to be joined with the caller's next words

    async def on_frame(self, frame: bytes) -> None:
        for event in self.vad.feed(frame):
            if event == "speech_start":
                await self.stt.start_utterance()
                # VAD needs a little audio before it is sure, so it fires slightly late.
                # Send the frames from just before, so the first syllable reaches STT.
                for earlier in self.preroll:
                    await self.stt.send_audio(earlier)
                self.preroll.clear()
                self.voiced_frames = 0
                self.speech_started_at = time.monotonic()
                await self.send_json({"type": "vad", "speaking": True})
            elif event == "speech_end":
                await self.stt.end_utterance()
                now = time.monotonic()
                self.ending = TurnTimer(caller_stopped=self.last_voiced_at or now, speech_end=now)
                if self.held:
                    self.hold_timeout = asyncio.create_task(self._resume_if_still_held(self.paused_at))
                await self.send_json({"type": "vad", "speaking": False})

        if self.vad.speaking:
            await self.stt.send_audio(frame)  # only speech goes to STT: less cost, less noise
            if self.vad.voiced:
                self.last_voiced_at = time.monotonic()
                self.voiced_frames += 1
                if self.voiced_frames == BARGE_IN_FRAMES and self.agent_busy():
                    await self.hold()
        else:
            self.preroll.append(frame)

    def agent_busy(self) -> bool:
        """Priya is thinking, speaking, or her audio is still playing, and may be interrupted."""
        now = time.monotonic()
        if now < self.no_barge_in_until:
            return False
        replying = self.reply_task is not None and not self.reply_task.done()
        return replying or now < self.playing_until

    async def hold(self) -> None:
        """300 ms of caller speech over Priya: pause her playback (it stays queued) until the
        final transcript says whether this is an interruption or just "haan"."""
        now = time.monotonic()
        self.held, self.paused_at = True, now
        self.remaining_at_pause = max(0.0, self.playing_until - now)
        self.hold_ms = round((now - self.speech_started_at) * 1000)
        log.info("paused for caller speech after %d ms", self.hold_ms)
        await self.send_json({"type": "pause"})

    async def resume(self) -> None:
        if not self.held:
            return
        self.held = False
        if self.hold_timeout is not None:
            self.hold_timeout.cancel()
        paused_for = time.monotonic() - self.paused_at
        if self.playing_until > self.paused_at:
            self.playing_until += paused_for  # the paused audio still has to play
        await self.send_json({"type": "resume"})

    async def _resume_if_still_held(self, paused_at: float) -> None:
        await asyncio.sleep(HOLD_TIMEOUT_S)
        if self.held and self.paused_at == paused_at:
            log.info("no transcript for the caller's speech: resuming")
            await self.resume()

    async def barge_in(self) -> None:
        """The caller interrupted: stop Priya's reply and drop the audio queued in the browser.
        If she hadn't said anything real yet, the caller was only pausing mid-sentence: roll
        the reply back and join their two fragments into one turn."""
        self.timer.barge_in_ms = self.hold_ms if self.held else round((time.monotonic() - self.speech_started_at) * 1000)
        # "Heard" means her real reply had started playing before the pause. Text sent while
        # she was paused, or a whole reply that finished generating meanwhile, was never heard.
        first_audio = self.timer.first_audio
        heard_something = first_audio is not None and (not self.held or first_audio < self.paused_at)
        self.held = False
        if self.hold_timeout is not None:
            self.hold_timeout.cancel()
        replying = self.reply_task is not None and not self.reply_task.done()
        continuation = self.reply_text is not None and not heard_something
        log.info("barge-in after %s ms (continuation=%s)", self.timer.barge_in_ms, continuation)
        if replying:
            self.reply_task.cancel()  # closes the TTS stream; the LLM thread finishes on its own
        if self.filler_task is not None:
            self.filler_task.cancel()
        self.playing_until = 0.0
        if continuation:
            self.continuation_text = self.reply_text
        else:
            self.interrupted = True
            if not replying and self.reply_text is not None:
                # Most interruptions land here: TTS finished sending (and the turn was saved)
                # seconds before the caller finished hearing it. Mark that saved turn.
                marked = asyncio.ensure_future(asyncio.to_thread(self.agent.mark_last_reply_interrupted, self.timer.barge_in_ms))
                marked.add_done_callback(_log_late_error)
        await self.send_json({"type": "barge_in", "continuation": continuation})

    async def on_final(self, text: str, timer: TurnTimer) -> None:
        """A final transcript: decide what it means if Priya was paused for it, then queue it."""
        if self.held:
            if not text.strip() or is_backchannel(text):
                # "haan" while she talks means "go on". In her last second it's probably the answer.
                answer = bool(text.strip()) and self.remaining_at_pause <= ANSWER_WINDOW_S
                await self.resume()
                if not answer:
                    return
            else:
                await self.barge_in()
        if text.strip():
            self.user_turns.put_nowait((text, timer))

    async def forward_transcripts(self) -> None:
        """Runs as its own task for the whole call, alongside the frame loop."""
        async for transcript in self.stt.transcripts():
            message = {"type": "transcript", "text": transcript.text, "final": transcript.final}
            timer = None
            if transcript.final:
                timer, self.ending = self.ending or TurnTimer(), None
                timer.stt_final = time.monotonic()
                # STT part of the latency budget: our end-of-speech -> provider's final text.
                if (stt_ms := timer.durations()["stt_ms"]) is not None:
                    message["stt_ms"] = stt_ms
            await self.send_json(message)
            if timer:
                await self.on_final(transcript.text, timer)

    async def converse(self) -> None:
        """Runs as its own task: the opening line, then one agent turn per final transcript.
        Turns are handled one at a time, in order."""
        # The opening carries the recording disclosure, so it must be heard in full:
        # no barge-in until it has finished playing. What the caller says meanwhile is queued.
        self.no_barge_in_until = float("inf")
        self.timer, self.said = TurnTimer(), []
        prepare = asyncio.create_task(self.prepare_filler())  # in parallel with the opening
        await self.speak(from_list([self.agent.first_line]))
        self.no_barge_in_until = self.playing_until
        await self.record(None, interrupted=False)
        await prepare
        while True:
            text, self.timer = await self.user_turns.get()
            replaces_previous = self.continuation_text is not None
            if replaces_previous:  # the caller paused mid-sentence: answer both parts together
                text, self.continuation_text = f"{self.continuation_text} {text}", None
            self.said, self.reply_text = [], text
            interrupted, self.interrupted = self.interrupted, False
            self.reply_task = asyncio.create_task(self.respond(text, interrupted, replaces_previous))
            try:
                await asyncio.wait({self.reply_task})  # unlike await, doesn't raise if the reply is cancelled
            except asyncio.CancelledError:  # the call itself is ending
                self.reply_task.cancel()
                raise
            cancelled = self.reply_task.cancelled()
            if not (cancelled and self.continuation_text is not None):
                await self.record(text, interrupted=cancelled)  # a rolled-back fragment isn't a turn
            if cancelled:
                continue  # barge-in: the caller's new words are already on their way as the next turn
            if error := self.reply_task.exception():  # LLM or DB error: say so and keep the call alive
                log.warning("agent turn failed: %r", error)
                await self.send_json({"type": "error", "message": f"Agent error: {error}"})
                continue
            if self.reply_task.result():
                await self.send_json({"type": "ended"})
                return

    async def record(self, caller_text: str | None, interrupted: bool) -> None:
        """Save the turn with its stage latencies, and show them in the browser."""
        timings = self.timer.durations()
        log.info("turn latency %s", {k: v for k, v in timings.items() if v is not None})
        await self.send_json({"type": "latency", **timings})
        try:
            await asyncio.to_thread(self.agent.save_turn, caller_text, " ".join(self.said), timings, interrupted)
        except Exception as e:  # metrics must never break the call
            log.warning("could not save turn: %r", e)

    async def prepare_filler(self) -> None:
        global _filler_pcm
        if _filler_pcm is not None:
            return
        try:
            _filler_pcm = b"".join([chunk async for chunk in self.tts.stream(from_list([FILLER_LINE]))])
        except Exception as e:  # no filler is fine: the caller just waits in silence as before
            log.warning("could not prepare filler: %r", e)

    def start_filler(self, tool: str) -> None:
        """The model started a tool call before saying anything this turn."""
        if tool in NO_FILLER_TOOLS or not _filler_pcm or self.said:
            return
        self.filler_task = asyncio.ensure_future(self.play_filler())

    async def play_filler(self) -> None:
        self.said.append(FILLER_LINE)
        await self.send_json({"type": "agent_text", "text": FILLER_LINE, "first": True, "filler": True})
        self.timer.filler_audio = time.monotonic()
        await self.send_audio(_filler_pcm)
        now = time.monotonic()
        self.playing_until = max(now, self.playing_until) + len(_filler_pcm) / PCM_BYTES_PER_SECOND

    async def respond(self, text: str, interrupted: bool = False, replaces_previous: bool = False) -> bool:
        """Run one agent turn in a worker thread and speak its sentences as they arrive.
        Returns True when the call has ended."""
        loop = asyncio.get_running_loop()
        sentences: asyncio.Queue[str | None] = asyncio.Queue()

        def on_text(sentence: str) -> None:
            # Called from the worker thread. asyncio objects aren't thread-safe, so hand
            # the sentence to the event loop instead of touching the queue directly.
            loop.call_soon_threadsafe(sentences.put_nowait, sentence)

        def on_tool_start(tool: str) -> None:
            loop.call_soon_threadsafe(self.start_filler, tool)  # also from the worker thread

        async def from_queue() -> AsyncIterator[str]:
            while (sentence := await sentences.get()) is not None:
                yield sentence

        self.timer.agent_start = time.monotonic()
        turn = asyncio.ensure_future(asyncio.to_thread(
            self.agent.turn, text, on_text, interrupted, on_tool_start, replaces_previous))
        turn.add_done_callback(lambda _: sentences.put_nowait(None))  # end of the reply
        try:
            await self.speak(from_queue())
        except asyncio.CancelledError:
            # Barge-in. The thread can't be stopped; when it finishes, read its result so a
            # late error is logged instead of becoming "exception was never retrieved".
            turn.add_done_callback(_log_late_error)
            raise
        return await turn

    async def speak(self, sentences: AsyncIterator[str]) -> None:
        """Say the agent's sentences: each one to the browser as text, and as audio once synthesized."""
        first = await anext(sentences, None)
        if first is None:
            return  # the turn produced nothing to say: don't open a TTS stream for it
        timer = self.timer
        timer.first_sentence = time.monotonic()

        async def shown(sentence_stream: AsyncIterator[str]) -> AsyncIterator[str]:
            is_first = True
            async for sentence in sentence_stream:
                await self.send_json({"type": "agent_text", "text": sentence.strip(), "first": is_first})
                is_first = False
                self.said.append(sentence.strip())
                yield sentence

        async def all_sentences() -> AsyncIterator[str]:
            yield first
            async for sentence in sentences:
                yield sentence

        try:
            async for chunk in self.tts.stream(shown(all_sentences())):
                if timer.first_audio is None:
                    # TTS part of the latency budget: first sentence ready -> first audio byte back.
                    timer.first_audio = time.monotonic()
                    first_audio_ms = timer.durations()["tts_first_byte_ms"]
                    log.info("tts first audio %d ms", first_audio_ms)
                    await self.send_json({"type": "tts", "first_audio_ms": first_audio_ms})
                await self.send_audio(chunk)
                now = time.monotonic()
                self.playing_until = max(now, self.playing_until) + len(chunk) / PCM_BYTES_PER_SECOND
        except Exception as e:  # bad key, network down: the caller sees the text and an error
            log.warning("TTS failed: %r", e)
            await self.send_json({"type": "error", "message": f"Text-to-speech unavailable: {e}"})
