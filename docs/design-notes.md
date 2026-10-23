# Design notes

## Day 4 · Fri 23 Oct 2026 — Audio out: the full spoken loop

**What got done**
- Day 3 finished: Deepgram nova-3 behind the same STT interface, phone-line simulation (8 kHz μ-law), and a WER comparison on synthetic and real recorded Hinglish. Sarvam translit won on key words (100%) and real-voice WER (6%).
- Streaming TTS: ElevenLabs Flash v2.5 over the stream-input WebSocket, 16 kHz PCM, gapless playback in the browser.
- The full loop: VAD → STT final → agent turn in a worker thread → each checked sentence to TTS while the LLM writes the next.
- Barge-in: 300 ms of caller speech over the agent cancels the reply and flushes browser audio; the next turn is told it was cut off. The disclosure opening can't be interrupted.
- Per-stage latency for every turn, saved in `turns`, with a p50/p95 summary.

**What broke, and why**
- The agent spoke the same question twice: the model returns a "commentary" and a reworded "final_answer" message in one response. Now only the first message item per response is spoken.
- The agent called `verify_identity` with digits the caller never said, burning an attempt. Deferred when the agent asked a question earlier in the same turn. Still open: once it invented digits without asking (full fix planned for Day 6: the factor must appear in the caller's words).
- Barge-in looked 2.8 s slow in a probe: the test recording was very quiet and Silero, right after loud speech, barely registered it. At browser-like level (auto gain control) it fired normally.
- Windows timing traps: `time.monotonic()` ticks every ~15 ms and `asyncio.sleep(0.02)` sleeps ~31 ms, which skewed a test and the probe's pacing.

**Numbers**
- Voice-to-voice 3–6.8 s in first calls; LLM to first sentence 2.5–5.4 s dominates. STT ~300 ms, TTS ~400 ms, barge-in ~300 ms.

**What's next**
- Day 5: cut LLM latency (model choice, prompt size, connection reuse, filler during tools), tune VAD silence with the recorded turn-taking clips, dashboard, deploy, demo video.
- Day 6: verify-factor check, Twilio, eval run with the judge, red-team.

## Day 5 (part 1) · Fri 23 Oct 2026 — Latency and turn-taking

**What got done**
- Profiled the LLM: six models answer a plain turn in ~1 s; tool turns take 3.7–5.3 s because of sequential round trips. Added a pre-synthesized filler at the first tool call (first sound on the verify turn ~4.9 s → ~2.5 s), one shared warm OpenAI client (~400 ms saved on the first request), and `filler_ms` alongside `voice_to_voice_ms`.
- Tuned endpointing on recorded clips: pauses inside a sentence are 320–1480 ms, so the silence stays 500 ms and continuations are merged (pending reply rolled back, both fragments answered together).
- Backchannels: pause at 300 ms of caller speech, decide on the final transcript (resume vs barge-in).

**What broke, and why**
- After a deferred tool, another LLM round made the model apologise and repeat itself; the turn now ends after a deferral.
- A reply generated while Priya was paused was counted as heard; "heard" now means audio played before the pause.
- Sarvam misheard digits and dates in synthesized speech ("21" → "24", "May" → "mein"), which fails verification. Day 6: read the DOB back before verifying.

**What's next**
- Dashboard (required), then deploy. Day 6: verify-factor check, eval run with judge, red-team.
