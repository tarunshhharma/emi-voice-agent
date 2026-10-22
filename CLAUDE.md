# EMI Voice Agent

Hinglish EMI reminder voice agent, synthetic data only. Full plan (architecture, tools, data model, 7-day roadmap) is in `docs/plan.md`; read it before starting work.

## Stack
Python 3.11, FastAPI, asyncio, Silero VAD, Sarvam/Deepgram STT, OpenAI LLM, ElevenLabs TTS, SQLite (SQLAlchemy).

## Environment
Windows, plain venv (`.\venv\Scripts\Activate.ps1`). Run the server with `uvicorn app.main:app --reload`. API keys live in `.env` (gitignored); never print or commit them.

## Goal
Voice AI interview prep. The user must be able to explain every design choice.

## Working rules
- Keep code simple and readable. No extra abstractions until needed.
- Do one task at a time, then stop and wait for the user to confirm it runs.
- After non-trivial code, explain in 3-5 bullets what it does and why, and what an interviewer might ask.
- Do not overwrite `docs/plan.md`, `docs/design-notes.md` or anything in `.cursor/`.

## Conventions from the plan
- Code decides the call state and which tools are allowed; the LLM only handles language inside a state.
- Guardrails (calling hours, no disclosure before verification, output filter) are enforced in code, not only in the prompt.
- Text mode first: test the agent brain without audio before adding VAD/STT/TTS.
- Confirm current provider model names before using them.

## Progress
- Before starting work, read the latest entry in `docs/design-notes.md` to see where we left off.
- `docs/design-notes.md` is append-only: add an end-of-day entry only when I ask (what got done, what broke, what's next).

## Dependencies
- Pin every version in `requirements.txt`. Ask before adding a new dependency and say why it's needed.
- Create the venv with `py -3.11 -m venv venv` so the Python version matches the plan.
- If `Activate.ps1` is blocked, use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Provider APIs
- Don't guess SDK methods or model names for Sarvam, ElevenLabs, Deepgram, Twilio or OpenAI. Check their current docs, or ask me to paste the relevant page.

## Code rules
- No blocking calls inside async code (no `time.sleep`, no sync HTTP clients, no heavy CPU work on the event loop); use async clients or `asyncio.to_thread`. This is the most common cause of voice latency bugs.
- Use `logging`, not `print`. Mask phone numbers and DOB in logs, even though the data is synthetic.
- Write pytest tests for tools, guardrails and the state machine. Run `pytest` before saying a task is done.

## Git
- After I confirm a task works, suggest a short commit message. Don't commit or push without my OK.

## Safety
- Synthetic data only. The only real phone number allowed is my verified test number, read from `.env`.