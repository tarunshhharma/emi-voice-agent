# The voice server as one container (Render free web service, or locally with docker compose).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY client ./client
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit

# Bake the Silero VAD model into the image: a cold start then doesn't depend on GitHub.
RUN python -c "from app.pipeline.vad import load_model; load_model()"

EXPOSE 8000
# Free hosting has no persistent disk, so seed the synthetic borrowers on every start
# (seed 42: the same data each time). Render sets PORT; locally it defaults to 8000.
# exec: uvicorn replaces the shell as process 1, so it receives stop signals and shuts down cleanly.
CMD ["sh", "-c", "python -m app.db.seed && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
