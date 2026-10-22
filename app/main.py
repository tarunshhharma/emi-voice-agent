from fastapi import FastAPI

app = FastAPI(title="EMI Voice Agent")


@app.get("/health")
def health():
    return {"status": "ok"}
