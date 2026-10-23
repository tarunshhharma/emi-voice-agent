"""Voice activity detection with Silero VAD (ONNX), turned into speech_start / speech_end events."""
import logging
import urllib.request
from pathlib import Path

import numpy as np

from app.pipeline.audio import SAMPLE_RATE

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"
MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
WINDOW = 512  # samples per model call at 16 kHz = 32 ms
CONTEXT = 64  # samples of the previous window the model needs in front of each new one
WINDOW_MS = WINDOW * 1000 // SAMPLE_RATE

_session = None  # one model shared by all calls; each call keeps its own state


def load_model():
    """Download the model on first use, then load it once. Call at startup, not mid-call."""
    global _session
    if _session is None:
        import onnxruntime as ort

        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading Silero VAD model to %s", MODEL_PATH)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1  # what Silero recommends
        _session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts)
    return _session


class SileroVAD:
    """Feed 20 ms frames; get back "speech_start" and "speech_end" events.

    threshold:  speech probability that starts speech
    silence_ms: how long the probability must stay low before we call the turn over.
                The main latency/interruption trade-off: shorter feels snappier but
                cuts people off when they pause mid-sentence. Tuned on Day 5.
    """

    def __init__(self, threshold: float = 0.5, silence_ms: int = 500, prob_fn=None):
        self.threshold = threshold
        self.low_threshold = max(threshold - 0.15, 0.01)  # hysteresis: harder to stop than to start
        self.silence_ms = silence_ms
        self.prob_fn = prob_fn or self._silero_prob  # tests pass a fake
        self.speaking = False
        self._silent_for_ms = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def _silero_prob(self, window: np.ndarray) -> float:
        model_input = np.concatenate([self._context, window])[np.newaxis, :]
        out, self._state = load_model().run(
            None, {"input": model_input, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self._context = model_input[0, -CONTEXT:]
        return float(out[0, 0])

    def feed(self, frame: bytes) -> list[str]:
        """One 20 ms frame in. Returns the events it caused (usually none)."""
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        self._pending = np.concatenate([self._pending, samples])
        events = []
        while len(self._pending) >= WINDOW:  # 20 ms frames, 32 ms windows: buffer the difference
            window, self._pending = self._pending[:WINDOW], self._pending[WINDOW:]
            prob = self.prob_fn(window)
            if not self.speaking and prob >= self.threshold:
                self.speaking = True
                self._silent_for_ms = 0
                events.append("speech_start")
            elif self.speaking:
                self._silent_for_ms = self._silent_for_ms + WINDOW_MS if prob < self.low_threshold else 0
                if self._silent_for_ms >= self.silence_ms:
                    self.speaking = False
                    events.append("speech_end")
        return events

    @property
    def voiced(self) -> bool:
        """Speech in the latest window, not just inside the silence tail of a turn.
        Barge-in counts these, so 300 ms of real speech is needed, not "haan" plus a pause."""
        return self.speaking and self._silent_for_ms == 0
