from app.pipeline.audio import FRAME_BYTES
from app.pipeline.vad import SileroVAD
from evals.split_take import find_segments, join


def scripted(probs):
    it = iter(probs)
    return lambda window: next(it, 0.0)


def test_two_sentences_give_two_segments():
    # 32 ms windows: speech, 1 s silence, speech, 1 s silence
    probs = [0.9] * 30 + [0.05] * 32 + [0.9] * 30 + [0.05] * 32
    vad = SileroVAD(silence_ms=700, prob_fn=scripted(probs))
    pcm = bytes(FRAME_BYTES * 250)  # 5 s, enough frames for every window
    segments = find_segments(pcm, vad)
    assert len(segments) == 2
    assert segments[0][1] <= segments[1][0] + 15 * FRAME_BYTES  # overlap at most the 300 ms pad


def test_pause_shorter_than_silence_setting_stays_one_segment():
    probs = [0.9] * 30 + [0.05] * 10 + [0.9] * 30 + [0.05] * 32  # 320 ms pause mid-sentence
    vad = SileroVAD(silence_ms=700, prob_fn=scripted(probs))
    assert len(find_segments(bytes(FRAME_BYTES * 250), vad)) == 1


def test_speech_until_the_end_is_still_a_segment():
    vad = SileroVAD(silence_ms=700, prob_fn=scripted([0.05] * 10 + [0.9] * 1000))
    segments = find_segments(bytes(FRAME_BYTES * 100), vad)
    assert len(segments) == 1 and segments[0][1] == 100 * FRAME_BYTES


def test_join_merges_a_segment_with_the_next():
    segments = [(0, 10), (20, 30), (40, 50), (60, 70)]
    assert join(segments, [2]) == [(0, 10), (20, 50), (60, 70)]
    assert join(segments, [1, 3]) == [(0, 30), (40, 70)]  # numbers refer to the original list
    assert join(segments, []) == segments
