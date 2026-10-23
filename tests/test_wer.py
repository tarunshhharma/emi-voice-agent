import json
import math
from array import array

import pytest

from app.pipeline.audio import mulaw_decode, mulaw_encode, phone_quality, resample
from app.pipeline.stt.deepgram import build_url, parse_results
from evals.wer import edit_distance, entity_hits, is_devanagari, normalize, reference_for, score
from evals.wer_clips import CLIPS

C13 = next(c for c in CLIPS if c["id"] == "c13")  # "teen hazaar paanch sau rupaye ... friday"


# --- normalisation and scoring ---


def test_normalize_drops_punctuation_and_case():
    assert normalize("Haan ji, main Rahul bol raha hoon!") == "haan ji main rahul bol raha hoon"


def test_normalize_unifies_devanagari_spellings():
    assert normalize("हाँ, फ़ोन।") == normalize("हां फोन")  # chandrabindu, nukta, danda


def test_normalize_numbers():
    assert normalize("₹3,500") == "3500"
    assert normalize("३५००") == "3500"  # Devanagari digits


def test_edit_distance():
    assert edit_distance("a b c".split(), "a b c".split()) == 0
    assert edit_distance("a b c".split(), "a x c".split()) == 1  # substitution
    assert edit_distance("a b c".split(), "a c".split()) == 1  # deletion
    assert edit_distance("a b c".split(), "a b b c".split()) == 1  # insertion
    assert edit_distance([], "a b".split()) == 2


def test_script_detection_picks_the_matching_reference():
    assert is_devanagari("तीन हजार पांच सौ रुपये")
    assert not is_devanagari("teen hazaar paanch sau")
    assert is_devanagari("मैं EMI परसों तक भर दूँगा")  # code-mixed but mostly Devanagari
    assert reference_for(C13, "teen hazaar") == C13["roman"]
    assert reference_for(C13, "तीन हजार") == C13["tts"]


def test_perfect_transcripts_score_zero_in_either_script():
    for clip in CLIPS:
        for hypothesis in (clip["roman"], clip["tts"]):
            s = score(clip, hypothesis)
            assert s["word_edits"] == 0 and s["char_edits"] == 0, (clip["id"], hypothesis)


def test_entity_matching_accepts_digits_words_and_either_script():
    assert entity_hits(C13, "3500 rupaye main friday ko de dunga") == 2
    assert entity_hits(C13, "तीन हजार पांच सौ रुपये मैं फ्राइडे को") == 2
    assert entity_hits(C13, "teen sau rupaye") == 0


def test_every_clip_finds_its_own_entities():
    for clip in CLIPS:
        assert entity_hits(clip, clip["roman"]) == len(clip["entities"]), clip["id"]
        assert entity_hits(clip, clip["tts"]) == len(clip["entities"]), clip["id"]


def test_twenty_unique_clips():
    assert len(CLIPS) == 20 and len({c["id"] for c in CLIPS}) == 20


# --- phone-line simulation ---


def tone(samples=16000):
    return array("h", (int(0.3 * 32767 * math.sin(2 * math.pi * 440 * n / 16000)) for n in range(samples))).tobytes()


def test_mulaw_known_values():
    assert mulaw_encode(array("h", [0]).tobytes()) == b"\xff"  # silence is 0xFF in G.711
    assert array("h", mulaw_decode(b"\xff"))[0] == 0


def test_mulaw_round_trip_is_close():
    original = array("h", tone(800))
    decoded = array("h", mulaw_decode(mulaw_encode(original.tobytes())))
    worst = max(abs(a - b) for a, b in zip(original, decoded))
    assert worst < 0.03 * 32768  # 8-bit, but small error on a mid-level signal


def test_resample_changes_length_only():
    assert len(resample(tone(16000), 16000, 8000)) == 8000 * 2


def test_phone_quality_keeps_length_and_format():
    assert len(phone_quality(tone(16000))) == 16000 * 2


# --- Deepgram parsing (shapes copied from a live session) ---


def test_deepgram_parse():
    msg = {"type": "Results", "is_final": True, "speech_final": True, "from_finalize": True,
           "channel": {"alternatives": [{"transcript": "haan ji"}]}}
    assert parse_results(json.dumps(msg)) == ("haan ji", True, True)
    assert parse_results(json.dumps({"type": "Metadata"})) is None


@pytest.mark.parametrize("language", ["hi", "multi"])
def test_deepgram_url(language):
    url = build_url(language)
    assert "model=nova-3" in url and f"language={language}" in url and "encoding=linear16" in url


def test_real_clip_is_scored_against_what_was_said():
    from evals.wer_clips import clip_for
    c16 = next(c for c in CLIPS if c["id"] == "c16")
    said = "mera naam priya iyer hai date of birth teen march unnees sau pachees"
    assert score(clip_for("real", c16), said)["word_edits"] == 0
    assert score(clip_for("real", c16), said)["entity_hits"] == 3
    assert score(clip_for("synthetic-clean", c16), said)["word_edits"] == 1  # script still says pachaasi
