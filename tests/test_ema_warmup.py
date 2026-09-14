"""Regression tests: a lead-in with no speech must not anchor the EMA.

RiskEngine.update() seeded _ema from the first window unconditionally.
A silent or noise-only lead-in therefore seeded it at its own near-zero
score and cost roughly two windows of climb before an alert could fire.
Windows with no speech are now reported but do not seed or move the
EMA, and the first window WITH speech becomes the seed.
"""

import os

import numpy as np
import pytest
import soundfile as sf

from src.audio_pipeline import has_speech, SPEECH_MIN_SECONDS
from src.risk_engine import RiskEngine, ALERT_THRESHOLD

SR = 16000
FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "fixtures", "live_clone_session")


def _silence(sec=5.0):
    return np.zeros(int(sec * SR), dtype=np.float32)


def _speech_like(sec=5.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(int(sec * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return (x + rng.randn(len(t)) * 0.01).astype(np.float32)


# ------------------------------------------------------ speech detector

def test_digital_silence_has_no_speech():
    assert has_speech(_silence(), SR) is False


def test_speech_like_audio_has_speech():
    assert has_speech(_speech_like(), SR) is True


def test_a_tiny_blip_is_not_speech():
    x = _silence()
    x[SR:SR + int(0.1 * SR)] = 0.5          # 0.1 s, under SPEECH_MIN_SECONDS
    assert SPEECH_MIN_SECONDS > 0.1
    assert has_speech(x, SR) is False


@pytest.mark.skipif(not os.path.exists(FIX), reason="fixture not present")
def test_real_fixture_windows_all_have_speech():
    """The captured session's window 0 was NOT silence -- it was speech
    the model judged genuine (acoustic 0.004). This pins that the
    detector agrees, so nobody later explains that failure away as a
    silent lead-in."""
    for i in range(6):
        p = os.path.join(FIX, f"chunk{i:03d}.wav")
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        assert has_speech(x, sr) is True, f"chunk{i:03d} should read as speech"


# --------------------------------------------------------- the warm-up

def test_silent_lead_in_costs_no_windows():
    """THE regression: a call with two silent windows before the clone
    must alert at the same window index (counting from the first speech
    window) as one with no lead-in at all."""
    with_lead = RiskEngine()
    fired_with = None
    for i, (score, speech) in enumerate([(0.0, False), (0.0, False),
                                         (0.90, True), (0.90, True), (0.90, True)]):
        if with_lead.update(score, speech_present=speech)["alert"] and fired_with is None:
            fired_with = i - 2                      # windows since speech started

    without = RiskEngine()
    fired_without = None
    for i, score in enumerate([0.90, 0.90, 0.90]):
        if without.update(score)["alert"] and fired_without is None:
            fired_without = i

    assert fired_with is not None and fired_without is not None
    assert fired_with == fired_without


def test_no_speech_window_does_not_move_the_ema():
    e = RiskEngine()
    e.update(0.80)
    before = e.summary()["final_ema"]
    r = e.update(0.0, speech_present=False)
    assert r["scored"] is False
    assert e.summary()["final_ema"] == before
    assert r["risk_score"] == before


def test_no_speech_window_neither_advances_nor_resets_the_streak():
    e = RiskEngine()
    e.update(0.90)                                   # streak 1
    quiet = e.update(0.0, speech_present=False)      # holds
    assert quiet["alert"] is False
    assert e.update(0.90)["alert"] is True           # streak 2 -> alert


def test_scored_flag_is_reported():
    e = RiskEngine()
    assert e.update(0.5)["scored"] is True
    assert e.update(0.05, speech_present=False)["scored"] is False


def test_alert_level_evidence_is_scored_even_when_speech_is_not_detected():
    """FAIL-SAFE: a speech-detection false negative must never suppress
    a window the detector scored at alert level. A 1.0 s In-the-Wild
    clone clip scored 0.996 synthetic and was reported LOW because the
    speech detector said 'no speech' -- the same bug class, from the
    other direction."""
    e = RiskEngine()
    r = e.update(0.90, speech_present=False)
    assert r["scored"] is True
    assert r["risk_score"] >= ALERT_THRESHOLD
    # ...while a quiet window with no evidence is still skipped
    e2 = RiskEngine()
    assert e2.update(0.05, speech_present=False)["scored"] is False
