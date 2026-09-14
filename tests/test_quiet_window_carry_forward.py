"""Regression tests for the three code items in the "smaller
correctness" commit.

1. A quiet window must not read as a perfect match. Below
   MIN_SPEECH_SECONDS of post-VAD speech, mismatch_score() returned
   0.0 -- indistinguishable from "confidently the enrolled speaker" --
   so a clone that pauses a lot got a free pass and dragged the EMA
   down. It now returns None, and _score_chunk carries the last
   confident mismatch forward within the call, flagging the window
   speaker_scored: False.
2. VoiceprintStore.enroll() is locked and writes atomically.
"""

import json
import os
import threading

import numpy as np
import pytest

from src.speaker_consistency import VoiceprintStore, MIN_SPEECH_SECONDS
from src.audio_pipeline import load_mono
from src.risk_engine import RiskEngine

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")
IMPOSTOR_B = os.path.join(SAMPLES, "impostor_speakerB.wav")
SR = 16000

pytestmark = pytest.mark.skipif(not os.path.exists(ENROLL_A), reason="samples not present")


@pytest.fixture
def store(tmp_path):
    s = VoiceprintStore(path=str(tmp_path / "vp.json"))
    s.enroll("ceo", *load_mono(ENROLL_A))
    return s


def _mostly_silent(seconds=5.0):
    """5 s window with ~0.4 s of speech-like noise: under MIN_SPEECH_SECONDS after VAD."""
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    burst = np.random.RandomState(0).randn(int(0.4 * SR)).astype(np.float32) * 0.05
    x[SR:SR + len(burst)] = burst
    return x


# ------------------------------------------------- 1. no opinion is None

def test_quiet_window_returns_none_not_zero(store):
    assert store.mismatch_score("ceo", _mostly_silent(), SR) is None


def test_unenrolled_speaker_returns_none(store):
    assert store.mismatch_score("nobody", *load_mono(IMPOSTOR_B)) is None


def test_speech_window_still_returns_a_number(store):
    m = store.mismatch_score("ceo", *load_mono(IMPOSTOR_B))
    assert m is not None and 0.0 <= m <= 1.0


def test_quiet_window_after_high_mismatch_carries_it_forward(store, monkeypatch):
    """THE regression case: impostor speaks (mismatch ~1.0), then a
    near-silent window. The silent window must keep the high mismatch,
    not reset to 0.0, and must be flagged as not assessed."""
    from src import api
    monkeypatch.setattr(api, "_voiceprints", store)
    ctx = {"context_score": 0.0, "reasons": []}
    engine, state = RiskEngine(), {}
    imp, _ = load_mono(IMPOSTOR_B)
    r1 = api._score_chunk(imp[:5 * SR], SR, 0.0, "ceo", True, ctx, engine, state)
    r2 = api._score_chunk(_mostly_silent(), SR, 5.0, "ceo", True, ctx, engine, state)

    assert r1["speaker_scored"] is True and r1["speaker_mismatch"] > 0.65
    assert r2["speaker_scored"] is False
    assert r2["speaker_carried"] is True
    assert r2["speaker_mismatch"] == r1["speaker_mismatch"]        # carried, not 0.0
    assert r2["speaker_mismatch"] != 0.0
    # and therefore the EMA is not dragged down by the pause
    assert r2["instant_score"] == pytest.approx(r1["instant_score"], abs=0.05)


def test_quiet_window_with_no_prior_opinion_is_neutral_and_flagged(store, monkeypatch):
    from src import api
    monkeypatch.setattr(api, "_voiceprints", store)
    r = api._score_chunk(_mostly_silent(), SR, 0.0, "ceo", True, {"context_score": 0.0, "reasons": []}, RiskEngine(), {})
    assert r["speaker_scored"] is False
    assert r["speaker_carried"] is False
    assert r["speaker_mismatch"] == 0.0


# ------------------------------------------------- 2. locked, atomic store

def test_enroll_writes_atomically_no_temp_left_behind(store, tmp_path):
    assert os.path.exists(store.path)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    data = json.load(open(store.path))
    assert "ceo" in data and len(data["ceo"]) == 256


def test_store_has_a_lock_and_concurrent_enrols_all_land(tmp_path):
    s = VoiceprintStore(path=str(tmp_path / "vp.json"))
    assert isinstance(s._lock, type(threading.Lock()))
    wav, sr = load_mono(ENROLL_A)
    errors = []
    def enrol(i):
        try:
            s.enroll(f"spk{i}", wav, sr)
        except Exception as e:      # pragma: no cover
            errors.append(e)
    threads = [threading.Thread(target=enrol, args=(i,)) for i in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors
    on_disk = json.load(open(s.path))
    assert set(on_disk) == {f"spk{i}" for i in range(4)}
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
