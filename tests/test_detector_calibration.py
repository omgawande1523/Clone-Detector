"""Calibration tests for the heuristic acoustic detector.

The thresholds were originally chosen against the synthetic sine waves
in demo_cli.py and were about an order of magnitude away from what real
speech measures, so on a 6.4-minute real recording EVERY sub-score came
out at exactly 0.000 for all 128 chunks — the detector was inert on real
audio, not merely weak.

These tests pin the two properties that matter:
  1. real recorded speech must not be flagged, and
  2. over-smoothed synthetic speech must still be flagged.

SCOPE: the genuine side is one real speaker; the synthetic side is a
vocoder-style signal plus Windows SAPI TTS. This establishes "does not
false-positive on real speech" and "still catches over-smoothed
output". It does NOT establish accuracy against modern neural voice
clones — that needs ASVspoof 2019/2021 LA or In-the-Wild data.
"""

import os

import numpy as np
import pytest

from src.detector import HeuristicDetector
from src.audio_pipeline import chunk_audio

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
REAL_SPEECH = os.path.join(SAMPLES, "Original Call Recording.wav")

SR = 16000


@pytest.fixture(scope="module")
def detector():
    return HeuristicDetector()


def _vocoder_signal(duration=9.0, sr=SR, f0=140.0):
    """Perfectly flat pitch and a smooth envelope — the classic
    over-smoothed vocoder artifact this detector exists to catch."""
    t = np.arange(int(duration * sr)) / sr
    phase = 2 * np.pi * f0 * t
    wav = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    wav = wav * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    return (wav / np.max(np.abs(wav)) * 0.8).astype(np.float32)


@pytest.fixture(scope="module")
def real_scores(detector):
    if not os.path.exists(REAL_SPEECH):
        pytest.skip("real speech sample not present in samples/")
    return np.array([
        detector.predict_detailed(c.waveform, c.sample_rate)["spoof_prob"]
        for c in chunk_audio(REAL_SPEECH, chunk_seconds=3.0)
        if len(c.waveform) >= c.sample_rate * 2.0
    ])


def test_real_speech_is_not_flagged(real_scores):
    """THE regression test for the false-positive direction."""
    assert np.median(real_scores) < 0.25
    assert np.percentile(real_scores, 90) < 0.35
    assert real_scores.max() < 0.60, (
        f"a chunk of genuine speech scored {real_scores.max():.3f}"
    )


def test_real_speech_false_positive_rate_is_zero(real_scores):
    assert np.mean(real_scores > 0.5) == 0.0


def test_detector_is_not_inert_on_real_audio(real_scores):
    """The original failure mode: constant 0.000 for every chunk. A
    detector that always says one thing is not a detector."""
    assert real_scores.std() > 0.0
    assert len(np.unique(np.round(real_scores, 3))) > 1


def test_oversmoothed_synthetic_is_flagged(detector):
    wav = _vocoder_signal()
    scores = [detector.predict_detailed(wav[i:i + 3 * SR], SR)["spoof_prob"]
              for i in range(0, len(wav), 3 * SR)]
    assert min(scores) > 0.6, f"vocoder signal scored only {scores}"


def test_synthetic_separates_from_real(detector, real_scores):
    wav = _vocoder_signal()
    synthetic = np.median([detector.predict_detailed(wav[i:i + 3 * SR], SR)["spoof_prob"]
                           for i in range(0, len(wav), 3 * SR)])
    assert synthetic - np.median(real_scores) > 0.5


def test_thresholds_bracket_the_real_speech_distribution(detector):
    """Each threshold must sit below where real speech actually lives,
    or the sub-score fires on genuine callers. These are the measured
    p10 values from the calibration recording."""
    assert detector.JITTER_SUSPICIOUS <= 0.063
    assert detector.ENV_VAR_SUSPICIOUS <= 0.0092
    assert detector.FLATNESS_TONAL <= 0.036
    assert detector.FLATNESS_NOISY >= 0.220


def test_flatness_is_two_sided(detector):
    """Real speech occupies roughly 0.017-0.220 spectral flatness.
    Both an unnaturally tonal signal and an unnaturally noise-like one
    are anomalous; the old one-sided check could only see the former,
    and the TTS measured here sat above the range, not below."""
    quiet = np.zeros(SR, dtype=np.float32)
    tonal = (np.sin(2 * np.pi * 200 * np.arange(SR) / SR) * 0.8).astype(np.float32)
    noisy = (np.random.RandomState(0).randn(SR) * 0.3).astype(np.float32)
    assert detector.predict_detailed(tonal, SR)["flatness_score"] > 0.5
    assert detector.predict_detailed(noisy, SR)["flatness_score"] > 0.5
    del quiet


def test_silence_is_uninformative_not_suspicious(detector):
    silence = np.zeros(int(1.5 * SR), dtype=np.float32)
    assert detector.predict_detailed(silence, SR)["jitter_score"] == 0.0


def test_too_short_input_returns_zeros(detector):
    tiny = np.zeros(int(0.1 * SR), dtype=np.float32)
    result = detector.predict_detailed(tiny, SR)
    assert result["spoof_prob"] == 0.0


def test_scores_stay_in_range(detector, real_scores):
    assert real_scores.min() >= 0.0 and real_scores.max() <= 1.0
