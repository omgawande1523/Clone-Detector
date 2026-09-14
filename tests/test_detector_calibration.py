"""Calibration tests for the heuristic acoustic detector, against
certainly-labelled audio: ASVspoof 2019 LA dev, labels from the
official protocol file (see fixtures/asvspoof2019/README.md).

Two properties are pinned:
  1. real, labelled-bonafide speech from several speakers must not be
     flagged — every threshold sits outside the observed bonafide range;
  2. crude over-smoothed synthetic speech must still be flagged.

What is deliberately NOT asserted: recall on the ASVspoof spoofs. On
that data the three hand-crafted features barely separate the classes
(see the calibration notes in detector.py). Asserting recall would
either fail or force thresholds that flag genuine callers. The spoof
scores are printed for visibility instead.

History, so nobody re-tunes on the wrong thing: thresholds tuned on
synthetic sine waves made the detector emit exactly 0.000 on all real
audio; thresholds tuned on one band-limited call recording then fired
on 100% of studio-quality bonafide speech, because spectral flatness
tracks recording conditions far more than synthesis.
"""

import os

import numpy as np
import pytest

from src.detector import HeuristicDetector
from src.audio_pipeline import load_mono

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "fixtures", "asvspoof2019")
PROTOCOL = os.path.join(FIX, "ASVspoof2019.LA.cm.dev.trl.subset.txt")
SR = 16000

pytestmark = pytest.mark.skipif(not os.path.exists(PROTOCOL), reason="ASVspoof fixtures not present")


@pytest.fixture(scope="module")
def detector():
    return HeuristicDetector()


def _labelled():
    """(utt, speaker, system, key) straight from the official protocol."""
    rows = []
    for line in open(PROTOCOL):
        spk, utt, _env, system, key = line.split()
        rows.append((utt, spk, system, key))
    return rows


def _score(detector, utt):
    wav, sr = load_mono(os.path.join(FIX, "flac", utt + ".flac"))
    assert sr == SR
    return detector.predict_detailed(wav, sr)


@pytest.fixture(scope="module")
def bonafide_scores(detector):
    rows = [r for r in _labelled() if r[3] == "bonafide"]
    return {utt: _score(detector, utt) for utt, *_ in rows}


@pytest.fixture(scope="module")
def spoof_scores(detector):
    rows = [r for r in _labelled() if r[3] == "spoof"]
    return {utt: (system, _score(detector, utt)) for utt, _spk, system, _key in rows}


def _vocoder_signal(duration=9.0, sr=SR, f0=140.0):
    """Perfectly flat pitch and a smooth envelope — the crude
    over-smoothed vocoder artefact this detector can still catch."""
    t = np.arange(int(duration * sr)) / sr
    phase = 2 * np.pi * f0 * t
    wav = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    wav = wav * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    return (wav / np.max(np.abs(wav)) * 0.8).astype(np.float32)


# ------------------------------------------------- false-positive side

def test_bonafide_batch_is_multi_speaker(bonafide_scores):
    speakers = {spk for utt, spk, *_ in _labelled() if utt in bonafide_scores}
    assert len(bonafide_scores) >= 40
    assert len(speakers) >= 5


def test_labelled_bonafide_speech_is_not_flagged(bonafide_scores):
    """THE regression test for the false-positive direction."""
    probs = np.array([s["spoof_prob"] for s in bonafide_scores.values()])
    assert np.median(probs) < 0.10
    assert probs.max() < 0.35, f"a labelled-bonafide utterance scored {probs.max():.3f}"
    assert np.mean(probs > 0.5) == 0.0


def test_no_single_cue_fires_on_bonafide(bonafide_scores):
    """Each sub-score individually, so a bad threshold on one cue
    cannot hide behind the others' weight."""
    for cue in ("jitter_score", "flatness_score", "smoothness_score"):
        vals = np.array([s[cue] for s in bonafide_scores.values()])
        assert np.percentile(vals, 95) < 0.25, f"{cue} p95 = {np.percentile(vals, 95):.2f} on bonafide"


def test_thresholds_sit_outside_the_bonafide_range(detector):
    """Measured on the 48 labelled-bonafide utterances:
    jitter min 0.076, flatness 0.118-0.567, envelope var 0.0028-0.0277.
    A separate call recording ran flatness ~0.09 and envelope var to
    ~0.054, so the bounds sit outside BOTH."""
    assert detector.JITTER_SUSPICIOUS < 0.076
    assert detector.FLATNESS_TONAL < 0.09
    assert detector.FLATNESS_NOISY > 0.567
    assert detector.ENV_VAR_SUSPICIOUS < 0.0028
    assert detector.ENV_VAR_EXCESS > 0.054


# ---------------------------------------------------- detection side

def test_oversmoothed_synthetic_is_flagged(detector):
    wav = _vocoder_signal()
    scores = [detector.predict_detailed(wav[i:i + 3 * SR], SR)["spoof_prob"]
              for i in range(0, len(wav), 3 * SR)]
    assert min(scores) > 0.6, f"vocoder signal scored only {scores}"


def test_synthetic_separates_from_labelled_bonafide(detector, bonafide_scores):
    wav = _vocoder_signal()
    synthetic = np.median([detector.predict_detailed(wav[i:i + 3 * SR], SR)["spoof_prob"]
                           for i in range(0, len(wav), 3 * SR)])
    bona = np.median([s["spoof_prob"] for s in bonafide_scores.values()])
    assert synthetic - bona > 0.5


def test_asvspoof_recall_is_reported_not_asserted(spoof_scores, bonafide_scores):
    """Visibility only. Run with -s to see it. The heuristic layer is
    known to be weak against these systems; this prints how weak."""
    bona_med = np.median([s["spoof_prob"] for s in bonafide_scores.values()])
    print(f"\n  bonafide median spoof_prob: {bona_med:.3f}")
    for system in sorted({sysid for sysid, _ in spoof_scores.values()}):
        vals = [s["spoof_prob"] for sysid, s in spoof_scores.values() if sysid == system]
        print(f"  {system} spoof_prob: " + ", ".join(f"{v:.3f}" for v in vals))
    caught = sum(s["spoof_prob"] > 0.3 for _, s in spoof_scores.values())
    print(f"  spoofs above 0.3: {caught}/{len(spoof_scores)}")
    assert True


# -------------------------------------------------------- edge cases

def test_flatness_only_fires_on_non_speech_extremes(detector):
    tonal = (np.sin(2 * np.pi * 200 * np.arange(SR) / SR) * 0.8).astype(np.float32)
    noise = (np.random.RandomState(0).randn(SR) * 0.3).astype(np.float32)
    assert detector.predict_detailed(tonal, SR)["flatness_score"] > 0.5
    assert detector.predict_detailed(noise, SR)["flatness_score"] > 0.5


def test_silence_is_uninformative_not_suspicious(detector):
    silence = np.zeros(int(1.5 * SR), dtype=np.float32)
    assert detector.predict_detailed(silence, SR)["jitter_score"] == 0.0


def test_too_short_input_returns_zeros(detector):
    tiny = np.zeros(int(0.1 * SR), dtype=np.float32)
    assert detector.predict_detailed(tiny, SR)["spoof_prob"] == 0.0


def test_scores_stay_in_range(bonafide_scores):
    probs = np.array([s["spoof_prob"] for s in bonafide_scores.values()])
    assert probs.min() >= 0.0 and probs.max() <= 1.0
