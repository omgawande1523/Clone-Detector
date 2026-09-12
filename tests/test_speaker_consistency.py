"""Regression tests for the speaker-consistency layer.

The original MFCC-style voiceprint compared log-band-energy vectors,
which are all the same sign — so cosine similarity was pinned near 1.0
for any two speech signals. A genuinely different speaker scored a
mismatch of 0.002 while the SAME speaker with flattened pitch scored
0.632: it measured synthesis artifacts, not speaker identity. These
tests assert real separation between speakers.

Requires the sample clips in samples/ (two distinct speakers) and
Resemblyzer; they are skipped if either is missing.
"""

import os

import pytest

from src.audio_pipeline import load_mono
from src.speaker_consistency import (
    VoiceprintStore, compute_voiceprint, cosine_similarity,
    mismatch_from_similarity, mismatch_between,
    SIM_MATCH, SIM_IMPOSTOR,
)

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")
GENUINE_A = os.path.join(SAMPLES, "genuine_speakerA.wav")
IMPOSTOR_B = os.path.join(SAMPLES, "impostor_speakerB.wav")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (ENROLL_A, GENUINE_A, IMPOSTOR_B)),
    reason="speaker samples not present in samples/",
)

# A mismatch below this means "confidently the enrolled speaker";
# above the second, "confidently somebody else". The gap between them
# is the dynamic range the old implementation did not have.
MATCH_CEILING = 0.35
IMPOSTOR_FLOOR = 0.65


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("vp") / "voiceprints.json"
    s = VoiceprintStore(path=str(path))
    wav, sr = load_mono(ENROLL_A)
    s.enroll("speakerA", wav, sr)
    return s


def test_same_speaker_scores_low_mismatch(store):
    wav, sr = load_mono(GENUINE_A)
    assert store.mismatch_score("speakerA", wav, sr) < MATCH_CEILING


def test_different_speaker_scores_high_mismatch(store):
    """THE regression test: this scored 0.002 before the fix."""
    wav, sr = load_mono(IMPOSTOR_B)
    mismatch = store.mismatch_score("speakerA", wav, sr)
    assert mismatch > IMPOSTOR_FLOOR, (
        f"a different speaker scored only {mismatch} — the layer is not "
        "discriminating between speakers"
    )


def test_speakers_are_actually_separated(store):
    """The two must not merely be ordered, but far apart."""
    same = store.mismatch_score("speakerA", *load_mono(GENUINE_A))
    other = store.mismatch_score("speakerA", *load_mono(IMPOSTOR_B))
    assert other - same > 0.5, f"separation is only {other - same:.3f}"


def test_raw_cosine_separates_speakers(store):
    same = store.similarity("speakerA", *load_mono(GENUINE_A))
    other = store.similarity("speakerA", *load_mono(IMPOSTOR_B))
    assert same > SIM_MATCH, f"same-speaker cosine {same} below SIM_MATCH"
    assert other < SIM_IMPOSTOR, f"different-speaker cosine {other} above SIM_IMPOSTOR"


def test_embedding_is_256_dim_and_unit_norm():
    wav, sr = load_mono(ENROLL_A)
    print_a = compute_voiceprint(wav, sr)
    assert len(print_a) == 256
    assert abs(sum(x * x for x in print_a) - 1.0) < 1e-3


def test_identical_audio_is_a_perfect_match():
    wav, sr = load_mono(GENUINE_A)
    vp = compute_voiceprint(wav, sr)
    assert cosine_similarity(vp, vp) == pytest.approx(1.0, abs=1e-6)
    assert mismatch_between(vp, vp) == 0.0


def test_unenrolled_speaker_stays_neutral(store):
    """No reference on file must mean 'no opinion', never a penalty."""
    assert store.mismatch_score("nobody_enrolled", *load_mono(GENUINE_A)) == 0.0
    assert store.similarity("nobody_enrolled", *load_mono(GENUINE_A)) is None


def test_too_short_audio_declines_to_score():
    """A sliver of audio must return None rather than a confident guess.
    At 0.6s these produced a 0.813 mismatch on a genuine caller and a
    0.000 mismatch on an impostor — confidently wrong both ways."""
    wav, sr = load_mono(GENUINE_A)
    assert compute_voiceprint(wav[: int(0.3 * sr)], sr) is None


def test_calibration_is_monotonic_and_saturates():
    assert mismatch_from_similarity(SIM_MATCH + 0.1) == 0.0
    assert mismatch_from_similarity(SIM_IMPOSTOR - 0.1) == 1.0
    midpoint = mismatch_from_similarity((SIM_MATCH + SIM_IMPOSTOR) / 2)
    assert 0.4 < midpoint < 0.6
    assert mismatch_from_similarity(None) == 0.0


def test_cosine_is_not_rescaled_into_uselessness():
    """Guards against reintroducing the (x+1)/2 rescale that compressed
    every comparison into a narrow band near 1.0."""
    a = [1.0] + [0.0] * 255
    b = [0.0, 1.0] + [0.0] * 254
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)
