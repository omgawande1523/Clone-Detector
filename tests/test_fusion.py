"""Regression tests for the fusion-weight / alert-threshold coupling.

These would have caught the bug where weights of 0.55/0.30/0.15 sat
under an alert threshold of 0.60, making it arithmetically impossible
for any single signal to raise an alert — the system advertised three
independent signals but in practice required several at once.
"""

import itertools

import pytest

from src.fusion import (fuse, SignalCaps, check_alertable, caps_for_backend,
                        detection_mode, TRANSFORMER_CAPS, HEURISTIC_CAPS)
from src.risk_engine import ALERT_THRESHOLD, LEVELS, level_for, RiskEngine
from src.detector import HeuristicDetector, TransformerDetector


@pytest.mark.parametrize("w", [TRANSFORMER_CAPS, HEURISTIC_CAPS])
def test_caps_need_not_sum_to_one(w):
    """They are per-signal caps in a noisy-OR, not weights in a mean.
    The sum-to-1 constraint is what forced a single confident signal to
    stay under its own weight -- see fusion.py and
    tests/test_fusion_noisy_or.py."""
    assert 0.0 <= w.acoustic <= 1.0
    assert 0.0 <= w.speaker <= 1.0
    assert 0.0 <= w.context <= 1.0


# ------------------------------------------------ backend-aware weights

def test_weights_follow_the_detector_not_a_flag():
    """Selection keys off the detector class's own `backend` attribute,
    so it cannot drift from whatever DETECTOR_BACKEND actually loaded."""
    assert caps_for_backend(HeuristicDetector.backend) is HEURISTIC_CAPS
    assert caps_for_backend(TransformerDetector.backend) is TRANSFORMER_CAPS


def test_unknown_backend_is_refused_not_defaulted():
    with pytest.raises(ValueError):
        caps_for_backend("some-new-model")


def test_heuristic_acoustic_weight_is_exactly_zero():
    """Measured AUC 0.25 (ASVspoof) and 0.14 (In-the-Wild): below
    chance. Any positive weight makes the fused score worse. If this
    test fails because someone restored 0.05 'to be safe', read the
    comment in fusion.py and revert."""
    assert HEURISTIC_CAPS.acoustic == 0.0
    assert fuse(1.0, 0.0, 0.0, HEURISTIC_CAPS)["fused_score"] == 0.0


def test_heuristic_mode_speaker_alone_still_alerts():
    assert fuse(0.0, 1.0, 0.0, HEURISTIC_CAPS)["fused_score"] >= ALERT_THRESHOLD
    c = check_alertable(ALERT_THRESHOLD, HEURISTIC_CAPS)
    assert c["speaker_can_alert"] is True
    assert c["acoustic_can_alert"] is False       # by design in this mode
    assert c["context_can_alert"] is False


def test_transformer_caps_are_the_measured_ones():
    assert TRANSFORMER_CAPS == SignalCaps(0.90, 0.90, 0.30)


def test_detection_mode_is_unambiguous():
    h, t = detection_mode("heuristic"), detection_mode("transformer")
    assert h["acoustic_enabled"] is False and h["security_posture"] == "reduced"
    assert t["acoustic_enabled"] is True and t["security_posture"] == "full"
    assert "no defence against a clone" in h["description"]
    assert h["weights"]["acoustic"] == 0.0


def test_weights_are_immutable():
    with pytest.raises(Exception):
        TRANSFORMER_CAPS.acoustic = 0.9


def test_acoustic_alone_can_reach_alert_level():
    """A confidently-synthetic voice is an alert on its own merits —
    in TRANSFORMER mode, where the acoustic signal is validated."""
    score = fuse(1.0, 0.0, 0.0, TRANSFORMER_CAPS)["fused_score"]
    assert score >= ALERT_THRESHOLD, (
        f"acoustic maxed reaches only {score}, below threshold {ALERT_THRESHOLD}"
    )


def test_speaker_mismatch_alone_can_reach_alert_level():
    """'This is definitively not the enrolled person' is an alert too."""
    score = fuse(0.0, 1.0, 0.0)["fused_score"]
    assert score >= ALERT_THRESHOLD, (
        f"speaker maxed reaches only {score}, below threshold {ALERT_THRESHOLD}"
    )


def test_context_alone_cannot_reach_alert_level():
    """Deliberate: metadata is not evidence that a voice was cloned.
    Context escalates a verdict, it does not make the accusation."""
    assert fuse(0.0, 0.0, 1.0)["fused_score"] < ALERT_THRESHOLD


def test_check_alertable_invariant():
    c = check_alertable(ALERT_THRESHOLD, TRANSFORMER_CAPS)
    assert c["acoustic_can_alert"] is True
    assert c["speaker_can_alert"] is True
    assert c["context_can_alert"] is False


def test_every_combination_including_acoustic_or_speaker_can_alert():
    names = ["acoustic", "speaker", "context"]
    for r in (1, 2, 3):
        for combo in itertools.combinations(range(3), r):
            vec = [1.0 if i in combo else 0.0 for i in range(3)]
            score = fuse(*vec)["fused_score"]
            label = "+".join(names[i] for i in combo)
            if combo == (2,):           # context alone
                assert score < ALERT_THRESHOLD, label
            else:
                assert score >= ALERT_THRESHOLD, f"{label} maxed reaches only {score}"


def test_alert_threshold_is_the_high_band_boundary():
    """'alert fired' and 'level is HIGH or worse' must never disagree."""
    high_boundary = dict((name, t) for t, name in LEVELS)["HIGH"]
    assert ALERT_THRESHOLD == high_boundary
    assert level_for(ALERT_THRESHOLD) == "HIGH"


def test_single_maxed_signal_actually_fires_through_the_risk_engine():
    """End-to-end through EMA smoothing and the sustain requirement,
    not just the raw arithmetic."""
    for acoustic, speaker in [(1.0, 0.0), (0.0, 1.0)]:
        engine = RiskEngine()
        fired = False
        for _ in range(6):
            if engine.update(fuse(acoustic, speaker, 0.0)["fused_score"])["alert"]:
                fired = True
        assert fired, f"acoustic={acoustic} speaker={speaker} never raised an alert"


def test_context_alone_never_fires_through_the_risk_engine():
    engine = RiskEngine()
    for _ in range(10):
        assert engine.update(fuse(0.0, 0.0, 1.0)["fused_score"])["alert"] is False


def test_dominant_signal_is_reported_for_explainability():
    assert fuse(0.9, 0.1, 0.1)["dominant_signal"] == "acoustic"
    assert fuse(0.1, 0.9, 0.1)["dominant_signal"] == "speaker"
