"""Regression tests for the fusion-weight / alert-threshold coupling.

These would have caught the bug where weights of 0.55/0.30/0.15 sat
under an alert threshold of 0.60, making it arithmetically impossible
for any single signal to raise an alert — the system advertised three
independent signals but in practice required several at once.
"""

import itertools

from src.fusion import fuse, FusionWeights, check_alertable
from src.risk_engine import ALERT_THRESHOLD, LEVELS, level_for, RiskEngine


def test_weights_form_a_weighted_average():
    w = FusionWeights()
    assert abs((w.acoustic + w.speaker + w.context) - 1.0) < 1e-9


def test_acoustic_alone_can_reach_alert_level():
    """A confidently-synthetic voice is an alert on its own merits."""
    score = fuse(1.0, 0.0, 0.0)["fused_score"]
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
    c = check_alertable(ALERT_THRESHOLD)
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
