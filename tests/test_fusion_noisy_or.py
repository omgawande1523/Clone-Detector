"""Regression tests for noisy-OR fusion.

A weighted mean structurally caps a single signal at its own weight.
With acoustic weighted 0.45 a 100%-confident synthetic detection
produced a fused score of 0.448 against a 0.40 threshold, and on a
captured live call (fixtures/live_clone_session/) four consecutive
windows at 0.984-0.996 never reached HIGH. Noisy-OR lets one confident
signal speak on its own, which is required for the targeted-clone
threat where the speaker layer is silent by design.
"""

import itertools

import pytest

from src.fusion import (fuse, SignalCaps, caps_for_backend, check_alertable,
                        TRANSFORMER_CAPS, HEURISTIC_CAPS)
from src.risk_engine import ALERT_THRESHOLD
from src.detector import HeuristicDetector, TransformerDetector


def F(a, s, c, caps=TRANSFORMER_CAPS):
    return fuse(a, s, c, caps)["fused_score"]


# --------------------------------------------------------- invariants

def test_acoustic_alone_maxed_can_alert():
    """THE case the weighted average could not express."""
    assert F(1.0, 0, 0) == pytest.approx(TRANSFORMER_CAPS.acoustic)
    assert F(1.0, 0, 0) >= ALERT_THRESHOLD


def test_speaker_alone_maxed_can_alert():
    assert F(0, 1.0, 0) == pytest.approx(TRANSFORMER_CAPS.speaker)
    assert F(0, 1.0, 0) >= ALERT_THRESHOLD


def test_context_alone_maxed_can_never_alert():
    """Metadata is not evidence that a voice was cloned."""
    assert F(0, 0, 1.0) == pytest.approx(TRANSFORMER_CAPS.context)
    assert F(0, 0, 1.0) < ALERT_THRESHOLD


def test_heuristic_backend_ignores_acoustic_entirely():
    assert HEURISTIC_CAPS.acoustic == 0.0
    assert F(1.0, 0, 0, HEURISTIC_CAPS) == 0.0
    assert fuse(1.0, 0, 0, HEURISTIC_CAPS)["contributions"]["acoustic"] == 0.0
    # an acoustic score cannot change a heuristic-mode verdict at all
    assert F(0.0, 0.5, 0.2, HEURISTIC_CAPS) == pytest.approx(F(1.0, 0.5, 0.2, HEURISTIC_CAPS))


def test_check_alertable_matches_the_caps():
    for caps in (TRANSFORMER_CAPS, HEURISTIC_CAPS):
        c = check_alertable(ALERT_THRESHOLD, caps)
        assert c["acoustic_alone_max"] == caps.acoustic
        assert c["speaker_alone_max"] == caps.speaker
        assert c["context_alone_max"] == caps.context
        assert c["speaker_can_alert"] is True
        assert c["context_can_alert"] is False
    assert check_alertable(ALERT_THRESHOLD, TRANSFORMER_CAPS)["acoustic_can_alert"] is True
    assert check_alertable(ALERT_THRESHOLD, HEURISTIC_CAPS)["acoustic_can_alert"] is False


def test_caps_follow_the_detector_not_a_flag():
    assert caps_for_backend(HeuristicDetector.backend) is HEURISTIC_CAPS
    assert caps_for_backend(TransformerDetector.backend) is TRANSFORMER_CAPS
    with pytest.raises(ValueError):
        caps_for_backend("some-new-model")


# ------------------------------------------------- noisy-OR properties

def test_a_single_maxed_signal_reaches_exactly_its_cap():
    for i, name in enumerate(("acoustic", "speaker", "context")):
        vec = [0.0, 0.0, 0.0]
        vec[i] = 1.0
        assert F(*vec) == pytest.approx(getattr(TRANSFORMER_CAPS, name))


def test_signals_compound_rather_than_average():
    """Two half-confident signals must beat either alone -- the property
    a mean does not have."""
    a, s = F(0.5, 0, 0), F(0, 0.5, 0)
    both = F(0.5, 0.5, 0)
    assert both > a and both > s
    assert both == pytest.approx(1 - (1 - 0.45) * (1 - 0.45))


def test_all_maxed_is_below_one_and_above_every_pair():
    allm = F(1, 1, 1)
    assert allm < 1.0
    assert allm == pytest.approx(1 - 0.10 * 0.10 * 0.70)
    for vec in ([1, 1, 0], [1, 0, 1], [0, 1, 1]):
        assert allm >= F(*vec)


def test_zero_everything_is_zero():
    assert fuse(0, 0, 0)["fused_score"] == 0.0
    assert fuse(0, 0, 0, HEURISTIC_CAPS)["fused_score"] == 0.0


def test_monotonic_in_every_input():
    """Raising any one input can never lower the fused score."""
    grid = [0.0, 0.2, 0.5, 0.8, 1.0]
    for a, s, c in itertools.product(grid, repeat=3):
        base = F(a, s, c)
        for i, v in enumerate((a, s, c)):
            if v >= 1.0:
                continue
            vec = [a, s, c]
            vec[i] = min(1.0, v + 0.2)
            assert F(*vec) >= base - 1e-12, (a, s, c, i)


def test_output_is_bounded_and_inputs_are_clamped():
    assert 0.0 <= F(1, 1, 1) <= 1.0
    assert F(5.0, -2.0, 0.0) == pytest.approx(F(1.0, 0.0, 0.0))


def test_dominant_signal_is_the_largest_contribution():
    assert fuse(0.9, 0.1, 0.1)["dominant_signal"] == "acoustic"
    assert fuse(0.1, 0.9, 0.1)["dominant_signal"] == "speaker"
    # context has the smallest cap, so it needs a big lead to dominate
    assert fuse(0.05, 0.05, 1.0)["dominant_signal"] == "context"
    # in heuristic mode acoustic can never dominate
    assert fuse(1.0, 0.5, 0.0, HEURISTIC_CAPS)["dominant_signal"] == "speaker"


def test_the_captured_failure_now_clears_the_threshold():
    """fixtures/live_clone_session/: acoustic 0.996, speaker 0.00,
    context 0.00 produced 0.448 under the weighted average."""
    assert F(0.996, 0.0, 0.0) == pytest.approx(0.90 * 0.996)
    assert F(0.996, 0.0, 0.0) >= ALERT_THRESHOLD


def test_genuine_speech_stays_far_below_threshold():
    """Measured worst-case acoustic on genuine fixtures is 0.007."""
    assert F(0.007, 0.0, 0.0) < 0.01
    assert F(0.007, 0.0, 0.0) < ALERT_THRESHOLD
