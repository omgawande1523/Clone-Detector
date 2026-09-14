"""Regression tests: final_level and alert_triggered must not read as a
contradiction.

peak_level is derived from the peak EMA with no sustain requirement;
alert_triggered needs sustain_required consecutive windows above
threshold. A single isolated spike therefore gave final_level "HIGH"
next to "No strong impersonation signal detected in this call." in the
same payload. The peak is still reported (that reasoning is right —
see risk_engine.py), but the payload now also carries sustained_level,
peak_windows_above and a three-way verdict_state, and the wording for
the peaked-but-not-sustained case names the peak instead of denying
it.
"""

import pytest

from src.risk_engine import RiskEngine, ALERT_THRESHOLD, level_for
from src.api import _verdict_payload, _verdict_state, _recommendation


def _run(scores):
    e = RiskEngine()
    for s in scores:
        e.update(s)
    return e.summary()


def test_single_isolated_spike_is_peaked_but_not_sustained():
    """THE regression case. The EMA initialises to the first window, so
    a first window at 0.50 is HIGH at once; the next window pulls the
    EMA to 0.35*0.05 + 0.65*0.50 = 0.34, below threshold, so the sustain
    rule never holds. (A first window at 0.95 would decay only to 0.64
    and legitimately sustain — the EMA carrying a strong spike across
    two windows is the alert working, not this bug.)"""
    summary = _run([0.50, 0.05, 0.05, 0.05, 0.05])
    assert summary["alert_triggered"] is False
    assert summary["peak_level"] == "HIGH"
    assert summary["peak_sustained_level"] == "LOW"
    assert summary["peak_windows_above"] == 1

    payload = _verdict_payload("x", True, {"reasons": []}, [], summary)
    assert payload["final_level"] == "HIGH"
    assert payload["sustained_level"] == "LOW"
    assert payload["verdict_state"] == "peaked_unsustained"
    assert "No strong impersonation signal" not in payload["recommendation"]
    assert "briefly crossed" in payload["recommendation"]
    assert f"{summary['peak_risk_score']:.2f}" in payload["recommendation"]


def test_sustained_spike_is_an_alert_and_levels_agree():
    summary = _run([0.05, 0.9, 0.9, 0.9, 0.05])
    assert summary["alert_triggered"] is True
    assert summary["peak_sustained_level"] in ("HIGH", "CRITICAL")
    assert summary["peak_windows_above"] >= 2
    assert level_for(summary["peak_risk_score"]) == summary["peak_level"]
    payload = _verdict_payload("x", True, {"reasons": []}, [], summary)
    assert payload["verdict_state"] == "sustained_alert"
    assert "secondary verification" in payload["recommendation"]


def test_mid_call_spike_from_elevated_baseline_is_peaked_but_not_sustained():
    """Same thing mid-call: EMA 0.30 -> 0.35*1.0 + 0.65*0.30 = 0.545 (HIGH)
    for one window, then back down. A spike from a LOW baseline cannot
    reach HIGH at all through the EMA (max 0.35), which is the smoothing
    doing its job."""
    summary = _run([0.30, 0.30, 1.0, 0.10, 0.10])
    assert summary["alert_triggered"] is False
    assert summary["peak_level"] == "HIGH"
    assert summary["peak_sustained_level"] == "LOW"
    assert summary["peak_windows_above"] == 1
    assert _verdict_state(summary) == "peaked_unsustained"


def test_never_crossed_is_clear():
    summary = _run([0.05, 0.1, 0.15, 0.1])
    assert summary["peak_level"] in ("LOW", "MEDIUM")
    assert summary["peak_sustained_level"] == "LOW"
    assert summary["peak_windows_above"] == 0
    payload = _verdict_payload("x", True, {"reasons": []}, [], summary)
    assert payload["verdict_state"] == "clear"
    assert "No strong impersonation signal" in payload["recommendation"]


def test_sustained_level_never_exceeds_peak_level():
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    for seq in ([1.0] * 6, [0.6, 0.6, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], [0.45] * 3 + [1.0]):
        s = _run(seq)
        assert order[s["peak_sustained_level"]] <= order[s["peak_level"]]


def test_sustained_level_only_counts_windows_while_alert_held():
    """The peak EMA can be reached on the first window above threshold
    (before sustain); sustained_level must reflect the EMA once the
    rule actually held, not that earlier peak."""
    s = _run([1.0, 0.45, 0.45])      # EMA: 1.0, 0.81, 0.68 -> alert holds from window 2
    assert s["alert_triggered"] is True
    assert s["peak_level"] == "CRITICAL"
    assert s["peak_sustained_level"] == level_for(0.35 * 0.45 + 0.65 * 1.0)


def test_verdict_state_and_recommendation_are_three_way():
    for state, seq in (("sustained_alert", [0.9, 0.9, 0.9]),
                       ("peaked_unsustained", [0.5, 0.0, 0.0]),
                       ("clear", [0.0, 0.0])):
        s = _run(seq)
        assert _verdict_state(s) == state
        assert _recommendation(s)
