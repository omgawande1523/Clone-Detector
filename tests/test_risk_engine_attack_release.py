"""Regression tests for the asymmetric EMA.

A single alpha of 0.35 meant a rising score needed six windows to
approach its asymptote. Combined with the weighted average capping a
maxed acoustic signal at 0.448, a confidently-detected clone took
~40 s to alert. Attack 0.60 / release 0.20: react fast to danger,
decay slowly so one quiet window cannot clear a bad call.
"""

import pytest

from src.risk_engine import (RiskEngine, ATTACK_ALPHA, RELEASE_ALPHA,
                             ALERT_THRESHOLD, level_for)


def test_attack_is_faster_than_release():
    assert ATTACK_ALPHA > RELEASE_ALPHA


def test_rising_sequence_crosses_within_two_windows():
    """THE headline: a maxed fused score must alert fast."""
    e = RiskEngine()
    e.update(0.0)                       # a benign first window
    rows = [e.update(0.90) for _ in range(3)]
    assert rows[0]["risk_score"] >= ALERT_THRESHOLD, "first clone window must already cross"
    assert rows[1]["alert"] is True, "sustain must be met by the second clone window"


def test_rising_uses_attack_alpha_and_falling_uses_release():
    e = RiskEngine()
    e.update(0.20)
    up = e.update(0.80)["risk_score"]
    assert up == pytest.approx(ATTACK_ALPHA * 0.80 + (1 - ATTACK_ALPHA) * 0.20, abs=1e-4)
    down = e.update(0.0)["risk_score"]
    assert down == pytest.approx((1 - RELEASE_ALPHA) * up, abs=1e-4)


def test_falling_decays_slowly_so_one_quiet_window_does_not_clear_a_call():
    e = RiskEngine()
    for _ in range(3):
        e.update(0.90)
    before = e.update(0.90)["risk_score"]
    after = e.update(0.0)["risk_score"]
    assert after > ALERT_THRESHOLD, "one silent window must not drop the call below threshold"
    assert after == pytest.approx(before * (1 - RELEASE_ALPHA), abs=1e-4)


def test_single_isolated_spike_still_does_not_alert():
    """The sustain rule is what protects against a one-window spike,
    and a fast attack must not weaken it."""
    e = RiskEngine()
    rows = [e.update(s) for s in (0.0, 0.0, 0.95, 0.0, 0.0, 0.0)]
    assert not any(r["alert"] for r in rows)
    s = e.summary()
    assert s["alert_triggered"] is False
    assert s["peak_level"] in ("HIGH", "CRITICAL")       # the peak is still reported
    assert s["peak_sustained_level"] == "LOW"


def test_three_way_verdict_still_reads_correctly_with_slow_release():
    """A slow release keeps the LEVEL elevated for several windows after
    a spike without alert_triggered becoming true -- exactly the case
    the sustained_level split exists for."""
    from src.api import _verdict_state
    e = RiskEngine()
    for s in (0.0, 0.95, 0.0, 0.0, 0.0):
        e.update(s)
    summary = e.summary()
    assert summary["alert_triggered"] is False
    assert _verdict_state(summary) == "peaked_unsustained"
    assert summary["peak_sustained_level"] == "LOW"


def test_peak_reporting_is_unchanged():
    e = RiskEngine()
    for s in (0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0):
        e.update(s)
    s = e.summary()
    assert s["alert_triggered"] is True
    assert s["peak_risk_score"] == pytest.approx(0.9, abs=1e-6)
    assert s["peak_level"] == level_for(0.9)
    assert s["final_ema"] < s["peak_risk_score"]
