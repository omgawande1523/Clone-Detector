"""Tests for the context engine's keyword matching and the risk
engine's peak-vs-final reporting."""

import pytest

from src.context_engine import score_context
from src.risk_engine import RiskEngine, level_for, ALERT_THRESHOLD


# --------------------------------------------------------------- context

def test_substring_false_positives_are_gone():
    """'pin' used to fire on 'shipping' and 'wire' on 'wireless'."""
    result = score_context(transcript="The shipping update came over the wireless hotspot.")
    assert result["context_score"] == 0.0
    assert result["reasons"] == []


def test_real_keywords_still_match():
    result = score_context(transcript="I need an urgent wire transfer right now")
    assert result["context_score"] > 0.0
    assert "urgent" in result["reasons"][0]


@pytest.mark.parametrize("innocent", [
    "shipping", "wireless", "hotspot", "opinion", "transferable notes",
    "passwordless login", "spinning", "urgently"])
def test_innocent_words_do_not_trigger(innocent):
    assert score_context(transcript=f"we discussed {innocent} today")["context_score"] == 0.0


@pytest.mark.parametrize("guilty", [
    "pin", "wire", "otp", "transfer", "urgent", "one time password",
    "gift card", "account number"])
def test_genuine_risk_phrases_do_trigger(guilty):
    assert score_context(transcript=f"please send the {guilty} now")["context_score"] > 0.0


def test_keyword_matching_is_case_insensitive():
    assert score_context(transcript="URGENT WIRE TRANSFER")["context_score"] > 0.0


def test_multiword_phrase_matches_across_whitespace():
    assert score_context(transcript="read me the one   time\npassword")["context_score"] > 0.0


def test_flags_accumulate_and_are_explained():
    result = score_context(caller_known=False, off_hours=True, first_contact=True,
                           transcript="urgent transfer")
    assert result["context_score"] > 0.5
    assert len(result["reasons"]) == 4


def test_context_score_is_bounded():
    result = score_context(caller_known=False, off_hours=True, first_contact=True,
                           transcript=" ".join(["urgent", "wire", "transfer", "otp",
                                                "pin", "password", "gift card"]))
    assert result["context_score"] <= 1.0


def test_missing_metadata_degrades_to_neutral():
    assert score_context()["context_score"] == 0.0


# ------------------------------------------------------------ risk engine

def test_peak_is_reported_not_the_trailing_ema():
    """A call cloned early and clean later is still a cloned call. The
    old code reported the last chunk's EMA, which could decay back to
    LOW while alert_triggered stayed true."""
    engine = RiskEngine()
    for score in [0.95] * 5 + [0.0] * 15:      # spike, then long clean tail
        engine.update(score)
    summary = engine.summary()

    assert summary["alert_triggered"] is True
    assert summary["peak_level"] in ("HIGH", "CRITICAL")
    assert summary["peak_risk_score"] >= ALERT_THRESHOLD
    # The bug: the trailing EMA has decayed below the alert threshold.
    assert summary["final_ema"] < ALERT_THRESHOLD
    assert summary["peak_risk_score"] > summary["final_ema"]


def test_alert_and_level_never_disagree():
    """If an alert fired, the reported level must be HIGH or worse."""
    engine = RiskEngine()
    for score in [0.9] * 4 + [0.0] * 20:
        engine.update(score)
    summary = engine.summary()
    assert summary["alert_triggered"] is True
    assert level_for(summary["peak_risk_score"]) in ("HIGH", "CRITICAL")


def test_score_at_alert_is_captured():
    engine = RiskEngine()
    for score in [0.9] * 5 + [0.0] * 10:
        engine.update(score)
    assert engine.summary()["score_at_alert"] >= ALERT_THRESHOLD


def test_single_spike_does_not_alert():
    """The sustain requirement should absorb one noisy chunk."""
    engine = RiskEngine()
    alerts = [engine.update(s)["alert"] for s in [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    assert not any(alerts)


def test_clean_call_reports_no_alert():
    engine = RiskEngine()
    for _ in range(10):
        engine.update(0.05)
    summary = engine.summary()
    assert summary["alert_triggered"] is False
    assert summary["peak_level"] == "LOW"
    assert summary["score_at_alert"] is None
