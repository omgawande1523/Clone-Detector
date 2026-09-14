"""Regression tests: the system must never be SILENTLY incapable of
alerting.

In heuristic mode the weights are acoustic 0.00 / speaker 0.90 /
context 0.10. With no voiceprint enrolled, mismatch_score() returns a
neutral 0.0, so the highest fused score any window can reach is 0.10 —
against an alert threshold of 0.40. A fresh clone-and-run with no
enrolment could never fire, and nothing said so. check_alertable()
did not catch it because it reports weight capacity, not runtime
state. reachable_max() reports runtime state; /health, /analyze and
the live ready message surface it.
"""

import os

import pytest
from fastapi.testclient import TestClient

from src.fusion import (reachable_max, capability_warning,
                        HEURISTIC_WEIGHTS, TRANSFORMER_WEIGHTS)
from src.risk_engine import ALERT_THRESHOLD

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
GENUINE_A = os.path.join(SAMPLES, "genuine_speakerA.wav")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")


# --------------------------------------------------------------- unit

def test_heuristic_without_enrolment_can_never_alert():
    """THE dead configuration."""
    cap = reachable_max(HEURISTIC_WEIGHTS, speaker_enrolled=False)
    assert cap["can_ever_alert"] is False
    assert cap["max_fused"] == pytest.approx(0.10)
    assert cap["max_fused"] < ALERT_THRESHOLD


def test_heuristic_with_enrolment_can_alert():
    cap = reachable_max(HEURISTIC_WEIGHTS, speaker_enrolled=True)
    assert cap["can_ever_alert"] is True
    assert cap["max_fused"] == pytest.approx(1.0)


def test_transformer_without_enrolment_can_alert():
    cap = reachable_max(TRANSFORMER_WEIGHTS, speaker_enrolled=False)
    assert cap["can_ever_alert"] is True
    assert cap["max_fused"] == pytest.approx(0.55)


def test_warning_text_names_the_fix():
    w = capability_warning("heuristic", HEURISTIC_WEIGHTS, speaker_enrolled=False)
    assert w is not None
    assert "CANNOT raise an alert" in w
    assert "enrol" in w.lower()
    assert "DETECTOR_BACKEND=transformer" in w
    assert capability_warning("heuristic", HEURISTIC_WEIGHTS, speaker_enrolled=True) is None
    assert capability_warning("transformer", TRANSFORMER_WEIGHTS, speaker_enrolled=False) is None


# --------------------------------------------------------------- API

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("capstate")
    from src import api
    from src.speaker_consistency import VoiceprintStore
    from src.audit_log import AuditLog
    api._voiceprints = VoiceprintStore(path=str(tmp / "voiceprints.json"))
    api._audit = AuditLog(path=str(tmp / "audit_log.jsonl"))
    return TestClient(api.app)


def test_health_reports_live_capability_for_both_cases(client):
    d = client.get("/health").json()
    cap = d["live_capability"]
    assert set(cap) >= {"enrolled", "unenrolled", "enrolled_speakers"}
    assert cap["enrolled"]["can_ever_alert"] is True
    # what the unenrolled case says must match the loaded weights
    w = d["detection_mode"]["weights"]
    expected = (w["acoustic"] + w["context"]) >= d["alert_threshold"]
    assert cap["unenrolled"]["can_ever_alert"] is expected


@pytest.mark.skipif(not os.path.exists(GENUINE_A), reason="samples not present")
def test_analyze_in_dead_configuration_returns_the_warning(client):
    """End to end: default (heuristic) backend, no speaker enrolled."""
    from src import api
    if api._detector.backend != "heuristic":
        pytest.skip("dead configuration only exists in heuristic mode")
    with open(GENUINE_A, "rb") as f:
        d = client.post("/analyze", data={"speaker_id": "nobody"}, files={"file": ("g.wav", f, "audio/wav")}).json()
    assert d["speaker_enrolled"] is False
    assert d["can_ever_alert"] is False
    assert d["max_reachable_score"] == pytest.approx(0.10)
    assert "CANNOT raise an alert" in d["capability_warning"]


@pytest.mark.skipif(not os.path.exists(ENROLL_A), reason="samples not present")
def test_analyze_with_enrolment_has_no_warning(client):
    with open(ENROLL_A, "rb") as f:
        assert client.post("/enroll", data={"speaker_id": "ceo"}, files={"file": ("e.wav", f, "audio/wav")}).status_code == 200
    with open(GENUINE_A, "rb") as f:
        d = client.post("/analyze", data={"speaker_id": "ceo"}, files={"file": ("g.wav", f, "audio/wav")}).json()
    assert d["can_ever_alert"] is True
    assert d["capability_warning"] is None


def test_live_ready_message_carries_capability(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"speaker_id": "nobody_enrolled"})
        ready = ws.receive_json()
        ws.send_json({"type": "stop"}); ws.receive_json()
    assert "can_ever_alert" in ready
    assert "capability_warning" in ready
    from src import api
    if api._detector.backend == "heuristic":
        assert ready["can_ever_alert"] is False
        assert ready["capability_warning"]
