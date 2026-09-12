"""End-to-end tests for the HTTP layer, including the error paths a
judge is most likely to hit by accident during a live demo.

test_corrupt_wav_returns_400 is a regression test: chunk_audio()
decodes lazily, so a corrupt file raised inside the analysis loop
rather than at upload time and surfaced as an HTTP 500.
"""

import io
import os
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")
GENUINE_A = os.path.join(SAMPLES, "genuine_speakerA.wav")
IMPOSTOR_B = os.path.join(SAMPLES, "impostor_speakerB.wav")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Point the API's state files at a temp dir so tests never touch
    the real voiceprints.json / audit_log.jsonl."""
    tmp = tmp_path_factory.mktemp("apistate")
    from src import api
    from src.speaker_consistency import VoiceprintStore
    from src.audit_log import AuditLog
    api._voiceprints = VoiceprintStore(path=str(tmp / "voiceprints.json"))
    api._audit = AuditLog(path=str(tmp / "audit_log.jsonl"))
    return TestClient(api.app)


def _wav_bytes(seconds=3.0, sr=16000, freq=180.0):
    buf = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    data = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return buf.getvalue()


def _upload(path_or_bytes, name="call.wav"):
    if isinstance(path_or_bytes, bytes):
        return {"file": (name, path_or_bytes, "audio/wav")}
    return {"file": (os.path.basename(path_or_bytes), open(path_or_bytes, "rb"), "audio/wav")}


needs_samples = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (ENROLL_A, GENUINE_A, IMPOSTOR_B)),
    reason="speaker samples not present in samples/",
)


# ------------------------------------------------------------- health

def test_health_reports_the_fusion_invariant(client):
    d = client.get("/health").json()
    assert d["status"] == "ok"
    assert d["fusion_check"]["acoustic_can_alert"] is True
    assert d["fusion_check"]["speaker_can_alert"] is True
    assert d["fusion_check"]["context_can_alert"] is False


# -------------------------------------------------------- happy paths

@needs_samples
def test_enroll_then_genuine_call_is_not_flagged(client):
    r = client.post("/enroll", data={"speaker_id": "ceo"}, files=_upload(ENROLL_A))
    assert r.status_code == 200, r.text
    assert r.json()["speaker_id"] == "ceo"

    r = client.post("/analyze", data={"speaker_id": "ceo"}, files=_upload(GENUINE_A))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["speaker_enrolled"] is True
    assert d["alert_triggered"] is False
    assert d["final_level"] == "LOW"
    assert max(t["speaker_mismatch"] for t in d["timeline"]) < 0.35


@needs_samples
def test_impostor_call_is_flagged_and_audited(client):
    client.post("/enroll", data={"speaker_id": "ceo"}, files=_upload(ENROLL_A))
    r = client.post("/analyze",
                    data={"speaker_id": "ceo", "caller_known": "false",
                          "off_hours": "true", "first_contact": "true",
                          "transcript": "urgent wire transfer, read me the one time password"},
                    files=_upload(IMPOSTOR_B))
    d = r.json()
    assert d["alert_triggered"] is True
    assert d["final_level"] in ("HIGH", "CRITICAL")
    assert d["audit_logged"] is True
    assert d["score_at_alert"] is not None
    assert max(t["speaker_mismatch"] for t in d["timeline"]) > 0.65
    assert client.get("/audit/verify").json()["valid"] is True


@needs_samples
def test_first_contact_flag_reaches_the_context_engine(client):
    r = client.post("/analyze", data={"first_contact": "true"}, files=_upload(GENUINE_A))
    reasons = r.json()["context_reasons"]
    assert any("first recorded contact" in x for x in reasons)


@needs_samples
def test_unenrolled_speaker_is_reported_as_not_checked(client):
    r = client.post("/analyze", data={"speaker_id": "never_enrolled"}, files=_upload(GENUINE_A))
    d = r.json()
    assert d["speaker_enrolled"] is False
    assert all(t["speaker_mismatch"] == 0.0 for t in d["timeline"])


@needs_samples
def test_peak_not_trailing_ema_is_reported(client):
    """final_risk_score must be the peak, so it can never sit below the
    alert threshold while alert_triggered is true."""
    client.post("/enroll", data={"speaker_id": "ceo"}, files=_upload(ENROLL_A))
    r = client.post("/analyze", data={"speaker_id": "ceo"}, files=_upload(IMPOSTOR_B))
    d = r.json()
    assert d["final_risk_score"] >= d["final_ema"]
    if d["alert_triggered"]:
        assert d["final_risk_score"] >= 0.40
        assert d["final_level"] in ("HIGH", "CRITICAL")


# ------------------------------------------------------- error paths

def test_non_audio_extension_returns_400(client):
    r = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "Unsupported audio format" in r.json()["detail"]


def test_corrupt_wav_returns_400_not_500(client):
    """THE regression test: this used to raise LibsndfileError inside
    the analysis loop and surface as an HTTP 500."""
    r = client.post("/analyze", files={"file": ("broken.wav", b"RIFFxxxxWAVEjunk", "audio/wav")})
    assert r.status_code == 400, f"expected 400, got {r.status_code}"
    assert "decode" in r.json()["detail"].lower()


def test_empty_file_returns_400(client):
    r = client.post("/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_oversize_upload_returns_413(client):
    from src.api import MAX_ANALYZE_BYTES
    blob = b"RIFF" + b"\0" * (MAX_ANALYZE_BYTES + 1024)
    r = client.post("/analyze", files={"file": ("huge.wav", blob, "audio/wav")})
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


def test_enroll_rejects_audio_with_too_little_speech(client):
    r = client.post("/enroll", data={"speaker_id": "shorty"},
                    files={"file": ("tiny.wav", _wav_bytes(seconds=0.4), "audio/wav")})
    assert r.status_code == 400
    assert "too little speech" in r.json()["detail"]


def test_enroll_rejects_blank_speaker_id(client):
    r = client.post("/enroll", data={"speaker_id": "   "},
                    files={"file": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 400


def test_silent_file_reports_no_analysable_audio(client):
    """A file of pure digital silence has no chunks worth scoring."""
    sr = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.zeros(int(0.1 * sr), dtype=np.int16).tobytes())
    r = client.post("/analyze", files={"file": ("silence.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 400
    assert "no analysable audio" in r.json()["detail"]
