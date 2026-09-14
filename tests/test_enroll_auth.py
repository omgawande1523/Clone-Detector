"""Regression tests: enrolment is the root of trust for the speaker
layer, so it must not be silently open.

POST /enroll accepted any speaker_id from anyone who could reach the
port; overwriting the protected person's voiceprint with a clone drives
speaker_mismatch to 0.0 and disables the one layer a clone cannot beat.
Now: if ENROLL_TOKEN is set, /enroll needs a matching X-Enroll-Token
(401 otherwise); if unset, enrolment stays open but /health says so;
and re-enrolling an existing speaker_id appends a hash-chained audit
record with the old and new voiceprint hashes.
"""

import os

import pytest
from fastapi.testclient import TestClient

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")
GENUINE_A = os.path.join(SAMPLES, "genuine_speakerA.wav")

pytestmark = pytest.mark.skipif(not os.path.exists(ENROLL_A), reason="samples not present")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ENROLL_TOKEN", raising=False)
    from src import api
    from src.speaker_consistency import VoiceprintStore
    from src.audit_log import AuditLog
    api._voiceprints = VoiceprintStore(path=str(tmp_path / "voiceprints.json"))
    api._audit = AuditLog(path=str(tmp_path / "audit_log.jsonl"))
    return TestClient(api.app)


def _enrol(client, sid="ceo", path=ENROLL_A, headers=None):
    with open(path, "rb") as f:
        return client.post("/enroll", data={"speaker_id": sid},
                           files={"file": ("e.wav", f, "audio/wav")}, headers=headers or {})


def test_token_set_and_no_header_is_401(client, monkeypatch):
    monkeypatch.setenv("ENROLL_TOKEN", "s3cret")
    r = _enrol(client)
    assert r.status_code == 401
    assert "X-Enroll-Token" in r.json()["detail"]


def test_token_set_and_wrong_header_is_401(client, monkeypatch):
    monkeypatch.setenv("ENROLL_TOKEN", "s3cret")
    assert _enrol(client, headers={"X-Enroll-Token": "nope"}).status_code == 401


def test_token_set_and_correct_header_is_200(client, monkeypatch):
    monkeypatch.setenv("ENROLL_TOKEN", "s3cret")
    r = _enrol(client, headers={"X-Enroll-Token": "s3cret"})
    assert r.status_code == 200
    assert r.json()["enroll_protected"] is True
    assert client.get("/health").json()["enroll_protected"] is True


def test_token_unset_is_open_and_health_says_so(client):
    assert client.get("/health").json()["enroll_protected"] is False
    r = _enrol(client)
    assert r.status_code == 200
    assert r.json()["enroll_protected"] is False
    assert r.json()["overwrote_existing"] is False


def test_reenrolment_is_audited_and_chain_still_verifies(client):
    from src import api
    assert _enrol(client, path=ENROLL_A).status_code == 200
    before = api._audit.verify_chain()["record_count"]
    r = _enrol(client, path=GENUINE_A)                       # different audio, same speaker_id
    assert r.status_code == 200
    assert r.json()["overwrote_existing"] is True
    assert r.json()["audit_logged"] is True

    chain = api._audit.verify_chain()
    assert chain["valid"] is True
    assert chain["record_count"] == before + 1
    import json
    last = [json.loads(l) for l in open(api._audit.path) if l.strip()][-1]["payload"]
    assert last["event"] == "voiceprint_overwritten"
    assert last["speaker_id"] == "ceo"
    assert last["old_voiceprint_sha256"] != last["new_voiceprint_sha256"]
    assert len(last["old_voiceprint_sha256"]) == 64


def test_first_enrolment_is_not_audited(client):
    from src import api
    before = api._audit.verify_chain()["record_count"]
    _enrol(client, sid="fresh")
    assert api._audit.verify_chain()["record_count"] == before
