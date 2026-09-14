"""Tests for the /ws/live streaming path.

The property under test is *streaming*: each chunk's result must come
back before the next chunk is sent, through ONE RiskEngine so EMA and
sustain carry across chunks, and an alert must be audited the moment
it fires — not at session end. scripts/live_latency_probe.py does the
same against a real server with a real 5 s cadence; this runs the same
code in-process so it is part of the suite.
"""

import io
import json
import os
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")
GENUINE_A = os.path.join(SAMPLES, "genuine_speakerA.wav")
IMPOSTOR_B = os.path.join(SAMPLES, "impostor_speakerB.wav")
SR = 16000

needs_samples = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (ENROLL_A, GENUINE_A, IMPOSTOR_B)),
    reason="speaker samples not present")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("livestate")
    from src import api
    from src.speaker_consistency import VoiceprintStore
    from src.audit_log import AuditLog
    api._voiceprints = VoiceprintStore(path=str(tmp / "voiceprints.json"))
    api._audit = AuditLog(path=str(tmp / "audit_log.jsonl"))
    c = TestClient(api.app)
    with open(ENROLL_A, "rb") as f:
        assert c.post("/enroll", data={"speaker_id": "ceo"}, files={"file": ("e.wav", f, "audio/wav")}).status_code == 200
    return c


def _wav(x, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chunks(path, seconds=5.0):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    n = int(seconds * sr)
    return [x[i:i + n] for i in range(0, len(x), n) if len(x[i:i + n]) >= sr * 1.0]


@needs_samples
def test_each_chunk_is_answered_before_the_next_is_sent(client):
    chunks = _chunks(GENUINE_A)
    assert len(chunks) >= 2
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"speaker_id": "ceo"})
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["speaker_enrolled"] is True

        gaps = []
        for i, chunk in enumerate(chunks):
            t0 = time.time()
            ws.send_bytes(_wav(chunk))
            reply = ws.receive_json()          # must be THIS chunk's result, now
            gaps.append(time.time() - t0)
            assert reply["type"] == "chunk"
            assert reply["index"] == i
            assert reply["compute_ms"] < 5000
            assert reply["server_sent"] >= reply["server_recv"]

        ws.send_json({"type": "stop"})
        summary = ws.receive_json()
        assert summary["type"] == "summary"
        assert len(summary["timeline"]) == len(chunks)
    print(f"\n  in-process per-chunk gaps: {[round(g, 3) for g in gaps]}")


@needs_samples
def test_one_risk_engine_carries_across_chunks(client):
    """The EMA must carry state: the second chunk's risk_score should be
    a blend, not a fresh engine's first-value initialisation."""
    chunks = _chunks(IMPOSTOR_B)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"speaker_id": "ceo"})
        ws.receive_json()
        rows = []
        for chunk in chunks:
            ws.send_bytes(_wav(chunk))
            rows.append(ws.receive_json())
        ws.send_json({"type": "stop"})
        summary = ws.receive_json()
    assert rows[0]["start_time"] == 0.0
    assert rows[1]["start_time"] == pytest.approx(rows[0]["duration"], abs=0.05)
    # fresh engine would set risk == instant on chunk 1; a carried engine blends
    assert rows[1]["risk_score"] == pytest.approx(
        0.35 * rows[1]["instant_score"] + 0.65 * rows[0]["risk_score"], abs=1e-3)
    assert summary["final_risk_score"] == max(r["risk_score"] for r in rows)


@needs_samples
def test_impostor_alerts_live_and_is_audited_immediately(client):
    from src import api
    before = api._audit.verify_chain()["record_count"]
    chunks = _chunks(IMPOSTOR_B)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"speaker_id": "ceo", "caller_known": False, "transcript": "urgent wire transfer"})
        ws.receive_json()
        first_alert_index, audited_at = None, None
        for i, chunk in enumerate(chunks):
            ws.send_bytes(_wav(chunk))
            r = ws.receive_json()
            if r["alert"] and first_alert_index is None:
                first_alert_index = i
                # audit record must already exist at this instant
                audited_at = api._audit.verify_chain()["record_count"]
                assert r["audit_logged"] is True
        ws.send_json({"type": "stop"})
        summary = ws.receive_json()
    assert first_alert_index is not None, "impostor never alerted"
    assert audited_at == before + 1
    assert summary["alert_triggered"] is True
    assert api._audit.verify_chain()["record_count"] == before + 1   # once, not again at stop


@needs_samples
def test_live_and_batch_paths_agree(client):
    """Same audio through /analyze and /ws/live must give the same
    per-chunk scores — they share _score_chunk."""
    with open(GENUINE_A, "rb") as f:
        batch = client.post("/analyze", data={"speaker_id": "ceo"}, files={"file": ("g.wav", f, "audio/wav")}).json()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"speaker_id": "ceo"})
        ws.receive_json()
        live = []
        for chunk in _chunks(GENUINE_A):
            ws.send_bytes(_wav(chunk))
            live.append(ws.receive_json())
        ws.send_json({"type": "stop"})
        ws.receive_json()
    for b, l in zip(batch["timeline"], live):
        assert l["acoustic_score"] == pytest.approx(b["acoustic_score"], abs=0.02)
        assert l["speaker_mismatch"] == pytest.approx(b["speaker_mismatch"], abs=0.02)
        assert l["risk_score"] == pytest.approx(b["risk_score"], abs=0.02)


@needs_samples
def test_raw_pcm_and_wav_are_both_accepted(client):
    x = _chunks(GENUINE_A)[0]
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"speaker_id": "ceo"})
        ws.receive_json()
        ws.send_bytes(_wav(x)); a = ws.receive_json()
        ws.send_bytes(pcm);     b = ws.receive_json()
        ws.send_json({"type": "stop"}); ws.receive_json()
    assert a["type"] == b["type"] == "chunk"
    assert b["acoustic_score"] == pytest.approx(a["acoustic_score"], abs=0.02)


def test_bad_chunk_is_reported_not_fatal(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({})
        ready = ws.receive_json()
        assert ready["speaker_enrolled"] is False
        ws.send_bytes(b"RIFF" + b"junk" * 8)
        r = ws.receive_json()
        assert r["type"] == "error"
        ws.send_bytes(b"\x00\x00" * 100)            # 100 samples: too short
        r = ws.receive_json()
        assert r["type"] == "error" and "short" in r["detail"]
        ws.send_json({"type": "stop"})
        s = ws.receive_json()
        assert s["type"] == "summary" and s["timeline"] == []
