"""
api.py — HTTP layer.

Run with:
    uvicorn src.api:app --reload --port 8000

Then open web/index.html in a browser (it calls http://localhost:8000).

POST /enroll
    multipart/form-data: speaker_id (str) + file (a clean reference
    .wav of that person's voice, 10-20 seconds). Builds the voiceprint
    used later for cross-checking.

POST /analyze
    multipart/form-data: file (the call recording) + optional
    speaker_id, caller_known, off_hours, first_contact, transcript.
    Returns a per-chunk timeline where each entry is a FUSED score
    (acoustic + speaker-consistency + context), plus the call verdict.
    Any triggered alert is appended to the tamper-evident audit log.

WS /ws/live
    True real-time path. Client sends one JSON "start" message (same
    fields as /analyze's form), then binary audio chunks (WAV, or raw
    int16 PCM @ 16 kHz mono) as they are captured. Each chunk is scored
    the moment it arrives — the same detector -> speaker-consistency ->
    context -> fusion -> risk-engine code as /analyze, with ONE
    RiskEngine for the whole call so EMA/sustain carry across chunks —
    and the result is sent back immediately. A JSON {"type": "stop"}
    ends the call and returns the same verdict block /analyze returns.

GET /audit/verify
    Confirms the audit log's hash chain has not been tampered with —
    a live demo of the "Blockchain & Cybersecurity" compliance angle.

GET /health
    Backend in use, plus the fusion/threshold invariant check, so a
    misconfiguration is visible before a demo rather than during one.

Uploads are streamed to a temp file with a hard size cap rather than
being read into memory whole, so a large or hostile upload cannot
exhaust RAM.
"""

import io
import json
import os
import tempfile
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .audio_pipeline import chunk_audio, load_mono, DEFAULT_CHUNK_SECONDS
from .risk_engine import RiskEngine, ALERT_THRESHOLD
from .detector import get_detector
from .speaker_consistency import VoiceprintStore
from .context_engine import score_context
from .fusion import fuse, check_alertable, weights_for_backend, detection_mode
from .audit_log import AuditLog

# Upload caps. 16 kHz mono 16-bit PCM is ~32 KB/s, so 50 MB is roughly
# 26 minutes of call audio — generous for a real call, bounded enough
# that a hostile upload cannot fill the disk or exhaust memory.
MAX_ANALYZE_BYTES = 50 * 1024 * 1024
MAX_ENROLL_BYTES = 25 * 1024 * 1024
_READ_CHUNK = 1024 * 1024

# Only extensions we can actually decode. Also stops a crafted filename
# from steering the temp-file suffix somewhere unexpected.
ALLOWED_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".aiff", ".aif"}

app = FastAPI(title="SIH26104 Voice-Clone Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo only — lock this down for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector = get_detector()          # loaded once at startup
# Fusion weights follow the loaded backend automatically. There is no
# flag for this: the heuristic backend's acoustic weight is zero because
# it measured at/below chance on labelled data (see fusion.py).
_weights = weights_for_backend(_detector.backend)
_mode = detection_mode(_detector.backend)
_voiceprints = VoiceprintStore()    # path anchored to project root
_audit = AuditLog()                 # path anchored to project root


def _safe_suffix(filename: str) -> str:
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix or 'unknown'}'. "
                   f"Supported: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )
    return suffix


async def _save_upload(file: UploadFile, max_bytes: int) -> str:
    """Streams an upload to a temp file, aborting past max_bytes.
    Returns the temp path; the caller is responsible for deleting it."""
    suffix = _safe_suffix(file.filename)
    size = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while True:
            block = await file.read(_READ_CHUNK)
            if not block:
                break
            size += len(block)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large — limit is {max_bytes // (1024 * 1024)} MB.",
                )
            tmp.write(block)
    except BaseException:
        tmp.close()
        os.remove(tmp.name)
        raise
    tmp.close()
    if size == 0:
        os.remove(tmp.name)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return tmp.name


def _load_or_400(path: str):
    try:
        return load_mono(path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode that audio file — is it a valid .wav? ({e})",
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": _detector.name,
        "detection_mode": _mode,
        "alert_threshold": ALERT_THRESHOLD,
        "fusion_check": check_alertable(ALERT_THRESHOLD, _weights),
    }


@app.post("/enroll")
async def enroll(speaker_id: str = Form(...), file: UploadFile = File(...)):
    speaker_id = speaker_id.strip()
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id must not be empty.")

    tmp_path = await _save_upload(file, MAX_ENROLL_BYTES)
    try:
        wav, sr = _load_or_400(tmp_path)
        try:
            _voiceprints.enroll(speaker_id, wav, sr)
        except ValueError as e:
            # Too little speech to build a usable voiceprint.
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp_path)
    return {
        "status": "enrolled",
        "speaker_id": speaker_id,
        "duration_seconds": round(len(wav) / sr, 2),
    }


def _score_chunk(waveform, sample_rate, start_time, speaker_id, speaker_enrolled, ctx, engine):
    """The one place a chunk is scored. Both /analyze (batch) and
    /ws/live (streaming) call this, so the two paths cannot drift."""
    acoustic = _detector.predict_detailed(waveform, sample_rate)
    speaker_mismatch = (
        _voiceprints.mismatch_score(speaker_id, waveform, sample_rate)
        if speaker_enrolled else 0.0
    )
    fused = fuse(acoustic["spoof_prob"], speaker_mismatch, ctx["context_score"], _weights)
    result = engine.update(fused["fused_score"])
    return {
        "start_time": round(start_time, 2),
        "duration": round(len(waveform) / sample_rate, 2),
        **result,
        "acoustic_score": round(fused["acoustic_score"], 3),
        "speaker_mismatch": round(fused["speaker_mismatch"], 3),
        "context_score": round(fused["context_score"], 3),
        "dominant_signal": fused["dominant_signal"],
    }


def _audit_alert(speaker_id, verdict, ctx):
    # Log only the outcome + reasons, never raw audio — keeps the audit
    # trail privacy-preserving while still being useful to a reviewer.
    _audit.append({
        "speaker_id": speaker_id,
        "peak_risk_score": verdict["peak_risk_score"],
        "peak_level": verdict["peak_level"],
        "score_at_alert": verdict["score_at_alert"],
        "context_reasons": ctx["reasons"],
    })


def _verdict_payload(speaker_id, speaker_enrolled, ctx, timeline, verdict):
    return {
        "backend": _detector.name,
        "detection_mode": _mode,
        "speaker_id": speaker_id,
        "speaker_enrolled": speaker_enrolled,
        "context_reasons": ctx["reasons"],
        "timeline": timeline,
        "final_risk_score": verdict["peak_risk_score"],
        "final_level": verdict["peak_level"],
        "alert_triggered": verdict["alert_triggered"],
        "score_at_alert": verdict["score_at_alert"],
        "final_ema": verdict["final_ema"],
        "recommendation": (
            "Recommend secondary verification (call-back on a known number) "
            "before proceeding with any sensitive action."
            if verdict["alert_triggered"] else
            "No strong impersonation signal detected in this call."
        ),
        "audit_logged": verdict["alert_triggered"],
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    speaker_id: str = Form(None),
    caller_known: bool = Form(True),
    off_hours: bool = Form(False),
    first_contact: bool = Form(False),
    transcript: str = Form(""),
):
    speaker_id = (speaker_id or "").strip() or None
    tmp_path = await _save_upload(file, MAX_ANALYZE_BYTES)

    try:
        ctx = score_context(
            caller_known=caller_known,
            off_hours=off_hours,
            first_contact=first_contact,
            transcript=transcript,
        )

        # Surfaced to the UI so a judge can tell "matches the enrolled
        # voice" apart from "we had nothing to compare against" — those
        # both score 0.0 mismatch but mean very different things.
        speaker_enrolled = bool(speaker_id and _voiceprints.has(speaker_id))

        engine = RiskEngine()
        timeline = []
        # chunk_audio() decodes lazily, so a corrupt file raises on the
        # first iteration of this loop rather than at _save_upload time.
        # Without this guard that surfaces as an HTTP 500 instead of a
        # "that file isn't valid audio" message.
        try:
            chunks = list(chunk_audio(tmp_path, chunk_seconds=DEFAULT_CHUNK_SECONDS))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode that audio file — is it a valid .wav? ({e})",
            )

        for chunk in chunks:
            timeline.append(_score_chunk(chunk.waveform, chunk.sample_rate, chunk.start_time,
                                         speaker_id, speaker_enrolled, ctx, engine))
    finally:
        os.remove(tmp_path)

    if not timeline:
        raise HTTPException(
            status_code=400,
            detail="That file contained no analysable audio (too short or silent).",
        )

    # Report the PEAK risk over the call, not the last chunk's EMA. A
    # call cloned early and clean later is still a cloned call; the old
    # behaviour could return alert_triggered=true alongside level=LOW.
    verdict = engine.summary()
    if verdict["alert_triggered"]:
        _audit_alert(speaker_id, verdict, ctx)
    return _verdict_payload(speaker_id, speaker_enrolled, ctx, timeline, verdict)


# ------------------------------------------------------------ live path

# Hard cap keeps a hostile client from streaming a 50 MB frame; a 5 s
# 16 kHz int16 chunk is ~160 KB, a 48 kHz float WAV of the same is ~1 MB.
MAX_LIVE_CHUNK_BYTES = 4 * 1024 * 1024
LIVE_SR = 16000


def _decode_live_chunk(data: bytes):
    """WAV (any rate; downmixed and resampled to 16 kHz) or raw int16
    PCM already at 16 kHz mono. Returns (float32 waveform, 16000)."""
    if data[:4] == b"RIFF":
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != LIVE_SR:
            n = int(len(wav) * LIVE_SR / sr)
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
        return wav, LIVE_SR
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0, LIVE_SR


# DIAGNOSTIC ONLY. When LIVE_CAPTURE_DIR is set, every chunk received on
# /ws/live is written to disk as WAV alongside its sub-scores, so a
# live-call verdict can be reproduced and inspected offline
# (scripts/live_capture_report.py). Off by default: it stores raw audio,
# which the audit trail deliberately never does.
_LIVE_CAPTURE_DIR = os.environ.get("LIVE_CAPTURE_DIR")


def _capture_live_chunk(session_tag, index, waveform, sr, row):
    if not _LIVE_CAPTURE_DIR:
        return
    d = os.path.join(_LIVE_CAPTURE_DIR, session_tag)
    os.makedirs(d, exist_ok=True)
    sf.write(os.path.join(d, f"chunk{index:03d}.wav"), waveform, sr, subtype="PCM_16")
    with open(os.path.join(d, "scores.jsonl"), "a") as f:
        f.write(json.dumps({"index": index, "detection_mode": _mode["mode"], **row}) + chr(10))


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    # 1. call setup: same fields as /analyze's form
    try:
        start = await ws.receive_json()
    except Exception:
        await ws.close(code=1003, reason="first message must be a JSON start message")
        return
    speaker_id = (start.get("speaker_id") or "").strip() or None
    ctx = score_context(
        caller_known=bool(start.get("caller_known", True)),
        off_hours=bool(start.get("off_hours", False)),
        first_contact=bool(start.get("first_contact", False)),
        transcript=start.get("transcript", "") or "",
    )
    speaker_enrolled = bool(speaker_id and _voiceprints.has(speaker_id))
    engine = RiskEngine()               # ONE engine for the whole call
    session_tag = time.strftime("%Y%m%d-%H%M%S") + f"-{speaker_id or 'nospeaker'}"
    timeline = []
    call_clock = 0.0
    audit_logged = False
    await ws.send_json({
        "type": "ready",
        "detection_mode": _mode,
        "speaker_id": speaker_id,
        "speaker_enrolled": speaker_enrolled,
        "context_reasons": ctx["reasons"],
        "chunk_seconds": DEFAULT_CHUNK_SECONDS,
    })

    # 2. score each chunk the moment it lands, reply immediately
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                t_recv = time.time()
                data = msg["bytes"]
                if len(data) > MAX_LIVE_CHUNK_BYTES:
                    await ws.send_json({"type": "error", "detail": "chunk too large"})
                    continue
                try:
                    waveform, sr = _decode_live_chunk(data)
                except Exception as e:
                    await ws.send_json({"type": "error", "detail": f"could not decode chunk ({e})"})
                    continue
                if len(waveform) < sr * 0.2:
                    await ws.send_json({"type": "error", "detail": "chunk too short to score"})
                    continue
                # Scoring is CPU-bound (~0.2-0.4 s with the transformer);
                # run it off the event loop so the socket stays responsive.
                row = await run_in_threadpool(
                    _score_chunk, waveform, sr, call_clock, speaker_id, speaker_enrolled, ctx, engine)
                call_clock += len(waveform) / sr
                timeline.append(row)
                _capture_live_chunk(session_tag, len(timeline) - 1, waveform, sr, row)
                if row["alert"] and not audit_logged:
                    # Same audit write as /analyze, but at the moment the
                    # alert first fires — a live call may never "end".
                    _audit_alert(speaker_id, engine.summary(), ctx)
                    audit_logged = True
                t_sent = time.time()
                await ws.send_json({
                    "type": "chunk",
                    "index": len(timeline) - 1,
                    **row,
                    "audit_logged": audit_logged,
                    "server_recv": t_recv,
                    "server_sent": t_sent,
                    "compute_ms": round((t_sent - t_recv) * 1000, 1),
                })
            elif msg.get("text"):
                try:
                    cmd = json.loads(msg["text"])
                except Exception:
                    cmd = {}
                if cmd.get("type") == "stop":
                    verdict = engine.summary()
                    payload = _verdict_payload(speaker_id, speaker_enrolled, ctx, timeline, verdict)
                    payload["type"] = "summary"
                    payload["audit_logged"] = audit_logged
                    await ws.send_json(payload)
                    break
    except WebSocketDisconnect:
        pass


@app.get("/audit/verify")
def audit_verify():
    return _audit.verify_chain()
