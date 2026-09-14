"""
demo_cli.py — sanity-check the whole pipeline with ZERO internet and
ZERO real audio files, by synthesizing two toy signals:

  1. "genuine-like": a harmonic voice-ish tone with natural pitch
     jitter and energy micro-variation (mimics real speech's
     imperfections).
  2. "robotic-like": the same base tone but with pitch and energy
     smoothed to be suspiciously perfect (mimics an over-clean
     vocoder/TTS artifact — a real, published anti-spoofing cue).

This is NOT a substitute for testing on real deepfake audio — before
your actual submission, replace these with genuine clips + samples
from a real dataset (ASVspoof 2019/2021 LA, or the "In-the-Wild"
deepfake audio dataset) to get a real accuracy number. This script
just proves the pipeline (chunking -> detector -> risk engine -> alert)
runs end to end.

Usage:
    python demo/demo_cli.py
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import get_detector
from src.risk_engine import RiskEngine
from src.speaker_consistency import compute_voiceprint, mismatch_between
from src.context_engine import score_context
from src.fusion import fuse, caps_for_backend
from src.audit_log import AuditLog


SR = 16000
DURATION = 15.0  # seconds -> 3 chunks of 5s


def synth_voice(duration, sr, f0=140.0, jitter_amt=0.02, smooth=False):
    n = int(duration * sr)
    t = np.arange(n) / sr

    if smooth:
        f0_track = np.full(n, f0)  # perfectly flat pitch -> "robotic"
        env_noise = 0.0
    else:
        # natural pitch wobble ~2% frame to frame + slow drift
        drift = 3 * np.sin(2 * np.pi * 0.3 * t)
        micro = np.cumsum(np.random.randn(n) * jitter_amt * f0) / sr * 50
        f0_track = f0 + drift + micro
        env_noise = 0.03

    phase = 2 * np.pi * np.cumsum(f0_track) / sr
    # a few harmonics to sound vaguely voice-like, not a pure sine
    wav = (
        1.0 * np.sin(phase)
        + 0.5 * np.sin(2 * phase)
        + 0.25 * np.sin(3 * phase)
    )
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)
    if env_noise:
        envelope = envelope * (1 + np.random.randn(n) * env_noise)
    wav = wav * envelope
    wav = wav / (np.max(np.abs(wav)) + 1e-9) * 0.8
    return wav.astype(np.float32)


def run_case(label, wav, sr, detector, reference_print=None, ctx=None, audit=None):
    print(f"\n=== {label} ===")
    weights = caps_for_backend(detector.backend)   # same rule as the API
    engine = RiskEngine()
    ctx = ctx or {"context_score": 0.0, "reasons": []}
    chunk_len = int(sr * 5.0)
    for start in range(0, len(wav), chunk_len):
        segment = wav[start:start + chunk_len]
        if len(segment) < sr * 0.5:
            break
        acoustic = detector.predict_detailed(segment, sr)
        if reference_print is not None:
            live_print = compute_voiceprint(segment, sr)
            # Calibrated mapping, NOT a raw 1-cosine: see
            # speaker_consistency.mismatch_from_similarity().
            speaker_mismatch = mismatch_between(reference_print, live_print)
        else:
            speaker_mismatch = 0.0
        fused = fuse(acoustic["spoof_prob"], speaker_mismatch, ctx["context_score"], weights)
        result = engine.update(fused["fused_score"])
        t = start / sr
        print(
            f"  t={t:5.1f}s  acoustic={fused['acoustic_score']:.2f}  "
            f"speaker_mismatch={fused['speaker_mismatch']:.2f}  "
            f"context={fused['context_score']:.2f}  "
            f"-> risk={result['risk_score']:.2f}  level={result['level']:<8}"
            f"{'  <-- ALERT' if result['alert'] else ''}"
        )
        if result["alert"] and audit is not None:
            audit.append({"case": label, "risk_score": result["risk_score"], "level": result["level"]})


def main():
    detector = get_detector("heuristic")
    print(f"Backend: {detector.name}  (fusion weights follow the backend: {caps_for_backend(detector.backend)})")

    genuine = synth_voice(DURATION, SR, smooth=False)
    robotic = synth_voice(DURATION, SR, smooth=True)
    impostor = synth_voice(DURATION, SR, f0=210.0, smooth=False)  # different pitch range

    # Enroll "genuine" as the reference voiceprint for a speaker, then
    # show that even a NATURAL-sounding but different voice (impostor)
    # gets caught by the speaker-consistency layer even though it might
    # pass the acoustic-only check.
    reference_print = compute_voiceprint(genuine, SR)
    # tempfile.gettempdir() instead of a hardcoded /tmp, which does
    # not exist on Windows.
    audit_path = os.path.join(tempfile.gettempdir(), "demo_audit_log.jsonl")
    if os.path.exists(audit_path):
        os.remove(audit_path)   # fresh chain each demo run
    audit = AuditLog(path=audit_path)

    run_case("Genuine call (matches enrolled voiceprint)", genuine, SR, detector,
              reference_print=reference_print, audit=audit)
    run_case("Robotic/cloned-like voice", robotic, SR, detector,
              reference_print=reference_print, audit=audit)
    run_case("Different real person's voice (not the enrolled speaker)", impostor, SR, detector,
              reference_print=reference_print, audit=audit)
    run_case("Robotic voice + risky context (unknown caller, 'urgent transfer')", robotic, SR, detector,
              reference_print=reference_print,
              ctx=score_context(caller_known=False, off_hours=True, transcript="urgent wire transfer needed"),
              audit=audit)

    print(f"\nAudit chain check: {audit.verify_chain()}")
    print(
        "\nNote: these are synthetic toy signals for pipeline testing only. "
        "Validate real accuracy on ASVspoof / In-the-Wild samples, or switch "
        "DETECTOR_BACKEND=transformer for the real wav2vec2 classifier."
    )


if __name__ == "__main__":
    main()
