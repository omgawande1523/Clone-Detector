"""
Diagnose one audio file through the exact fixture-evaluation harness:
format/duration/chunking, per-window acoustic score from BOTH backends,
whole-file transformer score, and the fused verdict.

    venv/Scripts/python scripts/score_file.py <audio file> [--speaker ID --enrol <ref file>]
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio_pipeline import load_mono, DEFAULT_CHUNK_SECONDS          # noqa: E402
from src.detector import HeuristicDetector, TransformerDetector         # noqa: E402
from src.fusion import weights_for_backend                              # noqa: E402
from src.evaluation import run_call, SR                                  # noqa: E402
from src.speaker_consistency import compute_voiceprint, MIN_SPEECH_SECONDS, _preprocess  # noqa: E402

path = sys.argv[1]
spk = sys.argv[sys.argv.index("--speaker") + 1] if "--speaker" in sys.argv else None
ref = sys.argv[sys.argv.index("--enrol") + 1] if "--enrol" in sys.argv else None

info = sf.info(path)
wav, sr = load_mono(path)
print("=" * 96)
print(f"FILE {path}")
print("=" * 96)
print(f"  container/subtype : {info.format} / {info.subtype}   channels {info.channels}   {info.samplerate} Hz")
print(f"  duration          : {info.duration:.2f} s   -> {int(info.duration // DEFAULT_CHUNK_SECONDS)} full {DEFAULT_CHUNK_SECONDS:.0f} s window(s)"
      + (f" + {info.duration % DEFAULT_CHUNK_SECONDS:.2f} s tail" if info.duration % DEFAULT_CHUNK_SECONDS >= 0.2 else ""))
if sr != SR:
    n = int(len(wav) * SR / sr)
    wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
    print(f"  resampled {sr} -> {SR} Hz for the pipeline")
speech = len(_preprocess(wav, SR)) / SR
print(f"  speech after VAD  : {speech:.2f} s   (speaker layer needs >= {MIN_SPEECH_SECONDS} s per window)")
print(f"  level             : peak {np.abs(wav).max():.3f}  rms {np.sqrt(np.mean(wav**2)):.4f}  "
      f"clipped {100*np.mean(np.abs(wav) > 0.99):.2f}%")
spec = np.abs(np.fft.rfft(wav)); freqs = np.fft.rfftfreq(len(wav), 1 / SR); cum = np.cumsum(spec ** 2)
print(f"  99% energy below  : {freqs[np.searchsorted(cum, 0.99 * cum[-1])]/1000:.2f} kHz")
if info.duration < DEFAULT_CHUNK_SECONDS:
    print(f"  !! shorter than one {DEFAULT_CHUNK_SECONDS:.0f} s window: scored as a single partial window; the 2-window sustain rule cannot fire")

reference = None
if spk and ref:
    rwav, rsr = load_mono(ref)
    reference = compute_voiceprint(rwav, rsr)
    print(f"  enrolled '{spk}' from {os.path.basename(ref)} ({len(rwav)/rsr:.1f} s) -> voiceprint {'OK' if reference else 'FAILED (too little speech)'}")

T, H = TransformerDetector(), HeuristicDetector()
print(f"\n  whole-file transformer spoof_prob : {T.predict(wav, SR):.3f}   (0 = human, 1 = synthetic)")
print(f"  whole-file heuristic  spoof_prob : {H.predict(wav, SR):.3f}")

for det in (T, H):
    w = weights_for_backend(det.backend)
    summary, rows, scored = run_call(wav, reference, det, w)
    print(f"\n  {det.backend.upper()} backend, weights a/s/c {w.acoustic:.2f}/{w.speaker:.2f}/{w.context:.2f}")
    print(f"    {'#':>2} {'t':>5} {'acoustic':>8} {'speaker':>8} {'context':>7} {'risk':>6} {'level':<9} alert")
    for i, r in enumerate(rows):
        print(f"    {i:>2} {r['t']:>5.1f} {r['acoustic']:>8.3f} {r['speaker']:>8.3f} {r['context']:>7.2f} "
              f"{r['risk']:>6.3f} {r['level']:<9} {'YES' if r['alert'] else '-'}")
    print(f"    verdict: {summary['peak_level']} peak {summary['peak_risk_score']:.3f} alert={summary['alert_triggered']}"
          + ("" if reference else "   (no enrolment: speaker layer off)"))
