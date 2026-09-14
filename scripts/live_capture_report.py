"""
Prints the raw per-chunk sub-score table for live sessions captured with
LIVE_CAPTURE_DIR (see api._capture_live_chunk), plus signal-level stats
of the audio the detector actually saw, and the same chunks re-scored
through the batch path so live-vs-upload can be compared on identical
samples.

    venv/Scripts/python scripts/live_capture_report.py <capture_dir> [session_tag ...]
"""

import glob
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cap = sys.argv[1]
sessions = sys.argv[2:] or sorted(os.path.basename(d) for d in glob.glob(os.path.join(cap, "*")) if os.path.isdir(d))
if not sessions:
    sys.exit(f"no captured sessions under {cap}")


def stats(x, sr):
    spec = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1 / sr)
    cum = np.cumsum(spec ** 2)
    bw = freqs[np.searchsorted(cum, 0.99 * cum[-1])] / 1000 if cum[-1] > 0 else 0
    return dict(peak=float(np.abs(x).max()), rms=float(np.sqrt(np.mean(x ** 2))),
                clip=float(np.mean(np.abs(x) > 0.99)), bw=float(bw))


for tag in sessions:
    d = os.path.join(cap, tag)
    rows = [json.loads(l) for l in open(os.path.join(d, "scores.jsonl")) if l.strip()]
    print("=" * 112)
    print(f"SESSION {tag}   detection_mode={rows[0]['detection_mode']}   chunks={len(rows)}")
    print("=" * 112)
    print(f"  {'#':>2} {'t':>6} {'secs':>5} | {'acoustic':>8} {'speaker':>8} {'context':>7} {'instant':>7} "
          f"{'risk(EMA)':>9} {'level':<9} {'alert':<5} | {'peak':>5} {'rms':>6} {'clip%':>5} {'bw kHz':>6}")
    print("  " + "-" * 108)
    for r in rows:
        x, sr = sf.read(os.path.join(d, f"chunk{r['index']:03d}.wav"), dtype="float32")
        st = stats(x, sr)
        print(f"  {r['index']:>2} {r['start_time']:>6.1f} {r['duration']:>5.2f} | {r['acoustic_score']:>8.3f} "
              f"{r['speaker_mismatch']:>8.3f} {r['context_score']:>7.2f} {r['instant_score']:>7.3f} "
              f"{r['risk_score']:>9.3f} {r['level']:<9} {'YES' if r['alert'] else '-':<5} | "
              f"{st['peak']:>5.2f} {st['rms']:>6.3f} {100*st['clip']:>5.2f} {st['bw']:>6.2f}")

    # the same audio through the batch path, for a like-for-like check
    try:
        import httpx
        joined = np.concatenate([sf.read(os.path.join(d, f"chunk{r['index']:03d}.wav"), dtype="float32")[0] for r in rows])
        buf = io.BytesIO(); sf.write(buf, joined, 16000, format="WAV", subtype="PCM_16")
        sid = tag.split("-", 2)[-1]
        data = {} if sid == "nospeaker" else {"speaker_id": sid}
        up = httpx.post("http://localhost:8000/analyze", data=data,
                        files={"file": ("joined.wav", buf.getvalue(), "audio/wav")}, timeout=300).json()
        print(f"\n  same chunks joined and sent through /analyze: {up['final_level']} peak {up['final_risk_score']:.3f}  "
              f"acoustic per window {[t['acoustic_score'] for t in up['timeline']]}")
    except Exception as e:
        print(f"\n  (/analyze cross-check skipped: {e})")
    print()
