"""
Proves /ws/live is genuinely real-time, not batched.

Feeds a live call to the running server one 5 s chunk at a time with a
REAL wall-clock delay between chunks (time.sleep, simulating the cadence
a microphone produces), and logs for every chunk:

    t_send      client wall-clock when the chunk left
    t_result    client wall-clock when that chunk's result arrived
    gap         t_result - t_send   <- the number that matters
    compute_ms  server-side decode+score time, from the reply itself

If the server were batching, `gap` would grow with each chunk (results
piling up until the session ends). If it is streaming, `gap` stays flat
at roughly the per-chunk scoring time (~0.2-0.4 s with the transformer)
regardless of how many chunks have gone before.

    venv/Scripts/python scripts/live_latency_probe.py            # genuine call
    venv/Scripts/python scripts/live_latency_probe.py --clone    # targeted clone

Requires a running server (uvicorn src.api:app --port 8000) and
`pip install websockets`.
"""

import asyncio
import io
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import httpx
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "http://localhost:8000"
WS = "ws://localhost:8000/ws/live"
SR = 16000
CHUNK_S = 5.0
CADENCE_S = 5.0            # real delay between chunks, like a live mic
CLONE = "--clone" in sys.argv


def wav_bytes(x):
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, (path, sr)
    return x


def build_call():
    """Returns (speaker_id, enrol_audio, [chunk_audio...])."""
    man = json.load(open(os.path.join(ROOT, "fixtures", "asvspoof2019", "manifest.json")))
    flac = lambda u: os.path.join(ROOT, "fixtures", "asvspoof2019", "flac", u + ".flac")
    spk = "LA_0075"
    enrol = np.concatenate([load(flac(m["utt"])) for m in man if m["speaker"] == spk and m["role"] == "enrol"])
    if CLONE:
        # every spoof of LA_0075 in the fixture set, back to back: a live
        # call in which the caller IS a clone of the enrolled person
        utts = [m["utt"] for m in man if m["speaker"] == spk and m["key"] == "spoof"]
    else:
        # the held-out genuine utterances, repeated to fill six chunks
        utts = [m["utt"] for m in man if m["speaker"] == spk and m["role"] == "test"] * 3
    audio = np.concatenate([load(flac(u)) for u in utts])
    n = int(CHUNK_S * SR)
    chunks = [audio[i:i + n] for i in range(0, len(audio), n) if len(audio[i:i + n]) >= SR * 1.0]
    return spk, enrol, chunks[:6]


async def main():
    spk, enrol, chunks = build_call()
    with httpx.Client(timeout=120) as c:
        r = c.post(f"{API}/enroll", data={"speaker_id": spk}, files={"file": ("enrol.wav", wav_bytes(enrol), "audio/wav")})
        print(f"enrolled {spk}: {r.json()}")
        mode = c.get(f"{API}/health").json()["detection_mode"]["mode"]

    label = "TARGETED CLONE of the enrolled speaker" if CLONE else "GENUINE caller (held-out bonafide)"
    print(f"\nlive call: {label}  |  {len(chunks)} chunks x {CHUNK_S:.0f}s, sent every {CADENCE_S:.0f}s  |  server mode: {mode}")
    print(f"\n  {'#':>2} {'t_send':>8} {'t_result':>9} {'GAP s':>7} {'compute_ms':>10} {'risk':>6} {'level':<9} alert")
    print("  " + "-" * 70)

    gaps = []
    t0 = time.time()
    async with websockets.connect(WS, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"speaker_id": spk, "caller_known": True}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        assert ready["speaker_enrolled"], "speaker not enrolled?"

        for i, chunk in enumerate(chunks):
            if i:
                await asyncio.sleep(CADENCE_S)          # the live cadence
            t_send = time.time()
            await ws.send(wav_bytes(chunk))
            reply = json.loads(await ws.recv())          # blocks only until THIS chunk's result
            t_result = time.time()
            assert reply["type"] == "chunk" and reply["index"] == i, reply
            gap = t_result - t_send
            gaps.append(gap)
            print(f"  {i:>2} {t_send - t0:>8.2f} {t_result - t0:>9.2f} {gap:>7.3f} {reply['compute_ms']:>10.1f} "
                  f"{reply['risk_score']:>6.3f} {reply['level']:<9} {'ALERT' if reply['alert'] else '-'}"
                  f"{'   <- audit written' if reply['audit_logged'] and reply['alert'] else ''}")

        await ws.send(json.dumps({"type": "stop"}))
        summary = json.loads(await ws.recv())
        assert summary["type"] == "summary"

    print(f"\n  verdict: {summary['final_level']} peak {summary['final_risk_score']:.3f}  "
          f"alert={summary['alert_triggered']}  audit_logged={summary['audit_logged']}")
    print(f"\n  chunk-arrival-to-result gap: min {min(gaps):.3f}s  median {sorted(gaps)[len(gaps)//2]:.3f}s  "
          f"max {max(gaps):.3f}s   (cadence {CADENCE_S:.0f}s; a batching server would show gaps growing toward "
          f"{CADENCE_S * len(chunks):.0f}s)")
    drift = gaps[-1] - gaps[0]
    print(f"  drift first->last chunk: {drift:+.3f}s  ->  {'FLAT: streaming' if abs(drift) < 1.0 else 'GROWING: something is batching'}")


if __name__ == "__main__":
    asyncio.run(main())
