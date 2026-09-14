"""Regression fixture from the real failure.

fixtures/live_clone_session/ holds four windows captured from a live
/ws/live run in which the transformer scored a clone of the enrolled
speaker 0.984-0.996 synthetic on every window and the system never
alerted (see the fixture README). This replays those exact windows
through the full api._score_chunk -> RiskEngine path with the speaker
enrolled and asserts an alert fires by window 3 at the latest.

It failed on the pre-fix code (weighted-average fusion capped acoustic
at 0.45, symmetric EMA never reached 0.40).
"""

import json
import os

import numpy as np
import pytest

from src.audio_pipeline import load_mono
from src.risk_engine import RiskEngine, ALERT_THRESHOLD
from src.speaker_consistency import VoiceprintStore, compute_voiceprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures", "live_clone_session")
ASV = os.path.join(ROOT, "fixtures", "asvspoof2019")


def _transformer_cached():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    from src.detector import TransformerDetector
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.isdir(os.path.join(hub, "models--" + TransformerDetector.MODEL_ID.replace("/", "--")))


pytestmark = pytest.mark.skipif(
    not (_transformer_cached() and os.path.exists(os.path.join(FIX, "manifest.json"))),
    reason="transformer weights not cached, or fixture missing")


@pytest.fixture(scope="module")
def live_api(tmp_path_factory):
    """api module with the transformer backend and LA_0075 enrolled the
    way the captured session had it."""
    from src import api
    from src.detector import TransformerDetector
    from src.fusion import caps_for_backend, detection_mode
    det = TransformerDetector()
    store = VoiceprintStore(path=str(tmp_path_factory.mktemp("vp") / "vp.json"))
    man = json.load(open(os.path.join(ASV, "manifest.json")))
    enrol = [m["utt"] for m in man if m["speaker"] == "LA_0075" and m["role"] == "enrol"]
    joined = np.concatenate([load_mono(os.path.join(ASV, "flac", u + ".flac"))[0] for u in enrol])
    store.enroll("LA_0075", joined, 16000)
    saved = (api._detector, api._weights, api._mode, api._voiceprints)
    api._detector, api._voiceprints = det, store
    api._weights, api._mode = caps_for_backend(det.backend), detection_mode(det.backend)
    yield api
    api._detector, api._weights, api._mode, api._voiceprints = saved


def test_captured_clone_windows_alert_by_window_3(live_api):
    man = json.load(open(os.path.join(FIX, "manifest.json")))
    ctx = {"context_score": 0.0, "reasons": []}
    # ALL six windows, in order, exactly as the session streamed them --
    # window 0 included, because it is what anchored the EMA low.
    engine, state, rows = RiskEngine(), {}, []
    for i, w in enumerate(man["windows"]):
        wav, sr = load_mono(os.path.join(FIX, w["file"]))
        rows.append(live_api._score_chunk(wav, sr, i * 5.0, "LA_0075", True, ctx, engine, state))
    print("\n  win  acoustic  speaker  instant   EMA    level     alert")
    for i, r in enumerate(rows):
        print(f"  {i:>3}  {r['acoustic_score']:>8.3f} {r['speaker_mismatch']:>8.3f} {r['instant_score']:>8.3f} "
              f"{r['risk_score']:>6.3f}  {r['level']:<9} {r['alert']}")
    # the model still sees what it saw in the session
    clone_windows = [i for i, r in enumerate(rows) if r["acoustic_score"] > 0.9]
    assert clone_windows[:4] == [1, 2, 3, 4], "transformer no longer flags the clone windows as it did"
    first_alert = next((i for i, r in enumerate(rows) if r["alert"]), None)
    assert first_alert is not None, "no alert on four consecutive windows of 0.98+ synthetic audio"
    # window 1 is the first clone window; allow one window of climb, no more
    assert first_alert <= 3, f"alert only at window {first_alert}; clone audio started at window 1"
    assert engine.summary()["alert_triggered"] is True
    assert max(r["risk_score"] for r in rows) >= ALERT_THRESHOLD
