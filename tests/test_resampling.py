"""Regression tests for the resampler.

TransformerDetector.predict() used np.interp with no anti-aliasing
filter and an off-by-one (linspace to len(wav) instead of len(wav)-1);
api._decode_live_chunk() had its own copy without the off-by-one. Every
committed fixture is 16 kHz, so the published AUC figures never
exercised either path — but an uploaded 44.1/48 kHz phone recording
did, and aliasing folds >8 kHz energy into exactly the band a spoof
classifier keys on. audio_pipeline.resample_to() (scipy resample_poly)
is now the only resampler.
"""

import io
import os

import numpy as np
import pytest
import soundfile as sf

from src.audio_pipeline import resample_to, chunk_audio, PIPELINE_SR
from src.detector import HeuristicDetector
from src.fusion import weights_for_backend
from src.evaluation import run_call, ASV, ITW, SR

FIX_ITW = os.path.join(ITW, "wav", "9532.wav")          # 16 kHz bonafide, 5.65 s
FIX_ASV = os.path.join(ASV, "flac", "LA_D_3876919.flac")  # 16 kHz A02 spoof, 8.7 s


def _transformer_cached():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    from src.detector import TransformerDetector
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.isdir(os.path.join(hub, "models--" + TransformerDetector.MODEL_ID.replace("/", "--")))


# ------------------------------------------------------------ aliasing

def test_tone_above_nyquist_does_not_alias_into_band():
    """A 12 kHz tone at 48 kHz is above the 8 kHz Nyquist limit of the
    16 kHz target. Without a filter, decimation folds it to 4 kHz. With
    one, it is removed."""
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    out = resample_to(tone, sr, 16000)
    assert abs(len(out) - 16000) <= 2
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / 16000)
    band = spec[(freqs > 3500) & (freqs < 4500)].max()        # where the alias would land
    total = np.sqrt(np.mean(out ** 2))
    assert band < 0.02 * len(out), f"aliased component present at ~4 kHz: {band}"
    assert total < 0.02, f"out-of-band tone should be removed, rms {total}"


def test_naive_interp_does_alias_which_is_why_this_exists():
    """The old path, reproduced here so the test documents the bug it
    prevents: linear interpolation lets the 12 kHz tone through as a
    4 kHz alias with substantial energy."""
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    n = 16000
    naive = np.interp(np.linspace(0, len(tone) - 1, n), np.arange(len(tone)), tone)
    assert np.sqrt(np.mean(naive ** 2)) > 0.1


def test_in_band_content_survives():
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    out = resample_to(tone, sr, 16000)
    assert np.sqrt(np.mean(out[1000:-1000] ** 2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


def test_length_is_exact_and_no_offbyone():
    x = np.random.RandomState(0).randn(48000).astype(np.float32)
    assert len(resample_to(x, 48000, 16000)) == 16000
    assert len(resample_to(x, 44100, 16000)) == round(48000 * 16000 / 44100)   # 44100/16000 is not integer
    assert len(resample_to(x, 16000, 16000)) == 48000               # identity, untouched


def test_same_rate_is_identity():
    x = np.random.RandomState(1).randn(1000).astype(np.float32)
    assert np.array_equal(resample_to(x, 16000, 16000), x)


# ---------------------------------------------------- pipeline round trip

def _roundtrip_verdict(path, detector):
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR
    w = weights_for_backend(detector.backend)
    s0, r0, _ = run_call(wav, None, detector, w)
    up = resample_to(wav, SR, 48000)
    back = resample_to(up, 48000, SR)
    s1, r1, _ = run_call(back, None, detector, w)
    return s0, s1, max(r["acoustic"] for r in r0), max(r["acoustic"] for r in r1)


@pytest.mark.skipif(not os.path.exists(FIX_ITW), reason="fixtures not present")
def test_48k_roundtrip_same_verdict_heuristic():
    s0, s1, a0, a1 = _roundtrip_verdict(FIX_ITW, HeuristicDetector())
    assert s1["peak_level"] == s0["peak_level"]
    assert abs(a1 - a0) < 0.05


@pytest.mark.skipif(not (_transformer_cached() and os.path.exists(FIX_ASV)), reason="transformer weights not cached")
def test_48k_roundtrip_same_verdict_transformer():
    """The number that says an uploaded phone recording is scored
    honestly: a spoof the model catches at 16 kHz must still be caught
    after a 48 kHz round trip, and a bonafide must stay clean."""
    from src.detector import TransformerDetector
    det = TransformerDetector()
    for path in (FIX_ASV, FIX_ITW):
        s0, s1, a0, a1 = _roundtrip_verdict(path, det)
        assert s1["peak_level"] == s0["peak_level"], path
        assert abs(a1 - a0) < 0.05, (path, a0, a1)


@pytest.mark.skipif(not os.path.exists(FIX_ITW), reason="fixtures not present")
def test_chunk_audio_resamples_uploads_to_pipeline_rate(tmp_path):
    """/analyze accepts any rate; chunk_audio must hand downstream 16 kHz."""
    wav, sr = sf.read(FIX_ITW, dtype="float32")
    up = resample_to(wav, sr, 48000)
    p = tmp_path / "up48.wav"
    sf.write(p, up, 48000, subtype="PCM_16")
    chunks = list(chunk_audio(str(p)))
    assert chunks and all(c.sample_rate == PIPELINE_SR for c in chunks)
    assert sum(len(c.waveform) for c in chunks) == pytest.approx(len(wav), abs=PIPELINE_SR * 0.01)


def test_live_chunk_decoder_uses_the_shared_resampler():
    """api._decode_live_chunk had its own np.interp copy; it must now
    produce the same samples as resample_to()."""
    from src.api import _decode_live_chunk
    sr = 44100
    t = np.arange(int(sr * 1.0)) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    got, gsr = _decode_live_chunk(buf.getvalue())
    ref = resample_to(sf.read(io.BytesIO(buf.getvalue()), dtype="float32")[0], sr, 16000)
    assert gsr == 16000
    assert np.allclose(got, ref, atol=1e-6)
