"""
audio_pipeline.py — turns a recorded call (wav file) into a sequence
of timestamped chunks, standing in for a live call feed.

In production this generator would be fed by the enterprise's SIP/VoIP
media stream instead of a file (see README "How will you listen to
calls?" section) — everything downstream (detector, risk engine) is
identical either way, since it only ever sees numpy chunks.
"""

from dataclasses import dataclass
import logging
from math import gcd

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


def resample_to(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """The ONE resampler. Polyphase FIR (scipy.signal.resample_poly),
    which applies a proper anti-aliasing low-pass before decimation.

    Why it matters: downsampling 44.1/48 kHz phone audio to 16 kHz
    without a filter folds everything above 8 kHz back into the band —
    exactly the spectral region a spoof classifier keys on. Every
    committed fixture is already 16 kHz, so the published AUC figures
    never exercised the old np.interp path; an uploaded phone recording
    did. That path also had an off-by-one (linspace to len(wav) instead
    of len(wav) - 1) that a second copy in api.py did not, so the two
    disagreed.

    Falls back to a corrected linear interpolation (no anti-aliasing)
    only if scipy is unavailable, and logs a warning when it does.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if sr == target_sr or len(wav) == 0:
        return wav
    try:
        from scipy.signal import resample_poly
    except ImportError:
        log.warning("scipy not available: resampling %d -> %d Hz with linear interpolation "
                    "and NO anti-aliasing filter; install scipy for correct results", sr, target_sr)
        n = int(round(len(wav) * target_sr / sr))
        return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
    g = gcd(int(sr), int(target_sr))
    return resample_poly(wav, int(target_sr) // g, int(sr) // g).astype(np.float32)


@dataclass
class AudioChunk:
    start_time: float   # seconds from call start
    duration: float
    waveform: np.ndarray
    sample_rate: int


def load_mono(path: str):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # downmix to mono
    return wav.astype(np.float32), sr


# Scoring window. Ablated at 3/5/6 s on the transformer backend over the
# committed fixture sets (scripts/evaluate_fused.py): 3 s missed an A03
# clone that the model catches on the whole utterance (0.119 vs 0.948),
# 5 s and 6 s both caught it (ASVspoof clone recall 7/12 -> 8/12) with
# zero new false positives and identical results to each other. 5 s is
# the smaller latency. Cost: a sustained alert (2 consecutive windows)
# now needs 10 s of audio rather than 6 -- immaterial on a real call.
DEFAULT_CHUNK_SECONDS = 5.0


PIPELINE_SR = 16000

# --- speech presence -------------------------------------------------
# Deliberately dependency-light: short-time RMS above a floor for long
# enough. This runs on the heuristic path too, which must NOT gain a
# torch/Resemblyzer dependency, so it cannot use the VAD in
# speaker_consistency.
# An ABSOLUTE floor only. A relative floor (a fraction of the window's
# loudest frame) was tried first and rejected: on a 1.0 s In-the-Wild
# clone clip it reported "no speech" for audio the transformer scored
# 0.996 synthetic, which silently suppressed a correct detection --
# the same bug class this whole change set exists to remove. Measured
# across the 28 committed fixture clips, the quietest real clip yields
# 0.38 s above this floor; digital silence and -60 dBFS line noise
# yield 0.00 s.
SPEECH_RMS_FLOOR = 0.004      # below this a frame is line noise, not speech
SPEECH_MIN_SECONDS = 0.25     # that much voiced audio makes a window scorable


def has_speech(wav: np.ndarray, sr: int, frame_ms: int = 20) -> bool:
    """True if the window holds at least SPEECH_MIN_SECONDS of audio
    above the noise floor. Used to keep a lead-in with no speech from
    seeding the risk EMA — see risk_engine.RiskEngine.update()."""
    wav = np.asarray(wav, dtype=np.float32)
    frame = max(1, int(sr * frame_ms / 1000))
    if len(wav) < frame:
        return False
    n = len(wav) // frame
    env = np.sqrt(np.mean(wav[:n * frame].reshape(n, frame) ** 2, axis=1))
    if env.size == 0 or env.max() <= 0:
        return False
    return float(np.sum(env > SPEECH_RMS_FLOOR)) * frame_ms / 1000.0 >= SPEECH_MIN_SECONDS


def chunk_audio(path: str, chunk_seconds: float = DEFAULT_CHUNK_SECONDS, overlap: float = 0.0,
                target_sr: int = PIPELINE_SR):
    """Yields AudioChunk objects covering the whole file in order.

    Audio at any rate is resampled once, up front, to target_sr with the
    anti-aliased resampler, so every downstream stage sees the rate the
    detectors and the speaker encoder were calibrated at.

    overlap: fraction of the previous chunk to re-include (0-0.9).
    Small overlap helps catch spoofing artifacts that straddle a
    chunk boundary, at the cost of more compute.
    """
    wav, sr = load_mono(path)
    if target_sr and sr != target_sr:
        wav, sr = resample_to(wav, sr, target_sr), target_sr
    chunk_len = int(sr * chunk_seconds)
    step = int(chunk_len * (1 - overlap)) or chunk_len
    total = len(wav)

    for start in range(0, total, step):
        end = min(start + chunk_len, total)
        segment = wav[start:end]
        if len(segment) < sr * 0.2:
            break  # trailing sliver too short to score meaningfully
        yield AudioChunk(
            start_time=start / sr,
            duration=len(segment) / sr,
            waveform=segment,
            sample_rate=sr,
        )
        if end == total:
            break
