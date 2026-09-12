"""
audio_pipeline.py — turns a recorded call (wav file) into a sequence
of timestamped chunks, standing in for a live call feed.

In production this generator would be fed by the enterprise's SIP/VoIP
media stream instead of a file (see README "How will you listen to
calls?" section) — everything downstream (detector, risk engine) is
identical either way, since it only ever sees numpy chunks.
"""

from dataclasses import dataclass
import numpy as np
import soundfile as sf


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


def chunk_audio(path: str, chunk_seconds: float = 3.0, overlap: float = 0.0):
    """Yields AudioChunk objects covering the whole file in order.

    overlap: fraction of the previous chunk to re-include (0-0.9).
    Small overlap helps catch spoofing artifacts that straddle a
    chunk boundary, at the cost of more compute.
    """
    wav, sr = load_mono(path)
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
