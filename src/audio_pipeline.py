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


# Scoring window. Ablated at 3/5/6 s on the transformer backend over the
# committed fixture sets (scripts/evaluate_fused.py): 3 s missed an A03
# clone that the model catches on the whole utterance (0.119 vs 0.948),
# 5 s and 6 s both caught it (ASVspoof clone recall 7/12 -> 8/12) with
# zero new false positives and identical results to each other. 5 s is
# the smaller latency. Cost: a sustained alert (2 consecutive windows)
# now needs 10 s of audio rather than 6 -- immaterial on a real call.
DEFAULT_CHUNK_SECONDS = 5.0


def chunk_audio(path: str, chunk_seconds: float = DEFAULT_CHUNK_SECONDS, overlap: float = 0.0):
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
