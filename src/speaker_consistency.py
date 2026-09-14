"""
speaker_consistency.py — "does this voice actually match the person
it claims to be?"

This is a fundamentally different question from "is this synthetic
speech?" (that's detector.py's job) — and it matters because it's the
one check a sufficiently good clone cannot simply get better at
passing. A clone can beat a generic artifact detector by getting
cleaner over time; it can only defeat THIS check by actually
reproducing one specific enrolled person's vocal identity. This
implements the PS's "cross-session consistency checks... against
historical genuine samples" requirement.

IMPLEMENTATION: Resemblyzer's VoiceEncoder (GE2E d-vector, 256-dim).

    Why not the previous MFCC-style voiceprint: log-band-energy
    vectors are all the same sign, so their cosine similarity is
    pinned near 1.0 for ANY two speech signals. Measured on the old
    code, a completely different speaker scored a mismatch of 0.002
    while the *same* speaker with flattened pitch scored 0.632 — i.e.
    it was measuring synthesis artifacts, not speaker identity, and
    so was not independent of the acoustic detector at all.

    Resemblyzer is trained for speaker verification specifically. The
    pretrained weights ship inside the pip package, so after install
    it runs fully offline — no download at demo time.

CALIBRATION (see the thresholds below): raw cosine on these
embeddings does not span [0,1] in practice — same-speaker pairs
cluster high and different-speaker pairs sit near 0.5 — so the raw
value is remapped onto a usable [0,1] risk score against a measured
decision band.
"""

import json
import os
import threading
import numpy as np

# Cosine-similarity decision band, used to turn a raw similarity into
# a [0,1] mismatch RISK score. Measured on 16 kHz samples with this
# encoder: same-speaker pairs scored 0.91-0.97, different-speaker
# pairs 0.52-0.54 (see tests/test_speaker_consistency.py).
#
# The band is deliberately wider than those measurements because real
# human enrollment audio — recorded weeks apart, on different
# handsets, through a phone codec — is less self-consistent than
# clean audio, and wrongly telling a genuine customer "you are not
# you" is the more costly error here. Retune SIM_MATCH downward if
# genuine callers are being flagged in your own data.
SIM_MATCH = 0.80      # at/above this -> confidently same speaker (mismatch 0.0)
SIM_IMPOSTOR = 0.55   # at/below this -> confidently different speaker (mismatch 1.0)

# Resemblyzer's internal sample rate. Everything is resampled to this.
TARGET_SR = 16000

# Below this much *speech* (after VAD trimming) a speaker embedding is
# not stable enough to trust, so we decline to score rather than guess.
#
# This is not a theoretical number: at 0.6s the trailing sliver of a
# call produced a 0.813 mismatch on a genuine caller AND a 0.000
# mismatch on an impostor — i.e. confidently wrong in both directions.
# 1.5s of real speech was enough for both to score correctly.
MIN_SPEECH_SECONDS = 1.5

_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():
    """Lazily load the voice encoder once per process (loading it per
    chunk would dominate the runtime)."""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                try:
                    from resemblyzer import VoiceEncoder
                except ImportError as e:
                    raise RuntimeError(
                        "speaker_consistency needs Resemblyzer: "
                        "pip install resemblyzer webrtcvad-wheels "
                        "(on Windows use webrtcvad-wheels, not webrtcvad "
                        "— the latter needs a C++ compiler)."
                    ) from e
                _encoder = VoiceEncoder(device="cpu", verbose=False)
    return _encoder


def _preprocess(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz, normalise loudness and trim silence using
    Resemblyzer's own front end, so the input matches what the encoder
    was trained on."""
    from resemblyzer import preprocess_wav
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return preprocess_wav(wav, source_sr=sr)


def compute_voiceprint(wav: np.ndarray, sr: int):
    """Returns a 256-dim L2-normalised speaker embedding as a plain
    list (JSON-serialisable for the voiceprint store).

    Returns None when the clip holds too little actual speech to
    embed — callers must treat that as "no opinion", not as a match.
    Longer, cleaner enrollment audio gives a more reliable print;
    10-20 seconds is a good target.
    """
    processed = _preprocess(wav, sr)
    if len(processed) < MIN_SPEECH_SECONDS * TARGET_SR:
        return None
    return _get_encoder().embed_utterance(processed).tolist()


def cosine_similarity(a, b):
    """Raw cosine similarity between two voiceprints.

    Resemblyzer embeddings are non-negative and unit-norm, so this
    already lands in [0, 1] — do NOT rescale it with (x+1)/2, which is
    what compressed the old implementation's dynamic range into
    uselessness. Interpret it via mismatch_from_similarity().

    Returns None if either voiceprint is None (nothing to compare).
    """
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def mismatch_from_similarity(similarity) -> float:
    """Maps a raw cosine similarity onto a [0,1] RISK score: 0 means
    "confidently the enrolled speaker", 1 means "confidently somebody
    else". Linear inside the decision band, saturating outside it —
    deliberately simple so an alert stays explainable to a compliance
    reviewer. A None similarity means we have no basis to judge, which
    scores a neutral 0.0 rather than penalising the caller."""
    if similarity is None:
        return 0.0
    span = SIM_MATCH - SIM_IMPOSTOR
    return float(np.clip((SIM_MATCH - similarity) / span, 0.0, 1.0))


def mismatch_between(print_a, print_b) -> float:
    """Convenience: two voiceprints in, calibrated risk score out."""
    return mismatch_from_similarity(cosine_similarity(print_a, print_b))


class VoiceprintStore:
    """Tiny local JSON-backed store — swap for a real DB in production.
    Keyed by speaker_id (e.g. "ceo", "cust_00123").

    The default path is anchored to the project root rather than the
    current working directory, so enrollments do not silently vanish
    when uvicorn is launched from a different folder.
    """

    DEFAULT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "voiceprints.json",
    )

    def __init__(self, path: str = None):
        self.path = path or self.DEFAULT_PATH
        self._lock = threading.Lock()
        self._data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # A corrupt store must not take the whole service down;
                # start clean and let the operator re-enroll.
                self._data = {}

    def enroll(self, speaker_id: str, wav: np.ndarray, sr: int):
        voiceprint = compute_voiceprint(wav, sr)
        if voiceprint is None:
            raise ValueError(
                "Enrollment audio contains too little speech to build a "
                "voiceprint — provide 10-20 seconds of clean speech."
            )
        # Read-modify-write under a lock (AuditLog already had one; this
        # store did not), and an atomic replace so a crash mid-write
        # cannot leave a truncated voiceprints.json behind.
        with self._lock:
            self._data[speaker_id] = voiceprint
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)

    def has(self, speaker_id: str) -> bool:
        return speaker_id in self._data

    def similarity(self, speaker_id: str, wav: np.ndarray, sr: int):
        """Raw cosine similarity against the enrolled print, or None if
        we cannot say (speaker not enrolled / not enough speech)."""
        if speaker_id not in self._data:
            return None
        live_print = compute_voiceprint(wav, sr)
        if live_print is None:
            return None
        return cosine_similarity(self._data[speaker_id], live_print)

    def mismatch_score(self, speaker_id: str, wav: np.ndarray, sr: int):
        """Returns a RISK score in [0, 1] — 0 means the voice matches
        the enrolled speaker well, 1 means it is very unlike them — or
        None when there is genuinely no opinion: no reference on file,
        or fewer than MIN_SPEECH_SECONDS of speech in the window.

        None used to be returned as 0.0, which is indistinguishable from
        "confidently the enrolled speaker". A clone that pauses a lot
        got a free pass on every quiet window and dragged the EMA down.
        Callers must treat None as "not assessed" — api._score_chunk
        carries the last confident value forward within a call.
        """
        sim = self.similarity(speaker_id, wav, sr)
        return None if sim is None else mismatch_from_similarity(sim)
