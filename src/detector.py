"""
detector.py — Voice authenticity scoring backends for SIH26104.

Two interchangeable backends:

1. HeuristicDetector (default, zero downloads, runs anywhere)
   Uses classic anti-spoofing signal cues from the speech-forensics
   literature: pitch (F0) micro-jitter, spectral flatness, and energy
   variance. Genuine human speech has natural micro-instability in
   pitch and energy; many TTS/vocoder outputs are unnaturally smooth
   and spectrally "too clean". This is a legitimate, explainable
   baseline — NOT a trained classifier — good for offline demos and
   for explaining to judges *why* the system flags what it flags.

2. TransformerDetector (real accuracy path, needs internet once)
   Wraps a HuggingFace wav2vec2 model fine-tuned for deepfake/spoof
   audio classification. Needs `pip install torch transformers` and
   an internet connection the FIRST time it runs (to download
   weights, then it's cached locally and works offline).

Swap backends via DETECTOR_BACKEND env var or the `backend=` argument
to get_detector().
"""

import os
import numpy as np


class HeuristicDetector:
    """Offline, dependency-light anti-spoofing baseline."""

    name = "heuristic-signal-baseline"

    def _f0_track(self, wav: np.ndarray, sr: int, frame_ms=32, hop_ms=16):
        """Very small autocorrelation-based pitch tracker. Returns an
        array of F0 estimates (Hz), one per frame; 0 where unvoiced."""
        frame_len = int(sr * frame_ms / 1000)
        hop_len = int(sr * hop_ms / 1000)
        f0s = []
        min_lag = int(sr / 400)   # 400 Hz upper bound
        max_lag = int(sr / 60)    # 60 Hz lower bound
        for start in range(0, max(1, len(wav) - frame_len), hop_len):
            frame = wav[start:start + frame_len]
            if len(frame) < frame_len:
                break
            frame = frame - np.mean(frame)
            energy = np.sum(frame ** 2)
            if energy < 1e-6:
                f0s.append(0.0)
                continue
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            search = corr[min_lag:max_lag]
            if len(search) == 0:
                f0s.append(0.0)
                continue
            peak_lag = np.argmax(search) + min_lag
            if corr[peak_lag] / (corr[0] + 1e-9) < 0.3:
                f0s.append(0.0)  # unvoiced / too noisy
            else:
                f0s.append(sr / peak_lag)
        return np.array(f0s)

    def _spectral_flatness(self, wav: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(wav)) + 1e-12
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        return float(geo_mean / arith_mean)

    # ---- calibration constants -------------------------------------
    #
    # These were previously chosen against the synthetic sine-wave
    # signals in demo_cli.py, and were roughly an order of magnitude
    # away from what real speech measures. Measured over 128 chunks of
    # a 6.4-minute real 16 kHz recording, every sub-score came out at
    # exactly 0.000 for every chunk — the detector was inert on real
    # audio, not merely weak:
    #
    #     measurement        real speech (p10/median/p90)   old threshold
    #     f0 jitter          0.063 / 0.151 / 0.260          < 0.010
    #     spectral flatness  0.036 / 0.091 / 0.165          < 0.050
    #     envelope variance  0.0092 / 0.0185 / 0.0400       < 0.0020
    #
    # The constants below are anchored on that measured distribution:
    # each fires only once a measurement leaves the range real speech
    # actually occupies. See tests/test_detector_calibration.py.
    #
    # HONEST SCOPE: calibrated against one real speaker for the
    # genuine side, and Windows SAPI TTS for the synthetic side. That
    # establishes "does not false-positive on real speech" and "still
    # catches over-smoothed vocoder output". It does NOT establish
    # accuracy against modern neural voice clones — that needs
    # ASVspoof 2019/2021 LA or In-the-Wild data, or the transformer
    # backend. Do not quote an accuracy number off this baseline.

    # Real speech sits at p10=0.063 and up. Below 0.060 is flatter
    # pitch than a human sustains; 0.020 is vocoder-flat.
    JITTER_SUSPICIOUS = 0.060
    JITTER_SATURATED = 0.020

    # Two-sided, because the measurement contradicted the assumption
    # the old code encoded. Real speech occupies 0.017-0.220
    # (p01-p99). Classic vocoder output sits BELOW that (too tonal),
    # but the TTS measured here sat well ABOVE it (median 0.396,
    # i.e. 4x more spectrally flat than real speech, not less). So
    # this scores "outside the range real speech occupies", in either
    # direction, rather than assuming a direction.
    FLATNESS_TONAL = 0.020          # below this: unnaturally tonal
    FLATNESS_TONAL_SAT = 0.005
    FLATNESS_NOISY = 0.250          # above this: unnaturally noise-like
    FLATNESS_NOISY_SAT = 0.350

    # Real speech sits at p10=0.0092 and up.
    ENV_VAR_SUSPICIOUS = 0.008
    ENV_VAR_SATURATED = 0.002

    def predict_detailed(self, wav: np.ndarray, sr: int) -> dict:
        """Returns the combined spoof_prob PLUS each sub-score, so the
        fusion engine (and the demo dashboard) can show *why* a chunk
        was flagged instead of just a single opaque number."""
        if len(wav) < sr * 0.2:
            return {"spoof_prob": 0.0, "jitter_score": 0.0,
                    "flatness_score": 0.0, "smoothness_score": 0.0}

        f0 = self._f0_track(wav, sr)
        voiced = f0[f0 > 0]

        if len(voiced) < 3:
            # No stable pitch found at all — could be silence/noise;
            # don't flag it as spoof, just uninformative.
            jitter_score = 0.0
        else:
            # Natural speech: F0 wobbles frame-to-frame (jitter).
            # Suspiciously flat pitch -> higher spoof probability.
            diffs = np.abs(np.diff(voiced))
            jitter = np.mean(diffs) / (np.mean(voiced) + 1e-6)
            jitter_score = float(np.clip(
                (self.JITTER_SUSPICIOUS - jitter)
                / (self.JITTER_SUSPICIOUS - self.JITTER_SATURATED), 0, 1))

        # Spectral flatness outside the band real speech occupies, in
        # either direction (see FLATNESS_* above).
        flatness = self._spectral_flatness(wav)
        too_tonal = np.clip(
            (self.FLATNESS_TONAL - flatness)
            / (self.FLATNESS_TONAL - self.FLATNESS_TONAL_SAT), 0, 1)
        too_noisy = np.clip(
            (flatness - self.FLATNESS_NOISY)
            / (self.FLATNESS_NOISY_SAT - self.FLATNESS_NOISY), 0, 1)
        flatness_score = float(max(too_tonal, too_noisy))

        # Energy-envelope smoothness: natural speech has micro-variation
        # in short-time energy; some vocoders over-smooth it.
        frame = sr // 50
        if len(wav) > frame * 4:
            env = np.array([
                np.sum(wav[i:i + frame] ** 2)
                for i in range(0, len(wav) - frame, frame)
            ])
            env = env / (np.max(env) + 1e-9)
            env_var = float(np.var(np.diff(env)))
            smoothness_score = float(np.clip(
                (self.ENV_VAR_SUSPICIOUS - env_var)
                / (self.ENV_VAR_SUSPICIOUS - self.ENV_VAR_SATURATED), 0, 1))
        else:
            smoothness_score = 0.0

        spoof_prob = (
            0.55 * jitter_score
            + 0.20 * flatness_score
            + 0.25 * smoothness_score
        )
        return {
            "spoof_prob": float(np.clip(spoof_prob, 0, 1)),
            "jitter_score": jitter_score,
            "flatness_score": flatness_score,
            "smoothness_score": smoothness_score,
        }

    def predict(self, wav: np.ndarray, sr: int) -> float:
        """Returns spoof probability in [0, 1]. Higher = more likely
        synthetic/cloned. Kept for backward compatibility — prefer
        predict_detailed() when you need the sub-scores."""
        return self.predict_detailed(wav, sr)["spoof_prob"]


class TransformerDetector:
    """Real pretrained wav2vec2-based spoof/deepfake classifier.

    Default model: abhishtagatya/wav2vec2-base-960h-itw-deepfake
    (fine-tuned on the In-The-Wild deepfake-audio dataset; ~98% val
    accuracy per the model card). Swap MODEL_ID for a multilingual
    XLSR-based model when you get to the Indian-accent fine-tuning
    stage — that's the natural upgrade path for this project.

    Requires: pip install torch transformers
    Requires internet on FIRST call (downloads + caches weights).
    """

    MODEL_ID = os.environ.get(
        "SPOOF_MODEL_ID", "abhishtagatya/wav2vec2-base-960h-itw-deepfake"
    )
    name = f"transformer:{MODEL_ID}"

    def __init__(self):
        try:
            import torch
            from transformers import AutoConfig, Wav2Vec2FeatureExtractor, \
                Wav2Vec2ForSequenceClassification
        except ImportError as e:
            raise RuntimeError(
                "TransformerDetector needs `pip install torch transformers`. "
                "Use HeuristicDetector for a dependency-free run."
            ) from e

        self._torch = torch
        self.device = torch.device("cpu")  # CPU-only laptop target
        self.config = AutoConfig.from_pretrained(self.MODEL_ID)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.MODEL_ID, config=self.config
        ).to(self.device)
        self.model.eval()

    def predict(self, wav: np.ndarray, sr: int) -> float:
        target_sr = self.feature_extractor.sampling_rate
        if sr != target_sr:
            # simple linear resample; swap for torchaudio.resample if
            # you need higher quality
            duration = len(wav) / sr
            n_target = int(duration * target_sr)
            wav = np.interp(
                np.linspace(0, len(wav), n_target),
                np.arange(len(wav)),
                wav,
            )

        inputs = self.feature_extractor(
            wav, sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0]

        # Label order depends on the model's config.id2label — assume
        # index 1 is "spoof"/"fake"; verify against config on first run.
        id2label = self.model.config.id2label
        spoof_idx = 1
        for idx, label in id2label.items():
            if str(label).lower() in ("spoof", "fake", "1"):
                spoof_idx = int(idx)
                break
        return float(probs[spoof_idx])

    def predict_detailed(self, wav: np.ndarray, sr: int) -> dict:
        """Interface parity with HeuristicDetector. The transformer is
        a single end-to-end model, so there are no separate sub-scores
        — just spoof_prob three times over for a uniform schema."""
        p = self.predict(wav, sr)
        return {"spoof_prob": p, "jitter_score": p, "flatness_score": p, "smoothness_score": p}


def get_detector(backend: str = None):
    backend = backend or os.environ.get("DETECTOR_BACKEND", "heuristic")
    if backend == "transformer":
        return TransformerDetector()
    return HeuristicDetector()
