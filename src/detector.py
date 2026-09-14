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

from .audio_pipeline import resample_to


class HeuristicDetector:
    """Offline, dependency-light anti-spoofing baseline."""

    name = "heuristic-signal-baseline"
    backend = "heuristic"     # selects fusion weights -- see fusion.weights_for_backend

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
    # Calibrated against ASVspoof 2019 LA (dev partition, labels from
    # the official protocol file): 48 bonafide utterances from 8
    # speakers and 12 spoofed utterances across all six attack systems
    # A01-A06. See tests/test_detector_calibration.py and
    # fixtures/asvspoof2019/README.md.
    #
    # READ THIS BEFORE QUOTING ANY NUMBER OFF THIS DETECTOR.
    #
    # On that labelled data the three features barely separate the
    # classes. P(a spoof looks more spoof-like than a random bonafide):
    #
    #     f0 jitter            0.54   (coin flip)
    #     spectral flatness    0.64
    #     envelope variance    0.29 in the direction the old code
    #                          assumed -- i.e. BACKWARDS; 0.71 reversed
    #
    # The strongest achievable operating point on this set is roughly
    # a third of spoofs caught at zero false positives, with the median
    # spoof scoring about the same as the median bonafide. That is not
    # a detector, it is a weak prior. It is kept because (a) it needs
    # no model download, (b) it still catches the crude over-smoothed
    # vocoder case, and (c) the fusion design lets the speaker layer
    # carry the load. The accuracy path is the TransformerDetector.
    #
    # What the constants below DO guarantee: genuine speech scores
    # ~0. Every threshold sits outside the observed bonafide range
    # with margin, so a real caller does not get flagged by this
    # layer. That property, not recall, is what they are tuned for.
    #
    # Two earlier calibrations are worth remembering as cautionary
    # tales. Thresholds tuned on synthetic sine waves made the detector
    # emit exactly 0.000 on all real audio. Thresholds then tuned on a
    # single band-limited call recording (spectral flatness ~0.09)
    # fired on 100% of studio-quality bonafide speech (flatness ~0.39),
    # because flatness tracks recording conditions far more than it
    # tracks synthesis. Flatness is therefore demoted to an extreme
    # outlier check only.

    # Bonafide f0 jitter: min 0.076, p05 0.083, median 0.115. Below
    # 0.070 is flatter pitch than any bonafide sample sustained; the
    # only spoofs caught at zero FP were A01/A03 in this range.
    JITTER_SUSPICIOUS = 0.070
    JITTER_SATURATED = 0.035

    # Bonafide spectral flatness spanned 0.12-0.57 on studio speech
    # and ~0.09 on a call recording. Nothing inside that span is
    # evidence of anything. Only fire on values no speech recording
    # produces: near-pure-tone or near-white-noise.
    FLATNESS_TONAL = 0.010
    FLATNESS_TONAL_SAT = 0.003
    FLATNESS_NOISY = 0.700
    FLATNESS_NOISY_SAT = 0.850

    # Envelope variance. ASVspoof bonafide: min 0.0028, max 0.0277; a
    # separate call recording ran to 0.054 in normal speech. The lower
    # bound is the classic over-smoothed-vocoder cue and is safely
    # below both. An upper bound anchored on the ASVspoof max fired on
    # ~15% of that other recording while catching one spoof in twelve,
    # so it is set far out — a guard against non-speech input, not a
    # spoof cue.
    ENV_VAR_SUSPICIOUS = 0.0025      # below: over-smoothed
    ENV_VAR_SATURATED = 0.0008
    ENV_VAR_EXCESS = 0.080           # above: not speech-like at all
    ENV_VAR_EXCESS_SAT = 0.120

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
            too_smooth = np.clip(
                (self.ENV_VAR_SUSPICIOUS - env_var)
                / (self.ENV_VAR_SUSPICIOUS - self.ENV_VAR_SATURATED), 0, 1)
            too_erratic = np.clip(
                (env_var - self.ENV_VAR_EXCESS)
                / (self.ENV_VAR_EXCESS_SAT - self.ENV_VAR_EXCESS), 0, 1)
            smoothness_score = float(max(too_smooth, too_erratic))
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


# Label vocabulary for resolving which output class means "spoof".
# Matched case-insensitively against the model's config.id2label.
SPOOF_LABELS = {"spoof", "fake", "deepfake", "synthetic", "cloned", "clone", "tts", "spoofed", "generated"}
BONAFIDE_LABELS = {"bonafide", "bona-fide", "bona_fide", "real", "genuine", "human", "original", "authentic", "natural"}


def resolve_spoof_index(id2label: dict, override=None) -> dict:
    """Decide, ONCE, which class index is the spoof class.

    The old code scanned id2label per prediction for "spoof" | "fake" |
    "1" and silently fell back to index 1. SPOOF_MODEL_ID is an
    environment variable and the roadmap is to swap in a multilingual
    XLSR model; if that model labels its classes "deepfake"/"bonafide"
    or orders them the other way, the detector would run confidently
    inverted with no error anywhere.

    Rules: exactly one class must match the spoof vocabulary and every
    other class must match the bonafide vocabulary. Anything else raises
    at startup naming the model's actual labels. SPOOF_LABEL_INDEX (or
    `override`) bypasses matching entirely. Never guess.
    """
    labels = {int(k): str(v) for k, v in dict(id2label).items()}
    if override is not None and str(override).strip() != "":
        idx = int(override)
        if idx not in labels:
            raise RuntimeError(
                f"SPOOF_LABEL_INDEX={override} is not a class of this model; id2label is {labels}")
        return {"spoof_index": idx, "spoof_label": labels[idx], "id2label": labels, "source": "SPOOF_LABEL_INDEX override"}
    norm = {i: lab.strip().lower().replace(" ", "") for i, lab in labels.items()}
    spoof = [i for i, lab in norm.items() if lab in SPOOF_LABELS]
    bona = [i for i, lab in norm.items() if lab in BONAFIDE_LABELS]
    if len(spoof) == 1 and len(bona) == len(labels) - 1:
        return {"spoof_index": spoof[0], "spoof_label": labels[spoof[0]], "id2label": labels, "source": "matched from id2label"}
    raise RuntimeError(
        f"Cannot tell which output class of this model means 'spoof'. Its id2label is {labels}; "
        f"recognised spoof-side names are {sorted(SPOOF_LABELS)} and bonafide-side names are "
        f"{sorted(BONAFIDE_LABELS)}. Set SPOOF_LABEL_INDEX=<index> explicitly. Refusing to guess, "
        f"because a wrong guess runs the detector inverted with no error."
    )


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
    backend = "transformer"   # selects fusion weights -- see fusion.weights_for_backend

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
        # Resolve the label mapping once, here, and fail loudly if it is
        # ambiguous -- see resolve_spoof_index().
        self.label_mapping = resolve_spoof_index(
            self.model.config.id2label, os.environ.get("SPOOF_LABEL_INDEX"))
        self.spoof_index = self.label_mapping["spoof_index"]

    def predict(self, wav: np.ndarray, sr: int) -> float:
        target_sr = self.feature_extractor.sampling_rate
        if sr != target_sr:
            # anti-aliased polyphase resample -- see audio_pipeline.resample_to
            wav = resample_to(wav, sr, target_sr)

        inputs = self.feature_extractor(
            wav, sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0]

        return float(probs[self.spoof_index])

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
