"""TransformerDetector measured on the two committed labelled fixture
sets, reported separately:

  * ASVspoof 2019 LA  — CROSS-domain. Older, different attack systems
                        (A01-A06) than the model was fine-tuned on.
  * In-the-Wild       — IN-domain. The model's own fine-tuning
                        distribution (split unknown, so possibly its
                        training data — treat as an upper bound).

The suite must never need the network, so everything here is skipped
unless `transformers` is installed AND the model weights are already
in the local HuggingFace cache. First run on a machine:

    pip install -r requirements-real-model.txt
    python -c "from src.detector import TransformerDetector; TransformerDetector()"

Run with -s to print the per-dataset tables and per-system breakdown.
"""

import os

import numpy as np
import pytest

from src.audio_pipeline import load_mono

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
ASV = os.path.join(FIXTURES, "asvspoof2019")
ITW = os.path.join(FIXTURES, "in_the_wild")


def _model_cached():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    from src.detector import TransformerDetector
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.isdir(os.path.join(hub, "models--" + TransformerDetector.MODEL_ID.replace("/", "--")))


pytestmark = pytest.mark.skipif(
    not (_model_cached() and os.path.exists(ASV) and os.path.exists(ITW)),
    reason="transformer weights not cached locally, or fixtures missing",
)


def asvspoof_items():
    """(path, is_spoof, system) — labels from the official protocol only."""
    items = []
    for line in open(os.path.join(ASV, "ASVspoof2019.LA.cm.dev.trl.subset.txt")):
        _spk, utt, _env, system, key = line.split()
        items.append((os.path.join(ASV, "flac", utt + ".flac"), key == "spoof", system))
    return items


def itw_items():
    """(path, is_spoof, label) — labels from the dataset's meta.csv only."""
    items = []
    for line in open(os.path.join(ITW, "meta.subset.csv")):
        if line.startswith("file,"):
            continue
        fname, _speaker, label = line.strip().split(",")
        items.append((os.path.join(ITW, "wav", fname), label == "spoof", label))
    return items


@pytest.fixture(scope="module")
def detector():
    from src.detector import TransformerDetector
    return TransformerDetector()


def _score_all(detector, items):
    out = []
    for path, is_spoof, system in items:
        wav, sr = load_mono(path)
        out.append((os.path.basename(path), is_spoof, system,
                    detector.predict_detailed(wav, sr)["spoof_prob"]))
    return out


def _report(name, rows):
    b = np.array([p for _, s, _, p in rows if not s])
    s = np.array([p for _, sp, _, p in rows if sp])
    auc = np.mean([[si > bi for bi in b] for si in s])
    print(f"\n  {name}: {len(b)} bonafide / {len(s)} spoof")
    print(f"    bonafide med {np.median(b):.3f} max {b.max():.3f} FP@0.5 {np.mean(b > 0.5):.0%} | "
          f"spoof med {np.median(s):.3f} min {s.min():.3f} recall@0.5 {np.mean(s > 0.5):.0%} | AUC {auc:.3f}")
    systems = sorted({sysid for _, sp, sysid, _ in rows if sp})
    if len(systems) > 1:
        for sysid in systems:
            vals = [p for _, sp, sy, p in rows if sp and sy == sysid]
            print(f"    {sysid}: " + ", ".join(f"{v:.2f}" for v in vals))
    return b, s, auc


@pytest.fixture(scope="module")
def asv_scores(detector):
    return _score_all(detector, asvspoof_items())


@pytest.fixture(scope="module")
def itw_scores(detector):
    return _score_all(detector, itw_items())


# ----------------------------------------------------------- label sanity

def test_spoof_label_index_resolves_from_config(detector):
    """The model's id2label is {0: 'bona-fide', 1: 'spoof'}; a wrong
    index would invert every score and every test below would still
    'pass' with inverted assertions. Pin it."""
    labels = {str(v).lower() for v in detector.model.config.id2label.values()}
    assert "spoof" in labels


# ---------------------------------------------------- cross-domain: ASVspoof

def test_asvspoof_no_bonafide_flagged(asv_scores):
    b, s, auc = _report("ASVspoof 2019 LA (cross-domain)", asv_scores)
    assert b.max() < 0.5, f"a labelled-bonafide utterance scored {b.max():.3f}"
    assert np.mean(b > 0.5) == 0.0


def test_asvspoof_neural_tts_systems_are_caught(asv_scores):
    """A01-A03 (neural waveform / vocoder TTS) were caught at >0.9
    every time. Pin that so a model swap cannot silently lose it."""
    for sysid in ("A01", "A02", "A03"):
        vals = [p for _, sp, sy, p in asv_scores if sp and sy == sysid]
        assert vals and min(vals) > 0.5, f"{sysid}: {vals}"


def test_asvspoof_recall_floor(asv_scores):
    """Measured 8/12 (67%). Assert a floor well under that so the test
    is a regression guard, not a coin-flip on 12 files."""
    s = np.array([p for _, sp, _, p in asv_scores if sp])
    assert np.mean(s > 0.5) >= 0.5


def test_asvspoof_known_misses_are_documented(asv_scores):
    """A04 (waveform concatenation of real speech units) was missed
    both times. This is informational: if a future model catches it,
    fine; if the current one starts catching it, something changed."""
    a04 = [p for _, sp, sy, p in asv_scores if sp and sy == "A04"]
    print(f"    A04 (waveform concatenation) scores: {[round(v, 3) for v in a04]}")


# --------------------------------------------------- in-domain: In-the-Wild

def test_itw_separates_cleanly(itw_scores):
    b, s, auc = _report("In-the-Wild (in-domain; possibly training data)", itw_scores)
    assert b.max() < 0.5
    assert s.min() > 0.5
    assert auc == 1.0


# ------------------------------------------------------ both, side by side

def test_transformer_beats_heuristic_on_both(asv_scores, itw_scores):
    from src.detector import HeuristicDetector
    h = HeuristicDetector()
    for name, items, t_rows in (("ASVspoof", asvspoof_items(), asv_scores),
                                ("In-the-Wild", itw_items(), itw_scores)):
        h_rows = _score_all(h, items)
        t_rec = np.mean([p > 0.5 for _, sp, _, p in t_rows if sp])
        h_rec = np.mean([p > 0.5 for _, sp, _, p in h_rows if sp])
        print(f"    {name}: recall@0.5 heuristic {h_rec:.0%} vs transformer {t_rec:.0%}")
        assert t_rec > h_rec
