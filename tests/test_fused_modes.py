"""System-level behaviour of the two detection modes through the FULL
fused pipeline, on the committed fixture sets.

Heuristic-mode tests always run. Transformer-mode tests are skipped
unless the weights are already cached locally (no network in the
suite). Run with -s to see the tables.
"""

import os

import numpy as np
import pytest

from src.fusion import (weights_for_backend, FusionWeights,
                        TRANSFORMER_WEIGHTS, HEURISTIC_WEIGHTS)
from src.evaluation import evaluate_asvspoof, evaluate_itw, metrics, ASV, ITW

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(ASV, "manifest.json")) and os.path.exists(os.path.join(ITW, "meta.subset.csv"))),
    reason="fixtures not present")


def _transformer_cached():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    from src.detector import TransformerDetector
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.isdir(os.path.join(hub, "models--" + TransformerDetector.MODEL_ID.replace("/", "--")))


needs_transformer = pytest.mark.skipif(not _transformer_cached(), reason="transformer weights not cached")


# ---------------------------------------------------------- heuristic

@pytest.fixture(scope="module")
def heuristic():
    from src.detector import HeuristicDetector
    return HeuristicDetector()


def test_heuristic_mode_zero_fp_on_both_sets(heuristic):
    w = weights_for_backend(heuristic.backend)
    for res in (evaluate_asvspoof(heuristic, w), evaluate_itw(heuristic, w)):
        m = metrics(res)
        print(f"\n  heuristic / {res['dataset']}: genuine HIGH+ {m['genuine_fp']}/{m['n_genuine']}, "
              f"clones HIGH+ {m['spoof_caught']}/{m['n_spoof']}")
        assert m["genuine_fp"] == 0
        assert m["genuine_alert"] == 0


def test_zeroing_heuristic_acoustic_cannot_raise_genuine_risk(heuristic):
    """Removing a weight from a signal that is ~0 on bonafide must not
    make any genuine clip score higher. (It can lower or raise clone
    scores; that is measured, not asserted.)"""
    old = FusionWeights(0.45, 0.45, 0.10)
    a = evaluate_asvspoof(heuristic, old)
    b = evaluate_asvspoof(heuristic, HEURISTIC_WEIGHTS)
    changed = 0
    for ga, gb in zip(a["genuine"], b["genuine"]):
        assert ga["utt"] == gb["utt"]
        # speaker weight went 0.45 -> 0.90, so a nonzero mismatch doubles;
        # the check is that nothing crosses into HIGH, not that nothing moves
        assert gb["level"] not in ("HIGH", "CRITICAL")
        changed += abs(ga["peak"] - gb["peak"]) > 1e-9
    print(f"\n  genuine clips whose peak moved when acoustic weight -> 0: {changed}/{len(a['genuine'])}")


# -------------------------------------------------------- transformer

@pytest.fixture(scope="module")
def transformer():
    from src.detector import TransformerDetector
    return TransformerDetector()


@pytest.fixture(scope="module")
def t_asv(transformer):
    return evaluate_asvspoof(transformer, TRANSFORMER_WEIGHTS)


@pytest.fixture(scope="module")
def t_itw(transformer):
    return evaluate_itw(transformer, TRANSFORMER_WEIGHTS)


@needs_transformer
def test_transformer_mode_zero_fp_on_both_sets(transformer, t_asv, t_itw):
    assert weights_for_backend(transformer.backend) is TRANSFORMER_WEIGHTS
    for res in (t_asv, t_itw):
        m = metrics(res)
        print(f"\n  transformer / {res['dataset']}: genuine HIGH+ {m['genuine_fp']}/{m['n_genuine']} "
              f"(max peak {m['genuine_peak_max']:.3f}), clones HIGH+ {m['spoof_caught']}/{m['n_spoof']}")
        assert m["genuine_fp"] == 0
        assert m["genuine_alert"] == 0
        assert m["genuine_peak_max"] < 0.2


@needs_transformer
def test_transformer_mode_targeted_clone_recall(t_asv, t_itw):
    """THE threat model: a clone of the enrolled speaker, scored against
    that speaker's own voiceprint. Measured 7/12 on ASVspoof (A01, A02
    caught 2/2; A04 0/2) and 7/7 on In-the-Wild. Floors are set below
    the measurement so this guards against regression, not noise."""
    asv, itw = metrics(t_asv), metrics(t_itw)
    print(f"\n  targeted-clone HIGH+: ASVspoof {asv['spoof_caught']}/12  "
          + "  ".join(f"{k}:{v['caught']}/{v['n']}" for k, v in asv["per_system"].items())
          + f"   | In-the-Wild {itw['spoof_caught']}/7")
    assert asv["spoof_caught"] >= 6
    assert asv["per_system"]["A01"]["caught"] == 2
    assert asv["per_system"]["A02"]["caught"] == 2
    assert itw["spoof_caught"] >= 6


@needs_transformer
def test_transformer_mode_beats_heuristic_mode_on_clones(t_asv, heuristic):
    t = metrics(t_asv)["spoof_caught"]
    h = metrics(evaluate_asvspoof(heuristic, HEURISTIC_WEIGHTS))["spoof_caught"]
    print(f"\n  targeted clones caught: heuristic mode {h}/12, transformer mode {t}/12")
    assert t > h


@needs_transformer
def test_chunking_can_lose_a_whole_file_detection(transformer):
    """Documented, not fixed: LA_D_5685369 (A03) scores 0.948 whole-file
    but 0.119 on its first 3 s chunk. If a chunk-size change fixes this,
    the assertion below should be flipped and the README updated."""
    from src.audio_pipeline import load_mono
    wav, sr = load_mono(os.path.join(ASV, "flac", "LA_D_5685369.flac"))
    whole = transformer.predict(wav, sr)
    first_chunk = transformer.predict(wav[: 3 * sr], sr)
    print(f"\n  LA_D_5685369 whole-file {whole:.3f} vs first 3 s chunk {first_chunk:.3f}")
    assert whole > 0.9
    assert first_chunk < 0.5
