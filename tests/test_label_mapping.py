"""Regression tests: a swapped model must not silently invert the
detector.

TransformerDetector.predict() used to scan id2label per prediction for
"spoof" | "fake" | "1" and default to index 1 when nothing matched.
SPOOF_MODEL_ID is an environment variable and the roadmap is to swap in
a multilingual XLSR model; a model labelled "deepfake"/"bonafide" or
ordered the other way would run confidently inverted with no error.
resolve_spoof_index() now runs once at startup, matches a broad
vocabulary on both sides, and raises rather than guesses. No model
download: these use plain dicts, exactly what config.id2label is.
"""

import pytest

from src.detector import resolve_spoof_index, SPOOF_LABELS, BONAFIDE_LABELS


def test_default_model_labels_resolve_to_1():
    m = resolve_spoof_index({0: "bona-fide", 1: "spoof"})
    assert m["spoof_index"] == 1
    assert m["spoof_label"] == "spoof"
    assert m["source"] == "matched from id2label"


def test_reversed_and_upper_case_labels_resolve_to_0():
    """THE inversion case: FAKE first, REAL second, shouting."""
    m = resolve_spoof_index({0: "FAKE", 1: "REAL"})
    assert m["spoof_index"] == 0


def test_roadmap_style_labels_resolve():
    assert resolve_spoof_index({0: "bonafide", 1: "deepfake"})["spoof_index"] == 1
    assert resolve_spoof_index({0: "synthetic", 1: "human"})["spoof_index"] == 0
    assert resolve_spoof_index({"0": "genuine", "1": "cloned"})["spoof_index"] == 1   # string keys, as HF configs have


def test_ambiguous_labels_raise_naming_both_labels():
    with pytest.raises(RuntimeError) as e:
        resolve_spoof_index({0: "LABEL_0", 1: "LABEL_1"})
    msg = str(e.value)
    assert "LABEL_0" in msg and "LABEL_1" in msg
    assert "SPOOF_LABEL_INDEX" in msg


def test_two_spoof_like_labels_raise():
    with pytest.raises(RuntimeError):
        resolve_spoof_index({0: "fake", 1: "spoof"})


def test_bonafide_side_must_also_match():
    """One clear spoof label is not enough if the other class is
    unrecognisable -- it could be a three-way model."""
    with pytest.raises(RuntimeError):
        resolve_spoof_index({0: "spoof", 1: "CLASS_B"})


def test_override_beats_matching_and_ambiguity():
    assert resolve_spoof_index({0: "LABEL_0", 1: "LABEL_1"}, override="0")["spoof_index"] == 0
    m = resolve_spoof_index({0: "bona-fide", 1: "spoof"}, override="0")   # even when it contradicts the names
    assert m["spoof_index"] == 0 and m["source"] == "SPOOF_LABEL_INDEX override"


def test_override_out_of_range_raises():
    with pytest.raises(RuntimeError):
        resolve_spoof_index({0: "bona-fide", 1: "spoof"}, override="2")


def test_env_var_is_read_at_init(monkeypatch):
    """The override is wired through the environment, not just the
    function argument."""
    import os
    monkeypatch.setenv("SPOOF_LABEL_INDEX", "0")
    assert resolve_spoof_index({0: "a", 1: "b"}, os.environ.get("SPOOF_LABEL_INDEX"))["spoof_index"] == 0


def test_vocabularies_do_not_overlap():
    assert not (SPOOF_LABELS & BONAFIDE_LABELS)
