"""False-positive batch check on real, labelled, multi-speaker audio,
through the FULL fused pipeline in HEURISTIC mode (the default, no
model download), with that mode's own weights (acoustic 0 / speaker
0.90 / context 0.10).

Data: ASVspoof 2019 LA dev; labels from the official protocol file
only. Batch construction is in src/evaluation.py: per speaker, 4
bonafide utterances joined for enrolment, 2 held out and scored
individually and as one joined ~6 s "call". Context is neutral.

Known limitation, stated rather than hidden: single ASVspoof
utterances are one 3 s chunk, so the sustain rule means an ALERT can
never fire on one; the per-chunk LEVEL still can, and that is what the
per-utterance rows are graded on. The joined calls exercise the full
alert path.

Run with -s to print the per-clip tables.
"""

import os

import numpy as np
import pytest

from src.detector import HeuristicDetector
from src.fusion import weights_for_backend
from src.risk_engine import ALERT_THRESHOLD
from src.evaluation import evaluate_asvspoof, metrics, print_table, ASV

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ASV, "manifest.json")), reason="ASVspoof fixtures not present")


@pytest.fixture(scope="module")
def batch():
    det = HeuristicDetector()
    return evaluate_asvspoof(det, weights_for_backend(det.backend))


def test_batch_uses_several_distinct_real_speakers(batch):
    assert len(batch["speakers"]) >= 5
    assert len(batch["genuine"]) >= 5


def test_no_genuine_utterance_reaches_high_or_critical(batch):
    """THE false-positive check, per utterance."""
    print_table(f"GENUINE UTTERANCES — heuristic mode, neutral context, alert threshold {ALERT_THRESHOLD}",
                batch["genuine"])
    m = metrics(batch)
    bad = [g for g in batch["genuine"] if g["level"] in ("HIGH", "CRITICAL")]
    print(f"\n  genuine utterances reaching HIGH/CRITICAL: {m['genuine_fp']}/{m['n_genuine']}  "
          f"(false-positive rate {m['genuine_fp']/m['n_genuine']:.0%}); max peak risk {m['genuine_peak_max']:.3f}")
    for g in bad:
        worst = max(g["rows"], key=lambda r: r["risk"])
        print(f"    FALSE POSITIVE {g['utt']} ({g['speaker']}): peak {g['peak']:.3f} {g['level']} "
              f"driven by {worst['dominant']} [acoustic {worst['acoustic']:.2f} speaker {worst['speaker']:.2f}]")
    assert not bad, [g["utt"] for g in bad]


def test_no_genuine_call_raises_an_alert(batch):
    print_table("GENUINE CALLS (each speaker's two test utterances joined)", batch["calls"])
    m = metrics(batch)
    print(f"\n  genuine calls raising an ALERT: {m['calls_alert']}/{m['n_calls']}   "
          f"reaching HIGH/CRITICAL: {m['calls_fp']}/{m['n_calls']}")
    assert m["calls_alert"] == 0
    assert m["calls_fp"] == 0


def test_speaker_layer_actually_scored_most_genuine_clips(batch):
    """A 0% FP rate is meaningless if the speaker layer abstained on
    everything. Some single utterances legitimately fall under
    MIN_SPEECH_SECONDS; require that most were scored."""
    scored = sum(g["speaker_scored"] for g in batch["genuine"])
    scored_calls = sum(c["speaker_scored"] for c in batch["calls"])
    print(f"\n  speaker layer had enough speech to score: {scored}/{len(batch['genuine'])} utterances, "
          f"{scored_calls}/{len(batch['calls'])} calls")
    assert scored >= len(batch["genuine"]) // 2
    assert scored_calls >= (3 * len(batch["calls"])) // 4


def test_targeted_clones_in_heuristic_mode_for_information(batch):
    """Not asserted — printed. In heuristic mode the acoustic weight is
    zero and the speaker layer scores a targeted clone at ~0.04
    mismatch, so this is expected to be 0/12. The README says so."""
    print_table("TARGETED CLONES (spoof of the enrolled speaker, vs that speaker's print) — heuristic mode",
                batch["spoof"], with_system=True)
    m = metrics(batch)
    print(f"\n  clones reaching HIGH/CRITICAL: {m['spoof_caught']}/{m['n_spoof']}   "
          f"median speaker-mismatch {np.median([s['worst_speaker'] for s in batch['spoof']]):.3f}")
