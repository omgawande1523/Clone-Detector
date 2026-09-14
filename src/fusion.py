"""
fusion.py — combines three independent signals into one risk score.

Why fusion instead of one model: each signal fails differently.
- Acoustic/prosody detector: fooled by a clone clean enough to pass
  as "not obviously synthetic".
- Speaker-consistency check: fooled by a genuine call from someone
  simply not yet enrolled (no reference to compare against).
- Context score: fooled by a scripted, patient scam that avoids
  obvious keywords and calls during business hours.
A real attack usually cannot be clean on all three at once — that is
the actual argument for an ensemble, not just "more signals = better".

WEIGHTING RULE (this is the part that used to be broken)

    The old weights were acoustic 0.55 / speaker 0.30 / context 0.15
    against an alert threshold of 0.60. Because the weights form a
    weighted *average*, a signal can never contribute more than its
    own weight — so acoustic maxed out reached only 0.55, and speaker
    plus context maxed out together reached only 0.45. Neither could
    ever raise an alert. The system advertised three signals but in
    practice required all of them at once.

    The rule now, asserted by tests and by check_alertable() below:

      * acoustic alone, maxed, MUST be able to raise an alert.
        A confidently-synthetic voice is an alert on its own merits.
      * speaker mismatch alone, maxed, MUST be able to raise an alert.
        "This is definitively not the enrolled person" is an alert on
        its own merits too.
      * context alone must NOT be able to raise an alert. An
        out-of-hours call from an unknown number that says "urgent
        transfer" is a suspicious call, but it is not evidence that
        anyone's voice was cloned, and firing a voice-clone alert on
        metadata alone would be the fastest way to lose a fraud
        team's trust in the system. Context escalates; it does not
        accuse.

    Consequently each of acoustic and speaker carries weight >= the
    alert threshold, and context carries weight < it.

Fixed weights are a reasonable starting point for a demo. The real
upgrade path once you have labeled call data: replace this with a
small trained meta-classifier (logistic regression over the three
sub-scores) — same three inputs, learned weights instead of chosen
ones. Any such retrain must preserve the rule above.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionWeights:
    acoustic: float = 0.45
    speaker: float = 0.45
    context: float = 0.10


# ---------------------------------------------------------------------
# BACKEND-AWARE WEIGHTS
#
# The acoustic signal's weight depends on which detector backend is
# loaded, and is selected automatically from the detector — it is not
# a flag a user sets, so it cannot be misconfigured.
#
# TRANSFORMER backend: the acoustic signal is validated. Measured on
# the committed fixtures (tests/test_transformer_backend.py):
#     ASVspoof 2019 LA, cross-domain   AUC 0.81, 0/48 false positives
#     In-the-Wild, in-domain           AUC 1.00, 0/7  false positives
# It earns an equal share with the speaker layer.
TRANSFORMER_WEIGHTS = FusionWeights(acoustic=0.45, speaker=0.45, context=0.10)

# HEURISTIC backend: the acoustic weight is EXACTLY ZERO. Not small.
# Zero. Do not "helpfully" restore 0.05 or 0.10 on the assumption that
# zero was an oversight. Measured on the same labelled fixtures:
#     ASVspoof 2019 LA   AUC 0.25   recall 0/12
#     In-the-Wild        AUC 0.14   recall 0/7
# An AUC below 0.5 means the score runs slightly AGAINST the label —
# a spoof is marginally more likely to score lower than a bonafide.
# Any positive weight on that signal makes the fused score worse, not
# just no better. Its only remaining job is the demo-only over-smoothed
# vocoder case, and it is still computed and shown in the UI for that;
# it is simply not allowed to move the risk score.
HEURISTIC_WEIGHTS = FusionWeights(acoustic=0.0, speaker=0.90, context=0.10)

_WEIGHTS_BY_BACKEND = {
    "transformer": TRANSFORMER_WEIGHTS,
    "heuristic": HEURISTIC_WEIGHTS,
}


def weights_for_backend(backend: str) -> FusionWeights:
    """`backend` is the detector's `.backend` attribute ("heuristic" or
    "transformer"). Unknown values raise rather than silently falling
    back — a new backend must be measured before it gets a weight."""
    try:
        return _WEIGHTS_BY_BACKEND[backend]
    except KeyError:
        raise ValueError(
            f"no fusion weights defined for backend {backend!r}; "
            f"measure it and add it to fusion._WEIGHTS_BY_BACKEND"
        )


def detection_mode(backend: str) -> dict:
    """Everything the API and UI need to say, unambiguously, which mode
    a result came from."""
    w = weights_for_backend(backend)
    if backend == "transformer":
        return {
            "mode": "transformer",
            "label": "Transformer (full signal set)",
            "acoustic_enabled": True,
            "security_posture": "full",
            "description": (
                "Acoustic analysis uses the wav2vec2 spoof classifier and "
                "carries full weight alongside the speaker-consistency check."
            ),
            "weights": {"acoustic": w.acoustic, "speaker": w.speaker, "context": w.context},
        }
    return {
        "mode": "heuristic",
        "label": "Heuristic-only (acoustic signal disabled)",
        "acoustic_enabled": False,
        "security_posture": "reduced",
        "description": (
            "REDUCED SECURITY POSTURE — demo/offline use only. The heuristic "
            "acoustic detector measured at or below chance on labelled spoof "
            "data, so its weight is zero; risk comes from the speaker-consistency "
            "check and call context only. This mode has no defence against a "
            "clone of the enrolled speaker."
        ),
        "weights": {"acoustic": w.acoustic, "speaker": w.speaker, "context": w.context},
    }


def fuse(acoustic_score: float, speaker_mismatch: float, context_score: float,
         weights: FusionWeights = TRANSFORMER_WEIGHTS) -> dict:
    fused = (
        weights.acoustic * acoustic_score
        + weights.speaker * speaker_mismatch
        + weights.context * context_score
    )
    fused = float(min(max(fused, 0.0), 1.0))

    # Identify the dominant contributor for an explainable alert message.
    contributions = {
        "acoustic": weights.acoustic * acoustic_score,
        "speaker": weights.speaker * speaker_mismatch,
        "context": weights.context * context_score,
    }
    dominant = max(contributions, key=contributions.get)

    return {
        "fused_score": fused,
        "acoustic_score": acoustic_score,
        "speaker_mismatch": speaker_mismatch,
        "context_score": context_score,
        "dominant_signal": dominant,
    }


def check_alertable(alert_threshold: float,
                    weights: FusionWeights = TRANSFORMER_WEIGHTS) -> dict:
    """Reports whether each signal can reach the alert threshold on its
    own. Exists so the weight/threshold mismatch that made this system
    unable to fire cannot silently come back — tests assert on it, and
    the API exposes it at /health so a misconfiguration is visible
    before a demo rather than during one.

    Under HEURISTIC_WEIGHTS acoustic_can_alert is False by design (its
    weight is zero); speaker_can_alert must be True in every mode."""
    return {
        "acoustic_alone_max": weights.acoustic,
        "speaker_alone_max": weights.speaker,
        "context_alone_max": weights.context,
        "alert_threshold": alert_threshold,
        "acoustic_can_alert": weights.acoustic >= alert_threshold,
        "speaker_can_alert": weights.speaker >= alert_threshold,
        # Intentionally False — see the weighting rule above.
        "context_can_alert": weights.context >= alert_threshold,
    }
