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


@dataclass
class FusionWeights:
    acoustic: float = 0.45
    speaker: float = 0.45
    context: float = 0.10


def fuse(acoustic_score: float, speaker_mismatch: float, context_score: float,
         weights: FusionWeights = FusionWeights()) -> dict:
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
                    weights: FusionWeights = FusionWeights()) -> dict:
    """Reports whether each signal can reach the alert threshold on its
    own. Exists so the weight/threshold mismatch that made this system
    unable to fire cannot silently come back — tests assert on it, and
    the API exposes it at /health so a misconfiguration is visible
    before a demo rather than during one."""
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
