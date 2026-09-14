"""
fusion.py — combines three independent signals into one risk score.

Why fusion instead of one model: each signal fails differently.
- Acoustic/prosody detector: fooled by a clone clean enough to pass
  as "not obviously synthetic".
- Speaker-consistency check: fooled by a genuine call from someone
  simply not yet enrolled (no reference to compare against), and by a
  good clone OF the enrolled person, which it is designed to match.
- Context score: fooled by a scripted, patient scam that avoids
  obvious keywords and calls during business hours.
A real attack usually cannot be clean on all three at once — that is
the actual argument for an ensemble, not just "more signals = better".

COMBINATION RULE: NOISY-OR OVER CAPPED EVIDENCE

    contribution_i = cap_i x score_i
    fused          = 1 - PRODUCT(1 - contribution_i)

Each signal's cap is the most fused risk that signal can produce ON
ITS OWN. Noisy-OR treats the signals as independent pieces of evidence
for the same conclusion: any one of them, if confident, is enough, and
several agreeing compound rather than average out.

    THIS REPLACED A WEIGHTED AVERAGE, AND THAT IS THE POINT.

    A weighted mean structurally caps a single signal at its own
    weight. With acoustic weighted 0.45, a detector that was 100%
    certain the audio was synthetic produced a fused score of 0.448
    against a 0.40 alert threshold — 12% of headroom. On a captured
    live call (fixtures/live_clone_session/) the transformer scored
    four consecutive windows 0.984-0.996 synthetic and the system
    never even reached HIGH, because the EMA approached that 0.448
    asymptote too slowly to cross. That is not a tuning error; no
    choice of weights fixes it while the rule is a mean.

    It matters specifically for THE threat this project exists for. A
    clone of the enrolled person passes the voiceprint check by
    design (measured median mismatch 0.00), so on that attack the
    acoustic signal is the ONLY live evidence. A rule that cannot let
    one signal speak alone cannot detect it.

    Under noisy-OR the same call reaches 1 - (1 - 0.90 x 0.996) = 0.896.

    Earlier cautionary history, kept: the weights were once
    0.55/0.30/0.15 against a threshold of 0.60, which made it
    arithmetically impossible for ANY single signal to raise an alert.
    Both bugs are the same shape — the combination rule silently
    deciding no single signal is ever enough.

THE INVARIANTS, asserted by tests and by check_alertable():

  * acoustic alone, maxed, MUST be able to raise an alert.
    A confidently-synthetic voice is an alert on its own merits.
  * speaker mismatch alone, maxed, MUST be able to raise an alert.
    "This is definitively not the enrolled person" is an alert too.
  * context alone must NOT be able to raise an alert. An out-of-hours
    call from an unknown number saying "urgent transfer" is a
    suspicious call, but it is not evidence that anyone's voice was
    cloned, and firing a voice-clone alert on metadata alone would be
    the fastest way to lose a fraud team's trust. Context escalates;
    it does not accuse.

Consequently each of acoustic and speaker carries a cap >= the alert
threshold, and context carries a cap < it.

Fixed caps are a reasonable starting point. The upgrade path once you
have labeled call data: fit the per-signal caps (or a small trained
meta-model) to real outcomes. Any such retrain must preserve the
invariants above.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalCaps:
    """The most fused risk each signal can produce on its own.

    Named caps, not weights: they do not have to sum to 1, and under
    noisy-OR a single maxed signal reaches exactly its own cap.
    """
    acoustic: float = 0.90
    speaker: float = 0.90
    context: float = 0.30


# ---------------------------------------------------------------------
# BACKEND-AWARE CAPS
#
# The acoustic cap depends on which detector backend is loaded, and is
# selected automatically from the detector — it is not a flag a user
# sets, so it cannot be misconfigured.
#
# TRANSFORMER backend: the acoustic signal is validated. Measured on
# the committed fixtures (tests/test_transformer_backend.py):
#     ASVspoof 2019 LA, cross-domain   AUC 0.81, 0/48 false positives
#     In-the-Wild, in-domain           AUC 1.00, 0/7  false positives
# It gets a full cap, equal to the speaker layer's.
TRANSFORMER_CAPS = SignalCaps(acoustic=0.90, speaker=0.90, context=0.30)

# HEURISTIC backend: the acoustic cap is EXACTLY ZERO. Not small.
# Zero. Do not "helpfully" restore 0.05 or 0.10 on the assumption that
# zero was an oversight. Measured on the same labelled fixtures:
#     ASVspoof 2019 LA   AUC 0.25   recall 0/12
#     In-the-Wild        AUC 0.14   recall 0/7
# An AUC below 0.5 means the score runs slightly AGAINST the label —
# a spoof is marginally more likely to score lower than a bonafide.
# Any positive cap on that signal makes the fused score worse, not
# just no better. Its only remaining job is the demo-only over-smoothed
# vocoder case, and it is still computed and shown in the UI for that;
# it is simply not allowed to move the risk score.
HEURISTIC_CAPS = SignalCaps(acoustic=0.0, speaker=0.90, context=0.30)

_CAPS_BY_BACKEND = {
    "transformer": TRANSFORMER_CAPS,
    "heuristic": HEURISTIC_CAPS,
}


def caps_for_backend(backend: str) -> SignalCaps:
    """`backend` is the detector's `.backend` attribute ("heuristic" or
    "transformer"). Unknown values raise rather than silently falling
    back — a new backend must be measured before it gets a cap."""
    try:
        return _CAPS_BY_BACKEND[backend]
    except KeyError:
        raise ValueError(
            f"no fusion caps defined for backend {backend!r}; "
            f"measure it and add it to fusion._CAPS_BY_BACKEND"
        )


def detection_mode(backend: str) -> dict:
    """Everything the API and UI need to say, unambiguously, which mode
    a result came from."""
    c = caps_for_backend(backend)
    caps = {"acoustic": c.acoustic, "speaker": c.speaker, "context": c.context}
    if backend == "transformer":
        return {
            "mode": "transformer",
            "label": "Transformer (full signal set)",
            "acoustic_enabled": True,
            "security_posture": "full",
            "description": (
                "Acoustic analysis uses the wav2vec2 spoof classifier and can "
                "raise an alert on its own, alongside the speaker-consistency check."
            ),
            "caps": caps,
            "weights": caps,          # back-compat key for older clients
        }
    return {
        "mode": "heuristic",
        "label": "Heuristic-only (acoustic signal disabled)",
        "acoustic_enabled": False,
        "security_posture": "reduced",
        "description": (
            "REDUCED SECURITY POSTURE — demo/offline use only. The heuristic "
            "acoustic detector measured at or below chance on labelled spoof "
            "data, so its cap is zero; risk comes from the speaker-consistency "
            "check and call context only. This mode has no defence against a "
            "clone of the enrolled speaker."
        ),
        "caps": caps,
        "weights": caps,
    }


def fuse(acoustic_score: float, speaker_mismatch: float, context_score: float,
         caps: SignalCaps = TRANSFORMER_CAPS) -> dict:
    """Noisy-OR over capped evidence. See the module docstring."""
    contributions = {
        "acoustic": caps.acoustic * _clamp(acoustic_score),
        "speaker": caps.speaker * _clamp(speaker_mismatch),
        "context": caps.context * _clamp(context_score),
    }
    product = 1.0
    for c in contributions.values():
        product *= (1.0 - c)
    fused = float(min(max(1.0 - product, 0.0), 1.0))

    # Largest single contribution, for an explainable alert message.
    dominant = max(contributions, key=contributions.get)

    return {
        "fused_score": fused,
        "acoustic_score": acoustic_score,
        "speaker_mismatch": speaker_mismatch,
        "context_score": context_score,
        "dominant_signal": dominant,
        "contributions": {k: round(v, 4) for k, v in contributions.items()},
    }


def _clamp(x) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def reachable_max(caps: SignalCaps, speaker_enrolled: bool,
                  alert_threshold: float = None) -> dict:
    """The highest fused score ACTUALLY attainable on a call, given
    whether a voiceprint exists to compare against.

    check_alertable() reports per-signal capacity; this reports runtime
    state. The difference matters: in heuristic mode (acoustic cap 0.00)
    with nobody enrolled, mismatch_score() has no opinion for every
    window, so the most the call can ever score is the context cap,
    0.30 — below the 0.40 threshold. Such a call is silently incapable
    of alerting, and before this function nothing told the operator.
    /health, /analyze and the live ready message all surface it now.
    """
    if alert_threshold is None:
        from .risk_engine import ALERT_THRESHOLD   # local: risk_engine does not import fusion
        alert_threshold = ALERT_THRESHOLD
    max_fused = fuse(1.0, 1.0 if speaker_enrolled else 0.0, 1.0, caps)["fused_score"]
    return {
        "speaker_enrolled": bool(speaker_enrolled),
        "max_fused": round(max_fused, 4),
        "alert_threshold": alert_threshold,
        "can_ever_alert": max_fused >= alert_threshold,
    }


def capability_warning(backend: str, caps: SignalCaps, speaker_enrolled: bool) -> str:
    """Plain-English warning when reachable_max() says a call cannot
    alert, or None when it can."""
    cap = reachable_max(caps, speaker_enrolled)
    if cap["can_ever_alert"]:
        return None
    fix = ("enrol a voiceprint for this speaker (step 1 in the dashboard, or POST /enroll)"
           if not speaker_enrolled else "")
    if backend != "transformer":
        fix = (fix + ", or " if fix else "") + "run the server with DETECTOR_BACKEND=transformer"
    return (f"This call CANNOT raise an alert in the current configuration: the highest risk score "
            f"it can reach is {cap['max_fused']:.2f}, below the alert threshold of {cap['alert_threshold']:.2f} "
            f"({backend} mode, {'no ' if not speaker_enrolled else ''}voiceprint enrolled). To fix: {fix}.")


def check_alertable(alert_threshold: float,
                    caps: SignalCaps = TRANSFORMER_CAPS) -> dict:
    """Reports whether each signal can reach the alert threshold on its
    own. Under noisy-OR a single maxed signal reaches exactly its cap,
    so this is a direct comparison of cap against threshold.

    Exists so the combination-rule bug that made this system unable to
    fire cannot silently come back — tests assert on it, and the API
    exposes it at /health so a misconfiguration is visible before a
    demo rather than during one.

    Under HEURISTIC_CAPS acoustic_can_alert is False by design (its cap
    is zero); speaker_can_alert must be True in every mode."""
    return {
        "acoustic_alone_max": caps.acoustic,
        "speaker_alone_max": caps.speaker,
        "context_alone_max": caps.context,
        "alert_threshold": alert_threshold,
        "acoustic_can_alert": caps.acoustic >= alert_threshold,
        "speaker_can_alert": caps.speaker >= alert_threshold,
        # Intentionally False — see the invariants above.
        "context_can_alert": caps.context >= alert_threshold,
    }


# Back-compat aliases. The concept changed from "weight in a mean" to
# "cap in a noisy-OR"; these keep older call sites importable.
FusionWeights = SignalCaps
TRANSFORMER_WEIGHTS = TRANSFORMER_CAPS
HEURISTIC_WEIGHTS = HEURISTIC_CAPS
weights_for_backend = caps_for_backend
