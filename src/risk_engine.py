"""
risk_engine.py — turns a stream of per-chunk fused risk scores into a
single live "risk gauge" for the call, plus a discrete alert level.

Design choices, explainable to judges:
- ASYMMETRIC EMA: fast attack, slow release. A rising score is smoothed
  with ATTACK_ALPHA (0.60) and a falling one with RELEASE_ALPHA (0.20).
  This is standard alarm behaviour — react quickly to danger, and do
  not let one quiet window clear a call that already looked bad.

  A single symmetric alpha of 0.35 was the second half of the bug that
  let a confidently-detected clone go unalerted: with the fused score
  capped at 0.448 by the old weighted-average fusion, the EMA needed
  six windows (30 s) just to approach the 0.40 threshold, and seven to
  satisfy the sustain rule. Fusion alone was not enough to fix it;
  smoothing dynamics and the combination rule had to change together.
  See fixtures/live_clone_session/ for the captured failure.
- Alerts require the EVIDENCE to persist: `sustain_required`
  consecutive windows whose own fused score is at/above threshold,
  AND a smoothed EMA also at/above it. Reduces false positives on a
  single cough/laugh/noise burst. With a fast attack this no longer
  costs 30 seconds.

  The streak counts the per-window score, not the EMA. It used to
  count the EMA, which was safe only while the EMA fell fast: with a
  slow release a single spike keeps the EMA above threshold for
  several windows, and an EMA-based streak would reach `sustain` on
  its own and call one spike a sustained alert. Counting the instant
  score keeps "sustained" meaning what it says — the detector kept
  seeing it — while the EMA still decides the reported LEVEL.
- Windows with no speech do not move the EMA at all, and the first
  window WITH speech seeds it. A silent or noise-only lead-in used to
  initialise the EMA at its own near-zero score and cost roughly two
  windows of climb. Such windows are reported with scored=False so the
  UI can say "no speech" rather than showing a misleading 0.00 risk.

  FAIL-SAFE: a window carrying alert-level evidence is ALWAYS scored,
  whatever the speech detector says. A detector confident enough to
  cross the threshold has found something, and a speech-detection
  false negative must never be able to suppress it -- that would
  recreate, from a different direction, the exact bug this file's
  history is about.
- The engine tracks the PEAK risk over the call, not just the latest
  value. A call that is cloned in the first minute and genuine
  afterwards is a cloned call; reporting the trailing EMA would let
  that decay back to LOW and report a clean verdict on a call that
  actually fired an alert.

THRESHOLDS AND FUSION WEIGHTS ARE COUPLED. The alert threshold is
deliberately defined as the HIGH band boundary. Any change to these
numbers has to keep fusion.check_alertable() satisfied: acoustic alone
and speaker mismatch alone must each still be able to cross, and
context alone must not. tests/test_fusion.py enforces that.

LEVEL vs ALERT — they measure different things, and can differ:

    peak_level        the level of the highest EMA the call reached,
                      with NO sustain requirement. One spike sets it.
    alert_triggered   the EMA stayed at/above the threshold for
                      `sustain_required` consecutive windows.

So a single isolated window above threshold gives peak_level HIGH with
alert_triggered False. That is not a contradiction, it is the sustain
rule doing its job — but it used to be reported as "HIGH" next to "no
signal detected", which read as one. summary() therefore also returns
peak_sustained_level (the highest level reached WHILE the sustain
condition held) and peak_windows_above (the longest run of windows at
or above threshold), and the API's recommendation distinguishes
sustained alert / peaked-but-did-not-sustain / never crossed.
tests/test_risk_engine_consistency.py pins this.
"""

from dataclasses import dataclass, field


LEVELS = [
    (0.0, "LOW"),
    (0.20, "MEDIUM"),
    (0.40, "HIGH"),
    (0.60, "CRITICAL"),
]

# The HIGH boundary. Single source of truth — do not hardcode 0.40
# anywhere else.
ALERT_THRESHOLD = LEVELS[2][0]


def level_for(score: float) -> str:
    label = LEVELS[0][1]
    for threshold, name in LEVELS:
        if score >= threshold:
            label = name
    return label


# Asymmetric smoothing. attack > release on purpose.
ATTACK_ALPHA = 0.60       # score rising  -> respond fast
RELEASE_ALPHA = 0.20      # score falling -> decay slowly


@dataclass
class RiskEngine:
    attack_alpha: float = ATTACK_ALPHA
    release_alpha: float = RELEASE_ALPHA
    alert_threshold: float = ALERT_THRESHOLD
    sustain_required: int = 2                  # consecutive chunks above threshold

    _ema: float = field(default=0.0, init=False)
    _above_streak: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)
    _peak: float = field(default=0.0, init=False)
    _alert_fired: bool = field(default=False, init=False)
    _score_at_alert: float = field(default=None, init=False)
    _peak_sustained: float = field(default=0.0, init=False)   # highest EMA while alert held
    _longest_streak: int = field(default=0, init=False)

    def update(self, spoof_prob: float, speech_present: bool = True) -> dict:
        """Fold one window's fused score into the call's risk gauge.

        speech_present=False means the window held no assessable audio:
        it is reported, but does not seed or move the EMA and does not
        advance or reset the sustain streak. Silence is not evidence in
        either direction.
        """
        # See FAIL-SAFE in the module docstring.
        if not speech_present and spoof_prob >= self.alert_threshold:
            speech_present = True

        if not speech_present:
            return {
                "instant_score": round(float(spoof_prob), 4),
                "risk_score": round(float(self._ema), 4),
                "level": level_for(self._ema) if self._initialized else "LOW",
                "alert": (self._above_streak >= self.sustain_required
                          and self._ema >= self.alert_threshold),
                "scored": False,
            }

        if not self._initialized:
            self._ema = spoof_prob
            self._initialized = True
        else:
            alpha = self.attack_alpha if spoof_prob > self._ema else self.release_alpha
            self._ema = alpha * spoof_prob + (1 - alpha) * self._ema

        # Streak on the window's own score; EMA must agree before alerting.
        if spoof_prob >= self.alert_threshold:
            self._above_streak += 1
        else:
            self._above_streak = 0

        alert = (self._above_streak >= self.sustain_required
                 and self._ema >= self.alert_threshold)

        self._peak = max(self._peak, self._ema)
        self._longest_streak = max(self._longest_streak, self._above_streak)
        if alert:
            self._peak_sustained = max(self._peak_sustained, self._ema)
            if not self._alert_fired:
                self._alert_fired = True
                self._score_at_alert = self._ema

        return {
            "instant_score": round(float(spoof_prob), 4),
            "risk_score": round(float(self._ema), 4),
            "level": level_for(self._ema),
            "alert": alert,
            "scored": True,
        }

    def summary(self) -> dict:
        """Call-level verdict, for reporting once the call is over.

        Reports the PEAK risk rather than the final EMA — see the
        module docstring. `score_at_alert` is the EMA at the moment the
        alert first fired, which is what an investigator wants to cite.

        peak_level has no sustain requirement; peak_sustained_level is
        the highest level reached while the sustain condition held (LOW
        if it never did). They differ exactly when the call spiked
        above threshold without sustaining it.
        """
        peak = round(float(self._peak), 4)
        return {
            "peak_risk_score": peak,
            "peak_level": level_for(self._peak),
            "peak_sustained_level": level_for(self._peak_sustained) if self._alert_fired else "LOW",
            "peak_windows_above": self._longest_streak,
            "alert_triggered": self._alert_fired,
            "score_at_alert": (
                round(float(self._score_at_alert), 4)
                if self._score_at_alert is not None else None
            ),
            "final_ema": round(float(self._ema), 4),
        }
