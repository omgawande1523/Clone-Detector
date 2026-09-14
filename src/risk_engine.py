"""
risk_engine.py — turns a stream of per-chunk fused risk scores into a
single live "risk gauge" for the call, plus a discrete alert level.

Design choices, explainable to judges:
- EMA (exponential moving average) instead of the raw per-chunk score,
  so one noisy scoring window does not trigger a false alarm.
- Alerts require the EMA to *stay* above threshold for a few
  consecutive chunks (`sustain`), not just touch it once — reduces
  false positives on a single cough/laugh/noise burst.
- The engine tracks the PEAK risk over the call, not just the latest
  value. A call that is cloned in the first minute and genuine
  afterwards is a cloned call; reporting the trailing EMA would let
  that decay back to LOW and report a clean verdict on a call that
  actually fired an alert.

THRESHOLDS AND FUSION WEIGHTS ARE COUPLED. The alert threshold is
deliberately defined as the HIGH band boundary, so "alert fired" and
"level is HIGH or worse" always mean the same thing — they previously
could, and did, disagree. Any change to these numbers has to keep
fusion.check_alertable() satisfied: acoustic alone and speaker
mismatch alone must each still be able to cross, and context alone
must not. tests/test_fusion.py enforces that.
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


@dataclass
class RiskEngine:
    alpha: float = 0.35                        # EMA smoothing factor
    alert_threshold: float = ALERT_THRESHOLD
    sustain_required: int = 2                  # consecutive chunks above threshold

    _ema: float = field(default=0.0, init=False)
    _above_streak: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)
    _peak: float = field(default=0.0, init=False)
    _alert_fired: bool = field(default=False, init=False)
    _score_at_alert: float = field(default=None, init=False)

    def update(self, spoof_prob: float) -> dict:
        if not self._initialized:
            self._ema = spoof_prob
            self._initialized = True
        else:
            self._ema = self.alpha * spoof_prob + (1 - self.alpha) * self._ema

        if self._ema >= self.alert_threshold:
            self._above_streak += 1
        else:
            self._above_streak = 0

        alert = self._above_streak >= self.sustain_required

        self._peak = max(self._peak, self._ema)
        if alert and not self._alert_fired:
            self._alert_fired = True
            self._score_at_alert = self._ema

        return {
            "instant_score": round(float(spoof_prob), 4),
            "risk_score": round(float(self._ema), 4),
            "level": level_for(self._ema),
            "alert": alert,
        }

    def summary(self) -> dict:
        """Call-level verdict, for reporting once the call is over.

        Reports the PEAK risk rather than the final EMA — see the
        module docstring. `score_at_alert` is the EMA at the moment the
        alert first fired, which is what an investigator wants to cite.
        """
        peak = round(float(self._peak), 4)
        return {
            "peak_risk_score": peak,
            "peak_level": level_for(self._peak),
            "alert_triggered": self._alert_fired,
            "score_at_alert": (
                round(float(self._score_at_alert), 4)
                if self._score_at_alert is not None else None
            ),
            "final_ema": round(float(self._ema), 4),
        }
