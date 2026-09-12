"""
context_engine.py — implements the PS's explicit "contextual
enrichment using metadata such as call origin, known contact
information, transaction context, and historical fraud indicators"
requirement.

Deliberately rule-based and transparent (not a black-box model) —
for a fraud-adjacent system, being able to say exactly *why* a call
was flagged ("unknown caller ID + urgent fund-transfer language +
off-hours") matters more than squeezing out a little more accuracy
from an opaque model, and it's what a compliance/audit reviewer will
actually want to see.

Upgrade path: replace the fixed weights below with a logistic
regression (or small gradient-boosted tree) trained on real labeled
call outcomes once your organization has that data — the interface
(a dict of context flags in, a [0,1] score out) doesn't need to change.
"""

import re

RISK_KEYWORDS = [
    "transfer", "urgent", "immediately", "otp", "one time password",
    "verification code", "confidential", "wire", "gift card",
    "password", "pin", "account number",
]

# Matched on word boundaries, not as bare substrings. Plain `in`
# matching gave false flags on ordinary words — "pin" fired on
# "shipping", "wire" on "wireless", "otp" on "hotspot" — which in a
# fraud system means flagging innocent calls and eroding a fraud
# team's trust in the alert. Multi-word phrases tolerate any run of
# whitespace, so "one  time\npassword" still matches.
_KEYWORD_PATTERNS = [
    (kw, re.compile(r"\b" + r"\s+".join(map(re.escape, kw.split())) + r"\b",
                    re.IGNORECASE))
    for kw in RISK_KEYWORDS
]


def score_context(
    caller_known: bool = True,
    off_hours: bool = False,
    first_contact: bool = False,
    transcript: str = "",
) -> dict:
    """All inputs are optional/best-effort — the system should degrade
    gracefully to a neutral score when call metadata isn't available,
    not fail."""
    score = 0.0
    reasons = []

    if not caller_known:
        score += 0.25
        reasons.append("caller ID not in known-contacts list")

    if off_hours:
        score += 0.10
        reasons.append("call placed outside normal business hours")

    if first_contact:
        score += 0.10
        reasons.append("first recorded contact with this number")

    if transcript:
        hits = [kw for kw, pattern in _KEYWORD_PATTERNS if pattern.search(transcript)]
        if hits:
            score += min(0.4, 0.15 * len(hits))
            reasons.append(f"high-risk phrases mentioned: {', '.join(hits[:3])}")

    return {
        "context_score": float(min(score, 1.0)),
        "reasons": reasons,
    }
