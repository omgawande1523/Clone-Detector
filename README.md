# SIH26104 — AI-Powered Real-Time Voice-Cloning Detection (Prototype)

Working prototype for Smart India Hackathon 2026, Problem Statement
**SIH26104** (AICTE Cyber Security Cell): detect AI-cloned/synthetic
voices during a call in near real time and raise a risk alert before
sensitive actions (fund transfers, credential disclosure) are taken.

## Why this design, not "run one classifier on the audio"

Most teams will pitch a single spoof-detection model. That's fragile:
a clone only has to get past one check. This system instead asks
three independent questions that a real attack usually can't all
pass at once, then fuses the answers:

1. **Is this speech synthetic at all?** (acoustic/prosody analysis)
2. **Does it actually sound like the specific person it claims to be?**
   (speaker-consistency check against an enrolled voiceprint — this
   catches a clone that's clean enough to fool #1)
3. **Does the surrounding call context look risky?** (unknown caller,
   off-hours, urgent-transfer language — this catches a well-executed
   scam that avoids obvious acoustic tells)

## Architecture

```
 call audio (.wav file today; SIP/VoIP media stream in production)
        │
        ▼
 audio_pipeline.py        -- splits into 3s chunks (simulates live streaming)
        │
        ▼
 Multi-layer voice analysis engine
   ├─ detector.py              -- acoustic/prosody spoof probability
   │    ├─ HeuristicDetector       (default: offline, signal-processing baseline)
   │    └─ TransformerDetector     (real wav2vec2 model, needs internet once)
   ├─ speaker_consistency.py   -- mismatch vs an enrolled voiceprint
   └─ context_engine.py        -- risk from call metadata / transcript keywords
        │
        ▼
 fusion.py                -- weighted combination -> one explainable risk score
        │
        ▼
 risk_engine.py           -- EMA smoothing + sustained-alert logic -> LOW/MEDIUM/HIGH/CRITICAL
        │
        ├──────────────────────────────┐
        ▼                              ▼
 api.py (FastAPI)              audit_log.py
   /analyze, /enroll             hash-chain log of every alert
   /audit/verify                 (blockchain-anchoring prototype)
        │
        ▼
 web/index.html            -- live gauge, per-signal breakdown, alert banner
```

## Quick start

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

uvicorn src.api:app --reload --port 8000
# then open web/index.html in a browser:
#  1. (optional) enroll a reference voice for a speaker_id
#  2. (optional) fill in call-context flags
#  3. upload the call recording and analyze
```

No-internet sanity check (proves the whole fused pipeline — acoustic +
speaker-consistency + context + audit log — works with zero audio
files or downloads):

```bash
python demo/demo_cli.py
```

## The three signals, in more detail

**Acoustic/prosody (`detector.py`)** — pitch micro-jitter, spectral
flatness, energy-envelope smoothness by default (zero downloads); or
swap in a real pretrained wav2vec2 spoof classifier once you have
internet (see below). This is "does this sound synthetic."

**Speaker consistency (`speaker_consistency.py`)** — a lightweight
MFCC-style voiceprint compared via cosine similarity against an
enrolled reference sample of the specific person being protected
(e.g. a company's CEO, or a bank's known customer). This directly
implements the PS's "cross-session consistency checks... against
historical genuine samples" requirement, and it's the layer that
catches a clone good enough to fool a generic spoof detector but that
still doesn't match *this particular* person's voice. Honest
limitation: this MVP voiceprint captures spectral timbre, not pitch,
so it can under-discriminate two synthetic voices with similar timbre
but different pitch — the documented upgrade is a real speaker
embedding model (ECAPA-TDNN via speechbrain, or resemblyzer).

**Call context (`context_engine.py`)** — implements the PS's
"contextual enrichment using metadata such as call origin, known
contact information, transaction context, and historical fraud
indicators" bullet directly: unknown caller ID, off-hours, and
risk-keyword detection in a transcript, combined transparently (not a
black box) so a compliance reviewer can see exactly why a call was
flagged.

**Fusion (`fusion.py`)** — weighted combination of the three, with the
dominant contributing signal surfaced for an explainable alert message.
Swap the fixed weights for a trained logistic-regression meta-model
once you have labeled call outcomes.

## The audit trail — why "blockchain" isn't just a buzzword here

`audit_log.py` implements a local hash chain: every alert record embeds
the hash of the previous record, so any retroactive edit or deletion
breaks the chain and is instantly detectable via `GET /audit/verify`.
This is a genuine, honest prototype of the "Blockchain & Cybersecurity"
theme tag — **be upfront that it is not yet a real distributed ledger**
(no consensus, single machine). The production upgrade is periodically
anchoring the running hash to a permissioned blockchain (e.g.
Hyperledger Fabric) shared across a bank's nodes, so no single admin
can unilaterally rewrite the fraud-alert history. Say exactly this to
judges — it shows you understand what blockchain actually buys you,
instead of just namedropping it.

## Switching to the real ML model

```bash
pip install -r requirements-real-model.txt
set DETECTOR_BACKEND=transformer      # Windows
uvicorn src.api:app --reload --port 8000
```

Default model: `abhishtagatya/wav2vec2-base-960h-itw-deepfake`
(fine-tuned on the In-the-Wild deepfake-audio dataset, ~98% reported
val accuracy). For the Indian-accent angle, the natural next step is
fine-tuning an XLSR-53 multilingual variant (e.g.
`Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`) on Indic
speech — name this as roadmap/future work; it's a genuinely unsolved
gap that strengthens your pitch rather than weakens it.

## Testing with real audio

Replace the synthetic signals in `demo/demo_cli.py` with real samples
before your submission video:
- Genuine speech: record your own voice for enrollment + a genuine call.
- Cloned speech: samples from **ASVspoof 2019/2021 (LA)** or the
  **In-the-Wild deepfake-audio dataset**, or generate one yourself
  with a free TTS/voice-clone tool.

## Judges Q&A prep

**"How will you actually listen to the calls?"**
This targets enterprise call centers, banks, and government
helplines — environments that already record every call for
QA/compliance/fraud (the "this call may be recorded" disclaimer).
The system taps that same already-lawful, already-consented audio
stream via the contact-center platform's existing media/recording API
(Genesys, Avaya, Twilio, Asterisk, etc.) — it is not a new access
point or a covert wiretap on personal phones.

**"Why three signals instead of one model?"**
Each fails differently — a clone can get cleaner over time to beat an
acoustic detector, but it can't defeat a speaker-consistency check
without exactly reproducing one specific enrolled voice, and a
context-aware attacker who avoids acoustic tells still has to avoid
every risk flag at once. Ensembling independent signals is standard
practice in the anti-spoofing literature (ASVspoof challenge-winning
systems are ensembles, not single models).

**"Why trust a heuristic instead of a trained model?"**
It's an explainable baseline built on published anti-spoofing cues
that runs with zero downloads — useful for unreliable venue wifi. The
transformer backend is the accuracy path for your actual numbers;
mention both so judges see the trade-off was a choice, not a gap.

**"What about privacy / data protection?"**
Inference runs on-device/on-premises rather than sending audio to a
third-party cloud, and the audit log stores only outcomes and reasons
— never raw audio — which is what makes it privacy-preserving while
still being useful for compliance.

## Known limitations (be upfront about these)

- Heuristic acoustic backend is a signal-processing baseline, not a
  trained classifier — good for demos, not a claimed production accuracy.
- The MVP speaker-consistency voiceprint can under-discriminate voices
  with similar timbre but different pitch — documented upgrade path above.
- Transformer backend's out-of-the-box accuracy is strongest on
  English/Western voices; Indian-accent robustness needs fine-tuning
  data you don't yet have.
- The audit log is a local hash chain, not yet an anchored distributed
  ledger — say so explicitly, it's a prototype of the mechanism.
- `.wav` only for now; non-wav call formats need an `ffmpeg` decode
  step before `audio_pipeline.py`.
