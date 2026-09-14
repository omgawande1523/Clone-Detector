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

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

113 tests. The suite is built around the bugs that actually shipped, so
each of these would fail if the bug came back:

| test file | pins |
|---|---|
| `test_speaker_consistency.py` | a different speaker must score a high mismatch (this was 0.002) |
| `test_fusion.py` | acoustic alone and speaker alone must each be able to alert; context alone must not |
| `test_audit_log.py` | backdating a record must break the chain; blank lines must not crash verification |
| `test_detector_calibration.py` | labelled-bonafide ASVspoof speech must not be flagged; over-smoothed synthetic must be |
| `test_false_positive_rate_asvspoof.py` | full fused pipeline on 8 real speakers' genuine calls: none may reach HIGH/CRITICAL |
| `test_context_and_risk.py` | word-boundary keyword matching; peak risk reported, not the trailing EMA |
| `test_api.py` | error paths return 4xx with a readable message, never a 500 |
| `test_transformer_backend.py` | wav2vec2 backend on both fixture sets, cross- and in-domain reported separately |
| `test_fused_modes.py` | full fused pipeline per mode: zero FP in both; targeted-clone recall floor in transformer mode; heuristic acoustic weight is exactly zero |

## The three signals, in more detail

**Acoustic/prosody (`detector.py`)** — pitch micro-jitter, spectral
flatness, energy-envelope smoothness by default (zero downloads); or
swap in a real pretrained wav2vec2 spoof classifier once you have
internet (see below). This is "does this sound synthetic."

**Speaker consistency (`speaker_consistency.py`)** — a 256-dim
Resemblyzer speaker embedding (GE2E d-vector) compared by cosine
similarity against an enrolled reference sample of the specific person
being protected (e.g. a company's CEO, or a bank's known customer).
This directly implements the PS's "cross-session consistency
checks... against historical genuine samples" requirement, and it's
the layer that catches a clone good enough to fool a generic spoof
detector but that still doesn't match *this particular* person's
voice.

Measured separation on two distinct speakers: same speaker scores a
cosine of 0.97 (mismatch 0.00), a different speaker 0.54 (mismatch
1.00). The raw cosine is remapped onto a [0,1] risk score against a
measured decision band rather than used directly — see the
calibration constants in that module.

This replaced an MFCC-style voiceprint that did not work: log-band-
energy vectors are all the same sign, so their cosine similarity was
pinned near 1.0 for any two speech signals. A genuinely different
speaker scored a mismatch of 0.002 while the *same* speaker with
flattened pitch scored 0.632 — it was measuring synthesis artifacts,
not identity, which also meant it was not independent of the acoustic
detector at all.

**Call context (`context_engine.py`)** — implements the PS's
"contextual enrichment using metadata such as call origin, known
contact information, transaction context, and historical fraud
indicators" bullet directly: unknown caller ID, off-hours, and
risk-keyword detection in a transcript, combined transparently (not a
black box) so a compliance reviewer can see exactly why a call was
flagged.

**Fusion (`fusion.py`)** — weighted combination of the three
(acoustic 0.45 / speaker 0.45 / context 0.10), with the dominant
contributing signal surfaced for an explainable alert message. Swap
the fixed weights for a trained logistic-regression meta-model once
you have labeled call outcomes.

The weights obey a rule the tests enforce: acoustic alone and speaker
mismatch alone must each be able to cross the alert threshold on
their own merits, and context alone must never be able to — a call
with suspicious metadata is not evidence that anyone's voice was
cloned. An earlier version had weights of 0.55/0.30/0.15 against a
threshold of 0.60, which made it arithmetically impossible for ANY
single signal to raise an alert.

## The audit trail — why "blockchain" isn't just a buzzword here

`audit_log.py` implements a local hash chain: every alert record embeds
the hash of the previous record, so any retroactive edit or deletion
breaks the chain and is instantly detectable via `GET /audit/verify`.
The hash covers the timestamp as well as the payload, so *backdating*
an alert — moving it to after a transfer was approved, say — breaks
the chain too. That is exactly the edit such a log exists to catch.
This is a genuine, honest prototype of the "Blockchain & Cybersecurity"
theme tag — **be upfront that it is not yet a real distributed ledger**
(no consensus, single machine). The production upgrade is periodically
anchoring the running hash to a permissioned blockchain (e.g.
Hyperledger Fabric) shared across a bank's nodes, so no single admin
can unilaterally rewrite the fraud-alert history. Say exactly this to
judges — it shows you understand what blockchain actually buys you,
instead of just namedropping it.

## Two detection modes — they are not equivalent

The fusion weights follow the loaded detector backend automatically
(`fusion.weights_for_backend`; there is no user flag, so it cannot be
misconfigured), and every `/health` and `/analyze` response plus the
UI say which mode produced the result.

| | **Transformer mode** (`DETECTOR_BACKEND=transformer`) | **Heuristic-only mode** (default, no download) |
|---|---|---|
| fusion weights (acoustic / speaker / context) | 0.45 / 0.45 / 0.10 | **0.00** / 0.90 / 0.10 |
| acoustic signal | wav2vec2 spoof classifier, validated (AUC 0.81 cross-domain, 1.00 in-domain) | measured AUC 0.25 / 0.14 — below chance, so weight is **exactly zero** |
| defence against an impostor (different person) | yes — speaker layer | yes — speaker layer |
| defence against a **clone of the enrolled speaker** | yes — acoustic layer: 7/12 ASVspoof clones, 7/7 In-the-Wild clones reach HIGH+ | **none** — 0/12 ASVspoof clones; the clone passes the voiceprint (median mismatch 0.04) and nothing else can fire |
| false positives on genuine multi-speaker calls | 0/16 utterances, 0/8 calls (ASVspoof); 0/7 (In-the-Wild) | 0/16, 0/8; 0/7 |
| what it is | **the real system** | an offline fallback with a structurally reduced capability |

**Heuristic-only mode must not be presented or deployed as equivalent
to transformer mode.** This is not "a bit less accurate": in that
configuration the system has *no* defence against a targeted clone of
the enrolled speaker, per the measured data above, and the code
enforces the reduction rather than leaving it to a weight that someone
might nudge back up. Say this to judges before they find it. The
heuristic detector is still computed and shown in the UI (it catches a
crude over-smoothed vocoder, which is a good demo moment) but it is
not allowed to move the risk score.

Reproduce every number in this section with
`venv/Scripts/python scripts/evaluate_fused.py` (add `--files` for
per-clip tables). Two things it also shows, kept as documented
limitations rather than fixed here:

- Zeroing heuristic-acoustic changed nothing that matters: with the
  old 0.45/0.45/0.10 weights the heuristic mode was already at 0/12
  clones and 0 false positives. The acoustic layer's contribution to
  the fused output was already too small to alter any verdict; the
  change removes a below-chance signal from the sum, it does not
  remove a working one.
- Scoring 3-second chunks can lose a detection the model makes on the
  whole utterance: `LA_D_5685369` (A03) scores 0.948 whole-file and
  0.119 on its first chunk, which is why fused recall on ASVspoof is
  7/12 where raw whole-file recall was 8/12.

## Measured accuracy of the two acoustic backends

Both backends, both committed fixture sets, same threshold (0.5).
`fixtures/asvspoof2019/` is cross-domain for the transformer model;
`fixtures/in_the_wild/` is its own fine-tuning distribution (split
unknown, so treat it as an upper bound). The two are reported
separately on purpose — spoof detection is known to generalise poorly
across attack types, and one averaged number would hide that.

| backend | dataset | n (bona/spoof) | bonafide max | false-pos | spoof recall | AUC |
|---|---|---|---|---|---|---|
| heuristic | ASVspoof 2019 LA (cross-domain) | 48 / 12 | 0.000 | 0% | **0%** | 0.25 |
| heuristic | In-the-Wild (in-domain) | 7 / 7 | 0.063 | 0% | **0%** | 0.14 |
| transformer | ASVspoof 2019 LA (cross-domain) | 48 / 12 | 0.007 | 0% | **67%** | 0.81 |
| transformer | In-the-Wild (in-domain) | 7 / 7 | 0.004 | 0% | **100%** | 1.00 |

Transformer misses on ASVspoof are structured, not random: neural /
vocoder TTS (A01–A03) is caught at >0.93 every time; waveform
concatenation of real speech units (A04) is missed every time; the two
voice-conversion systems (A05, A06) go 1-for-2 each. Cost: 378 MB
download once, ~0.2–0.3 s per 3 s chunk on a laptop CPU.
`tests/test_transformer_backend.py` re-measures this (skipped unless
the weights are cached locally).

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

- **Heuristic acoustic backend does not detect modern spoofs.** Measured
  on ASVspoof 2019 LA (labels from the official protocol; see
  `fixtures/asvspoof2019/`): 0 of 48 bonafide utterances from 8 speakers
  are flagged (max score 0.000), but only 1 of 12 spoofs across systems
  A01-A06 scores above 0.3, and the median spoof scores the same as the
  median bonafide. Each of its three cues is close to a coin flip on
  that data, and the envelope cue runs in the opposite direction to the
  one the literature suggests. It is kept as a zero-download baseline
  that catches crude over-smoothed vocoder output and never cries wolf;
  it is not the accuracy path. Use `DETECTOR_BACKEND=transformer`.
- **The speaker-consistency layer catches impostors, not clones.** It
  separates two different people cleanly (mismatch 0.00 vs 1.00), but on
  ASVspoof spoofs — which are TTS/VC systems built to *sound like* the
  target speaker — the median mismatch is 0.04. A good clone of the
  enrolled person passes this check. That is the well-known ASV
  vulnerability that ASVspoof exists to study, and it is exactly why the
  acoustic layer has to carry real weight, which today it does not.
- **Measured end-to-end on ASVspoof 2019 LA, heuristic backend: 0%
  false positives on genuine multi-speaker calls, and 0 of 12 spoofs
  reach HIGH/CRITICAL.** Say this to judges before they find it.
- Transformer backend's out-of-the-box accuracy is strongest on
  English/Western voices; Indian-accent robustness needs fine-tuning
  data you don't yet have.
- The audit log is a local hash chain, not yet an anchored distributed
  ledger — say so explicitly, it's a prototype of the mechanism.
- `.wav` only for now; non-wav call formats need an `ffmpeg` decode
  step before `audio_pipeline.py`.
