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
 audio_pipeline.py        -- splits into 5s chunks (simulates live streaming)
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
#  3. either "Live call (microphone)" and start talking, or upload a recording
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

153 tests. The suite is built around the bugs that actually shipped, so
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
| `test_label_mapping.py` | the spoof class is resolved once from id2label or an explicit override, and ambiguous labels raise instead of guessing |
| `test_resampling.py` | one anti-aliased resampler everywhere: a >8 kHz tone does not alias into band; a 48 kHz round trip keeps its verdict |
| `test_risk_engine_consistency.py` | a single unsustained crossing reports final_level HIGH, sustained_level LOW, and never says "no signal detected" |
| `test_alert_capability.py` | heuristic mode with no enrolment can never alert and must say so in /health, /analyze and the live ready message |
| `test_live_ws.py` | `/ws/live` answers each window before the next is sent, one risk engine carries across windows, an alert is audited the instant it fires, and live and batch paths score identically |

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

## Two ways to feed audio in: live call vs upload

| | **Live call (microphone)** | **Upload a recording** |
|---|---|---|
| path | `WS /ws/live` | `POST /analyze` |
| what it is | the true real-time path — the thing the PS asks for | batch analysis of a file; useful for testing against the fixture sets |
| audio in | browser captures the mic in 5 s segments (`getUserMedia` + `MediaRecorder`), decodes each in-page to 16 kHz WAV and sends it the moment it is recorded | whole file, split into 5 s windows server-side |
| scoring | each window scored on arrival, result pushed back immediately; **one `RiskEngine` for the whole call**, so EMA and the 2-window sustain rule carry across windows | same code — both paths call `api._score_chunk()`, and `tests/test_live_ws.py` asserts they agree |
| audit | written the instant an alert first fires (a live call may never "end") | written once at the end if an alert fired |
| measured latency | **0.29–0.35 s** from a window leaving the client to its result arriving, flat across a call (transformer, laptop CPU) | n/a |

The latency number is measured, not claimed. `scripts/live_latency_probe.py`
feeds a running server one 5 s window every 5 s with a real
`time.sleep` between them and logs, per window, when it was sent and
when its result came back:

```
   #   t_send  t_result   GAP s compute_ms   risk level     alert
   0     2.03      2.38   0.351      342.7  0.002 LOW       -
   1     7.38      7.68   0.296      285.8  0.158 LOW       -
   2    12.68     12.99   0.313      302.4  0.260 MEDIUM    -
   3    17.99     18.30   0.309      302.0  0.326 MEDIUM    -
   4    23.30     23.59   0.292      285.1  0.367 MEDIUM    -
   5    28.59     28.91   0.319      310.5  0.275 MEDIUM    -
  chunk-arrival-to-result gap: min 0.292s  median 0.313s  max 0.351s
  drift first->last chunk: -0.032s  ->  FLAT: streaming
```

A batching server would show that gap growing toward 30 s by the last
window.

**Level vs alert — they measure different things.** `final_level` is
the level of the highest EMA the call reached and has *no* sustain
requirement; `alert_triggered` needs the EMA at or above the threshold
for two consecutive windows. A single window that crosses and falls
straight back therefore gives `final_level: HIGH` with
`alert_triggered: false`. That is the sustain rule working, not a
contradiction — but it used to be reported next to "no signal
detected", which read as one. Every verdict now also carries
`sustained_level` (highest level reached while the sustain rule held),
`peak_windows_above` (longest run at or above threshold) and a
three-way `verdict_state`: `sustained_alert` / `peaked_unsustained` /
`clear`. The peaked-but-not-sustained wording names the peak value and
recommends caution; it never says "no signal". The dashboard headline
follows the same split (`tests/test_risk_engine_consistency.py`). Run it with `--clone` to stream a targeted clone of the enrolled
speaker instead: the alert fires on window 1 (7.7 s into the call) and
the audit record is written in the same instant.

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
| defence against a **clone of the enrolled speaker** | yes — acoustic layer: 8/12 ASVspoof clones, 7/7 In-the-Wild clones reach HIGH+ | **none** — 0/12 ASVspoof clones; the clone passes the voiceprint (median mismatch 0.04) and nothing else can fire |
| false positives on genuine multi-speaker calls | 0/16 utterances, 0/8 calls (ASVspoof); 0/7 (In-the-Wild) | 0/16, 0/8; 0/7 |
| what it is | **the real system** | an offline fallback with a structurally reduced capability |

**A call that cannot alert says so.** In heuristic mode with no
voiceprint enrolled the highest reachable fused score is 0.10 against
an alert threshold of 0.40 — the call is structurally incapable of
alerting. `fusion.reachable_max()` computes this from runtime state
(not weight capacity), `/health` reports it for both the enrolled and
unenrolled case under `live_capability`, every `/analyze` response and
live `ready` message carries `can_ever_alert` plus a plain-English
`capability_warning`, and the dashboard shows it as a red banner above
the verdict. It is not possible to run a demo in that state without
seeing it (`tests/test_alert_capability.py`).

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
- **Scoring window: 5 s** (was 3 s). Ablated at 3/5/6 s on the
  transformer backend: a 3 s window missed an A03 clone the model
  catches on the whole utterance (`LA_D_5685369`: 0.119 vs 0.948); 5 s
  and 6 s both caught it, taking ASVspoof clone recall from 7/12 to
  8/12 with zero new false positives and identical results to each
  other, so 5 s was adopted for the lower latency. General statement
  that still holds: sub-clip real-time scoring can under-detect
  relative to whole-file scoring for some attack types; a production
  system would buffer and re-score on a rolling window. Trade-off: a
  sustained alert (two consecutive windows) now needs 10 s of audio
  instead of 6 — immaterial on a real call, visible on 1–9 s fixture
  clips, where sustained alerts on spoofs dropped from 2/12 to 1/12
  even as level-based detection rose.

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
download once, ~0.2–0.4 s per 5 s chunk on a laptop CPU.
`tests/test_transformer_backend.py` re-measures this (skipped unless
the weights are cached locally).

**Sample rate — the path an uploaded phone recording actually takes.**
Every committed fixture is 16 kHz, so the figures above never
exercised resampling. The old resampler (`np.interp`, no anti-aliasing
filter, plus an off-by-one that `api.py`'s separate copy did not have)
would have folded >8 kHz energy from a 44.1/48 kHz recording into the
band a spoof classifier keys on. There is now one resampler,
`audio_pipeline.resample_to()` (scipy polyphase FIR), used by the
transformer, the upload path and the live path alike. Measured with
`scripts/evaluate_fused.py`'s 48 kHz round-trip section — every test
clip upsampled to 48 kHz, resampled back by the pipeline, re-scored:

| backend | clips | same level | same alert | max Δ peak | max Δ acoustic |
|---|---|---|---|---|---|
| heuristic | 42 | 42/42 | 42/42 | 0.017 | 0.015 |
| transformer | 42 | **41/42** | 42/42 | 0.042 | 0.093 |

The one change is a borderline A03 clip (`LA_D_2169320`: HIGH 0.429 →
MEDIUM 0.388, acoustic 0.983 → 0.890) sitting on the 0.40 boundary; no
alert changed. `tests/test_resampling.py` pins that a 12 kHz tone does
not alias into the band and that a 48 kHz round trip keeps its verdict.

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

**Swapping the model cannot silently invert the detector.** Which
output class means "spoof" is resolved once at startup from the
model's `id2label` against a vocabulary on both sides (spoof / fake /
deepfake / synthetic / cloned / tts … vs bonafide / real / genuine /
human / original …). If exactly one class is spoof-side and the rest
bonafide-side it is used; anything else — `LABEL_0`/`LABEL_1`, two
spoof-like names, a third class — raises at startup naming the model's
actual labels, and `SPOOF_LABEL_INDEX=<n>` is the explicit override. It
never falls back to a guess (the old code defaulted to index 1). The
resolved mapping is shown in `/health` as `label_mapping`
(`tests/test_label_mapping.py`).

## Testing with real audio

Replace the synthetic signals in `demo/demo_cli.py` with real samples
before your submission video:
- Genuine speech: record your own voice for enrollment + a genuine call.
- Cloned speech: samples from **ASVspoof 2019/2021 (LA)** or the
  **In-the-Wild deepfake-audio dataset**, or generate one yourself
  with a free TTS/voice-clone tool.

## Judges Q&A prep

**"Is this actually real-time, or does it analyse a recording afterwards?"**
Both, and the live one is the real system. `POST /analyze` takes a
recording and is what we test fixture files against. `WS /ws/live`
takes the call as it happens: the dashboard's "Live call (microphone)"
mode captures 5-second segments and each one is scored the moment it
arrives — the gauge moves while the caller is still talking. Measured
window-to-result latency is 0.29–0.35 s and stays flat for the whole
call (`scripts/live_latency_probe.py` prints the per-window timestamps).
On a clone of the enrolled speaker the alert fires at the second window,
under 8 seconds into the call, and the audit record is written at that
moment, not when the call ends.

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
- Container formats: WAV/FLAC/OGG/AIFF (whatever `libsndfile` reads);
  MP3/M4A/WebM call formats need an `ffmpeg` decode step before
  `audio_pipeline.py`. Any sample rate is accepted and resampled once,
  up front, with an anti-aliased resampler.
