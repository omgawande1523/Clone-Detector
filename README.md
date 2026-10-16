# SwarKavach · On-device Voice-Clone Shield (SIH26104 simulation)

A self-contained, judge-facing simulation of **SwarKavach**, our Android app that checks a phone call for an AI-cloned or impersonated voice **entirely on the phone**. Smart India Hackathon 2026, problem statement SIH26104 (AICTE Cyber Security Cell).

The page walks a call through the real architecture, window by window: policy decision, capture, cleaning, the three checks, noisy-OR fusion, the risk engine, the alert and the tamper-evident log. The scores are the Python reference scores for the same audio the app's demo mode plays, and each scenario quotes the result measured on the test phone.

---

## 1. Open it

- **Offline:** double-click `index.html`. It is a single file with no CDN, no web fonts and no network calls, so it runs from a USB stick.
- **Hosted:** the repo deploys as a static site on Vercel (`vercel.json`), or run `npm start` to serve it locally.
- **Sound:** the alert plays a short triple beep. Browsers need one click or keypress on the page before audio is allowed.

## 2. Keys

| Key | Action |
| :-- | :-- |
| `Space` | Play / pause |
| `→` / `←` | Step forward / back (ring → answer → each 5 s window → call ended) |
| `1`–`7` | Pick a scenario |
| `R` | Restart the scenario |
| `V` | Switch between **Simulation** and **Architecture** views |
| `T` | Light / dark theme |

## 3. Scenarios

| Key | Scenario | Caller policy | What it teaches | Measured on phone (demo mode) |
| :-- | :-- | :-- | :-- | :-- |
| 1 | Genuine: Papa | Protect → verify voice | No false alarm on the real person | CLEAR, peak 0.004 |
| 2 | Impostor on Papa's number | Protect → verify voice | Human voice passes the deepfake check; the **voiceprint** catches it | Alert on 2nd window, peak 0.900, speaker |
| 3 | AI clone of Papa *(default headline)* | Protect → verify voice | Clone passes the voiceprint; the **deepfake model** catches it; noisy-OR lets one signal alert | Alert on 3rd window, peak 0.869, acoustic |
| 4 | Scam script, real voice | Unknown → scan | Context (OTP, transfer…) raises to MEDIUM but **never accuses alone** | CLEAR, peak 0.258, MEDIUM |
| 5 | Trusted contact (Ignore) | Skip | The user chooses who to check; nothing runs | — |
| 6 | Commercial AI voice | Unknown → scan | **Known miss**, shown on purpose | CLEAR, peak 0.079 |
| 7 | Internet deepfake | Unknown → scan | Acoustic alone alerts without any voiceprint | Alert on 2nd window, peak 0.904, acoustic |

## 4. The architecture (phone edition)

1. **Policy:** a call-screening service and a contacts lookup decide *Skip / Scan / Verify voice*. Unknown numbers are always checked; contacts set to *Ignore* are skipped; *Protect* contacts get their voice verified.
2. **Capture:** *Live* (accessibility service + microphone while on speaker, sideloaded/enterprise build), *after-call* (scan the dialer's own recording; Play-Store-safe), or *built-in* (OEM/operator pre-install with system call-audio access). Ordinary Android apps cannot read call audio, and we designed around that.
3. **Clean:** 5 s windows, Silero VAD, and the owner's voiceprint removes the user's own side of the call.
4. **Three checks:** wav2vec2 deepfake model (ONNX) · WeSpeaker voiceprint (256 numbers) · call context (number rules + offline English-India scam-phrase recognition with Vosk).
5. **Fusion:** noisy-OR over capped evidence, `fused = 1 − (1 − 0.9·a)(1 − 0.9·s)(1 − 0.3·c)`. Acoustic alone or speaker alone can alert; context alone never can.
6. **Risk engine:** asymmetric EMA (attack 0.60, release 0.20); an alert needs two windows in a row ≥ 0.40.
7. **Act:** full-screen warning ("Do NOT share OTP / UPI PIN / money"), report to 1930 / cybercrime.gov.in / Sanchar Saathi (Chakshu), and a SHA-256 hash-chained audit record. The page lets you verify the chain and tamper with a record to watch verification fail.

Audio never leaves the phone. Windows are scored in memory and discarded; only scores, reasons and a hashed number are stored. Learning is opt-in: voiceprints of *Protect* contacts are built from the user's own past call recordings.

## 5. Measured numbers used on the page

From the phone project's `docs/phone/RESULTS.md` (OnePlus CPH2707, Android 16):

- 959 ms median, 1174 ms p95 per 5 s window for all three models (real-time factor 0.19).
- Phone vs Python reference scores: max difference 5.3e-08.
- ASVspoof 2019 LA (cross-domain): AUC 0.825 per window, 0 of 48 genuine clips flagged.
- Voiceprint similarity: same person median 0.89, different person 0.55.
- English (India) scam-phrase recognition adds about 0.9 s at p95; Hindi does not fit the real-time budget on this phone yet.

Known limits are shown in the Architecture view: some commercial AI voices slip past the acoustic model, one real-person phone recording was falsely flagged, and narrowband 8 kHz audio flagged 4 of 48 genuine clips.

## 6. 60-second walk-through (scenario 3)

1. **0:00** "When a cloned voice of Papa calls, a single model is not enough. Here is what happens on the phone." Press `→`: Papa is a *Protect* contact, so the policy says *verify voice*.
2. **0:10** Press `→`: answered on speaker, capture starts, nothing leaves the phone.
3. **0:20** Press `→` twice: the clone passes the voiceprint (mismatch 0.00) but the deepfake model scores 0.996. Noisy-OR carries it to 0.90; one window is not enough yet.
4. **0:35** Press `→`: second clone window, the alert fires, the phone shows the warning and the record is sealed in the hash chain.
5. **0:45** Press `2`: an impostor does the opposite: human voice, voiceprint mismatch 1.00, alert. Press `4`: scam words alone stay at MEDIUM. "Three checks, because each fails differently."

## 7. Editing the data

All scenario data is the `SC` array at the top of the `<script>` in `index.html`: per-window acoustic scores `a`, speaker mismatch `s` (`null` when there is no saved voice), context `c`, and the teaching and measured text. The engine recomputes fusion, the EMA, levels and alerts from those numbers.
