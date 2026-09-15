# SIH26104 · Voice-Clone Detection Pipeline Simulation

A self-contained visual simulation designed for a panel of Smart India Hackathon judges (~90 seconds attention span). Explains how our AI-cloned voice defense system works through three independent multi-modal signals and asymmetric risk accumulation.

---

## 1. How to Open

1. **Directly in Any Modern Browser**:
   Double-click `index.html` or drag it into Chrome, Edge, Firefox, or Safari.
2. **Offline & Air-Gapped**:
   The entire simulation is 100% self-contained in a single file with zero external dependencies, no CDN links, no Google Fonts, and no network calls. It runs directly off a USB drive without internet access.
3. **Sound**:
   An offline Web Audio API tone generates a clean, non-intrusive alert beep the moment a call crosses the 0.40 consecutive risk threshold. (Browsers require one click or keypress anywhere on the page before web audio is permitted).

---

## 2. Keyboard Shortcuts

Designed for quick, confident presenting without looking down at a trackpad:

| Key | Action | Presenter Use Case |
| :--- | :--- | :--- |
| **`Space`** | Play / Pause | Start the playback or freeze immediately to speak. |
| **`→`** (Right Arrow) | Step Forward (+1 Window) | Step through window-by-window during a pitch. |
| **`←`** (Left Arrow) | Step Backward (-1 Window) | Return to the exact moment an alert fired. |
| **`3`** | Scenario 3: *an AI copy of the CEO* | Headline scenario (AI clone vs real person). |
| **`2`** | Scenario 2: *someone else pretending* | Impostor scenario (human impersonation). |
| **`1`** | Scenario 1: *the real CEO* | Genuine caller baseline (zero false alarms). |
| **`T`** | Toggle Light / Dark Theme | Switch for dark presentation auditoriums or bright projector halls. |
| **`Esc`** | Close Info Popover | Dismiss any open inspection card. |

---

## 3. Where to Edit `SCENARIOS` Data

All measurements displayed on screen come strictly from real benchmark runs on **ASVspoof 2019 LA** and **In-the-Wild** deepfake audio datasets.

To update or replace numbers with new experimental runs:
1. Open `index.html` in any text editor.
2. Locate the `SCENARIOS` constant near the top of the `<script>` section (~line 520):
   ```javascript
   const SCENARIOS = {
     1: {
       name: "the real CEO",
       teachingPoint: "Every window: acoustic 0.002, speaker 0.00, context 0.00. The system does not cry wolf.",
       windows: [
         { window: 1, time: "0.0s – 5.0s", acoustic: 0.002, speaker: 0.00, context: 0.00, flags: [false, false, false] },
         ...
       ]
     },
     2: { ... },
     3: { ... }
   };
   ```
3. Update the `acoustic`, `speaker`, `context`, or `flags` arrays. The mathematical engine automatically recalculates:
   - **Noisy-OR Fusion**: $\text{fused} = 1 - \prod (1 - c_i s_i)$
   - **Asymmetric EMA**: Attack $\alpha = 0.60$, Release $\alpha = 0.20$
   - **Consecutive Alert Logic**: 2 consecutive windows $\ge 0.40$
   - **Tamper-Evident SHA Hash Chain** blocks

---

## 4. 60-Second Presenter Pitch Script (Walking Scenario 3)

> **Setup:** Simulation loads by default on **Scenario 3 ("an AI copy of the CEO")** at Window 1. Have your thumb on the **Spacebar** or **Right Arrow**.

* **0:00 – 0:12 | The Problem & Architecture**
  > *"Judges, when an attacker clones a CEO's voice to authorize a wire transfer, single-model defenses fail. Look at our pipeline: audio enters, splits into 5-second scoring windows, and evaluates three independent signals simultaneously."*

* **0:12 – 0:25 | Surfacing Our Limitation Honestly (The Pivot)**
  > *(Press `→` to advance to Window 1)*
  > *"Notice the middle stage — Speaker verification. The clone matches the CEO's voiceprint with a mismatch score of 0.00. The speaker check is completely blind here. We openly document this because an attacker with high-fidelity cloning will always pass a voiceprint."*

* **0:25 – 0:42 | Why Multi-Modal Fusion Saves the Victim**
  > *"Now look at the top branch — Acoustic. Our wav2vec2 classifier catches vocoder synthesis artifacts at 1.00. Because we fuse using noisy-OR over capped evidence rather than a naive average, the acoustic signal isn't diluted by the speaker check. It carries the fused risk straight to 0.90."*

* **0:42 – 0:60 | The Risk Engine & Verdict**
  > *(Press `→` to advance to Window 2 — [Beep sounds, red CRITICAL alert locks in])*
  > *"Our asymmetric risk engine requires two consecutive windows above 0.40 to prevent brief glitches from crying wolf. At just 10 seconds into the call — well before OTPs or bank credentials can be spoken — the critical alert locks in, and a cryptographic hash block seals the audit chain. 
  > 
  > Switch to Scenario 2 with key '2', and the exact inverse happens: an impostor passes acoustic checks, but the speaker check flags them instantly. Three independent checks, because each fails differently. That is our defense."*
