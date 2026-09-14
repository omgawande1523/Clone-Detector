"""False-positive batch check on real, labelled, multi-speaker audio.

The fusion weights were rebalanced so that a single strong signal can
raise an alert on its own. That fixed an unreachable threshold, but it
also made the system more sensitive — so this test runs a batch of
DEFINITELY-genuine calls from DIFFERENT real speakers through the
full fused pipeline and asserts none of them reaches HIGH/CRITICAL.

Data: ASVspoof 2019 LA, dev partition. Labels come from the official
protocol file (fixtures/asvspoof2019/ASVspoof2019.LA.cm.dev.trl.subset.txt),
never from filenames. See fixtures/asvspoof2019/README.md for
provenance and licence.

How the batch is built
    For each of 8 speakers the protocol lists as bonafide:
      * ENROL  — 4 bonafide utterances, concatenated, become that
                 speaker's voiceprint (real audio joined end to end;
                 nothing is padded or synthesised).
      * TEST   — 2 further bonafide utterances, scored individually,
                 and also joined into one ~6 s "call" so the risk
                 engine's 2-chunk sustain rule can actually fire.
    Context is neutral (known caller, business hours, no transcript).

Known limitation, stated rather than hidden
    ASVspoof utterances are single sentences (median 3.0 s). A single
    one is often one 3-second chunk, so the sustain rule means an
    *alert* can never fire on it; the per-chunk *level* still can and
    that is what per-utterance rows are graded on. The joined "call"
    rows exercise the full alert path.

Run `pytest tests/test_false_positive_rate_asvspoof.py -s` to print
the full per-clip table.
"""

import json
import os

import numpy as np
import pytest

from src.audio_pipeline import load_mono
from src.detector import get_detector
from src.fusion import fuse
from src.risk_engine import RiskEngine, level_for, ALERT_THRESHOLD
from src.context_engine import score_context
from src.speaker_consistency import VoiceprintStore, compute_voiceprint, mismatch_between

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "fixtures", "asvspoof2019")
PROTOCOL = os.path.join(FIX, "ASVspoof2019.LA.cm.dev.trl.subset.txt")
MANIFEST = os.path.join(FIX, "manifest.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(PROTOCOL) and os.path.exists(MANIFEST)),
    reason="ASVspoof fixtures not present",
)

CHUNK_S = 3.0
SR = 16000


# ------------------------------------------------------------ helpers

def _labels():
    """Ground truth from the official protocol: utt -> (speaker, system, key)."""
    out = {}
    for line in open(PROTOCOL):
        spk, utt, _env, system, key = line.split()
        out[utt] = (spk, system, key)
    return out


def _wav(utt):
    wav, sr = load_mono(os.path.join(FIX, "flac", utt + ".flac"))
    assert sr == SR
    return wav


def _chunks(wav):
    n = int(SR * CHUNK_S)
    for start in range(0, len(wav), n):
        seg = wav[start:start + n]
        if len(seg) < SR * 0.2:
            break
        yield start / SR, seg


def run_call(wav, reference_print, detector, ctx_score=0.0):
    """Full fused pipeline over one piece of audio. Returns the risk
    engine summary plus per-chunk detail and whether the speaker layer
    ever had enough speech to give an opinion."""
    engine = RiskEngine()
    rows, speaker_scored = [], False
    for t, seg in _chunks(wav):
        acoustic = detector.predict_detailed(seg, SR)["spoof_prob"]
        live = compute_voiceprint(seg, SR) if reference_print is not None else None
        if live is not None:
            speaker_scored = True
            mismatch = mismatch_between(reference_print, live)
        else:
            mismatch = 0.0
        fused = fuse(acoustic, mismatch, ctx_score)
        r = engine.update(fused["fused_score"])
        rows.append(dict(t=t, acoustic=acoustic, speaker=mismatch, context=ctx_score,
                         risk=r["risk_score"], level=r["level"], alert=r["alert"],
                         dominant=fused["dominant_signal"]))
    return engine.summary(), rows, speaker_scored


@pytest.fixture(scope="module")
def batch():
    labels = _labels()
    manifest = json.load(open(MANIFEST))
    # cross-check the manifest against the protocol: the protocol wins
    for m in manifest:
        spk, system, key = labels[m["utt"]]
        assert (spk, key) == (m["speaker"], m["key"]), f"manifest disagrees with protocol for {m['utt']}"

    detector = get_detector("heuristic")
    speakers = sorted({m["speaker"] for m in manifest if m["key"] == "bonafide"})
    prints = {}
    for spk in speakers:
        enrol = [m["utt"] for m in manifest if m["speaker"] == spk and m["role"] == "enrol"]
        joined = np.concatenate([_wav(u) for u in enrol])
        prints[spk] = compute_voiceprint(joined, SR)
        assert prints[spk] is not None, f"enrolment for {spk} had too little speech"

    ctx = score_context()["context_score"]          # neutral: 0.0
    genuine_rows, call_rows, spoof_rows = [], [], []
    for spk in speakers:
        tests = [m["utt"] for m in manifest if m["speaker"] == spk and m["role"] == "test" and m["key"] == "bonafide"]
        for u in tests:
            wav = _wav(u)
            summary, rows, scored = run_call(wav, prints[spk], detector, ctx)
            genuine_rows.append(dict(utt=u, speaker=spk, secs=len(wav) / SR, chunks=len(rows),
                                     speaker_scored=scored, summary=summary, rows=rows))
        wav = np.concatenate([_wav(u) for u in tests])
        summary, rows, scored = run_call(wav, prints[spk], detector, ctx)
        call_rows.append(dict(utt="+".join(tests), speaker=spk, secs=len(wav) / SR, chunks=len(rows),
                              speaker_scored=scored, summary=summary, rows=rows))
    for m in manifest:
        if m["key"] != "spoof":
            continue
        wav = _wav(m["utt"])
        ref = prints.get(m["speaker"])
        summary, rows, scored = run_call(wav, ref, detector, ctx)
        spoof_rows.append(dict(utt=m["utt"], speaker=m["speaker"], system=m["system"], secs=len(wav) / SR,
                               chunks=len(rows), speaker_scored=scored, summary=summary, rows=rows))
    return dict(speakers=speakers, genuine=genuine_rows, calls=call_rows, spoof=spoof_rows)


def _worst(rows, k):
    return max(r[k] for r in rows)


def _print_table(title, items, with_system=False):
    print(f"\n{title}")
    hdr = (f"  {'clip':<28} {'speaker':<8} " + (f"{'sys':<4} " if with_system else "")
           + f"{'secs':>5} {'chk':>3} {'acoustic':>8} {'speaker':>8} {'context':>7} "
           f"{'peak risk':>9} {'level':<9} alert")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for it in items:
        s = it["summary"]
        spk_col = f"{_worst(it['rows'], 'speaker'):>8.3f}" if it["speaker_scored"] else f"{'n/a':>8}"
        print(f"  {it['utt']:<28} {it['speaker']:<8} " + (f"{it['system']:<4} " if with_system else "")
              + f"{it['secs']:>5.2f} {it['chunks']:>3} {_worst(it['rows'], 'acoustic'):>8.3f} {spk_col} "
              f"{it['rows'][0]['context']:>7.2f} {s['peak_risk_score']:>9.3f} {s['peak_level']:<9} "
              f"{'ALERT' if s['alert_triggered'] else '-'}")


# -------------------------------------------------------------- tests

def test_batch_uses_several_distinct_real_speakers(batch):
    assert len(batch["speakers"]) >= 5
    assert len(batch["genuine"]) >= 5


def test_no_genuine_utterance_reaches_high_or_critical(batch):
    """THE false-positive check, per utterance."""
    _print_table(f"GENUINE UTTERANCES (bonafide per protocol) — fused pipeline, neutral context, "
                 f"alert threshold {ALERT_THRESHOLD}", batch["genuine"])
    bad = [g for g in batch["genuine"] if g["summary"]["peak_level"] in ("HIGH", "CRITICAL")]
    n = len(batch["genuine"])
    print(f"\n  genuine utterances reaching HIGH/CRITICAL: {len(bad)}/{n}  "
          f"(false-positive rate {len(bad)/n:.0%})")
    for g in bad:
        worst = max(g["rows"], key=lambda r: r["risk"])
        print(f"    FALSE POSITIVE {g['utt']} ({g['speaker']}): peak {g['summary']['peak_risk_score']:.3f} "
              f"{g['summary']['peak_level']} driven by {worst['dominant']} "
              f"[acoustic {worst['acoustic']:.2f} speaker {worst['speaker']:.2f} context {worst['context']:.2f}]")
    assert not bad, f"{len(bad)} genuine utterance(s) flagged HIGH/CRITICAL: {[g['utt'] for g in bad]}"


def test_no_genuine_call_raises_an_alert(batch):
    """Same check on the joined ~6 s calls, where the sustain rule can fire."""
    _print_table("GENUINE CALLS (each speaker's two test utterances joined)", batch["calls"])
    alerted = [c for c in batch["calls"] if c["summary"]["alert_triggered"]]
    high = [c for c in batch["calls"] if c["summary"]["peak_level"] in ("HIGH", "CRITICAL")]
    print(f"\n  genuine calls raising an ALERT: {len(alerted)}/{len(batch['calls'])}   "
          f"reaching HIGH/CRITICAL: {len(high)}/{len(batch['calls'])}")
    assert not alerted
    assert not high


def test_speaker_layer_actually_scored_most_genuine_clips(batch):
    """If the speaker layer abstained on everything, a 0% FP rate would
    be meaningless. Report and require that it had an opinion on most."""
    scored = sum(g["speaker_scored"] for g in batch["genuine"])
    scored_calls = sum(c["speaker_scored"] for c in batch["calls"])
    print(f"\n  speaker layer had enough speech to score: {scored}/{len(batch['genuine'])} utterances, "
          f"{scored_calls}/{len(batch['calls'])} calls")
    # Single ASVspoof utterances are short; some legitimately fall under
    # MIN_SPEECH_SECONDS. Require that most were scored, so the 0% FP
    # figure above is not just the layer declining to answer.
    assert scored >= len(batch["genuine"]) // 2
    assert scored_calls >= (3 * len(batch["calls"])) // 4


def test_spoofs_for_information_only(batch):
    """Not an assertion on recall — the heuristic acoustic layer is
    known-weak on ASVspoof (see detector.py). Printed so the detection
    side is visible next to the false-positive side, not hidden."""
    _print_table("SPOOFED UTTERANCES (same pipeline, against the targeted speaker's voiceprint) — "
                 "for information", batch["spoof"], with_system=True)
    caught = [s for s in batch["spoof"] if s["summary"]["peak_level"] in ("HIGH", "CRITICAL")]
    med_acoustic = np.median([_worst(s["rows"], "acoustic") for s in batch["spoof"]])
    med_speaker = np.median([_worst(s["rows"], "speaker") for s in batch["spoof"] if s["speaker_scored"]])
    print(f"\n  spoofs reaching HIGH/CRITICAL: {len(caught)}/{len(batch['spoof'])}   "
          f"median acoustic {med_acoustic:.3f}   median speaker-mismatch {med_speaker:.3f}")
