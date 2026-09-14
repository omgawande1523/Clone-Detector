"""
evaluation.py — runs the FULL fused pipeline (acoustic + speaker-
consistency against an enrolled voiceprint + context -> fusion -> risk
engine) over the committed labelled fixture sets, for any backend and
any weight vector. Used by the tests and by scripts/evaluate_fused.py,
so the numbers in the README come from the same code path the API
executes, not from a per-signal shortcut.

Datasets (see fixtures/*/README.md for provenance and licence):

  ASVspoof 2019 LA  — per speaker: 4 bonafide utterances joined for
                      enrolment, 2 held-out bonafide for testing (also
                      joined into one ~4-9 s "call"), and the spoofs
                      that TARGET that speaker scored against that
                      speaker's print. Every spoof in the fixture set
                      targets an enrolled speaker, so the spoof rows
                      ARE the targeted-clone / CEO-fraud scenario.

  In-the-Wild       — one bonafide + one spoof clip per speaker. The
                      bonafide clip is the only genuine audio for that
                      speaker, so it is used to enrol AND (self-)tested;
                      the speaker layer will trivially match it. The
                      spoof is scored against that print = targeted
                      clone again. Treat the bonafide rows as the
                      acoustic layer alone, effectively.

Labels come from the datasets' own protocol/meta files, never from
filenames.
"""

import json
import os

import numpy as np

from .audio_pipeline import load_mono, resample_to, DEFAULT_CHUNK_SECONDS
from .fusion import fuse, FusionWeights
from .risk_engine import RiskEngine
from .context_engine import score_context
from .speaker_consistency import compute_voiceprint, mismatch_between

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASV = os.path.join(ROOT, "fixtures", "asvspoof2019")
ITW = os.path.join(ROOT, "fixtures", "in_the_wild")
SR = 16000
CHUNK_S = DEFAULT_CHUNK_SECONDS


def _chunks(wav, chunk_s=CHUNK_S):
    n = int(SR * chunk_s)
    for start in range(0, len(wav), n):
        seg = wav[start:start + n]
        if len(seg) < SR * 0.2:
            break
        yield start / SR, seg


def run_call(wav, reference_print, detector, weights: FusionWeights, ctx_score=0.0,
             chunk_s=CHUNK_S):
    """One piece of audio through the full pipeline. Returns
    (risk-engine summary, per-chunk rows, speaker_layer_had_an_opinion)."""
    engine = RiskEngine()
    rows, speaker_scored = [], False
    for t, seg in _chunks(wav, chunk_s):
        acoustic = detector.predict_detailed(seg, SR)["spoof_prob"]
        live = compute_voiceprint(seg, SR) if reference_print is not None else None
        if live is not None:
            speaker_scored = True
            mismatch = mismatch_between(reference_print, live)
        else:
            mismatch = 0.0
        f = fuse(acoustic, mismatch, ctx_score, weights)
        r = engine.update(f["fused_score"])
        rows.append(dict(t=t, acoustic=acoustic, speaker=mismatch, context=ctx_score,
                         risk=r["risk_score"], level=r["level"], alert=r["alert"],
                         dominant=f["dominant_signal"]))
    return engine.summary(), rows, speaker_scored


def _row(utt, speaker, key, system, wav, summary, rows, scored):
    return dict(utt=utt, speaker=speaker, key=key, system=system, secs=len(wav) / SR,
                chunks=len(rows), speaker_scored=scored, summary=summary, rows=rows,
                worst_acoustic=max(r["acoustic"] for r in rows),
                worst_speaker=max(r["speaker"] for r in rows),
                peak=summary["peak_risk_score"], level=summary["peak_level"],
                alert=summary["alert_triggered"])


# ------------------------------------------------------------ ASVspoof

def asvspoof_labels():
    out = {}
    for line in open(os.path.join(ASV, "ASVspoof2019.LA.cm.dev.trl.subset.txt")):
        spk, utt, _env, system, key = line.split()
        out[utt] = (spk, system, key)
    return out


def _asv_wav(utt):
    wav, sr = load_mono(os.path.join(ASV, "flac", utt + ".flac"))
    assert sr == SR
    return wav


def evaluate_asvspoof(detector, weights: FusionWeights, chunk_s=CHUNK_S):
    labels = asvspoof_labels()
    manifest = json.load(open(os.path.join(ASV, "manifest.json")))
    for m in manifest:                       # protocol is the authority
        spk, system, key = labels[m["utt"]]
        assert (spk, key) == (m["speaker"], m["key"]), m["utt"]

    speakers = sorted({m["speaker"] for m in manifest if m["key"] == "bonafide"})
    prints = {}
    for spk in speakers:
        enrol = [m["utt"] for m in manifest if m["speaker"] == spk and m["role"] == "enrol"]
        prints[spk] = compute_voiceprint(np.concatenate([_asv_wav(u) for u in enrol]), SR)
        assert prints[spk] is not None

    ctx = score_context()["context_score"]
    genuine, calls, spoof = [], [], []
    for spk in speakers:
        tests = [m["utt"] for m in manifest
                 if m["speaker"] == spk and m["role"] == "test" and m["key"] == "bonafide"]
        for u in tests:
            wav = _asv_wav(u)
            genuine.append(_row(u, spk, "bonafide", "-", wav, *run_call(wav, prints[spk], detector, weights, ctx, chunk_s)))
        wav = np.concatenate([_asv_wav(u) for u in tests])
        calls.append(_row("+".join(tests), spk, "bonafide", "-", wav, *run_call(wav, prints[spk], detector, weights, ctx, chunk_s)))
    for m in manifest:
        if m["key"] != "spoof":
            continue
        wav = _asv_wav(m["utt"])
        ref = prints[m["speaker"]]           # every spoof targets an enrolled speaker
        spoof.append(_row(m["utt"], m["speaker"], "spoof", m["system"], wav,
                          *run_call(wav, ref, detector, weights, ctx, chunk_s)))
    return dict(dataset="ASVspoof 2019 LA", speakers=speakers, genuine=genuine, calls=calls, spoof=spoof)


# --------------------------------------------------------- In-the-Wild

def evaluate_itw(detector, weights: FusionWeights, chunk_s=CHUNK_S):
    rows = []
    for line in open(os.path.join(ITW, "meta.subset.csv")):
        if line.startswith("file,"):
            continue
        fname, speaker, label = line.strip().split(",")
        rows.append((fname, speaker, label))
    by_spk = {}
    for fname, speaker, label in rows:
        by_spk.setdefault(speaker, {})[label] = fname

    ctx = score_context()["context_score"]
    genuine, spoof = [], []
    for speaker, d in sorted(by_spk.items()):
        bwav, sr = load_mono(os.path.join(ITW, "wav", d["bona-fide"]))
        ref = compute_voiceprint(bwav, sr)
        genuine.append(_row(d["bona-fide"], speaker, "bonafide", "-", bwav,
                            *run_call(bwav, ref, detector, weights, ctx, chunk_s)))
        swav, sr = load_mono(os.path.join(ITW, "wav", d["spoof"]))
        spoof.append(_row(d["spoof"], speaker, "spoof", "itw", swav,
                          *run_call(swav, ref, detector, weights, ctx, chunk_s)))
    return dict(dataset="In-the-Wild", speakers=sorted(by_spk), genuine=genuine, calls=[], spoof=spoof)


# ------------------------------------------------------ 48 kHz round trip

def roundtrip_48k(detector, weights: FusionWeights, items):
    """Does a verdict survive being delivered at phone/laptop sample
    rates? Each 16 kHz clip is upsampled to 48 kHz (what a phone or
    browser would hand us) and resampled back through the pipeline's
    own resample_to(); the fused verdict is compared with the original.

    items: iterable of (label, wav16k, reference_print_or_None).
    Returns per-item rows and a summary."""
    rows = []
    ctx = score_context()["context_score"]
    for label, wav, ref in items:
        s0, r0, _ = run_call(wav, ref, detector, weights, ctx)
        wav48 = resample_to(wav, SR, 48000)
        back = resample_to(wav48, 48000, SR)
        s1, r1, _ = run_call(back, ref, detector, weights, ctx)
        rows.append(dict(label=label, level0=s0["peak_level"], level1=s1["peak_level"],
                         peak0=s0["peak_risk_score"], peak1=s1["peak_risk_score"],
                         alert0=s0["alert_triggered"], alert1=s1["alert_triggered"],
                         acoustic0=max(r["acoustic"] for r in r0), acoustic1=max(r["acoustic"] for r in r1)))
    same_level = sum(r["level0"] == r["level1"] for r in rows)
    same_alert = sum(r["alert0"] == r["alert1"] for r in rows)
    max_dpeak = max((abs(r["peak0"] - r["peak1"]) for r in rows), default=0.0)
    max_dac = max((abs(r["acoustic0"] - r["acoustic1"]) for r in rows), default=0.0)
    return rows, dict(n=len(rows), same_level=same_level, same_alert=same_alert,
                      max_delta_peak=max_dpeak, max_delta_acoustic=max_dac)


def fixture_items_for_roundtrip():
    """Every test clip in both fixture sets with its proper reference
    print, as (label, wav, ref) for roundtrip_48k()."""
    items = []
    manifest = json.load(open(os.path.join(ASV, "manifest.json")))
    speakers = sorted({m["speaker"] for m in manifest if m["key"] == "bonafide"})
    prints = {}
    for spk in speakers:
        enrol = [m["utt"] for m in manifest if m["speaker"] == spk and m["role"] == "enrol"]
        prints[spk] = compute_voiceprint(np.concatenate([_asv_wav(u) for u in enrol]), SR)
    for m in manifest:
        if m["role"] == "test":
            items.append((f"asv:{m['utt']}:{m['key']}:{m['system']}", _asv_wav(m["utt"]), prints[m["speaker"]]))
    by_spk = {}
    for line in open(os.path.join(ITW, "meta.subset.csv")):
        if line.startswith("file,"):
            continue
        fname, speaker, label = line.strip().split(",")
        by_spk.setdefault(speaker, {})[label] = fname
    for speaker, d in sorted(by_spk.items()):
        bwav, _ = load_mono(os.path.join(ITW, "wav", d["bona-fide"]))
        ref = compute_voiceprint(bwav, SR)
        items.append((f"itw:{d['bona-fide']}:bonafide", bwav, ref))
        swav, _ = load_mono(os.path.join(ITW, "wav", d["spoof"]))
        items.append((f"itw:{d['spoof']}:spoof", swav, ref))
    return items


# ------------------------------------------------------------ metrics

def metrics(result):
    g, s = result["genuine"], result["spoof"]
    hi = lambda r: r["level"] in ("HIGH", "CRITICAL")
    per_system = {}
    for sysid in sorted({r["system"] for r in s}):
        rs = [r for r in s if r["system"] == sysid]
        per_system[sysid] = dict(n=len(rs), caught=sum(hi(r) for r in rs), alerted=sum(r["alert"] for r in rs),
                                 peaks=[r["peak"] for r in rs])
    return dict(
        n_genuine=len(g), genuine_fp=sum(hi(r) for r in g), genuine_alert=sum(r["alert"] for r in g),
        genuine_peak_max=max(r["peak"] for r in g),
        n_calls=len(result["calls"]), calls_fp=sum(hi(r) for r in result["calls"]),
        calls_alert=sum(r["alert"] for r in result["calls"]),
        n_spoof=len(s), spoof_caught=sum(hi(r) for r in s), spoof_alert=sum(r["alert"] for r in s),
        spoof_peak_med=float(np.median([r["peak"] for r in s])),
        speaker_scored=sum(r["speaker_scored"] for r in g),
        per_system=per_system,
    )


def print_table(title, items, with_system=False):
    print(f"\n{title}")
    hdr = (f"  {'clip':<28} {'speaker':<18} " + (f"{'sys':<4} " if with_system else "")
           + f"{'secs':>5} {'chk':>3} {'acoustic':>8} {'speaker':>8} {'peak risk':>9} {'level':<9} alert")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for it in items:
        spk = f"{it['worst_speaker']:>8.3f}" if it["speaker_scored"] else f"{'n/a':>8}"
        print(f"  {it['utt']:<28} {it['speaker']:<18} " + (f"{it['system']:<4} " if with_system else "")
              + f"{it['secs']:>5.2f} {it['chunks']:>3} {it['worst_acoustic']:>8.3f} {spk} "
              f"{it['peak']:>9.3f} {it['level']:<9} {'ALERT' if it['alert'] else '-'}")
