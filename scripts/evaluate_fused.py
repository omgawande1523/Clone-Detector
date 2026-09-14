"""
Full fused-pipeline evaluation: every backend x every fixture set,
under the backend-aware weights, plus (a) the heuristic backend under
its OLD weights to show what zeroing acoustic changed, and (b) the
targeted-clone subset isolated.

    venv/Scripts/python scripts/evaluate_fused.py            # summary
    venv/Scripts/python scripts/evaluate_fused.py --files    # + per-clip tables

Needs the transformer weights cached locally for the transformer rows
(python -c "from src.detector import TransformerDetector; TransformerDetector()").
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import get_detector                                   # noqa: E402
from src.fusion import (weights_for_backend, FusionWeights,             # noqa: E402
                        TRANSFORMER_WEIGHTS, HEURISTIC_WEIGHTS)
from src.risk_engine import ALERT_THRESHOLD                             # noqa: E402
from src.evaluation import (evaluate_asvspoof, evaluate_itw, metrics, print_table,  # noqa: E402
                            roundtrip_48k, fixture_items_for_roundtrip)

FILES = "--files" in sys.argv


def w(x: FusionWeights):
    return f"{x.acoustic:.2f}/{x.speaker:.2f}/{x.context:.2f}"


def line(backend, weights, res, m, note=""):
    ps = "  ".join(f"{k}:{v['caught']}/{v['n']}" for k, v in m["per_system"].items()) if len(m["per_system"]) > 1 else ""
    return (f"  {backend:<12} {res['dataset']:<17} {w(weights):<15} "
            f"{m['genuine_fp']}/{m['n_genuine']:<4} {m['genuine_alert']}/{m['n_genuine']:<4} "
            f"{m['genuine_peak_max']:>8.3f} | {m['spoof_caught']}/{m['n_spoof']:<4} {m['spoof_alert']}/{m['n_spoof']:<4} "
            f"{m['spoof_peak_med']:>8.3f} | {ps}{note}")


hdr = (f"  {'backend':<12} {'dataset':<17} {'weights a/s/c':<15} "
       f"{'gen HI+':<6} {'gen ALT':<6} {'gen max':>8} | {'spf HI+':<6} {'spf ALT':<6} {'spf med':>8} | per system HIGH+")

print("=" * 118)
print(f"FULL FUSED PIPELINE  (acoustic + speaker vs enrolled print + neutral context -> EMA -> level; "
      f"HI+ = HIGH/CRITICAL, alert threshold {ALERT_THRESHOLD})")
print("=" * 118)
print(hdr)
print("  " + "-" * 116)

results = {}
for backend in ("heuristic", "transformer"):
    try:
        det = get_detector(backend)
    except Exception as e:
        print(f"  {backend:<12} SKIPPED: {e}")
        continue
    weights = weights_for_backend(det.backend)
    for evaluator in (evaluate_asvspoof, evaluate_itw):
        t0 = time.time()
        res = evaluator(det, weights)
        m = metrics(res)
        results[(backend, res["dataset"])] = (res, m, weights)
        print(line(backend, weights, res, m, f"   ({time.time()-t0:.0f}s)"))

# --- what did zeroing heuristic-acoustic actually change? --------------
print("\n" + "=" * 118)
print("HEURISTIC BACKEND: OLD weights (0.45/0.45/0.10) vs NEW (0.00/0.90/0.10) — same clips, same pipeline")
print("=" * 118)
print(hdr)
print("  " + "-" * 116)
det = get_detector("heuristic")
old = FusionWeights(0.45, 0.45, 0.10)
for evaluator in (evaluate_asvspoof, evaluate_itw):
    r_old, r_new = evaluator(det, old), evaluator(det, HEURISTIC_WEIGHTS)
    print(line("heuristic", old, r_old, metrics(r_old), "   <- old"))
    print(line("heuristic", HEURISTIC_WEIGHTS, r_new, metrics(r_new), "   <- new"))
    # per-clip delta
    delta = [(a["utt"], a["peak"], b["peak"]) for a, b in zip(r_old["genuine"] + r_old["spoof"], r_new["genuine"] + r_new["spoof"])
             if abs(a["peak"] - b["peak"]) > 1e-9]
    print(f"      clips whose peak risk changed at all: {len(delta)}/{len(r_old['genuine']) + len(r_old['spoof'])}"
          + ("" if not delta else "   max |delta| = %.3f" % max(abs(a - b) for _, a, b in delta)))

# --- targeted clone / CEO-fraud scenario ------------------------------
print("\n" + "=" * 118)
print("TARGETED-CLONE SCENARIO: spoof is a clone OF THE ENROLLED SPEAKER, scored against that speaker's own voiceprint")
print("=" * 118)
for backend in ("heuristic", "transformer"):
    key = (backend, "ASVspoof 2019 LA")
    if key not in results:
        continue
    res, m, weights = results[key]
    s = res["spoof"]
    caught = [r for r in s if r["level"] in ("HIGH", "CRITICAL")]
    alerted = [r for r in s if r["alert"]]
    print(f"\n  {backend} backend, weights {w(weights)}:  ASVspoof A01-A06, all 12 target an enrolled speaker")
    print(f"    reach HIGH/CRITICAL : {len(caught)}/12  ({len(caught)/12:.0%})")
    print(f"    raise an ALERT      : {len(alerted)}/12  (sustain rule needs 2 chunks; {sum(r['chunks'] >= 2 for r in s)}/12 clips have >= 2)")
    print(f"    speaker-mismatch    : median {sorted(r['worst_speaker'] for r in s)[6]:.3f}  -- the clone passes the voiceprint check")
    print(f"    per system          : " + "  ".join(f"{k}:{v['caught']}/{v['n']}" for k, v in m["per_system"].items()))
    key2 = (backend, "In-the-Wild")
    if key2 in results:
        res2, m2, _ = results[key2]
        print(f"  {backend} backend, In-the-Wild: 7 clones of enrolled speakers -> HIGH/CRITICAL {m2['spoof_caught']}/7, "
              f"median speaker-mismatch {sorted(r['worst_speaker'] for r in res2['spoof'])[3]:.3f}")

# --- 48 kHz round trip: does an uploaded phone/laptop recording get the same verdict?
print("\n" + "=" * 118)
print("48 kHz ROUND TRIP: every test clip upsampled to 48 kHz, resampled back by the pipeline's resample_to(), re-scored")
print("=" * 118)
for backend in ("heuristic", "transformer"):
    try:
        det = get_detector(backend)
    except Exception:
        continue
    rows, m = roundtrip_48k(det, weights_for_backend(det.backend), fixture_items_for_roundtrip())
    changed = [r for r in rows if r["level0"] != r["level1"] or r["alert0"] != r["alert1"]]
    print(f"  {backend:<12} {m['n']} clips: same level {m['same_level']}/{m['n']}, same alert {m['same_alert']}/{m['n']}, "
          f"max |delta peak| {m['max_delta_peak']:.3f}, max |delta acoustic| {m['max_delta_acoustic']:.3f}")
    for r in changed:
        print(f"      CHANGED {r['label']}: {r['level0']} {r['peak0']:.3f} -> {r['level1']} {r['peak1']:.3f}")

if FILES:
    for (backend, dataset), (res, m, weights) in results.items():
        print("\n" + "#" * 118 + f"\n# {backend} x {dataset}  weights {w(weights)}\n" + "#" * 118)
        print_table("GENUINE", res["genuine"])
        if res["calls"]:
            print_table("GENUINE CALLS (joined)", res["calls"])
        print_table("SPOOF (targeted clones)", res["spoof"], with_system=True)
