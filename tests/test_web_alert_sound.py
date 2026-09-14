"""Static smoke checks on web/index.html for the audible alert.

This is deliberately minimal: it asserts the structural properties that
keep the beep from becoming a per-window nuisance or a page-breaking
exception, and nothing more. It is not a substitute for pressing
"Test alert sound" before the demo.
"""

import os
import re

HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")


def _js():
    html = open(HTML, encoding="utf-8").read()
    return re.search(r"<script>(.*)</script>", html, re.S).group(1)


def _function_body(js, name):
    start = js.index(f"function {name}(")
    depth, i = 0, js.index("{", start)
    for j in range(i, len(js)):
        if js[j] == "{":
            depth += 1
        elif js[j] == "}":
            depth -= 1
            if depth == 0:
                return js[i:j + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def test_transition_guard_variable_exists():
    js = _js()
    assert "alertFloor" in js
    assert "prevLevel" in js
    assert "function levelTransitionSound(" in js


def test_render_live_chunk_never_beeps_unconditionally():
    """Inside renderLiveChunk, every playAlertSound call must sit behind
    a condition; the per-level sound goes through the guard function."""
    body = _function_body(_js(), "renderLiveChunk")
    for line in body.splitlines():
        if "playAlertSound(" in line:
            assert re.search(r"\bif\s*\(", line), f"unconditional beep in renderLiveChunk: {line.strip()}"
    assert "levelTransitionSound(live, m.level)" in body


def test_guard_only_fires_on_rise_above_floor():
    body = _function_body(_js(), "levelTransitionSound")
    assert "cur > floor" in body
    assert "alertFloor = 'LOW'" in body          # dropping below HIGH re-arms


def test_audio_calls_are_wrapped_in_try_catch():
    js = _js()
    for fn in ("playAlertSound", "resumeAudio"):
        body = _function_body(js, fn)
        assert "try {" in body and "catch" in body, fn


def test_two_distinct_patterns_and_modest_gain():
    body = _function_body(_js(), "playAlertSound")
    assert "880" in body and "1175" in body
    gains = [float(g) for g in re.findall(r"tone\(ac, \d+, [^,]+, [^,]+, ([0-9.]+)\)", body)]
    assert gains and max(gains) <= 0.2


def test_envelope_is_ramped_not_hard_switched():
    body = _function_body(_js(), "tone")
    assert "exponentialRampToValueAtTime" in body


def test_controls_and_persistence():
    html = open(HTML, encoding="utf-8").read()
    assert 'id="soundToggle"' in html and 'id="soundTest"' in html
    js = _js()
    assert "localStorage.getItem('sih_alert_sound')" in js
    assert "localStorage.setItem('sih_alert_sound'" in js
    # both storage accesses are guarded
    for m in re.finditer(r"localStorage\.(get|set)Item", js):
        window = js[max(0, m.start() - 120):m.start()]
        assert "try" in window, "localStorage access not wrapped in try"


def test_audio_resumed_from_user_gestures():
    js = _js()
    assert "resumeAudio();" in _function_body(js, "startLive")
    analyze = js[js.index("$('analyzeBtn').onclick"):js.index("$('analyzeBtn').onclick") + 200]
    assert "resumeAudio();" in analyze


def test_upload_path_beeps_once_on_alert_only():
    body = _function_body(_js(), "render")
    lines = [l for l in body.splitlines() if "playAlertSound(" in l]
    assert len(lines) == 1 and "alert_triggered" in lines[0] and "if (" in lines[0]
