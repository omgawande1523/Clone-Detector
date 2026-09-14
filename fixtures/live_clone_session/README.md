# Live clone session — regression fixture

Four 5-second windows (`chunk001..004.wav`, 16 kHz mono PCM) captured
from a real `WS /ws/live` run on 2026-09-14 (session
`20260914-221742-LA_0075`, `LIVE_CAPTURE_DIR` enabled), transformer
backend, speaker `LA_0075` enrolled from the ASVspoof 2019 LA fixture
utterances. The audio streamed was the A02/A03 clones of `LA_0075` from
`fixtures/asvspoof2019/` — a clone OF THE ENROLLED SPEAKER, the threat
this system exists for.

The transformer scored every one of these windows **0.984–0.996
synthetic** and the system never alerted; it never reached HIGH. With the
weighted-average fusion in force at the time (acoustic cap 0.45) and a
symmetric EMA seeded by a near-silent window 0, the fused score could not
exceed 0.448 and the EMA reached only 0.367 (`manifest.json` holds the
per-window scores as captured).

`tests/test_live_clone_replay.py` replays these windows through the full
`api._score_chunk -> RiskEngine` path with `LA_0075` enrolled and asserts
an alert fires by window 3 at the latest. It failed on the pre-fix code.
