"""Regression tests: enrolments must survive another process writing.

VoiceprintStore read the file once at construction and wrote back its
own in-memory dict. Two server processes (a restart while another was
running) each held a stale snapshot, and the second write dropped what
the first had added. That is not hypothetical: "ceo" was enrolled and
working at 22:02 on 2026-09-14 (live_capture/20260914-220233-ceo/ shows
speaker_mismatch 1.00 and an alert on window 1) and was gone from
voiceprints.json by 22:18, leaving every later "ceo" session running
with the speaker layer entirely off.

The atomic write added earlier made the file safe against CORRUPTION,
not against being CLOBBERED. enroll() now re-reads and merges from disk
under the lock, and reads refresh the cache when the file's mtime moves.
"""

import json
import os
import threading
import time

import pytest

from src.audio_pipeline import load_mono
from src.speaker_consistency import VoiceprintStore

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ENROLL_A = os.path.join(SAMPLES, "enroll_speakerA.wav")
GENUINE_A = os.path.join(SAMPLES, "genuine_speakerA.wav")

pytestmark = pytest.mark.skipif(not os.path.exists(ENROLL_A), reason="samples not present")


@pytest.fixture(scope="module")
def audio():
    return load_mono(ENROLL_A)


def test_two_stores_over_one_path_do_not_clobber_each_other(tmp_path, audio):
    """THE regression case, in miniature: two stores constructed before
    either wrote, as two server processes would be."""
    path = str(tmp_path / "vp.json")
    a = VoiceprintStore(path=path)
    b = VoiceprintStore(path=path)          # stale snapshot: empty
    wav, sr = audio

    a.enroll("ceo", wav, sr)
    b.enroll("LA_0075", wav, sr)            # this used to drop "ceo"

    on_disk = json.load(open(path))
    assert set(on_disk) == {"ceo", "LA_0075"}


def test_a_store_sees_an_enrolment_made_elsewhere(tmp_path, audio):
    path = str(tmp_path / "vp.json")
    reader = VoiceprintStore(path=path)
    assert reader.speaker_ids() == []
    assert reader.has("ceo") is False

    time.sleep(0.01)                        # ensure a distinct mtime
    writer = VoiceprintStore(path=path)
    writer.enroll("ceo", *audio)

    # same instance, no reconstruction
    assert reader.has("ceo") is True
    assert "ceo" in reader.speaker_ids()
    assert reader.similarity("ceo", *load_mono(GENUINE_A)) is not None


def test_concurrent_enrolments_all_land(tmp_path, audio):
    path = str(tmp_path / "vp.json")
    wav, sr = audio
    stores = [VoiceprintStore(path=path) for _ in range(4)]
    errors = []

    def enrol(store, name):
        try:
            store.enroll(name, wav, sr)
        except Exception as e:                       # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=enrol, args=(s, f"spk{i}")) for i, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert set(json.load(open(path))) == {f"spk{i}" for i in range(4)}
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_reenrolment_replaces_rather_than_duplicates(tmp_path, audio):
    path = str(tmp_path / "vp.json")
    s = VoiceprintStore(path=path)
    s.enroll("ceo", *audio)
    first = json.load(open(path))["ceo"]
    time.sleep(0.01)
    s.enroll("ceo", *load_mono(GENUINE_A))
    data = json.load(open(path))
    assert set(data) == {"ceo"}
    assert data["ceo"] != first


def test_external_deletion_is_noticed(tmp_path, audio):
    path = str(tmp_path / "vp.json")
    s = VoiceprintStore(path=path)
    s.enroll("ceo", *audio)
    assert s.has("ceo") is True
    os.remove(path)
    assert s.has("ceo") is False
    assert s.speaker_ids() == []
