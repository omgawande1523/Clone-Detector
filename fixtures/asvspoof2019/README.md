# ASVspoof 2019 LA — test fixture subset

60 utterances (3.4 MB, FLAC, 16 kHz mono 16-bit) from the **ASVspoof 2019
Logical Access (LA) dev partition**, used by the test suite as
certainly-labelled genuine and spoofed speech.

## Source

- Dataset: ASVspoof 2019, University of Edinburgh DataShare,
  https://datashare.ed.ac.uk/handle/10283/3336 (`LA.zip`, 7.1 GB).
- Files were extracted from `LA/ASVspoof2019_LA_dev/flac/` in that
  archive, byte-for-byte, via HTTP range requests on the zip — nothing
  was re-encoded or converted. They are FLAC because that is the
  dataset's native format and `soundfile` reads it directly.
- Citation: Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed
  and Fake Audio Detection", Interspeech 2019; Wang et al., "ASVspoof
  2019: a large-scale public database of synthesized, converted and
  replayed speech", Computer Speech & Language, 2020.

## Labels

`ASVspoof2019.LA.cm.dev.trl.subset.txt` holds the 60 matching lines of
the official protocol file `ASVspoof2019.LA.cm.dev.trl.txt`, verbatim.
**It is the only source of ground truth.** Column layout:

    SPEAKER_ID  UTTERANCE_ID  -  SYSTEM_ID  KEY

`KEY` is `bonafide` or `spoof`; `SYSTEM_ID` is `-` for bonafide and
A01–A06 for spoof (A01–A04 text-to-speech, A05–A06 voice conversion).
`manifest.json` records which role each file plays in the tests and is
cross-checked against the protocol at test time; the protocol wins if
they ever disagree.

## Selection

Deterministic (seed 20190), from the protocol:

- 8 speakers with bonafide utterances; per speaker 4 for enrolment and
  2 held out for testing = 48 bonafide.
- 2 spoofed utterances per attack system A01–A06 = 12 spoof, chosen
  where possible to target one of the 8 enrolled speakers so the
  speaker-consistency layer has a genuine voiceprint to compare
  against.

## Licence

The ASVspoof 2019 database is released under the Open Data Commons
Attribution License (ODC-By) v1.0 — see `LICENSE.txt` alongside this
file. Redistribution of this subset is permitted with attribution,
which the section above provides.
