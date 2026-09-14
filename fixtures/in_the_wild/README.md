# In-the-Wild — test fixture subset

14 clips (2.7 MB, WAV 16 kHz mono 32-bit float, 1.0–5.7 s) from the
**In-the-Wild audio deepfake dataset**: 7 bona-fide and 7 spoofed
utterances, one of each for 7 different public figures.

## Source

- Dataset: Müller et al., "Does Audio Deepfake Detection Generalize?",
  Interspeech 2022 (arXiv:2203.16263). Mirror used:
  https://huggingface.co/datasets/mueller91/In-The-Wild
  (`release_in_the_wild.zip`, 8.16 GB).
- Files were extracted byte-for-byte from that archive via HTTP range
  requests; nothing was re-encoded.
- The dataset's own `attribution.txt` is included alongside.

## Why this set exists next to ASVspoof

The default `TransformerDetector` model
(`abhishtagatya/wav2vec2-base-960h-itw-deepfake`) was fine-tuned on
In-the-Wild. So this set measures the model **in-domain** — "how good is
it at its own job" — while `fixtures/asvspoof2019/` measures it
**cross-domain** against older, different attack systems. The two
numbers are reported separately on purpose; averaging them would be
misleading.

**Caveat:** the model card does not state which In-the-Wild split it
was fine-tuned on, so these 14 clips may overlap its training data.
Treat the in-domain figure as an upper bound, closer to training
accuracy than to a held-out score.

## Labels

`meta.subset.csv` holds the 14 matching rows of the dataset's official
`meta.csv`, verbatim: `file,speaker,label` with `label` ∈
{`bona-fide`, `spoof`}. It is the only source of ground truth.

## Selection

Deterministic (seed 2022): 7 speakers who have both classes, one
bona-fide and one spoof clip each, preferring clips of 2–8 s so they fit
this pipeline's 3-second chunking.

## Licence

CC BY-SA 4.0, per the dataset card. This subset is redistributed with
attribution under the same terms.
