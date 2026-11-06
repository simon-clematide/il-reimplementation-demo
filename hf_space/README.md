---
title: Neural Transducer Inflection
emoji: 🔤
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.48.0
app_file: app.py
pinned: false
license: cc-by-sa-4.0
short_description: Multilingual inflection with neural transducers
---

# Neural Transducer: Morphological Inflection Demo

An interactive demonstration of transition-based neural string transduction (Makarov & Clematide, 2020) trained with imitation learning.

- **Benchmark:** CoNLL-SIGMORPHON 2017 Shared Task 1 (English, German, French, Italian)
- **Regimes:** 100, 1,000, and 10,000 training examples
- **Features:** Precomputed test gallery, live custom inputs, greedy decoding, and edit action inspection (`COPY`, `DELETE`, `INSERT`, `SUBSTITUTE`).


## Exploration gallery

The Explore tab starts with precomputed predictions for 320 held-out test items
(80 per language, excluding unchanged forms). Filter by model disagreement,
improvement from 100 to 10,000 examples, largest-model errors, or all-correct
cases. Selected cases illustrate behavior; they do not estimate accuracy.
Custom words and random test samples run live.

Regenerate from the local bundles after changing model weights:

```bash
python scripts/precompute_examples.py --per-language 80
```

Run that command from the repository root with the project dependencies installed.
The JSON records a deterministic sampling seed and SHA-256 fingerprints of each
local bundle, plus predictions and action traces. Commit the regenerated JSON
with model updates. No models are loaded to browse the gallery.

Custom inputs offer language-specific part-of-speech / word-form selectors,
then readable grammatical-form choices. These use the unique full feature
bundles already precomputed from the 10,000-example training split in
`data_index.json`; no unobserved combinations are synthesized. Participles
and converbs are grouped consistently even when the dataset also includes a
`V` tag. Manual UniMorph entry remains available. Regenerate the data index
with `python scripts/build_data_index.py` when the training data changes.
