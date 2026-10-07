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
short_description: Multilingual morphological inflection with neural transducers
---

# Neural Transducer: Morphological Inflection Demo

An interactive demonstration of transition-based neural string transduction (Makarov & Clematide, 2020) trained with imitation learning.

- **Benchmark:** CoNLL-SIGMORPHON 2017 Shared Task 1 (English, German, French, Italian)
- **Regimes:** 100 examples (Low) vs. 1,000 examples (Medium)
- **Features:** Side-by-side comparison, greedy and beam search decoding, and complete edit action alignment inspection (`COPY`, `DELETE`, `INSERT`, `SUBSTITUTE`).
