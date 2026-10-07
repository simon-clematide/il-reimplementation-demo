---
language:
- en
- de
- fr
- it
license: cc-by-sa-4.0
tags:
- morphology
- inflection
- neural-transducer
- imitation-learning
- sigmorphon
datasets:
- conll-sigmorphon-2017
---

# Neural Transducer Inflection Models

This repository contains fine-tuned neural string transducer models (`neural_transducer` / package `trans`) for **morphological inflection generation**, trained with imitation learning on the **CoNLL-SIGMORPHON 2017 Shared Task 1** benchmark.

## Repository Contents
Each subfolder contains a complete, self-contained model bundle:
- `eng_100/` & `eng_1000/`: English inflection (100 vs. 1,000 examples)
- `deu_100/` & `deu_1000/`: German inflection (100 vs. 1,000 examples)
- `fra_100/` & `fra_1000/`: French inflection (100 vs. 1,000 examples)
- `ita_100/` & `ita_1000/`: Italian inflection (100 vs. 1,000 examples)

Each bundle directory includes:
- `best.model`: PyTorch state dict checkpoint
- `best.model.json`: Complete training parameters and architecture metadata
- `vocabulary.pkl`: UniMorph character and feature vocabularies
- `sed.pkl` & `sed.pkl.json`: Stochastic Edit Distance parameters and metadata
- `bundle_manifest.json`: Verification hashes and benchmark accuracy scores

## Training Setup
- **Architecture:** Transition-based neural string transducer (Makarov & Clematide, 2020)
- **Encoder:** 1-layer BiLSTM (hidden dimension 200, output locked dropout 0.3)
- **Decoder:** 1-layer LSTM (hidden dimension 200)
- **Optimizer:** AdamW (learning rate 0.001) with `ReduceLROnPlateau` scheduler
- **Seed:** 42

## Interactive Space Demo
Try the interactive demo space: [simon-clmtd/morphological-inflection](https://huggingface.co/spaces/simon-clmtd/morphological-inflection)
