# Neural Transducer: Morphological Inflection Demo Ecosystem

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simon-clematide/il-reimplementation-demo/blob/main/notebooks/morphological_inflection.ipynb)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/simon-clmtd/morphological-inflection)

This repository provides a complete demonstration ecosystem for transition-based neural string transducers ([Makarov & Clematide, 2020](https://aclanthology.org/2020.sigmorphon-1.19)) using imitation learning on the **CoNLL-SIGMORPHON 2017 Shared Task 1** benchmark.

It showcases morphological inflection generation across four languages:
- **English (`eng`)**
- **German (`deu`)**
- **French (`fra`)**
- **Italian (`ita`)**

Comparing generalization under:
- **Low-resource regime:** 100 training examples
- **Medium-resource regime:** 1,000 training examples

---

## Ecosystem Components

1. **[Hugging Face Space (`hf_space/`)](https://huggingface.co/spaces/simon-clmtd/morphological-inflection):**
   - Interactive Gradio web application.
   - Side-by-side comparison (100 vs. 1,000 examples).
   - Real-time step-by-step edit action traces (`COPY`, `DELETE`, `INSERT`, `SUBSTITUTE`, `END`).
   - Curated linguistic examples for regular rules and irregular stem alternations.

2. **[Colab Tutorial Notebook (`notebooks/morphological_inflection.ipynb`)](https://colab.research.google.com/github/simon-clematide/il-reimplementation-demo/blob/main/notebooks/morphological_inflection.ipynb):**
   - Step-by-step interactive walkthrough in Google Colab.
   - Instant inference with pre-trained Hugging Face bundles.
   - Fast from-scratch training with `trans-train`.

3. **Pre-trained Models (`simon-clmtd/neural-transducer-inflection`):**
   - 8 fine-tuned self-contained model bundles hosted on the Hugging Face Model Hub.
   - Includes state dicts, vocabularies, Stochastic Edit Distance (SED) parameters, and architecture metadata.

---

## Project Structure

```text
il-reimplementation-demo/
├── README.md                          # This overview
├── IMPLEMENTATION_PLAN.md             # Complete design specification and decisions
├── REPRODUCIBILITY.md                 # Benchmark results across all 8 configurations
├── Makefile                           # Development automation commands
├── configs/
│   └── train.yaml                     # Frozen training hyperparameters
├── data/
│   ├── MANIFEST.json                  # Pinned data verification hashes
│   └── README.md                      # Data provenance & UniMorph / CC BY-SA 3.0 license
├── demo_core/                         # Shared inference & presentation layer
│   ├── loader.py                      # Metadata-driven bundle loader
│   ├── predict.py                     # Inference engine & action trace formatter
│   └── features.py                    # UniMorph tag parser & OOV validator
├── hf_space/                          # Gradio Space repository files
│   ├── app.py                         # Web interface
│   ├── inference.py                   # On-demand Hub bundle downloader & LRU cache
│   └── requirements.txt               # Space environment requirements
├── notebooks/
│   └── morphological_inflection.ipynb # Colab-runnable tutorial notebook
└── scripts/
    ├── fetch_data.py                  # Pinned benchmark downloader
    ├── validate_data.py               # Data schema, leakage, and hash validation
    ├── train_bundle.py                # Single-bundle training CLI
    ├── train_all_models.py            # Orchestrator for all 8 bundles + REPRODUCIBILITY.md
    ├── package_bundle.py              # Bundle manifest packaging
    └── upload_hf.py                   # Model Hub release uploader
```

---

## Local Setup & Quickstart

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r hf_space/requirements.txt
```

### 2. Fetch and Validate Benchmark Data
```bash
python scripts/fetch_data.py
python scripts/validate_data.py
```

### 3. Run the Gradio Demo Locally
```bash
cd hf_space
python app.py
```
Visit `http://localhost:7860` in your browser.

---

## Citation & References
- **Peter Makarov and Simon Clematide (2020):** *CLUZH at SIGMORPHON 2020 Shared Task 0: Exploration of Architecture and Training Innovations in Neural Transducers*. ACL Anthology.
- **Silvan Wehrli, Simon Clematide, and Peter Makarov (2022):** *CLUZH at SIGMORPHON 2022 Shared Tasks on Morpheme Segmentation and Inflection Generation*. ACL Anthology.
- **CoNLL-SIGMORPHON 2017 Shared Task 1:** Universal Morphological Reinflection.
