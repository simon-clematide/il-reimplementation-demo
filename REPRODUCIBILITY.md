# Reproducibility & Benchmark Results

This table summarizes training runs across all 8 configurations on the **CoNLL-SIGMORPHON 2017 Task 1** benchmark.
- **Training seed:** 42
- **Data commit:** `519bb1677a7c747b3863d37069ffd395062288d8`
- **Model code:** `neural_transducer` (`trans`), commit `f71833192acd505bc1d3ba47d75798cfb778fc3d`

| Language | Regime | Best Epoch | Dev Str Acc | Dev Sym Acc | Test Str Acc | Test Sym Acc | Model Checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|
| **ENG** | 100 | 14 | 0.8610 | 0.9785 | 0.8830 | 0.9818 | `bd022f711ef9...` |
| **ENG** | 1000 | 19 | 0.9050 | 0.9830 | 0.9320 | 0.9875 | `33b1242f8ab5...` |
| **DEU** | 100 | 35 | 0.5200 | 0.8962 | 0.5040 | 0.9033 | `49966e9cf28d...` |
| **DEU** | 1000 | 16 | 0.7750 | 0.9286 | 0.7720 | 0.9447 | `45092d0ee7fd...` |
| **FRA** | 100 | 33 | 0.5090 | 0.8933 | 0.5200 | 0.8894 | `32b84f251d72...` |
| **FRA** | 1000 | 16 | 0.7710 | 0.9509 | 0.7530 | 0.9513 | `c25a03e611fe...` |
| **ITA** | 100 | 36 | 0.3490 | 0.8622 | 0.3130 | 0.8543 | `017f996634b4...` |
| **ITA** | 1000 | 20 | 0.8640 | 0.9781 | 0.8610 | 0.9768 | `cfeb9818e605...` |
