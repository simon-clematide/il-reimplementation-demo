# CoNLL-SIGMORPHON 2017 Shared Task 1 Benchmark Data

This dataset directory contains the official morphological inflection benchmark data splits from the **CoNLL-SIGMORPHON 2017 Shared Task 1: Universal Morphological Reinflection**.

## Pinned Source & Provenance
- **Repository:** [https://github.com/sigmorphon/conll2017](https://github.com/sigmorphon/conll2017)
- **Commit:** `519bb1677a7c747b3863d37069ffd395062288d8`
- **Languages:**
  - English (`eng`)
  - German (`deu`)
  - French (`fra`)
  - Italian (`ita`)

## License
The original datasets in CoNLL-SIGMORPHON 2017 were harvested and curated from [UniMorph](https://unimorph.github.io/) / Wiktionary, licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)).

## Data Splits & Characteristics
For each language:
- **Low-resource train:** 100 triples (`<iso>-train-low.tsv`)
- **Medium-resource train:** 1,000 triples (`<iso>-train-medium.tsv`)
- **Development set:** 1,000 triples (`<iso>-dev.tsv`)
- **Test set (Gold):** 1,000 triples (`<iso>-test.tsv`, from `answers/task1/<lang>-uncovered-test`)

### Validation Summary (`scripts/validate_data.py`)
- **Subset property:** `train-low` is a 100% strict subset of `train-medium` for all 4 languages.
- **Triple leakage:** Exactly 0 overlap between `train`, `dev`, and `test` across all languages.
- **Tag OOV:** 0 unseen feature tags in `dev` and `test` relative to `train-medium`.
- All hashes and counts are recorded in `MANIFEST.json`.
