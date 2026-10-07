#!/usr/bin/env python3
"""
Build lightweight data index for the demo:
- Training set coverage: unique lemmas and unique feature bundles for 100 and 1000 regimes.
- Dev and test splits for random sampling and evaluation with gold targets.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
HF_SPACE = REPO_ROOT / "hf_space"
DEMO_CORE = REPO_ROOT / "demo_core"

LANGUAGES = ["eng", "deu", "fra", "ita"]

def load_tsv(path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                items.append({
                    "lemma": parts[0],
                    "target": parts[1],
                    "features": parts[2],
                })
    return items

def build_index():
    index = {}

    for lang in LANGUAGES:
        low_items = load_tsv(DATA_RAW / f"{lang}-train-low.tsv")
        med_items = load_tsv(DATA_RAW / f"{lang}-train-medium.tsv")
        dev_items = load_tsv(DATA_RAW / f"{lang}-dev.tsv")
        test_items = load_tsv(DATA_RAW / f"{lang}-test.tsv")

        index[lang] = {
            "training_coverage": {
                "100": {
                    "lemmas": sorted(list({x["lemma"] for x in low_items})),
                    "features": sorted(list({x["features"] for x in low_items})),
                },
                "1000": {
                    "lemmas": sorted(list({x["lemma"] for x in med_items})),
                    "features": sorted(list({x["features"] for x in med_items})),
                },
            },
            "dev": dev_items,
            "test": test_items,
        }

    out_space = HF_SPACE / "data_index.json"
    with open(out_space, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Wrote index to {out_space} ({out_space.stat().st_size / 1024:.1f} KB)")

    out_core = DEMO_CORE / "data_index.json"
    with open(out_core, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Wrote index to {out_core} ({out_core.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    build_index()
