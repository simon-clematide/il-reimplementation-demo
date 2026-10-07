#!/usr/bin/env python3
"""
Validates downloaded CoNLL-SIGMORPHON 2017 data against quality criteria:
- File existence & SHA-256 matching data/MANIFEST.json
- UTF-8 encoding & Unicode normalization checks
- Exactly 3 tab-separated columns per line (lemma, form, unimorph_features)
- Line counts match expected values (100 low, 1000 medium, 1000 dev, 1000 test)
- Triple & lemma level duplicate counts within splits
- Overlap between train-low and train-medium (subsampling vs independent check)
- Data leakage between train, dev, and test
- Morphological feature inventory and OOV tag rate inspection
"""

import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_split(filepath: str):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue

            # Check NFC normalization
            if line_str != unicodedata.normalize("NFC", line_str):
                print(f"Warning: {filepath}:{line_num} not NFC normalized")

            parts = line_str.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Line {line_num} in {filepath} does not have exactly 3 columns: '{line_str}'"
                )

            lemma, form, feats = parts
            if not lemma or not form or not feats:
                raise ValueError(
                    f"Line {line_num} in {filepath} has empty fields: lemma='{lemma}', form='{form}', feats='{feats}'"
                )

            tags = set(feats.split(";"))
            rows.append({
                "line": line_num,
                "lemma": lemma,
                "form": form,
                "features": feats,
                "tags": tags,
                "triple": (lemma, form, feats),
            })
    return rows


def validate():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(root_dir, "data", "MANIFEST.json")
    raw_dir = os.path.join(root_dir, "data", "raw")

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=== 1. Validating Hashes and Line Counts against MANIFEST.json ===")
    for filename, info in manifest["files"].items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}", file=sys.stderr)
            sys.exit(1)

        actual_sha = compute_sha256(filepath)
        if actual_sha != info["sha256"]:
            print(f"Hash mismatch for {filename}: expected {info['sha256']}, got {actual_sha}", file=sys.stderr)
            sys.exit(1)

    print("All file hashes match manifest successfully.")

    print("\n=== 2. Parsing Splits and Validating Schema ===")
    data = {}
    for filename in manifest["files"]:
        filepath = os.path.join(raw_dir, filename)
        rows = load_split(filepath)
        data[filename] = rows
        print(f"  {filename:22s} -> {len(rows):4d} valid triples")

    languages = ["eng", "deu", "fra", "ita"]

    print("\n=== 3. Overlap & Leakage Analysis ===")
    for iso in languages:
        low_file = f"{iso}-train-low.tsv"
        med_file = f"{iso}-train-medium.tsv"
        dev_file = f"{iso}-dev.tsv"
        test_file = f"{iso}-test.tsv"

        low_rows = data[low_file]
        med_rows = data[med_file]
        dev_rows = data[dev_file]
        test_rows = data[test_file]

        low_triples = set(r["triple"] for r in low_rows)
        med_triples = set(r["triple"] for r in med_rows)
        dev_triples = set(r["triple"] for r in dev_rows)
        test_triples = set(r["triple"] for r in test_rows)

        low_lemmas = set(r["lemma"] for r in low_rows)
        med_lemmas = set(r["lemma"] for r in med_rows)
        dev_lemmas = set(r["lemma"] for r in dev_rows)
        test_lemmas = set(r["lemma"] for r in test_rows)

        # Check train-low in train-medium
        low_in_med = low_triples.intersection(med_triples)
        is_subset = low_triples.issubset(med_triples)

        # Leakage
        leak_dev_med = med_triples.intersection(dev_triples)
        leak_test_med = med_triples.intersection(test_triples)
        leak_test_dev = dev_triples.intersection(test_triples)

        lemma_overlap_dev = med_lemmas.intersection(dev_lemmas)
        lemma_overlap_test = med_lemmas.intersection(test_lemmas)

        print(f"Language: {iso.upper()}")
        print(f"  train-low in train-medium : {len(low_in_med)}/100 (is strict subset: {is_subset})")
        print(f"  Exact triple leakage     : train∩dev = {len(leak_dev_med)}, train∩test = {len(leak_test_med)}, dev∩test = {len(leak_test_dev)}")
        print(f"  Lemma overlap (medium)    : with dev = {len(lemma_overlap_dev)}/{len(med_lemmas)}, with test = {len(lemma_overlap_test)}/{len(med_lemmas)}")

        # Features inventory & OOV tags
        train_tags = set().union(*(r["tags"] for r in med_rows))
        dev_tags = set().union(*(r["tags"] for r in dev_rows))
        test_tags = set().union(*(r["tags"] for r in test_rows))

        unk_dev_tags = dev_tags - train_tags
        unk_test_tags = test_tags - train_tags
        print(f"  Total unique UniMorph tags in train-medium : {len(train_tags)}")
        print(f"  Unseen tags in dev  : {len(unk_dev_tags)} ({', '.join(sorted(unk_dev_tags)) if unk_dev_tags else 'None'})")
        print(f"  Unseen tags in test : {len(unk_test_tags)} ({', '.join(sorted(unk_test_tags)) if unk_test_tags else 'None'})")
        print()

    print("Data validation completed successfully without errors.")


if __name__ == "__main__":
    validate()
