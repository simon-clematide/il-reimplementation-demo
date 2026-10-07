#!/usr/bin/env python3
"""
Downloads pinned CoNLL-SIGMORPHON 2017 Shared Task 1 benchmark data splits
for English, German, French, and Italian.
Generates data/MANIFEST.json recording SHA-256 hashes, line counts, and URLs.
"""

import hashlib
import json
import os
import sys
import urllib.request

PINNED_REPO = "sigmorphon/conll2017"
PINNED_COMMIT = "519bb1677a7c747b3863d37069ffd395062288d8"
BASE_URL = f"https://raw.githubusercontent.com/{PINNED_REPO}/{PINNED_COMMIT}"

LANGUAGES = {
    "eng": "english",
    "deu": "german",
    "fra": "french",
    "ita": "italian",
}

SPLIT_FILES = {
    "train_low": ("all/task1/{lang}-train-low", "{iso}-train-low.tsv"),
    "train_medium": ("all/task1/{lang}-train-medium", "{iso}-train-medium.tsv"),
    "dev": ("all/task1/{lang}-dev", "{iso}-dev.tsv"),
    "test_gold": ("answers/task1/{lang}-uncovered-test", "{iso}-test.tsv"),
}


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def count_lines(filepath: str) -> int:
    with open(filepath, "rb") as f:
        return sum(1 for _ in f)


def fetch_data(output_dir: str, manifest_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "repository": PINNED_REPO,
        "commit": PINNED_COMMIT,
        "base_url": BASE_URL,
        "files": {},
    }

    print(f"Fetching data from {PINNED_REPO} at commit {PINNED_COMMIT[:8]}...")

    for iso, lang in LANGUAGES.items():
        for split_key, (src_pattern, dst_pattern) in SPLIT_FILES.items():
            src_path = src_pattern.format(lang=lang)
            dst_name = dst_pattern.format(iso=iso)
            dst_path = os.path.join(output_dir, dst_name)
            url = f"{BASE_URL}/{src_path}"

            print(f"  Downloading {url} -> {dst_name}")
            try:
                urllib.request.urlretrieve(url, dst_path)
            except Exception as e:
                print(f"Error downloading {url}: {e}", file=sys.stderr)
                sys.exit(1)

            file_hash = compute_sha256(dst_path)
            line_count = count_lines(dst_path)

            manifest["files"][dst_name] = {
                "language_iso": iso,
                "language_name": lang,
                "split": split_key,
                "source_path": src_path,
                "url": url,
                "sha256": file_hash,
                "line_count": line_count,
            }
            print(f"    Downloaded {dst_name}: {line_count} lines, SHA256: {file_hash[:10]}...")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote manifest with {len(manifest['files'])} entries to {manifest_path}")


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "data", "raw")
    manifest_path = os.path.join(root_dir, "data", "MANIFEST.json")
    fetch_data(output_dir, manifest_path)


if __name__ == "__main__":
    main()
