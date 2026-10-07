#!/usr/bin/env python3
"""
Adds bundle_manifest.json to a trained model bundle directory.
Records SHA-256 hashes of all artifacts, training metadata, and test evaluation metrics.
"""

import argparse
import hashlib
import json
import os
import sys
import torch
import trans


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_eval_file(eval_path: str) -> dict:
    metrics = {}
    if not os.path.exists(eval_path):
        return metrics
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(" ", "_")
                try:
                    metrics[key] = float(parts[1].strip())
                except ValueError:
                    metrics[key] = parts[1].strip()
    return metrics


def package_bundle(bundle_dir: str):
    best_json_path = os.path.join(bundle_dir, "best.model.json")
    if not os.path.exists(best_json_path):
        raise FileNotFoundError(f"Missing best.model.json in {bundle_dir}")

    with open(best_json_path, "r", encoding="utf-8") as f:
        best_meta = json.load(f)

    artifacts = [
        "best.model",
        "best.model.json",
        "vocabulary.pkl",
        "sed.pkl",
        "sed.pkl.json",
    ]

    file_hashes = {}
    for art in artifacts:
        p = os.path.join(bundle_dir, art)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact {art} missing from {bundle_dir}")
        file_hashes[art] = compute_sha256(p)

    dev_eval = parse_eval_file(os.path.join(bundle_dir, "dev_greedy.eval"))
    dev_beam = parse_eval_file(os.path.join(bundle_dir, "dev_beam4.eval"))
    test_eval = parse_eval_file(os.path.join(bundle_dir, "test_greedy.eval"))
    test_beam = parse_eval_file(os.path.join(bundle_dir, "test_beam4.eval"))

    manifest = {
        "bundle": os.path.basename(bundle_dir),
        "torch_version": torch.__version__,
        "seed": best_meta.get("args", {}).get("pytorch_seed", 42),
        "best_epoch": best_meta.get("epoch"),
        "dev_string_accuracy": best_meta.get("dev_string_accuracy"),
        "dev_symbol_accuracy": best_meta.get("dev_symbol_accuracy"),
        "dev_greedy": dev_eval,
        "dev_beam4": dev_beam,
        "test_greedy": test_eval,
        "test_beam4": test_beam,
        "files": file_hashes,
    }

    manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Packaged {bundle_dir} -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Package a trained model bundle with a manifest.")
    parser.add_argument("bundle_dir", help="Path to bundle directory")
    args = parser.parse_args()
    package_bundle(args.bundle_dir)


if __name__ == "__main__":
    main()
