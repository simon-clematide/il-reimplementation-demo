#!/usr/bin/env python3
"""
Orchestrates training and packaging of all 8 model bundles across 4 languages
(English, German, French, Italian) and 2 regimes (100, 1000).
Generates REPRODUCIBILITY.md summarizing accuracy metrics and hashes.
"""

import argparse
import json
import os
import subprocess
import sys
import torch

LANGUAGES = ["eng", "deu", "fra", "ita"]
REGIMES = ["100", "1000"]


def train_all(device: str = None):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_script = os.path.join(root_dir, "scripts", "train_bundle.py")
    package_script = os.path.join(root_dir, "scripts", "package_bundle.py")

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = []

    for lang in LANGUAGES:
        for regime in REGIMES:
            bundle_name = f"{lang}_{regime}"
            bundle_dir = os.path.join(root_dir, "models", bundle_name)

            # Check if already trained & packaged
            manifest_file = os.path.join(bundle_dir, "bundle_manifest.json")
            if not os.path.exists(manifest_file):
                print(f"\n==========================================")
                print(f" Training {bundle_name} on device={device}")
                print(f"==========================================")
                cmd = [
                    sys.executable,
                    train_script,
                    "--lang", lang,
                    "--regime", regime,
                    "--device", device,
                ]
                subprocess.run(cmd, check=True)

                # Package bundle
                cmd_pkg = [sys.executable, package_script, bundle_dir]
                subprocess.run(cmd_pkg, check=True)
            else:
                print(f"Bundle {bundle_name} already packaged, reading manifest.")

            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            results.append(manifest)

    # Generate REPRODUCIBILITY.md
    generate_reproducibility_report(root_dir, results)


def generate_reproducibility_report(root_dir: str, results: list):
    repro_path = os.path.join(root_dir, "REPRODUCIBILITY.md")

    lines = [
        "# Reproducibility & Benchmark Results",
        "",
        "This table summarizes training runs across all 8 configurations on the **CoNLL-SIGMORPHON 2017 Task 1** benchmark.",
        "- **Training seed:** 42",
        "- **Data commit:** `519bb1677a7c747b3863d37069ffd395062288d8`",
        "- **Model code:** `neural_transducer` (`trans`), commit `f71833192acd505bc1d3ba47d75798cfb778fc3d`",
        "",
        "| Language | Regime | Best Epoch | Dev Str Acc | Dev Sym Acc | Test Str Acc | Test Sym Acc | Model Checkpoint SHA-256 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for res in results:
        bundle = res["bundle"]
        lang, regime = bundle.split("_")
        epoch = res.get("best_epoch", "N/A")

        dev_greedy = res.get("dev_greedy", {}).get("dev_string_accuracy", res.get("dev_string_accuracy", "N/A"))
        dev_sym = res.get("dev_greedy", {}).get("dev_symbol_accuracy", res.get("dev_symbol_accuracy", "N/A"))
        test_greedy = res.get("test_greedy", {}).get("test_string_accuracy", "N/A")
        test_sym = res.get("test_greedy", {}).get("test_symbol_accuracy", "N/A")

        model_sha = res.get("files", {}).get("best.model", "")[:12] + "..."

        dev_g_str = f"{dev_greedy:.4f}" if isinstance(dev_greedy, float) else str(dev_greedy)
        dev_s_str = f"{dev_sym:.4f}" if isinstance(dev_sym, float) else str(dev_sym)
        test_g_str = f"{test_greedy:.4f}" if isinstance(test_greedy, float) else str(test_greedy)
        test_s_str = f"{test_sym:.4f}" if isinstance(test_sym, float) else str(test_sym)

        lines.append(
            f"| **{lang.upper()}** | {regime} | {epoch} | {dev_g_str} | {dev_s_str} | {test_g_str} | {test_s_str} | `{model_sha}` |"
        )

    lines.append("")
    with open(repro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nGenerated reproducibility report at {repro_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate all 8 model bundles.")
    parser.add_argument("--device", default=None, help="Device (mps/cpu/cuda)")
    args = parser.parse_args()
    train_all(args.device)


if __name__ == "__main__":
    main()
