#!/usr/bin/env python3
"""
Uploads trained model bundles and model card to the Hugging Face model repository.
Only runs when explicitly invoked.
"""

import argparse
import os
import sys
from huggingface_hub import HfApi

HF_MODEL_REPO = "simon-clmtd/neural-transducer-inflection"
LANGUAGES = ["eng", "deu", "fra", "ita"]
REGIMES = ["100", "1000"]


def upload_models(repo_id: str = HF_MODEL_REPO, revision: str = "main"):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "models")
    card_path = os.path.join(root_dir, "model_card", "README.md")

    api = HfApi()

    print(f"Uploading model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    for lang in LANGUAGES:
        for regime in REGIMES:
            bundle_name = f"{lang}_{regime}"
            bundle_dir = os.path.join(models_dir, bundle_name)
            if not os.path.exists(bundle_dir):
                print(f"Warning: bundle directory {bundle_dir} does not exist, skipping.", file=sys.stderr)
                continue

            print(f"Uploading bundle {bundle_name} to {repo_id}/{bundle_name}...")
            api.upload_folder(
                folder_path=bundle_dir,
                path_in_repo=bundle_name,
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[
                    "best.model",
                    "best.model.json",
                    "vocabulary.pkl",
                    "sed.pkl",
                    "sed.pkl.json",
                    "bundle_manifest.json",
                    "*.eval",
                ],
            )

    print(f"\nAll bundles uploaded successfully to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload trained bundles to Hugging Face Model Hub.")
    parser.add_argument("--repo", default=HF_MODEL_REPO, help="Target HF repo ID")
    args = parser.parse_args()
    upload_models(args.repo)


if __name__ == "__main__":
    main()
