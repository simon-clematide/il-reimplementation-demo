"""
Metadata-driven model loader for neural transducer bundles.
Reconstructs Transducer architecture exclusively from best.model.json metadata,
ensuring zero configuration hard-coding in client apps or scripts.
"""

import argparse
import hashlib
import json
import logging
import os
from typing import Optional, Tuple

import torch

from trans import optimal_expert_substitutions
from trans import sed
from trans import transducer
from trans import vocabulary


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_bundle_manifest(bundle_dir: str) -> dict:
    """Verifies that all files in bundle_dir match bundle_manifest.json hashes."""
    manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing bundle_manifest.json in {bundle_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for filename, expected_hash in manifest.get("files", {}).items():
        filepath = os.path.join(bundle_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bundle file missing: {filepath}")
        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Bundle file integrity compromised for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return manifest


def load_bundle(bundle_dir: str, device: Optional[str] = None) -> Tuple[transducer.Transducer, vocabulary.FeatureVocabularies, dict]:
    """
    Loads a fine-tuned Transducer model bundle.

    Returns:
        (model, vocabulary, metadata)
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    metadata_path = os.path.join(bundle_dir, "best.model.json")
    model_path = os.path.join(bundle_dir, "best.model")
    voc_path = os.path.join(bundle_dir, "vocabulary.pkl")
    sed_path = os.path.join(bundle_dir, "sed.pkl")

    for path in (metadata_path, model_path, voc_path, sed_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required bundle artifact not found: {path}")

    # Optional integrity check if manifest is present
    manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    if os.path.exists(manifest_path):
        verify_bundle_manifest(bundle_dir)

    # 1. Load metadata and build args Namespace
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_args_dict = metadata.get("args", {})
    # Override device to current inference device
    train_args_dict["device"] = device
    args = argparse.Namespace(**train_args_dict)

    # 2. Load FeatureVocabularies and SED aligner
    vocab = vocabulary.FeatureVocabularies.from_pickle(voc_path)
    sed_aligner = sed.StochasticEditDistance.from_pickle(sed_path)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)

    # 3. Instantiate Transducer using metadata-driven args
    model = transducer.Transducer(vocab, expert, args)

    # 4. Load model weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logging.info("Successfully loaded bundle from %s to device %s", bundle_dir, device)
    return model, vocab, metadata
