#!/usr/bin/env python3
"""
Trains a single model bundle (language + regime) using frozen hyperparameters.
Outputs:
  models/<lang>_<size>/
    best.model
    best.model.json
    vocabulary.pkl
    sed.pkl
    sed.pkl.json
    train.log
"""

import argparse
import os
import subprocess
import sys
import yaml
import torch


def train_bundle(lang_iso: str, regime: str, device: str = None):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "configs", "train.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    raw_data_dir = os.path.join(root_dir, "data", "raw")
    if regime == "100":
        train_file = os.path.join(raw_data_dir, f"{lang_iso}-train-low.tsv")
        epochs = cfg["epochs"]["low"]
    elif regime == "1000":
        train_file = os.path.join(raw_data_dir, f"{lang_iso}-train-medium.tsv")
        epochs = cfg["epochs"]["medium"]
    else:
        raise ValueError(f"Unknown regime: {regime}. Must be '100' or '1000'.")

    dev_file = os.path.join(raw_data_dir, f"{lang_iso}-dev.tsv")
    test_file = os.path.join(raw_data_dir, f"{lang_iso}-test.tsv")

    bundle_dir = os.path.join(root_dir, "models", f"{lang_iso}_{regime}")
    os.makedirs(bundle_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m", "trans.train",
        "--train", train_file,
        "--dev", dev_file,
        "--test", test_file,
        "--output", bundle_dir,
        "--device", device,
        "--pytorch-seed", str(cfg["seed"]),
        "--epochs", str(epochs),
        "--enc-type", cfg["encoder"]["type"],
        "--enc-hidden-dim", str(cfg["encoder"]["hidden_dim"]),
        "--enc-layers", str(cfg["encoder"]["layers"]),
        "--enc-output-dropout", str(cfg["encoder"]["output_dropout"]),
        "--enc-output-dropout-type", cfg["encoder"]["output_dropout_type"],
        "--char-dim", str(cfg["embeddings"]["char_dim"]),
        "--feat-dim", str(cfg["embeddings"]["feat_dim"]),
        "--action-dim", str(cfg["embeddings"]["action_dim"]),
        "--dec-hidden-dim", str(cfg["decoder"]["hidden_dim"]),
        "--dec-layers", str(cfg["decoder"]["layers"]),
        "--output-feedback-dim", str(cfg["decoder"]["output_feedback_dim"]),
        "--sed-em-mode", cfg["alignment"]["sed_em_mode"],
        "--sed-em-damping", str(cfg["alignment"]["sed_em_damping"]),
        "--sed-em-iterations", str(cfg["alignment"]["sed_em_iterations"]),
        "--expert-loss", cfg["imitation_learning"]["expert_loss"],
        "--expert-temperature", str(cfg["imitation_learning"]["expert_temperature"]),
        "--rollin-prob", str(cfg["imitation_learning"]["rollin_prob"]),
        "--optimizer", cfg["optimization"]["optimizer"],
        "--lr", str(cfg["optimization"]["lr"]),
        "--weight-decay", str(cfg["optimization"]["weight_decay"]),
        "--scheduler", cfg["optimization"]["scheduler"],
        "--factor", str(cfg["optimization"]["factor"]),
        "--lrs-patience", str(cfg["optimization"]["lrs_patience"]),
        "--patience", str(cfg["optimization"]["patience"]),
        "--batch-size", str(cfg["optimization"]["batch_size"]),
        "--eval-batch-size", str(cfg["optimization"]["eval_batch_size"]),
        "--loss-reduction", cfg["optimization"]["loss_reduction"],
        "--beam-width", str(cfg["decoding"]["beam_width"]),
    ]

    print(f"Training bundle: {lang_iso}_{regime} on device={device} (epochs={epochs})...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Bundle successfully trained at: {bundle_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a single Transducer bundle.")
    parser.add_argument("--lang", required=True, choices=["eng", "deu", "fra", "ita"], help="Language code")
    parser.add_argument("--regime", required=True, choices=["100", "1000"], help="Training data size")
    parser.add_argument("--device", default=None, help="Device (mps/cpu/cuda)")
    args = parser.parse_args()
    train_bundle(args.lang, args.regime, args.device)


if __name__ == "__main__":
    main()
