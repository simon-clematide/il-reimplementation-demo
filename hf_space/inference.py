"""
Core inference engine for Hugging Face Space.
Downloads bundles on-demand from the Hugging Face Hub (or local directory)
and maintains an in-memory LRU cache of loaded models.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from demo_core.loader import load_bundle
from demo_core.predict import PredictionResult, predict_single

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "simon-clmtd/neural-transducer-inflection")
LOCAL_MODELS_DIR = os.environ.get("LOCAL_MODELS_DIR", "models")


class TransducerSpaceEngine:
    def __init__(self, model_repo: str = HF_MODEL_REPO, local_dir: str = LOCAL_MODELS_DIR):
        self.model_repo = model_repo
        self.local_dir = local_dir
        self._cache: Dict[str, Tuple] = {}
        self._cache_order: List[str] = []
        self.max_cache_size = 2
        self.device = "cpu"  # Space runs on cpu-basic

    def _resolve_bundle_path(self, bundle_name: str) -> str:
        # Check local path first (useful during local testing and development)
        local_path = os.path.join(self.local_dir, bundle_name)
        if os.path.exists(os.path.join(local_path, "best.model.json")):
            return local_path

        # Download from Hugging Face Hub snapshot
        logging.info("Downloading bundle %s from %s...", bundle_name, self.model_repo)
        download_dir = snapshot_download(
            repo_id=self.model_repo,
            allow_patterns=f"{bundle_name}/*",
            local_dir_use_symlinks=False,
        )
        target_dir = os.path.join(download_dir, bundle_name)
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Bundle {bundle_name} not found after download in {download_dir}")
        return target_dir

    def get_bundle(self, lang: str, regime: str) -> Tuple:
        bundle_name = f"{lang}_{regime}"
        if bundle_name in self._cache:
            # Move to end of LRU
            self._cache_order.remove(bundle_name)
            self._cache_order.append(bundle_name)
            return self._cache[bundle_name]

        # Evict oldest if cache limit reached
        if len(self._cache_order) >= self.max_cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
            logging.info("Evicted bundle %s from LRU cache.", oldest)

        path = self._resolve_bundle_path(bundle_name)
        model, vocab, meta = load_bundle(path, device=self.device)
        self._cache[bundle_name] = (model, vocab, meta)
        self._cache_order.append(bundle_name)
        return model, vocab, meta

    def predict(
        self,
        lang: str,
        regime: str,
        lemma: str,
        features: str,
        beam_width: int = 1,
    ) -> PredictionResult:
        model, vocab, _ = self.get_bundle(lang, regime)
        return predict_single(
            model=model,
            vocab=vocab,
            lemma=lemma,
            features_str=features,
            beam_width=beam_width,
        )
