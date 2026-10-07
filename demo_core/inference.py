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

    def _load_data_index(self) -> Dict:
        if not hasattr(self, "_data_index"):
            index_path = os.path.join(os.path.dirname(__file__), "data_index.json")
            if not os.path.exists(index_path):
                # Fallback to demo_core location if needed
                index_path = os.path.join(os.path.dirname(__file__), "..", "demo_core", "data_index.json")
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    self._data_index = json.load(f)
            else:
                self._data_index = {}
        return self._data_index

    def get_random_sample(self, lang: str, split: str = "dev") -> Optional[Dict[str, str]]:
        import random
        data = self._load_data_index()
        split_key = "dev" if split.lower().startswith("dev") else "test"
        samples = data.get(lang, {}).get(split_key, [])
        if not samples:
            return None
        return random.choice(samples)

    def check_training_coverage(self, lang: str, lemma: str, features_str: str) -> Dict[str, Dict[str, bool]]:
        data = self._load_data_index()
        lang_data = data.get(lang, {}).get("training_coverage", {})
        lemma = lemma.strip()
        features_norm = set(features_str.strip().split(";"))

        result = {}
        for regime in ["100", "1000"]:
            cov = lang_data.get(regime, {})
            seen_lemmas = set(cov.get("lemmas", []))
            raw_feats = cov.get("features", [])
            seen_feats_sets = [set(f.split(";")) for f in raw_feats]

            lemma_seen = lemma in seen_lemmas
            # Feat combination seen if exact string matches or tag set matches
            feats_seen = (features_str.strip() in raw_feats) or (features_norm in seen_feats_sets)

            result[regime] = {
                "lemma_seen": lemma_seen,
                "features_seen": feats_seen,
            }
        return result

