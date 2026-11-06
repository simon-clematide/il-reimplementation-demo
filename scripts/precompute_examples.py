"""Build a deterministic gallery from held-out test items, using local bundles.

Run: scratch/venv/bin/python scripts/precompute_examples.py --per-language 80
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hf_space'))
from inference import TransducerSpaceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-language', type=int, default=80)
    args = parser.parse_args()
    engine = TransducerSpaceEngine(local_dir=str(ROOT / 'models'))
    records = []
    for language in ['eng', 'deu', 'fra', 'ita']:
        samples = list(engine._load_data_index()[language]['test'])
        random.Random(42).shuffle(samples)
        samples = [s for s in samples if s['lemma'] != s['target']][:args.per_language]
        predictions = [[] for _ in samples]
        fingerprints = {}
        for regime in ['100', '1000', '10000']:
            bundle = ROOT / 'models' / f'{language}_{regime}'
            fingerprints[regime] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(bundle.iterdir()) if p.is_file()}
            for i, sample in enumerate(samples):
                predictions[i].append(asdict(engine.predict(language, regime, sample['lemma'], sample['features'])))
        for sample, results in zip(samples, predictions):
            forms = [r['predicted_form'] for r in results]
            correct = [f == sample['target'] for f in forms]
            groups = ['All examples']
            if len(set(forms)) > 1:
                groups.append('Models disagree')
            if correct[-1] and not correct[0]:
                groups.append('More data helps')
            if not correct[-1]:
                groups.append('Largest model misses')
            if all(correct):
                groups.append('All models correct')
            records.append(dict(sample, language=language, results=results, groups=groups,
                                bundle_sha256=fingerprints))
        print(f'{language}: {len(samples)} test examples', flush=True)
    destination = ROOT / 'hf_space' / 'exploration_examples.json'
    destination.write_text(json.dumps({'seed': 42, 'split': 'test', 'examples': records}, ensure_ascii=False))
    print(destination)


if __name__ == '__main__':
    main()
