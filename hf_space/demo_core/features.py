"""
Feature parsing, validation, and OOV checking against vocabulary.
"""

from typing import List, Set, Tuple
from trans import vocabulary


def parse_feature_string(features_str: str) -> List[str]:
    """Splits a semicolon-delimited UniMorph tag string into ordered feature tokens."""
    return [f.strip() for f in features_str.split(";") if f.strip()]


def validate_features(features_str: str, vocab: vocabulary.FeatureVocabularies) -> Tuple[List[str], List[str]]:
    """
    Validates features against the bundle's vocabulary.

    Returns:
        (known_tags, unknown_tags)
    """
    tags = parse_feature_string(features_str)
    known = []
    unknown = []
    for tag in tags:
        if tag in vocab.features.w2i:
            known.append(tag)
        else:
            unknown.append(tag)
    return known, unknown
