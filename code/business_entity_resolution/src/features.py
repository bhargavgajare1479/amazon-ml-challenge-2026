"""
Pairwise Feature Engineering Engine for Business Entity Resolution.
Extracts string distances, token overlaps, numeric matches, and cross-field interactions.
Complies with Amazon ML Challenge 2026 requirements.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

try:
    from .normalize import (
        clean_basic,
        normalize_business_name,
        normalize_business_address,
    )
except ImportError:
    from normalize import (
        clean_basic,
        normalize_business_name,
        normalize_business_address,
    )

# Standard feature column order for model training and inference
FEATURE_NAMES = [
    # Name string similarities
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_partial_ratio",
    "name_stripped_ratio",
    "name_exact_clean",
    "name_exact_stripped",
    "name_compact_match",
    "name_acronym_match",
    "name_len_diff",
    "name_len_ratio",
    
    # Address string similarities
    "addr_missing",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_exact_clean",
    "addr_shared_token_count",
    "addr_len_diff",
    
    # Numeric & Postal features
    "exact_number_match",
    "number_jaccard",
    "postal_code_match",
    "has_number_conflict",
    
    # Cross-field interaction features
    "name_x_addr_sim",
    "name_x_addr_sort",
    "strong_name_weak_addr",
    "weak_name_strong_addr",
    "name_match_and_num_match",
]


def extract_pair_features(
    s1_meta: Tuple[Dict[str, Any], Dict[str, Any]],
    target_meta: Tuple[Dict[str, Any], Dict[str, Any]],
) -> List[float]:
    """
    Extract pairwise features between an S1 entity and a target candidate entity.
    
    s1_meta: (s1_name_meta, s1_addr_meta) from src.normalize
    target_meta: (target_name_meta, target_addr_meta) from src.normalize
    
    Returns: List of float features matching FEATURE_NAMES.
    """
    n1, a1 = s1_meta
    n2, a2 = target_meta

    # 1. Name Features
    clean_name1 = n1["clean"]
    clean_name2 = n2["clean"]
    strip_name1 = n1["stripped_legal"]
    strip_name2 = n2["stripped_legal"]

    name_lev = fuzz.ratio(clean_name1, clean_name2) / 100.0
    name_sort = fuzz.token_sort_ratio(clean_name1, clean_name2) / 100.0
    name_set = fuzz.token_set_ratio(clean_name1, clean_name2) / 100.0
    name_jw = float(distance.JaroWinkler.similarity(clean_name1, clean_name2))
    name_part = fuzz.partial_ratio(clean_name1, clean_name2) / 100.0
    name_strip = fuzz.ratio(strip_name1, strip_name2) / 100.0

    name_exact = 1.0 if (clean_name1 == clean_name2 and len(clean_name1) > 0) else 0.0
    name_exact_strip = 1.0 if (strip_name1 == strip_name2 and len(strip_name1) > 0) else 0.0

    # Compact match (handles web domains e.g. 'telefutureindia' vs 'tele future india')
    comp1 = n1["compact"]
    comp2 = n2["compact"]
    name_compact = 1.0 if (comp1 == comp2 and len(comp1) >= 4) else 0.0

    # Acronym match (e.g. 'PC' vs 'Primary Care')
    acr1 = n1["acronym"]
    acr2 = n2["acronym"]
    name_acr = 0.0
    if len(acr1) >= 2 and (acr1 == clean_name2 or acr1 == acr2):
        name_acr = 1.0
    elif len(acr2) >= 2 and (acr2 == clean_name1 or acr2 == acr1):
        name_acr = 1.0

    len1 = len(clean_name1)
    len2 = len(clean_name2)
    name_diff = float(abs(len1 - len2))
    name_ratio = float(min(len1, len2) / max(len1, len2, 1))

    # 2. Address Features
    clean_addr1 = a1["clean"]
    clean_addr2 = a2["clean"]
    addr_miss = 1.0 if len(clean_addr2) == 0 else 0.0

    if addr_miss == 1.0:
        addr_lev = 0.0
        addr_sort = 0.0
        addr_set = 0.0
        addr_jaccard = 0.0
        addr_exact = 0.0
        shared_tokens = 0.0
        addr_diff = float(len(clean_addr1))
    else:
        addr_lev = fuzz.ratio(clean_addr1, clean_addr2) / 100.0
        addr_sort = fuzz.token_sort_ratio(clean_addr1, clean_addr2) / 100.0
        addr_set = fuzz.token_set_ratio(clean_addr1, clean_addr2) / 100.0

        tok1 = set(a1["tokens"])
        tok2 = set(a2["tokens"])
        union_tok = tok1 | tok2
        addr_jaccard = float(len(tok1 & tok2) / max(1, len(union_tok)))
        addr_exact = 1.0 if (clean_addr1 == clean_addr2 and len(clean_addr1) > 0) else 0.0
        shared_tokens = float(len(tok1 & tok2))
        addr_diff = float(abs(len(clean_addr1) - len(clean_addr2)))

    # 3. Numeric & Postal Features
    nums1 = a1["numbers"]
    nums2 = a2["numbers"]

    exact_num = 1.0 if (nums1 and nums2 and len(nums1 & nums2) > 0) else 0.0
    if nums1 or nums2:
        num_jacc = float(len(nums1 & nums2) / max(1, len(nums1 | nums2)))
    else:
        num_jacc = 0.0

    post1 = a1["postal_candidates"]
    post2 = a2["postal_candidates"]
    post_match = 1.0 if (post1 and post2 and len(post1 & post2) > 0) else 0.0

    # Number conflict: Both addresses have numbers, but NO numbers overlap
    # (High-signal false-positive discriminator for different businesses on the same street)
    has_num_conflict = 1.0 if (nums1 and nums2 and len(nums1 & nums2) == 0) else 0.0

    # 4. Cross-Field Interaction Features
    name_x_addr = name_set * (addr_set if addr_miss == 0.0 else 0.5)
    name_x_sort = name_sort * (addr_sort if addr_miss == 0.0 else 0.5)

    strong_name_weak_addr = 1.0 if (name_set > 0.85 and addr_set < 0.35 and addr_miss == 0.0) else 0.0
    weak_name_strong_addr = 1.0 if (name_set < 0.50 and addr_set > 0.85) else 0.0
    name_match_and_num = name_sort * exact_num

    return [
        name_lev,
        name_sort,
        name_set,
        name_jw,
        name_part,
        name_strip,
        name_exact,
        name_exact_strip,
        name_compact,
        name_acr,
        name_diff,
        name_ratio,
        addr_miss,
        addr_lev,
        addr_sort,
        addr_set,
        addr_jaccard,
        addr_exact,
        shared_tokens,
        addr_diff,
        exact_num,
        num_jacc,
        post_match,
        has_num_conflict,
        name_x_addr,
        name_x_sort,
        strong_name_weak_addr,
        weak_name_strong_addr,
        name_match_and_num,
    ]


def extract_batch_features(
    s1_records: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
    target_records: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
    candidate_pairs: List[Tuple[str, str]],
) -> np.ndarray:
    """
    Extract feature matrix for a list of candidate pairs (s1_id, target_id).
    
    Returns: 2D NumPy array of shape (N_pairs, len(FEATURE_NAMES)) with dtype float32.
    """
    n_pairs = len(candidate_pairs)
    feature_matrix = np.empty((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)

    for i, (s1_id, target_id) in enumerate(candidate_pairs):
        s1_meta = s1_records[s1_id]
        target_meta = target_records[target_id]
        feature_matrix[i] = extract_pair_features(s1_meta, target_meta)

    return feature_matrix


if __name__ == "__main__":
    # Self-test unit check
    n1 = normalize_business_name("Maure Williams Colombier Inc")
    a1 = normalize_business_address("85 Wayne Avenue, Ticonderoga, NY 12883")

    n2 = normalize_business_name("Maure Wilblims Colombier")
    a2 = normalize_business_address("85 Wanye Ave, Ticonderoga Townshiip, NY 12883")

    feats = extract_pair_features((n1, a1), (n2, a2))
    assert len(feats) == len(FEATURE_NAMES), f"Feature count mismatch: {len(feats)} vs {len(FEATURE_NAMES)}"
    print("Unit test passed! Extracted", len(feats), "features:")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:<28} = {val:.4f}")
