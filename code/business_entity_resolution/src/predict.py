"""
Prediction, Threshold Optimization, and Post-Processing Engine.
Evaluates candidate pairs with trained LightGBM matcher, optimizes tau*,
and enforces target 1-to-1 disjoint constraint.
"""

import os
import gc
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    from .evaluate import evaluate_predictions, compute_single_entity_f05
    from .features import extract_batch_features, FEATURE_NAMES
except ImportError:
    from evaluate import evaluate_predictions, compute_single_entity_f05
    from features import extract_batch_features, FEATURE_NAMES

MODEL_PATH = "models/lgbm_matcher.bin"


def score_candidate_pairs(
    booster: lgb.Booster,
    s1_meta: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
    target_meta: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
    candidates_dict: Dict[str, Set[str]],
    batch_size: int = 50000,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Score all candidate pairs for each S1 entity using the trained booster.
    
    Returns:
      scored_candidates: Dict mapping s1_id -> List of (target_id, match_probability)
    """
    print(f"Scoring candidates across {len(candidates_dict):,} S1 entities...")
    t0 = time.time()

    # Flatten candidate pairs
    all_pairs = []
    for s1_id, cands in candidates_dict.items():
        for tid in cands:
            if s1_id in s1_meta and tid in target_meta:
                all_pairs.append((s1_id, tid))

    total_pairs = len(all_pairs)
    print(f"Total candidate pairs to score: {total_pairs:,}")

    if total_pairs == 0:
        return {s1_id: [] for s1_id in candidates_dict}

    # Batch feature extraction and prediction
    probabilities = np.empty(total_pairs, dtype=np.float32)
    for start_idx in range(0, total_pairs, batch_size):
        end_idx = min(start_idx + batch_size, total_pairs)
        batch_pairs = all_pairs[start_idx:end_idx]
        X_batch = extract_batch_features(s1_meta, target_meta, batch_pairs)
        probabilities[start_idx:end_idx] = booster.predict(X_batch)

    # Reconstruct dictionary mapping s1_id -> [(target_id, prob), ...]
    scored_candidates = defaultdict(list)
    for (s1_id, target_id), prob in zip(all_pairs, probabilities):
        scored_candidates[s1_id].append((target_id, float(prob)))

    # Ensure every S1 is present
    for s1_id in candidates_dict:
        if s1_id not in scored_candidates:
            scored_candidates[s1_id] = []

    elapsed = round(time.time() - t0, 1)
    print(f"Scored {total_pairs:,} pairs in {elapsed}s ({round(total_pairs/max(0.1, elapsed), 1)} pairs/sec).")
    return dict(scored_candidates)


def apply_target_disjoint_constraint(
    scored_candidates: Dict[str, List[Tuple[str, float]]],
    threshold: float,
) -> Dict[str, Set[str]]:
    """
    Enforce target 1-to-1 disjoint constraint:
      Every target record S2/S3 can be assigned to AT MOST ONE S1 entity.
      If multiple S1 entities predict target T above threshold, T is assigned
      exclusively to the S1 entity with the highest predicted probability.
    """
    # 1. Filter candidates by threshold and group by target
    target_to_claims = defaultdict(list)
    for s1_id, pairs in scored_candidates.items():
        for target_id, prob in pairs:
            if prob >= threshold:
                target_to_claims[target_id].append((s1_id, prob))

    # 2. Resolve conflicts: assign each target to argmax S1
    s1_to_matches = defaultdict(set)
    for target_id, claims in target_to_claims.items():
        # Pick S1 with highest probability
        best_s1, best_prob = max(claims, key=lambda x: x[1])
        s1_to_matches[best_s1].add(target_id)

    # Ensure all S1 entities exist in output
    final_predictions = {}
    for s1_id in scored_candidates:
        final_predictions[s1_id] = s1_to_matches.get(s1_id, set())

    return final_predictions


def optimize_threshold(
    scored_candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    thresholds: List[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Grid-search optimal threshold tau* maximizing Macro F0.5 on validation split.
    """
    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]

    print("\n--- Optimizing Decision Threshold tau on Macro F0.5 ---")
    best_tau = 0.50
    best_f05 = -1.0
    best_metrics = {}

    for tau in thresholds:
        preds = apply_target_disjoint_constraint(scored_candidates, threshold=tau)
        metrics = evaluate_predictions(ground_truth, preds)
        f05 = metrics["macro_f05"]
        print(f"  tau = {tau:.2f} -> Macro F0.5 = {f05:.5f} (Prec: {metrics['macro_precision']:.4f}, Rec: {metrics['macro_recall']:.4f}, SingAcc: {metrics['singleton_accuracy']:.4f})")

        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau
            best_metrics = metrics

    print(f"\nOptimal Threshold: tau* = {best_tau:.2f} with Macro F0.5 = {best_f05:.5f}")
    return best_tau, best_metrics
