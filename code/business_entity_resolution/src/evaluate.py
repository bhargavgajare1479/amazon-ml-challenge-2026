"""
Official Metric Implementation: Macro F0.5 Score for Business Entity Resolution.
Complies with Amazon ML Challenge 2026 specifications.
"""

from typing import Dict, Set, Union, List, Any
import numpy as np


def compute_single_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Dict[str, float]:
    """
    Compute F_0.5 score for a single Source 1 entity.
    
    Formula:
      F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
      
    Singleton rule:
      If true_set is empty:
        - pred_set is empty -> score = 1.0 (correct singleton)
        - pred_set is not empty -> score = 0.0 (false merge)
      If true_set is not empty:
        - pred_set is empty -> score = 0.0
        - pred_set has no true positives -> score = 0.0
    """
    n_true = len(true_set)
    n_pred = len(pred_set)

    if n_true == 0:
        if n_pred == 0:
            return {"f05": 1.0, "precision": 1.0, "recall": 1.0, "tp": 0, "is_singleton": True}
        else:
            return {"f05": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "is_singleton": True}

    if n_pred == 0:
        return {"f05": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "is_singleton": False}

    tp = len(true_set & pred_set)
    if tp == 0:
        return {"f05": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "is_singleton": False}

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return {
        "f05": float(f05),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "is_singleton": False,
    }


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Compute Macro F_0.5 across all Source 1 entities in ground_truth.
    
    Parameters:
      ground_truth: Dict mapping source1_entity_id -> Set of true matched entity IDs
      predictions: Dict mapping source1_entity_id -> Set of predicted entity IDs
      
    Returns:
      Comprehensive dictionary of metrics including:
      - macro_f05 (Primary Competition Metric)
      - macro_precision
      - macro_recall
      - singleton_accuracy (rate of correctly predicting empty on true singletons)
      - non_singleton_f05 (macro F0.5 on entities with >= 1 true match)
      - total_predictions
      - total_true_matches
    """
    f05_scores = []
    precisions = []
    recalls = []

    singleton_correct = 0
    singleton_total = 0
    
    non_singleton_f05 = []
    total_tp = 0
    total_pred = 0
    total_true = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        res = compute_single_entity_f05(true_set, pred_set)

        f05_scores.append(res["f05"])
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        total_tp += res["tp"]
        total_pred += len(pred_set)
        total_true += len(true_set)

        if res["is_singleton"]:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_f05.append(res["f05"])

    n_entities = len(ground_truth)
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    
    singleton_acc = (
        float(singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    )
    ns_f05 = (
        float(np.mean(non_singleton_f05)) if non_singleton_f05 else 0.0
    )

    return {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_prec, 5),
        "macro_recall": round(macro_rec, 5),
        "singleton_accuracy": round(singleton_acc, 5),
        "singleton_count": singleton_total,
        "non_singleton_f05": round(ns_f05, 5),
        "non_singleton_count": len(non_singleton_f05),
        "total_entities": n_entities,
        "total_true_matches": total_true,
        "total_predicted_matches": total_pred,
        "total_true_positives": total_tp,
    }


def evaluate_blocking_candidates(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_possible_targets: int,
) -> Dict[str, Any]:
    """
    Evaluate candidate generation / blocking stage.
    
    Computes:
      - candidate_recall: Proportion of true pairs that exist in candidates
      - candidates_per_s1: Average candidate count per S1 entity
      - candidate_reduction_ratio: 1 - (total_candidates / total_possible_pairs)
      - s1_coverage: % of S1 entities with at least 1 candidate
    """
    total_true = 0
    covered_true = 0
    candidate_counts = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        candidate_counts.append(len(cand_set))
        total_true += len(true_set)
        covered_true += len(true_set & cand_set)

    n_entities = len(ground_truth)
    total_cands = sum(candidate_counts)
    total_possible_pairs = n_entities * total_possible_targets

    candidate_recall = (covered_true / total_true) if total_true > 0 else 1.0
    reduction_ratio = (
        1.0 - (total_cands / total_possible_pairs)
        if total_possible_pairs > 0
        else 0.0
    )

    return {
        "candidate_recall": round(candidate_recall, 5),
        "covered_true_matches": covered_true,
        "total_true_matches": total_true,
        "total_candidates": total_cands,
        "candidates_per_s1_mean": round(float(np.mean(candidate_counts)), 2),
        "candidates_per_s1_median": float(np.median(candidate_counts)),
        "candidates_per_s1_max": int(np.max(candidate_counts)) if candidate_counts else 0,
        "reduction_ratio": round(reduction_ratio, 7),
    }


if __name__ == "__main__":
    # Self-test against the official README example:
    # True = ['S2-00047', 'S3-00812']
    # Pred = ['S2-00047', 'S2-00193', 'S3-00812']
    # Expected F0.5 = 0.714
    example_true = {"S1-00001": {"S2-00047", "S3-00812"}}
    example_pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    res = evaluate_predictions(example_true, example_pred)
    print("README test case verification:", res)
    assert abs(res["macro_f05"] - 0.71429) < 0.001, f"Expected 0.714, got {res['macro_f05']}"

    # Singleton test case:
    # True = set()
    # Pred = set() -> score 1.0
    singleton_true = {"S1-00002": set()}
    singleton_pred_correct = {"S1-00002": set()}
    s_res = evaluate_predictions(singleton_true, singleton_pred_correct)
    assert s_res["macro_f05"] == 1.0, f"Expected 1.0, got {s_res['macro_f05']}"

    # Singleton false positive test case:
    # True = set()
    # Pred = {'S2-00001'} -> score 0.0
    singleton_pred_wrong = {"S1-00002": {"S2-00001"}}
    s_res_wrong = evaluate_predictions(singleton_true, singleton_pred_wrong)
    assert s_res_wrong["macro_f05"] == 0.0, f"Expected 0.0, got {s_res_wrong['macro_f05']}"

    print("All metric unit tests passed successfully!")
