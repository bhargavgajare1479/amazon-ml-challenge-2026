"""
Local F0.5 scorer, matching the official metric exactly:
per-entity F_beta (beta=0.5), macro-averaged, with correct singleton handling
(predicting empty for a true singleton scores 1.0; predicting anything scores 0.0).
"""
import numpy as np
import pandas as pd

from . import config


def f05_score(pred_set: set, true_set: set) -> float:
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    beta2 = 0.25
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f05(predictions: dict, gt_map: dict) -> float:
    scores = [f05_score(predictions.get(eid, set()), true_set) for eid, true_set in gt_map.items()]
    return float(np.mean(scores)) if scores else 0.0


def build_predictions_from_scores(df: pd.DataFrame, threshold: float, all_entities,
                                   prob_col: str = "pred_prob") -> dict:
    """all_entities must include entities with zero candidates too, so they correctly
    default to an empty prediction (right answer for true singletons)."""
    preds = {eid: set() for eid in all_entities}
    sub = df[df[prob_col] >= threshold]
    for eid, cid in zip(sub["source1_entity_id"], sub["candidate_entity_id"]):
        if eid in preds:
            preds[eid].add(cid)
    return preds


def tune_threshold(val_part: pd.DataFrame, val_entities, gt_map: dict,
                    grid=config.THRESHOLD_GRID):
    """Grid search the decision threshold directly against macro F0.5. Because F0.5
    weights precision 2x over recall, the optimum usually lands well above 0.5 -
    that's expected, not a bug."""
    best_threshold, best_score = 0.5, -1.0
    local_gt = {eid: gt_map[eid] for eid in val_entities if eid in gt_map}
    start, stop, step = grid
    for t in np.arange(start, stop, step):
        preds = build_predictions_from_scores(val_part, t, val_entities)
        score = macro_f05(preds, local_gt)
        if score > best_score:
            best_score, best_threshold = score, t
    return float(best_threshold), best_score
