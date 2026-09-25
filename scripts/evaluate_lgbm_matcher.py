"""
Evaluate LightGBM Matcher on Validation Split and Optimize Decision Threshold.
"""

import os
import sys
import gc
import json
import time
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import CandidateBlocker
from src.normalize import normalize_business_name, normalize_business_address
from src.predict import score_candidate_pairs, optimize_threshold, apply_target_disjoint_constraint
from src.evaluate import evaluate_predictions

VAL_DIR = "data/validation"
VAL_S1_TSV = os.path.join(VAL_DIR, "val_source1.tsv")
VAL_GT_JSON = os.path.join(VAL_DIR, "val_ground_truth.json")
MODEL_PATH = "models/lgbm_matcher.bin"

print("=== Evaluating LightGBM Matcher on Validation Split ===")
t0 = time.time()

# 1. Load validation S1 and ground truth
val_s1_df = pd.read_csv(VAL_S1_TSV, sep="\t", dtype=str)
val_s1_in = val_s1_df[val_s1_df["country"] == "India"].copy()
with open(VAL_GT_JSON) as f:
    val_gt = {k: set(v) for k, v in json.load(f).items()}
val_gt_in = {eid: val_gt[eid] for eid in val_s1_in["entity_id"]}

print(f"Evaluating on {len(val_s1_in):,} India validation entities.")

# Precompute S1 metadata
s1_meta = {}
for row in val_s1_in.itertuples(index=False):
    n_meta = normalize_business_name(row.business_name)
    a_meta = normalize_business_address(row.business_address)
    s1_meta[row.entity_id] = (n_meta, a_meta)

# 2. Load India targets and index with blocker
print("Loading India target records...")
s2_chunks = [c[c["country"] == "India"] for c in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, chunksize=500000)]
s3_chunks = [c[c["country"] == "India"] for c in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, chunksize=500000)]
targets = pd.concat(s2_chunks + s3_chunks, ignore_index=True)
del s2_chunks, s3_chunks
gc.collect()

blocker = CandidateBlocker(max_token_freq=15000, max_candidates_per_s1=100)
blocker.index_target_records(targets)

# 3. Query candidate sets
print("Querying candidate sets...")
candidates_dict = {}
for row in val_s1_in.itertuples(index=False):
    candidates_dict[row.entity_id] = blocker.generate_candidates_for_s1(row.business_name, row.business_address)

# Precompute target metadata for all generated candidates
needed_targets = set()
for cands in candidates_dict.values():
    needed_targets.update(cands)

print(f"Precomputing metadata for {len(needed_targets):,} unique candidate targets...")
target_meta = {}
for row in targets[targets["entity_id"].isin(needed_targets)].itertuples(index=False):
    n_meta = normalize_business_name(row.business_name)
    a_meta = normalize_business_address(row.business_address)
    target_meta[row.entity_id] = (n_meta, a_meta)

del targets, blocker
gc.collect()

# 4. Load trained LightGBM model and score candidates
print(f"Loading trained model from {MODEL_PATH}...")
booster = lgb.Booster(model_file=MODEL_PATH)

scored_candidates = score_candidate_pairs(booster, s1_meta, target_meta, candidates_dict, batch_size=50000)

# 5. Optimize threshold tau*
best_tau, best_metrics = optimize_threshold(
    scored_candidates,
    val_gt_in,
    thresholds=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95],
)

# 6. Compare with Baseline 0
print("\n=== COMPARISON WITH BASELINE 0 ===")
print(f"Baseline 0A (Exact Name Matcher):  Macro F0.5 = 0.38328")
print(f"Baseline 0B (Exact Name+Address):  Macro F0.5 = 0.12290")
print(f"LightGBM Matcher (Optimal tau={best_tau:.2f}): Macro F0.5 = {best_metrics['macro_f05']:.5f}")
diff = best_metrics['macro_f05'] - 0.38328
print(f"Absolute Gain over Baseline 0:    {'+' if diff >= 0 else ''}{diff:.5f} ({round(diff/0.38328*100, 1)}%)")

# Save evaluation results
with open("reports/lgbm_matcher_val_results.json", "w") as f:
    json.dump({
        "optimal_tau": best_tau,
        "metrics": best_metrics,
        "evaluation_time_sec": round(time.time() - t0, 1),
    }, f, indent=2)

print(f"\nEvaluation completed in {round(time.time() - t0, 1)}s.")
