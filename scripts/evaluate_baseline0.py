"""
Baseline 0 Evaluation: Exact Normalized Matcher.
Evaluates official Macro F0.5 on the validation split.
"""

import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.evaluate import evaluate_predictions
from src.normalize import normalize_business_name, normalize_business_address

VAL_DIR = "data/validation"
VAL_S1_TSV = os.path.join(VAL_DIR, "val_source1.tsv")
VAL_GT_JSON = os.path.join(VAL_DIR, "val_ground_truth.json")

print("Starting Baseline 0 Evaluation (Exact Normalized Matcher)...")
t0 = time.time()

# 1. Load validation S1 and ground truth
val_s1_df = pd.read_csv(VAL_S1_TSV, sep="\t", dtype=str)
val_s1_in = val_s1_df[val_s1_df["country"] == "India"].copy()
with open(VAL_GT_JSON) as f:
    val_gt = {k: set(v) for k, v in json.load(f).items()}
val_gt_in = {eid: val_gt[eid] for eid in val_s1_in["entity_id"]}

print(f"Loaded {len(val_s1_in):,} India validation entities.")

# 2. Build index of targets on exact normalized name
print("Loading India targets for Baseline 0...")
s2_chunks = [c[c["country"] == "India"] for c in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, chunksize=500000)]
s3_chunks = [c[c["country"] == "India"] for c in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, chunksize=500000)]
targets = pd.concat(s2_chunks + s3_chunks, ignore_index=True)
del s2_chunks, s3_chunks

# Build exact stripped name -> list of target entity IDs
name_to_target_eids = {}
target_meta = {}
for row in targets.itertuples(index=False):
    eid = row.entity_id
    raw_name = row.business_name if isinstance(row.business_name, str) else ""
    raw_addr = row.business_address if isinstance(row.business_address, str) else ""
    
    n_clean = normalize_business_name(raw_name)["stripped_legal"]
    a_clean = normalize_business_address(raw_addr)["clean"]
    
    target_meta[eid] = (n_clean, a_clean)
    if len(n_clean) >= 3:
        if n_clean not in name_to_target_eids:
            name_to_target_eids[n_clean] = []
        name_to_target_eids[n_clean].append(eid)

print(f"Indexed {len(name_to_target_eids):,} distinct business names.")

# 3. Predict matches using exact match logic:
# Match if exact stripped name AND (exact clean address OR high address token overlap)
predictions_exact_name_only = {}
predictions_exact_name_and_addr = {}

for row in val_s1_in.itertuples(index=False):
    s1_id = row.entity_id
    n_clean = normalize_business_name(row.business_name)["stripped_legal"]
    a_clean = normalize_business_address(row.business_address)["clean"]
    
    candidate_targets = name_to_target_eids.get(n_clean, [])
    
    # Policy A: Exact name only
    predictions_exact_name_only[s1_id] = set(candidate_targets[:10])
    
    # Policy B: Exact name + exact address
    matching_both = set()
    for tid in candidate_targets:
        t_name, t_addr = target_meta[tid]
        if t_name == n_clean and (t_addr == a_clean or not t_addr):
            matching_both.add(tid)
    predictions_exact_name_and_addr[s1_id] = matching_both

# Evaluate Policy A
res_a = evaluate_predictions(val_gt_in, predictions_exact_name_only)
print("\n=== Baseline 0A: Exact Normalized Name Only ===")
print(json.dumps(res_a, indent=2))

# Evaluate Policy B
res_b = evaluate_predictions(val_gt_in, predictions_exact_name_and_addr)
print("\n=== Baseline 0B: Exact Normalized Name + Address ===")
print(json.dumps(res_b, indent=2))

# Save results
baseline_results = {
    "baseline_0a_name_only": res_a,
    "baseline_0b_name_and_addr": res_b,
}
with open("reports/baseline_0_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print(f"\nBaseline 0 evaluated in {round(time.time() - t0, 1)}s.")
