import os
import sys
import gc
import json
import time
import re
import numpy as np
import pandas as pd
from collections import Counter

print("Starting dataset audit...")
start_time = time.time()

# File paths
TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
TRAIN_GT = "dataset/train/train_ground_truth.tsv"
TEST_S1 = "dataset/test/test_source1.tsv"
TEST_S2 = "dataset/test/test_source2.tsv"
TEST_S3 = "dataset/test/test_source3.tsv"

def get_file_stats(path):
    size_bytes = os.path.getsize(path)
    return {
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2)
    }

def normalize_simple(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r'[\s\W_]+', ' ', s).strip()
    return s

audit_results = {}

# 1. Audit Ground Truth first (fast and essential for GT patterns)
print("\n--- Auditing Ground Truth ---")
gt_df = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
print(f"Ground truth rows: {len(gt_df):,}")
print(f"Ground truth columns: {gt_df.columns.tolist()}")

s1_gt_ids = gt_df["source1_entity_id"]
matched_col = gt_df["matched_entity_ids"].fillna("")

gt_s1_unique = s1_gt_ids.nunique()
gt_s1_total = len(s1_gt_ids)
gt_duplicate_s1 = gt_s1_total - gt_s1_unique

# Parse matches
match_counts = []
s2_counts = []
s3_counts = []
all_matched_ids = []
all_s2_matched = []
all_s3_matched = []

only_s2 = 0
only_s3 = 0
both_s2_s3 = 0
zero_matches = 0

# For checking if S2/S3 IDs match multiple S1 entities
matched_target_to_s1 = {}
s2_target_to_s1 = Counter()
s3_target_to_s1 = Counter()

# Sample ground truth pairs for cross-referencing later
gt_pairs_dict = {} # s1 -> set of targets

for s1, m in zip(s1_gt_ids, matched_col):
    if not m.strip():
        match_counts.append(0)
        s2_counts.append(0)
        s3_counts.append(0)
        zero_matches += 1
        gt_pairs_dict[s1] = set()
    else:
        ids = [x.strip() for x in m.split(",") if x.strip()]
        s2_ids = [x for x in ids if x.startswith("S2-")]
        s3_ids = [x for x in ids if x.startswith("S3-")]
        other_ids = [x for x in ids if not (x.startswith("S2-") or x.startswith("S3-"))]
        
        match_counts.append(len(ids))
        s2_counts.append(len(s2_ids))
        s3_counts.append(len(s3_ids))
        
        gt_pairs_dict[s1] = set(ids)
        
        for mid in ids:
            if mid.startswith("S2-"):
                s2_target_to_s1[mid] += 1
            elif mid.startswith("S3-"):
                s3_target_to_s1[mid] += 1
                
        if len(s2_ids) > 0 and len(s3_ids) == 0:
            only_s2 += 1
        elif len(s3_ids) > 0 and len(s2_ids) == 0:
            only_s3 += 1
        elif len(s2_ids) > 0 and len(s3_ids) > 0:
            both_s2_s3 += 1

match_counts = np.array(match_counts)
s2_counts = np.array(s2_counts)
s3_counts = np.array(s3_counts)

gt_stats = {
    "total_s1_in_gt": gt_s1_total,
    "unique_s1_in_gt": gt_s1_unique,
    "duplicate_s1_rows": gt_duplicate_s1,
    "zero_matches_singletons": int(zero_matches),
    "singleton_percentage": float(round(100.0 * zero_matches / gt_s1_total, 4)),
    "only_s2": int(only_s2),
    "only_s3": int(only_s3),
    "both_s2_s3": int(both_s2_s3),
    "total_matched_pairs": int(match_counts.sum()),
    "total_s2_matches": int(s2_counts.sum()),
    "total_s3_matches": int(s3_counts.sum()),
    "unique_s2_matched": len(s2_target_to_s1),
    "unique_s3_matched": len(s3_target_to_s1),
    "match_count_distribution": {
        str(k): int(v) for k, v in Counter(match_counts).most_common(15)
    },
    "s2_count_distribution": {
        str(k): int(v) for k, v in Counter(s2_counts).most_common(10)
    },
    "s3_count_distribution": {
        str(k): int(v) for k, v in Counter(s3_counts).most_common(10)
    },
    "max_matches_per_s1": int(match_counts.max()),
    "mean_matches_per_s1": float(round(match_counts.mean(), 4)),
    "median_matches_per_s1": float(np.median(match_counts)),
    # Check if S2 or S3 matches link to multiple S1s
    "s2_max_s1_per_target": max(s2_target_to_s1.values()) if s2_target_to_s1 else 0,
    "s2_targets_matching_gt1_s1": sum(1 for c in s2_target_to_s1.values() if c > 1),
    "s3_max_s1_per_target": max(s3_target_to_s1.values()) if s3_target_to_s1 else 0,
    "s3_targets_matching_gt1_s1": sum(1 for c in s3_target_to_s1.values() if c > 1),
}

print("GT stats computed:", json.dumps(gt_stats, indent=2))
del gt_df, match_counts, s2_counts, s3_counts
gc.collect()

# Save checkpoint
audit_results["ground_truth"] = gt_stats

with open("reports/audit_checkpoint_gt.json", "w") as f:
    json.dump(audit_results, f, indent=2)

print("\n--- Ground Truth Audit Complete ---")
