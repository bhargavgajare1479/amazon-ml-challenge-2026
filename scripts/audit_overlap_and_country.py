import os
import sys
import gc
import json
import time
import re
import numpy as np
import pandas as pd

print("Checking country consistency and string overlap on true matches...")
start_time = time.time()

# We need:
# 1. Map entity_id -> (country, raw_name, norm_name, raw_addr, norm_addr)
# To save memory, let's load S1 train, then load GT, sample or full scan.
# Let's see: full scan of 7.6M pairs:
# We can load s1 country and s2/s3 country.

def normalize_simple(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r'[\s\W_]+', ' ', s).strip()
    return s

# Load S1 id -> country and names
print("Loading Train S1...")
s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str)
s1_country = dict(zip(s1_df["entity_id"], s1_df["country"]))
s1_name_raw = dict(zip(s1_df["entity_id"], s1_df["business_name"].fillna("")))
s1_addr_raw = dict(zip(s1_df["entity_id"], s1_df["business_address"].fillna("")))
del s1_df
gc.collect()

print("Loading Train S2...")
s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str)
s2_country = dict(zip(s2_df["entity_id"], s2_df["country"]))
s2_name_raw = dict(zip(s2_df["entity_id"], s2_df["business_name"].fillna("")))
s2_addr_raw = dict(zip(s2_df["entity_id"], s2_df["business_address"].fillna("")))
del s2_df
gc.collect()

print("Loading Train S3...")
s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str)
s3_country = dict(zip(s3_df["entity_id"], s3_df["country"]))
s3_name_raw = dict(zip(s3_df["entity_id"], s3_df["business_name"].fillna("")))
s3_addr_raw = dict(zip(s3_df["entity_id"], s3_df["business_address"].fillna("")))
del s3_df
gc.collect()

print("Reading Ground Truth and evaluating true pairs...")
# To keep runtime fast and memory light, we can evaluate a huge sample (e.g. 500,000 true pairs) or all.
# Let's evaluate across the full GT!
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)

total_true_pairs = 0
country_mismatches = 0
s2_count = 0
s3_count = 0

exact_raw_name_matches = 0
exact_norm_name_matches = 0
exact_raw_addr_matches = 0
exact_norm_addr_matches = 0
exact_norm_both = 0
exact_norm_neither = 0

# Sample check on 500,000 pairs to be extremely fast and compute robust percentages
sample_size = 500000
evaluated = 0

for s1, m in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
    if not m.strip():
        continue
    c1 = s1_country.get(s1, "")
    n1_raw = s1_name_raw.get(s1, "")
    a1_raw = s1_addr_raw.get(s1, "")
    n1_norm = normalize_simple(n1_raw)
    a1_norm = normalize_simple(a1_raw)
    
    ids = [x.strip() for x in m.split(",") if x.strip()]
    for mid in ids:
        total_true_pairs += 1
        if mid.startswith("S2-"):
            s2_count += 1
            c2 = s2_country.get(mid, "")
            n2_raw = s2_name_raw.get(mid, "")
            a2_raw = s2_addr_raw.get(mid, "")
        else:
            s3_count += 1
            c2 = s3_country.get(mid, "")
            n2_raw = s3_name_raw.get(mid, "")
            a2_raw = s3_addr_raw.get(mid, "")
            
        if c1 != c2:
            country_mismatches += 1
            
        if evaluated < sample_size:
            evaluated += 1
            name_exact_raw = (n1_raw == n2_raw)
            addr_exact_raw = (a1_raw == a2_raw)
            n2_norm = normalize_simple(n2_raw)
            a2_norm = normalize_simple(a2_raw)
            name_exact_norm = (n1_norm == n2_norm and len(n1_norm) > 0)
            addr_exact_norm = (a1_norm == a2_norm and len(a1_norm) > 0)
            
            if name_exact_raw:
                exact_raw_name_matches += 1
            if name_exact_norm:
                exact_norm_name_matches += 1
            if addr_exact_raw:
                exact_raw_addr_matches += 1
            if addr_exact_norm:
                exact_norm_addr_matches += 1
            if name_exact_norm and addr_exact_norm:
                exact_norm_both += 1
            if (not name_exact_norm) and (not addr_exact_norm):
                exact_norm_neither += 1

overlap_results = {
    "total_true_pairs_checked": total_true_pairs,
    "country_mismatches": country_mismatches,
    "country_match_rate": 100.0 * (1.0 - country_mismatches / max(1, total_true_pairs)),
    "evaluated_sample_size": evaluated,
    "exact_raw_name_match_pct": round(100.0 * exact_raw_name_matches / evaluated, 2),
    "exact_norm_name_match_pct": round(100.0 * exact_norm_name_matches / evaluated, 2),
    "exact_raw_addr_match_pct": round(100.0 * exact_raw_addr_matches / evaluated, 2),
    "exact_norm_addr_match_pct": round(100.0 * exact_norm_addr_matches / evaluated, 2),
    "exact_norm_both_pct": round(100.0 * exact_norm_both / evaluated, 2),
    "exact_norm_neither_pct": round(100.0 * exact_norm_neither / evaluated, 2)
}

print("Overlap & Country Results:", json.dumps(overlap_results, indent=2))

with open("reports/audit_checkpoint_overlap.json", "w") as f:
    json.dump(overlap_results, f, indent=2)

print("Done in", round(time.time() - start_time, 2), "s")
