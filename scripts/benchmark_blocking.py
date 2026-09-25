"""
Blocking Evaluation Benchmark.
Measures candidate recall and reduction ratio on the validation set.
"""

import os
import gc
import json
import time
import pandas as pd
from collections import defaultdict

import sys
sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

from src.blocking import CandidateBlocker
from src.evaluate import evaluate_blocking_candidates

VAL_DIR = "data/validation"
VAL_S1_TSV = os.path.join(VAL_DIR, "val_source1.tsv")
VAL_GT_JSON = os.path.join(VAL_DIR, "val_ground_truth.json")

print("Starting Blocking Benchmark...")
t0 = time.time()

# 1. Load validation S1 and ground truth
print("Loading validation dataset...")
val_s1_df = pd.read_csv(VAL_S1_TSV, sep="\t", dtype=str)
with open(VAL_GT_JSON) as f:
    val_gt = {k: set(v) for k, v in json.load(f).items()}

print(f"Validation S1 entities: {len(val_s1_df):,}")
total_true_matches = sum(len(v) for v in val_gt.values())
print(f"Total true matches in validation: {total_true_matches:,}")

# We will benchmark country by country (India first, then US)
# To demonstrate fast iteration, let's test India partition first
# Val S1 India:
val_s1_in = val_s1_df[val_s1_df["country"] == "India"].copy()
val_gt_in = {eid: val_gt[eid] for eid in val_s1_in["entity_id"]}
in_true_matches = sum(len(v) for v in val_gt_in.values())
print(f"\n--- Benchmarking India Partition ({len(val_s1_in):,} S1, {in_true_matches:,} true matches) ---")

print("Loading India Target records (S2 and S3)...")
t_load = time.time()
s2_chunks = []
for chunk in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, chunksize=500000):
    sub = chunk[chunk["country"] == "India"]
    s2_chunks.append(sub)
s2_in = pd.concat(s2_chunks, ignore_index=True)
del s2_chunks
gc.collect()

s3_chunks = []
for chunk in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, chunksize=500000):
    sub = chunk[chunk["country"] == "India"]
    s3_chunks.append(sub)
s3_in = pd.concat(s3_chunks, ignore_index=True)
del s3_chunks
gc.collect()

targets_in = pd.concat([s2_in, s3_in], ignore_index=True)
del s2_in, s3_in
gc.collect()
print(f"Loaded {len(targets_in):,} India target records in {round(time.time() - t_load, 1)}s.")

# Initialize blocker and index targets
blocker = CandidateBlocker(max_token_freq=8000, max_candidates_per_s1=100)
blocker.index_target_records(targets_in)

# Generate candidates for all validation S1 records in India
print(f"Generating candidates for {len(val_s1_in):,} validation S1 entities...")
t_cand = time.time()
candidates = {}
for row in val_s1_in.itertuples(index=False):
    s1_id = row.entity_id
    cands = blocker.generate_candidates_for_s1(row.business_name, row.business_address)
    candidates[s1_id] = cands

gen_time = round(time.time() - t_cand, 2)
print(f"Candidate generation completed in {gen_time}s ({round(len(val_s1_in)/gen_time, 1)} entities/sec).")

# Evaluate blocking metrics
eval_res = evaluate_blocking_candidates(val_gt_in, candidates, total_possible_targets=len(targets_in))
print("\n=== India Blocking Results ===")
print(json.dumps(eval_res, indent=2))

with open("reports/blocking_benchmark_india.json", "w") as f:
    json.dump(eval_res, f, indent=2)

print("\nBenchmark completed in", round(time.time() - t0, 1), "s.")
