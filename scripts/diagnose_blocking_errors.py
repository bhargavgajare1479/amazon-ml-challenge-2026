"""
Diagnose blocking false negatives (missed matches) on a fast sample.
"""

import os
import sys
import gc
import json
import time
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import CandidateBlocker
from src.normalize import normalize_business_name, normalize_business_address

VAL_S1_TSV = "data/validation/val_source1.tsv"
VAL_GT_JSON = "data/validation/val_ground_truth.json"

# Load validation
val_s1_df = pd.read_csv(VAL_S1_TSV, sep="\t", dtype=str)
val_s1_in = val_s1_df[val_s1_df["country"] == "India"].head(1000).copy()
sample_s1_ids = set(val_s1_in["entity_id"])

with open(VAL_GT_JSON) as f:
    full_gt = json.load(f)

sample_gt = {eid: set(full_gt[eid]) for eid in sample_s1_ids}
sample_true_target_ids = set()
for s in sample_gt.values():
    sample_true_target_ids.update(s)

print(f"Sample S1: {len(sample_s1_ids):,}, True target IDs to find: {len(sample_true_target_ids):,}")

# Load targets that correspond to these, plus a background of S2/S3
s2_chunks = []
for chunk in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, chunksize=300000):
    sub = chunk[chunk["country"] == "India"]
    # keep rows that are either true targets or sample
    hit = sub[sub["entity_id"].isin(sample_true_target_ids)]
    s2_chunks.append(hit)
    if len(s2_chunks) > 10: break

s3_chunks = []
for chunk in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, chunksize=300000):
    sub = chunk[chunk["country"] == "India"]
    hit = sub[sub["entity_id"].isin(sample_true_target_ids)]
    s3_chunks.append(hit)
    if len(s3_chunks) > 10: break

target_sample = pd.concat(s2_chunks + s3_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
print(f"Found {len(target_sample):,} matching target records for detailed error diagnosis.")

target_dict = target_sample.set_index("entity_id").to_dict("index")
s1_dict = val_s1_in.set_index("entity_id").to_dict("index")

blocker = CandidateBlocker(max_token_freq=8000, max_candidates_per_s1=100)
blocker.index_target_records(target_sample)

missed = []
for row in val_s1_in.itertuples(index=False):
    s1_id = row.entity_id
    true_set = sample_gt[s1_id]
    if not true_set: continue
    cands = blocker.generate_candidates_for_s1(row.business_name, row.business_address)
    missed_for_s1 = true_set - cands
    for m_id in missed_for_s1:
        if m_id in target_dict:
            missed.append((s1_id, m_id, s1_dict[s1_id], target_dict[m_id]))

print(f"\nFound {len(missed)} missed pairs among the indexed targets.")
print("\n=== Sample Missed Pairs ===")
for s1_id, m_id, s1_data, m_data in missed[:10]:
    print(f"\nS1 ({s1_id}):")
    print(f"  Name:    {s1_data['business_name']}")
    print(f"  Address: {s1_data['business_address']}")
    print(f"Target ({m_id}):")
    print(f"  Name:    {m_data['business_name']}")
    print(f"  Address: {m_data['business_address']}")
