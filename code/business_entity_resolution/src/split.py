"""
Validation Split Generator for Business Entity Resolution.
Creates a stratified, leak-free validation set at the Source 1 entity level.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Set

SEED = 42
VAL_SIZE = 30000

TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_GT = "dataset/train/train_ground_truth.tsv"
VAL_DIR = "data/validation"


def create_validation_split(
    val_size: int = VAL_SIZE,
    seed: int = SEED,
    output_dir: str = VAL_DIR,
):
    print(f"Creating stratified validation split of {val_size:,} S1 entities (seed={seed})...")
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load S1 metadata (entity_id, country)
    print("Loading S1 metadata...")
    s1_df = pd.read_csv(TRAIN_S1, sep="\t", usecols=["entity_id", "country"], dtype=str)
    
    # 2. Load ground truth to compute match count bins
    print("Loading ground truth...")
    gt_df = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    
    # Map S1 to match count
    match_counts = {}
    gt_dict = {}
    for s1, m in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
        if not m.strip():
            match_counts[s1] = 0
            gt_dict[s1] = set()
        else:
            ids = set(x.strip() for x in m.split(",") if x.strip())
            match_counts[s1] = len(ids)
            gt_dict[s1] = ids

    s1_df["match_count"] = s1_df["entity_id"].map(match_counts).fillna(0).astype(int)

    # Bin match counts for stratification: 0 (singleton), 1, 2, 3-4, 5+
    def get_bin(c):
        if c == 0:
            return "0_singleton"
        elif c == 1:
            return "1_match"
        elif c == 2:
            return "2_matches"
        elif c in (3, 4):
            return "3_4_matches"
        else:
            return "5plus_matches"

    s1_df["strata"] = s1_df["country"] + "__" + s1_df["match_count"].apply(get_bin)

    print("Stratification distribution across all S1:")
    strata_counts = s1_df["strata"].value_counts(normalize=True)
    for s, frac in strata_counts.items():
        print(f"  {s}: {frac*100:.2f}%")

    # Sample proportionally from each stratum
    sampled_dfs = []
    rng = np.random.default_rng(seed)
    
    for strata_name, group in s1_df.groupby("strata"):
        n_sample = int(round(len(group) / len(s1_df) * val_size))
        sampled_ids = rng.choice(group["entity_id"].values, size=n_sample, replace=False)
        sampled_dfs.append(sampled_ids)

    all_val_ids = np.concatenate(sampled_dfs)
    # Adjust if slight rounding discrepancy
    if len(all_val_ids) > val_size:
        all_val_ids = rng.choice(all_val_ids, size=val_size, replace=False)
    val_set = set(all_val_ids)
    print(f"Selected {len(val_set):,} validation S1 entities.")

    # Save validation IDs list
    val_ids_path = os.path.join(output_dir, "val_s1_ids.json")
    with open(val_ids_path, "w") as f:
        json.dump(sorted(list(val_set)), f)

    # Save validation ground truth
    val_gt = {s1: list(gt_dict[s1]) for s1 in val_set}
    val_gt_path = os.path.join(output_dir, "val_ground_truth.json")
    with open(val_gt_path, "w") as f:
        json.dump(val_gt, f)

    # Also extract and save val_source1.tsv (convenient for fast local loading)
    print("Extracting val_source1.tsv...")
    full_s1_df = pd.read_csv(TRAIN_S1, sep="\t", dtype=str)
    val_s1_df = full_s1_df[full_s1_df["entity_id"].isin(val_set)].copy()
    val_s1_tsv_path = os.path.join(output_dir, "val_source1.tsv")
    val_s1_df.to_csv(val_s1_tsv_path, sep="\t", index=False)

    # Summary statistics of validation set
    val_singletons = sum(1 for s1 in val_set if len(val_gt[s1]) == 0)
    val_true_matches = sum(len(val_gt[s1]) for s1 in val_set)
    val_countries = val_s1_df["country"].value_counts().to_dict()

    summary = {
        "val_size": len(val_set),
        "val_singletons": val_singletons,
        "val_singleton_pct": round(100.0 * val_singletons / len(val_set), 3),
        "val_true_matches": val_true_matches,
        "val_mean_matches": round(val_true_matches / len(val_set), 3),
        "country_distribution": val_countries,
        "creation_time_sec": round(time.time() - t0, 2),
    }

    print("\nValidation Set Summary:")
    print(json.dumps(summary, indent=2))
    
    with open(os.path.join(output_dir, "val_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return val_set


if __name__ == "__main__":
    create_validation_split()
