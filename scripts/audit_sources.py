import os
import sys
import gc
import json
import time
import re
import numpy as np
import pandas as pd
from collections import Counter

print("Starting source files audit...")
start_time = time.time()

# File paths
FILES = {
    "train_s1": "dataset/train/train_source1.tsv",
    "train_s2": "dataset/train/train_source2.tsv",
    "train_s3": "dataset/train/train_source3.tsv",
    "test_s1": "dataset/test/test_source1.tsv",
    "test_s2": "dataset/test/test_source2.tsv",
    "test_s3": "dataset/test/test_source3.tsv",
}

def clean_str(s):
    if not isinstance(s, str):
        return ""
    return s.strip()

def normalize_simple(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r'[\s\W_]+', ' ', s).strip()
    return s

source_stats = {}

for name, path in FILES.items():
    t0 = time.time()
    file_bytes = os.path.getsize(path)
    file_mb = round(file_bytes / (1024 * 1024), 2)
    print(f"\nProcessing {name} ({file_mb} MB)...")
    
    # Load with pandas
    df = pd.read_csv(path, sep="\t", dtype=str)
    n_rows = len(df)
    cols = df.columns.tolist()
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = round(mem_bytes / (1024 * 1024), 2)
    
    # Missing values
    missing = {}
    empty_or_whitespace = {}
    for c in cols:
        n_null = int(df[c].isna().sum())
        missing[c] = n_null
        # check empty or whitespace
        s_series = df[c].fillna("")
        n_empty = int((s_series.str.strip() == "").sum())
        empty_or_whitespace[c] = n_empty

    # Entity ID uniqueness
    unique_ids = int(df["entity_id"].nunique())
    duplicate_ids = n_rows - unique_ids
    
    # Prefix check
    id_prefixes = df["entity_id"].apply(lambda x: str(x)[:3]).value_counts().to_dict()
    
    # Duplicate business names and addresses
    unique_names = int(df["business_name"].nunique())
    duplicate_names = n_rows - unique_names
    
    unique_addresses = int(df["business_address"].nunique())
    duplicate_addresses = n_rows - unique_addresses
    
    # Exact duplicate rows (across all columns)
    exact_dups = int(df.duplicated().sum())
    
    # Exact duplicate across (business_name, business_address, country)
    exact_content_dups = int(df.duplicated(subset=["business_name", "business_address", "country"]).sum())
    
    # Country distribution
    country_counts = df["country"].value_counts(dropna=False).to_dict()
    country_dist = {str(k): int(v) for k, v in country_counts.items()}
    country_pct = {str(k): round(100.0 * int(v) / n_rows, 3) for k, v in country_counts.items()}
    
    # Length distributions
    name_lens = df["business_name"].fillna("").str.len().values
    addr_lens = df["business_address"].fillna("").str.len().values
    
    name_len_stats = {
        "min": int(name_lens.min()),
        "max": int(name_lens.max()),
        "mean": float(round(float(name_lens.mean()), 2)),
        "median": float(np.median(name_lens)),
        "p25": float(np.percentile(name_lens, 25)),
        "p75": float(np.percentile(name_lens, 75)),
        "p95": float(np.percentile(name_lens, 95)),
        "p99": float(np.percentile(name_lens, 99)),
        "len_0": int((name_lens == 0).sum()),
        "len_le_2": int((name_lens <= 2).sum()),
    }
    
    addr_len_stats = {
        "min": int(addr_lens.min()),
        "max": int(addr_lens.max()),
        "mean": float(round(float(addr_lens.mean()), 2)),
        "median": float(np.median(addr_lens)),
        "p25": float(np.percentile(addr_lens, 25)),
        "p75": float(np.percentile(addr_lens, 75)),
        "p95": float(np.percentile(addr_lens, 95)),
        "p99": float(np.percentile(addr_lens, 99)),
        "len_0": int((addr_lens == 0).sum()),
        "len_le_5": int((addr_lens <= 5).sum()),
    }
    
    # Sample top frequent business names
    top_names = df["business_name"].value_counts().head(5).to_dict()
    # Sample top frequent addresses
    top_addrs = df["business_address"].value_counts().head(5).to_dict()

    stats = {
        "file_size_mb": file_mb,
        "n_rows": n_rows,
        "columns": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
        "memory_usage_mb": mem_mb,
        "missing_nulls": missing,
        "missing_empty_whitespace": empty_or_whitespace,
        "unique_entity_ids": unique_ids,
        "duplicate_entity_ids": duplicate_ids,
        "id_prefixes": id_prefixes,
        "unique_business_names": unique_names,
        "duplicate_business_names": duplicate_names,
        "unique_business_addresses": unique_addresses,
        "duplicate_business_addresses": duplicate_addresses,
        "exact_duplicate_rows": exact_dups,
        "duplicate_name_addr_country": exact_content_dups,
        "country_counts": country_dist,
        "country_pct": country_pct,
        "name_length_stats": name_len_stats,
        "addr_length_stats": addr_len_stats,
        "top_frequent_names": {str(k): int(v) for k, v in top_names.items()},
        "top_frequent_addresses": {str(k): int(v) for k, v in top_addrs.items()},
        "processing_time_sec": round(time.time() - t0, 2)
    }
    
    source_stats[name] = stats
    print(f"Finished {name} in {stats['processing_time_sec']}s: {n_rows:,} rows, mem: {mem_mb} MB, countries: {country_dist}")
    
    del df, name_lens, addr_lens
    gc.collect()

with open("reports/audit_checkpoint_sources.json", "w") as f:
    json.dump(source_stats, f, indent=2)

print("\n--- Source Files Audit Complete in", round(time.time() - start_time, 2), "s ---")
