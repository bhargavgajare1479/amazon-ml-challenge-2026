"""
Feature engineering: pairwise similarity features (name + address) plus
entity-level aggregate features (rank, margin over runner-up, candidate count)
aimed specifically at the singleton-detection problem, since F0.5 rewards
correctly predicting "no match" as much as finding a real one.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("entity_id")


def compute_pair_features(eid1: str, eid2: str, sim_score: float,
                           lookup1: pd.DataFrame, lookup2: pd.DataFrame) -> dict:
    r1 = lookup1.loc[eid1]
    r2 = lookup2.loc[eid2]
    n1, n2 = r1["norm_name"], r2["norm_name"]
    a1, a2 = r1["norm_addr"], r2["norm_addr"]

    feats = {
        "tfidf_sim": sim_score,
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_jaccard": jaccard(r1["name_tokens"], r2["name_tokens"]),
        "addr_ratio": fuzz.ratio(a1, a2) / 100.0,
        "addr_token_set": fuzz.token_set_ratio(a1, a2) / 100.0,
        "addr_jaccard": jaccard(r1["addr_tokens"], r2["addr_tokens"]),
        "digit_jaccard": jaccard(r1["addr_digits"], r2["addr_digits"]),
        "country_match": 1.0 if r1["country_norm"] == r2["country_norm"] else 0.0,
    }
    len1, len2 = max(len(n1), 1), max(len(n2), 1)
    feats["name_len_ratio"] = min(len1, len2) / max(len1, len2)
    return feats


def build_feature_table(candidates: dict, s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                         s3_df: pd.DataFrame, gt_map: dict = None) -> pd.DataFrame:
    lookup1 = build_lookup(s1_df)
    lookup2 = build_lookup(s2_df)
    lookup3 = build_lookup(s3_df)
    rows = []
    for eid, cand_dict in candidates.items():
        for cid, sim in cand_dict.items():
            src_lookup = lookup2 if cid.startswith("S2-") else lookup3
            feats = compute_pair_features(eid, cid, sim, lookup1, src_lookup)
            feats["source1_entity_id"] = eid
            feats["candidate_entity_id"] = cid
            if gt_map is not None:
                feats["label"] = 1 if cid in gt_map.get(eid, set()) else 0
            rows.append(feats)
    return pd.DataFrame(rows)


def add_entity_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rank_in_entity, num_candidates, gap_to_top, top_margin - all zero-cost
    signals for whether an entity likely has a real match vs. is a singleton."""
    df = df.copy()
    df["combined_score"] = (df["tfidf_sim"] + df["name_token_set"] + df["addr_token_set"]) / 3.0
    df = df.sort_values(["source1_entity_id", "combined_score"], ascending=[True, False])
    df["rank_in_entity"] = df.groupby("source1_entity_id").cumcount()
    df["num_candidates"] = df.groupby("source1_entity_id")["combined_score"].transform("count")
    top_score = df.groupby("source1_entity_id")["combined_score"].transform("max")
    df["gap_to_top"] = top_score - df["combined_score"]
    second_score = df.groupby("source1_entity_id")["combined_score"].transform(
        lambda x: x.nlargest(2).min() if len(x) > 1 else x.max()
    )
    df["top_margin"] = np.where(df["rank_in_entity"] == 0, df["combined_score"] - second_score, np.nan)
    df["top_margin"] = df["top_margin"].fillna(0.0)
    return df
