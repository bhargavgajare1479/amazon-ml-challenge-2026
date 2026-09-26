"""
Blocking / candidate generation.

candidate_pairs.tsv now counts toward the final ranking on TWO axes: recall
(did you keep the true matches) AND size (a smaller average candidate set per
entity, at the same recall, ranks higher). This module optimizes both:

1. TF-IDF character n-gram cosine similarity within each country group, run
   TWICE as independent signals - once on business_name, once on
   business_address - and unioned together. This matters because a true match
   can have a very different name (DBA/trade name, heavy abbreviation,
   transliteration) but a matching address, or vice versa; blocking on name
   alone misses those entirely, which is the single biggest recall leak in
   practice. Each signal uses an ADAPTIVE per-entity cutoff (keep candidates
   within ADAPTIVE_MARGIN of the entity's own best match for that signal)
   instead of a flat top-K for every entity. This generalizes to unseen
   countries (e.g. France) automatically, since it groups by whatever country
   string is present rather than a hardcoded list.
2. A cheap token-overlap fallback (on name tokens) that only fires for
   entities the TF-IDF stages under-served, instead of padding every entity's
   candidate set unconditionally.
"""
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import config

STOP_TOKENS = {
    "the", "and", "of", "a", "an", "company", "private", "limited",
    "corporation", "inc", "llc", "group", "co",
}


def build_token_index(df: pd.DataFrame) -> dict:
    """token -> set(entity_id) inverted index, excluding very short/common tokens."""
    index = defaultdict(set)
    for eid, tokens in zip(df["entity_id"], df["name_tokens"]):
        for t in tokens:
            if len(t) >= 3 and t not in STOP_TOKENS:
                index[t].add(eid)
    return index


def token_fallback_candidates(eids_needing_fallback, s1_lookup: pd.DataFrame,
                               other_index: dict, top_n: int = None) -> dict:
    top_n = config.FALLBACK_TOP_TOKENS if top_n is None else top_n
    result = {}
    for eid in eids_needing_fallback:
        tokens = s1_lookup.loc[eid, "name_tokens"]
        counts = Counter()
        for t in tokens:
            if len(t) >= 3 and t not in STOP_TOKENS and t in other_index:
                for cand in other_index[t]:
                    counts[cand] += 1
        result[eid] = [c for c, _ in counts.most_common(top_n)]
    return result


def fit_vectorizer(*name_series, min_df: int = None, max_features: int = None) -> TfidfVectorizer:
    min_df = config.TFIDF_MIN_DF if min_df is None else min_df
    max_features = config.TFIDF_MAX_FEATURES if max_features is None else max_features
    corpus = pd.concat(name_series).tolist()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=min_df,
        max_features=max_features,
    )
    vec.fit(corpus)
    print(f"    vectorizer vocabulary size: {len(vec.vocabulary_)} "
          f"(capped at {max_features}, min_df={min_df})")
    return vec


def tfidf_country_blocking(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    text_col: str = "norm_name",
    top_k: int = None,
    min_sim: float = None,
    adaptive_margin: float = None,
    adaptive_min_keep: int = None,
    n_jobs: int = None,
) -> dict:
    """
    Fetch top_k nearest neighbors by cosine similarity within each country group,
    then prune per-entity down to only candidates within adaptive_margin of that
    entity's own best match (floor: adaptive_min_keep). This is what keeps the
    candidate set small without a flat, wasteful top-K applied to every entity.

    text_col selects which normalized field to block on ("norm_name" or
    "norm_addr") - run this twice with each, then union the results, since a
    true match can agree strongly on one field while disagreeing on the other.

    top_k/min_sim/adaptive_margin/adaptive_min_keep default to the current
    config.* values, READ AT CALL TIME (not import time) - this is what makes
    CLI overrides in pipeline.py (which mutate the config module before calling
    this) actually take effect. Don't give these params `= config.X` directly
    as defaults; that binds the value once at import and ignores later changes.
    """
    top_k = config.TOP_K_BLOCK if top_k is None else top_k
    min_sim = config.MIN_SIM if min_sim is None else min_sim
    adaptive_margin = config.ADAPTIVE_MARGIN if adaptive_margin is None else adaptive_margin
    adaptive_min_keep = config.ADAPTIVE_MIN_KEEP if adaptive_min_keep is None else adaptive_min_keep
    n_jobs = config.NN_N_JOBS if n_jobs is None else n_jobs

    s1_mat = vectorizer.transform(s1_df[text_col])
    other_mat = vectorizer.transform(other_df[text_col])
    sim_scores = defaultdict(dict)
    countries = set(s1_df["country_norm"]) & set(other_df["country_norm"])

    for country in countries:
        s1_idx = np.where(s1_df["country_norm"].values == country)[0]
        other_idx = np.where(other_df["country_norm"].values == country)[0]
        if len(s1_idx) == 0 or len(other_idx) == 0:
            continue
        print(f"    blocking [{text_col}] country='{country}': {len(s1_idx)} x {len(other_idx)} records ...")
        if len(s1_idx) * len(other_idx) > 5_000_000_000:
            print(f"      NOTE: this group is large ({len(s1_idx)} x {len(other_idx)} = "
                  f"{len(s1_idx) * len(other_idx):,} pairs) - this step may take a while. "
                  f"If it's too slow, rerun with a smaller --top-k-block or a higher --min-sim.")
        k = min(top_k, len(other_idx))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=n_jobs)
        nn.fit(other_mat[other_idx])
        dist, idx = nn.kneighbors(s1_mat[s1_idx])
        s1_ids = s1_df["entity_id"].values[s1_idx]
        other_ids = other_df["entity_id"].values[other_idx]

        for row_i, s1_id in enumerate(s1_ids):
            row_sims = 1.0 - dist[row_i]
            best = row_sims.max() if len(row_sims) else 0.0
            for col_j, sim in enumerate(row_sims):
                if sim < min_sim and sim < best - 1e-9:
                    continue  # below the global floor and not this entity's own best - skip
                cand_id = other_ids[idx[row_i, col_j]]
                sim_scores[s1_id][cand_id] = max(sim_scores[s1_id].get(cand_id, 0), sim)

    # adaptive per-entity prune
    for eid, cand_dict in sim_scores.items():
        if not cand_dict:
            continue
        best = max(cand_dict.values())
        kept = {c: s for c, s in cand_dict.items() if s >= best - adaptive_margin}
        if len(kept) < adaptive_min_keep:
            kept = dict(sorted(cand_dict.items(), key=lambda x: -x[1])[:adaptive_min_keep])
        sim_scores[eid] = kept
    return sim_scores


def _merge_sim_dicts(*sim_dicts) -> dict:
    """Union several {eid: {cand: sim}} dicts, keeping the max sim per candidate
    when the same candidate appears from more than one signal."""
    merged = defaultdict(dict)
    for d in sim_dicts:
        for eid, cand_dict in d.items():
            for cand, sim in cand_dict.items():
                merged[eid][cand] = max(merged[eid].get(cand, 0.0), sim)
    return merged


def generate_candidates(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
    name_vectorizer: TfidfVectorizer, addr_vectorizer: TfidfVectorizer,
) -> dict:
    """Returns {source1_entity_id: {candidate_entity_id: similarity_score}}.
    Unions name-based and address-based blocking, since a true match can agree
    strongly on one field while disagreeing on the other."""
    sim_s2_name = tfidf_country_blocking(s1_df, s2_df, name_vectorizer, text_col="norm_name")
    sim_s3_name = tfidf_country_blocking(s1_df, s3_df, name_vectorizer, text_col="norm_name")
    sim_s2_addr = tfidf_country_blocking(s1_df, s2_df, addr_vectorizer, text_col="norm_addr")
    sim_s3_addr = tfidf_country_blocking(s1_df, s3_df, addr_vectorizer, text_col="norm_addr")

    sim_s2 = _merge_sim_dicts(sim_s2_name, sim_s2_addr)
    sim_s3 = _merge_sim_dicts(sim_s3_name, sim_s3_addr)

    s1_lookup = s1_df.set_index("entity_id")
    idx_s2 = build_token_index(s2_df)
    idx_s3 = build_token_index(s3_df)

    all_eids = s1_df["entity_id"].tolist()
    needs_fb_s2 = [e for e in all_eids if len(sim_s2.get(e, {})) < config.FALLBACK_TRIGGER_BELOW]
    needs_fb_s3 = [e for e in all_eids if len(sim_s3.get(e, {})) < config.FALLBACK_TRIGGER_BELOW]
    fb_s2 = token_fallback_candidates(needs_fb_s2, s1_lookup, idx_s2)
    fb_s3 = token_fallback_candidates(needs_fb_s3, s1_lookup, idx_s3)

    candidates = {}
    for eid in all_eids:
        merged = {}
        merged.update(sim_s2.get(eid, {}))
        merged.update(sim_s3.get(eid, {}))
        for c in fb_s2.get(eid, []):
            merged.setdefault(c, 0.0)
        for c in fb_s3.get(eid, []):
            merged.setdefault(c, 0.0)
        if len(merged) > config.MAX_CANDIDATES_PER_ENTITY:
            merged = dict(sorted(merged.items(), key=lambda x: -x[1])[: config.MAX_CANDIDATES_PER_ENTITY])
        candidates[eid] = merged
    return candidates


def blocking_recall(candidates: dict, gt_map: dict, verbose: bool = True):
    """Diagnostic: what fraction of true matches survived blocking (the hard recall ceiling)."""
    total_true, found_true, ent_with, ent_full = 0, 0, 0, 0
    for eid, true_set in gt_map.items():
        if len(true_set) == 0:
            continue
        ent_with += 1
        total_true += len(true_set)
        cand_set = set(candidates.get(eid, {}).keys())
        found = len(true_set & cand_set)
        found_true += found
        if found == len(true_set):
            ent_full += 1
    pair_recall = found_true / total_true if total_true else 1.0
    entity_full_recall = ent_full / ent_with if ent_with else 1.0
    if verbose:
        print(f"Pair-level recall: {pair_recall:.4f}")
        print(f"Entity full-recall (all true matches captured): {entity_full_recall:.4f}")
    return pair_recall, entity_full_recall


def candidate_set_stats(candidates: dict, n_source2: int, n_source3: int) -> dict:
    """Reports the metrics now relevant to ranking: avg/median/max candidates, reduction ratio."""
    counts = [len(v) for v in candidates.values()]
    possible_pairs = len(candidates) * (n_source2 + n_source3)
    actual_pairs = sum(counts)
    reduction_ratio = 1 - (actual_pairs / possible_pairs) if possible_pairs else 0.0
    stats = {
        "avg_candidates": float(np.mean(counts)) if counts else 0.0,
        "median_candidates": float(np.median(counts)) if counts else 0.0,
        "max_candidates": int(np.max(counts)) if counts else 0,
        "reduction_ratio": reduction_ratio,
        "total_pairs_kept": actual_pairs,
        "total_pairs_possible": possible_pairs,
    }
    return stats