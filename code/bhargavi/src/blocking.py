"""
Candidate generation (blocking).

Idea: turn every record's name+address into a TF-IDF vector of words, then for
each S1 record find the K most similar S2/S3 records *in the same country*.
Very common words (city names, "road", "services") are dropped with max_df:
they barely help and would make the search slow.

Returns integer positions (not string IDs) to keep memory small.
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _topk_per_row(M, k):
    """Rows/cols/scores of the k largest entries in every row of sparse matrix M."""
    M = M.tocoo()
    if M.nnz == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32), e
    order = np.lexsort((-M.data, M.row))          # by row, then score descending
    r, c, d = M.row[order], M.col[order], M.data[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))
    keep = rank < k
    return r[keep], c[keep], d[keep], rank[keep]


def block(s1: pd.DataFrame, index: pd.DataFrame, k: int = 30,
          max_df: int = 2000, chunk: int = 2000, verbose: bool = True) -> pd.DataFrame:
    """
    s1, index: DataFrames with 'country' and 'block_text', and a default 0..n-1 index.
    index = S2 and S3 records together.
    Returns DataFrame(s1_pos, cand_pos, score, rank) with positions into s1 / index.
    """
    out = []
    for country in s1["country"].unique():          # open set: works for France too
        a_pos = np.flatnonzero((s1["country"] == country).to_numpy())
        b_pos = np.flatnonzero((index["country"] == country).to_numpy())
        if len(a_pos) == 0 or len(b_pos) == 0:
            continue
        t0 = time.time()
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, dtype=np.float32,
                              sublinear_tf=True, max_df=min(max_df, len(b_pos)))
        B = vec.fit_transform(index["block_text"].to_numpy()[b_pos])
        BT = B.T.tocsr()
        A = vec.transform(s1["block_text"].to_numpy()[a_pos])
        for i in range(0, A.shape[0], chunk):
            r, c, d, rk = _topk_per_row(A[i:i + chunk] @ BT, k)
            out.append(pd.DataFrame({
                "s1_pos": a_pos[i + r].astype(np.int32),
                "cand_pos": b_pos[c].astype(np.int32),
                "score": d.astype(np.float32),
                "rank": rk.astype(np.int16),
            }))
        if verbose:
            print(f"  {country}: {len(a_pos):,} S1 vs {len(b_pos):,} S2/S3, "
                  f"vocab {len(vec.vocabulary_):,}, {time.time() - t0:.0f}s")
    if not out:
        return pd.DataFrame(columns=["s1_pos", "cand_pos", "score", "rank"])
    return pd.concat(out, ignore_index=True)


def recall_at_k(cand: pd.DataFrame, s1: pd.DataFrame, index: pd.DataFrame,
                true_pairs: pd.DataFrame, ks=(5, 10, 20, 30)) -> pd.DataFrame:
    """
    true_pairs: DataFrame(source1_entity_id, ids) of true matches.
    Recall@k = share of true matches that appear in the top-k candidates.
    This is the ceiling for the final model's recall.
    """
    s1_map = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    ix_map = pd.Series(np.arange(len(index)), index=index["entity_id"])
    tp = true_pairs[true_pairs["source1_entity_id"].isin(s1_map.index)]
    t1 = s1_map.reindex(tp["source1_entity_id"]).to_numpy()
    t2 = ix_map.reindex(tp["ids"]).to_numpy()
    ok = ~np.isnan(t2)
    true_key = t1[ok].astype(np.int64) * len(index) + t2[ok].astype(np.int64)
    rows = []
    for k in ks:
        sub = cand[cand["rank"] < k]
        key = sub["s1_pos"].to_numpy(np.int64) * len(index) + sub["cand_pos"].to_numpy(np.int64)
        rows.append({"k": k, "recall": np.isin(true_key, key).mean(),
                     "cands_per_s1": len(sub) / len(s1)})
    return pd.DataFrame(rows)
