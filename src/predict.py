"""
Runs inference on the test candidate set and writes matching_results.tsv and
candidate_pairs.tsv in the exact format the challenge requires, plus a local
validator that mirrors the key rules from the official validate_submission.py.
"""
import os

import pandas as pd

from .evaluate import build_predictions_from_scores


def write_submission_files(test_s1_df: pd.DataFrame, test_candidates: dict,
                            test_feat_df: pd.DataFrame, threshold: float, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_test_s1_ids = test_s1_df["entity_id"].tolist()
    final_preds = build_predictions_from_scores(test_feat_df, threshold, all_test_s1_ids)

    matching_rows = []
    for eid in all_test_s1_ids:
        matched = sorted(final_preds.get(eid, set()))
        matching_rows.append({"source1_entity_id": eid, "matched_entity_ids": ",".join(matched)})
    matching_df = pd.DataFrame(matching_rows)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    matching_df.to_csv(matching_path, sep="\t", index=False, encoding="utf-8")

    candidate_rows = []
    for eid in all_test_s1_ids:
        cands = sorted(test_candidates.get(eid, {}).keys())
        candidate_rows.append({"source1_entity_id": eid, "candidate_entity_ids": ",".join(cands)})
    candidate_df = pd.DataFrame(candidate_rows)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    candidate_df.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8")

    return matching_path, candidate_path, final_preds


def local_validate(matching_path: str, candidate_path: str,
                    test_s1_ids, test_s2_ids, test_s3_ids) -> list:
    """Quick sanity check mirroring the official validator's key rules.
    Still run the real utils/validate_submission.py before submitting."""
    issues = []
    valid_ids = set(test_s2_ids) | set(test_s3_ids)
    s1_set = set(test_s1_ids)

    m_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    c_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)

    if set(m_df["source1_entity_id"]) != s1_set:
        issues.append("matching_results.tsv does not cover exactly all test Source-1 entities")
    if m_df["source1_entity_id"].duplicated().any():
        issues.append("Duplicate source1_entity_id rows in matching_results.tsv")

    cand_map = {}
    for row in c_df.itertuples(index=False):
        cand_map[row.source1_entity_id] = set(x for x in row.candidate_entity_ids.split(",") if x)

    for row in m_df.itertuples(index=False):
        eid = row.source1_entity_id
        matched = [x for x in row.matched_entity_ids.split(",") if x]
        if len(matched) != len(set(matched)):
            issues.append(f"Duplicate IDs within matched_entity_ids for {eid}")
        for mid in matched:
            if mid not in valid_ids:
                issues.append(f"{mid} for {eid} is not a valid test Source2/3 ID")
            if eid in cand_map and mid not in cand_map[eid]:
                issues.append(f"{mid} matched for {eid} but missing from its own candidate list")

    if issues:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for i in issues[:30]:
            print(" -", i)
    else:
        print("Local checks PASS. Still run the official utils/validate_submission.py before submitting.")
    return issues