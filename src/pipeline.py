"""
End-to-end entity-resolution pipeline: normalize -> block -> feature engineer
-> train -> calibrate -> tune threshold -> predict -> write submission files.

Run from the project root (the folder containing this src/ package):

    python -m src.pipeline --train-dir dataset/train --test-dir dataset/test --output-dir output

Or just:

    python -m src.pipeline

if your data already lives at ./dataset/train and ./dataset/test.
"""
import argparse
import os

from . import config
from .normalize import load_source, add_normalized_cols, parse_ground_truth
from .blocking import fit_vectorizer, generate_candidates, blocking_recall, candidate_set_stats
from .features import build_feature_table, add_entity_aggregate_features
from .split import split_entities
from .train import train_model, calibrate, predict_proba, feature_importance_report
from .evaluate import tune_threshold
from .predict import write_submission_files, local_validate


def parse_args():
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    p.add_argument("--train-dir", default="dataset/train", help="Folder with train_source1/2/3.tsv and train_ground_truth.tsv")
    p.add_argument("--test-dir", default="dataset/test", help="Folder with test_source1/2/3.tsv")
    p.add_argument("--output-dir", default="output", help="Where to write matching_results.tsv and candidate_pairs.tsv")

    # Blocking/candidate-generation overrides - all optional, fall back to config.py's
    # defaults when omitted. Tune these first when moving from a small sample to the
    # full dataset: memory (--tfidf-max-features) and recall-vs-candidate-set-size
    # (--min-sim / --adaptive-margin / --max-candidates-per-entity) are the knobs
    # that actually change behavior at scale, without editing any code.
    p.add_argument("--tfidf-max-features", type=int, default=None,
                    help="Cap on TF-IDF vocabulary size PER vectorizer (name/address each get their own). "
                         "Lower this if you hit memory pressure on the full dataset. "
                         f"(config default: {config.TFIDF_MAX_FEATURES})")
    p.add_argument("--tfidf-min-df", type=int, default=None,
                    help=f"Ignore n-grams appearing in fewer than this many names/addresses. "
                         f"(config default: {config.TFIDF_MIN_DF})")
    p.add_argument("--top-k-block", type=int, default=None,
                    help=f"Max candidates fetched per entity per source, per signal, before pruning. "
                         f"(config default: {config.TOP_K_BLOCK})")
    p.add_argument("--min-sim", type=float, default=None,
                    help=f"Min cosine similarity to consider a TF-IDF blocking candidate. "
                         f"(config default: {config.MIN_SIM})")
    p.add_argument("--adaptive-margin", type=float, default=None,
                    help=f"Keep candidates within this cosine-sim margin of an entity's best match. "
                         f"(config default: {config.ADAPTIVE_MARGIN})")
    p.add_argument("--adaptive-min-keep", type=int, default=None,
                    help=f"Always keep at least this many candidates per entity, per signal. "
                         f"(config default: {config.ADAPTIVE_MIN_KEEP})")
    p.add_argument("--max-candidates-per-entity", type=int, default=None,
                    help=f"Hard cap on final candidates per entity (candidate_pairs.tsv size). "
                         f"(config default: {config.MAX_CANDIDATES_PER_ENTITY})")
    p.add_argument("--fallback-trigger-below", type=int, default=None,
                    help=f"Fire the token fallback only if TF-IDF found fewer candidates than this. "
                         f"(config default: {config.FALLBACK_TRIGGER_BELOW})")
    p.add_argument("--fallback-top-tokens", type=int, default=None,
                    help=f"Max token-overlap fallback candidates per entity. "
                         f"(config default: {config.FALLBACK_TOP_TOKENS})")
    p.add_argument("--n-jobs", type=int, default=None,
                    help="CPU cores to use for the nearest-neighbor search (-1 = all cores). "
                         "The main speed lever on a strong multi-core machine at full dataset scale. "
                         f"(config default: {config.NN_N_JOBS})")
    return p.parse_args()


def apply_config_overrides(args):
    """Applies any CLI overrides onto the config module, IN PLACE, before any
    blocking/training code runs. blocking.py's functions read config.* fresh at
    call time (not at import time), so this is safe to do here."""
    overrides = {
        "TFIDF_MAX_FEATURES": args.tfidf_max_features,
        "TFIDF_MIN_DF": args.tfidf_min_df,
        "TOP_K_BLOCK": args.top_k_block,
        "MIN_SIM": args.min_sim,
        "ADAPTIVE_MARGIN": args.adaptive_margin,
        "ADAPTIVE_MIN_KEEP": args.adaptive_min_keep,
        "MAX_CANDIDATES_PER_ENTITY": args.max_candidates_per_entity,
        "FALLBACK_TRIGGER_BELOW": args.fallback_trigger_below,
        "FALLBACK_TOP_TOKENS": args.fallback_top_tokens,
        "NN_N_JOBS": args.n_jobs,
    }
    changed = {k: v for k, v in overrides.items() if v is not None}
    for key, value in changed.items():
        setattr(config, key, value)
    if changed:
        print("== Config overrides from CLI ==")
        for key, value in changed.items():
            print(f"  {key} = {value}")
        print()


def load_and_normalize(source_dir: str, prefix: str, with_gt: bool):
    print(f"  loading {prefix}_source1.tsv ...")
    s1 = add_normalized_cols(load_source(os.path.join(source_dir, f"{prefix}_source1.tsv")))
    print(f"  loading {prefix}_source2.tsv ...")
    s2 = add_normalized_cols(load_source(os.path.join(source_dir, f"{prefix}_source2.tsv")))
    print(f"  loading {prefix}_source3.tsv ...")
    s3 = add_normalized_cols(load_source(os.path.join(source_dir, f"{prefix}_source3.tsv")))
    gt_map = None
    if with_gt:
        print(f"  loading {prefix}_ground_truth.tsv ...")
        import pandas as pd
        gt_df = pd.read_csv(os.path.join(source_dir, f"{prefix}_ground_truth.tsv"), sep="\t",
                             dtype=str, keep_default_na=False, encoding="utf-8")
        gt_map = parse_ground_truth(gt_df)
    return s1, s2, s3, gt_map


def main():
    args = parse_args()
    apply_config_overrides(args)

    # ---------- Step 1-2: load + normalize ----------
    print("== Loading & normalizing train data ==")
    train_s1, train_s2, train_s3, gt_map = load_and_normalize(args.train_dir, "train", with_gt=True)
    print(f"train: s1={len(train_s1)} s2={len(train_s2)} s3={len(train_s3)} gt={len(gt_map)}")

    print("== Loading & normalizing test data ==")
    test_s1, test_s2, test_s3, _ = load_and_normalize(args.test_dir, "test", with_gt=False)
    print(f"test: s1={len(test_s1)} s2={len(test_s2)} s3={len(test_s3)}")

    # ---------- Step 3: blocking on train ----------
    print("\n== Blocking (train) ==")
    name_vec_train = fit_vectorizer(train_s1["norm_name"], train_s2["norm_name"], train_s3["norm_name"])
    addr_vec_train = fit_vectorizer(train_s1["norm_addr"], train_s2["norm_addr"], train_s3["norm_addr"])
    train_candidates = generate_candidates(train_s1, train_s2, train_s3, name_vec_train, addr_vec_train)
    stats = candidate_set_stats(train_candidates, len(train_s2), len(train_s3))
    print(f"avg candidates/entity: {stats['avg_candidates']:.2f}  "
          f"median: {stats['median_candidates']:.1f}  max: {stats['max_candidates']}  "
          f"reduction ratio: {stats['reduction_ratio']:.6f}")
    blocking_recall(train_candidates, gt_map)

    # ---------- Step 4: features ----------
    print("\n== Feature engineering (train) ==")
    train_feat_df = build_feature_table(train_candidates, train_s1, train_s2, train_s3, gt_map)
    train_feat_df = add_entity_aggregate_features(train_feat_df)
    print(f"feature table: {train_feat_df.shape}, positive rate: {train_feat_df['label'].mean():.4f}")

    # ---------- Step 5: split ----------
    train_entities, val_entities = split_entities(train_s1["entity_id"])
    train_part = train_feat_df[train_feat_df["source1_entity_id"].isin(train_entities)].reset_index(drop=True)
    val_part = train_feat_df[train_feat_df["source1_entity_id"].isin(val_entities)].reset_index(drop=True)
    print(f"train rows: {train_part.shape}  val rows: {val_part.shape}")

    # ---------- Step 6: train model ----------
    print("\n== Training LightGBM matcher ==")
    model = train_model(train_part, val_part)
    print(feature_importance_report(model))

    # ---------- Step 7: calibrate ----------
    iso = calibrate(model, val_part)
    val_part = val_part.copy()
    val_part["pred_prob"] = predict_proba(model, iso, val_part)

    # ---------- Step 9: tune threshold ----------
    print("\n== Tuning decision threshold against F0.5 ==")
    best_threshold, best_score = tune_threshold(val_part, val_entities, gt_map)
    print(f"best threshold: {best_threshold:.2f}   validation macro F0.5: {best_score:.4f}")

    # ---------- Step 10: test blocking + features ----------
    print("\n== Blocking + features (test) ==")
    name_vec_test = fit_vectorizer(test_s1["norm_name"], test_s2["norm_name"], test_s3["norm_name"])
    addr_vec_test = fit_vectorizer(test_s1["norm_addr"], test_s2["norm_addr"], test_s3["norm_addr"])
    test_candidates = generate_candidates(test_s1, test_s2, test_s3, name_vec_test, addr_vec_test)
    test_stats = candidate_set_stats(test_candidates, len(test_s2), len(test_s3))
    print(f"avg candidates/entity: {test_stats['avg_candidates']:.2f}  "
          f"median: {test_stats['median_candidates']:.1f}  max: {test_stats['max_candidates']}  "
          f"reduction ratio: {test_stats['reduction_ratio']:.6f}")

    test_feat_df = build_feature_table(test_candidates, test_s1, test_s2, test_s3, gt_map=None)
    test_feat_df = add_entity_aggregate_features(test_feat_df)
    test_feat_df["pred_prob"] = predict_proba(model, iso, test_feat_df)

    # ---------- Step 11: write outputs ----------
    print("\n== Writing submission files ==")
    matching_path, candidate_path, final_preds = write_submission_files(
        test_s1, test_candidates, test_feat_df, best_threshold, args.output_dir
    )
    n_matched = sum(1 for m in final_preds.values() if len(m) > 0)
    print(f"Wrote {matching_path}")
    print(f"Wrote {candidate_path}")
    print(f"Entities predicted with >=1 match: {n_matched} / {len(test_s1)}")

    # ---------- Step 12: local validation ----------
    print("\n== Local validation checks ==")
    local_validate(
        matching_path, candidate_path,
        test_s1["entity_id"].tolist(), test_s2["entity_id"].tolist(), test_s3["entity_id"].tolist(),
    )
    print("\nDone. Now run the official utils/validate_submission.py before submitting.")


if __name__ == "__main__":
    main()