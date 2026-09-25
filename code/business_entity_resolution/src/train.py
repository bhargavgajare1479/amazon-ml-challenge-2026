"""
Supervised Matcher Training Pipeline for Business Entity Resolution.
Trains a Gradient Boosted Decision Tree (LightGBM) on pairwise features.
"""

import os
import gc
import json
import time
import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, Any

try:
    from .labels import generate_training_pairs
    from .features import extract_batch_features, FEATURE_NAMES
except ImportError:
    from labels import generate_training_pairs
    from features import extract_batch_features, FEATURE_NAMES

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_matcher.bin")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")


def train_matcher(
    sample_s1_count: int = 20000,
    neg_to_pos_ratio: int = 6,
    seed: int = 42,
    model_dir: str = MODEL_DIR,
):
    print("=== Starting Supervised Matcher Training ===")
    t0 = time.time()
    os.makedirs(model_dir, exist_ok=True)

    # 1. Generate training candidate pairs
    pairs, labels, s1_meta, target_meta = generate_training_pairs(
        sample_s1_count=sample_s1_count,
        neg_to_pos_ratio=neg_to_pos_ratio,
        seed=seed,
    )

    # 2. Extract pairwise features
    print(f"\nExtracting features for {len(pairs):,} training pairs...")
    t_feat = time.time()
    X = extract_batch_features(s1_meta, target_meta, pairs)
    y = np.array(labels, dtype=np.int32)
    print(f"Feature matrix built in {round(time.time() - t_feat, 1)}s. Shape: {X.shape}")

    # 3. Split train / internal validation for early stopping (85% / 15%)
    n_total = len(X)
    n_train = int(0.85 * n_total)
    
    indices = np.arange(n_total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Train split: {len(X_train):,} pairs (pos: {int(y_train.sum()):,})")
    print(f"Val split:   {len(X_val):,} pairs (pos: {int(y_val.sum()):,})")

    # 4. Train LightGBM Binary Classifier
    print("\nTraining LightGBM classifier...")
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": seed,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    t_train = time.time()
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=600,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Model trained in {round(time.time() - t_train, 1)}s.")

    # 5. Feature Importances
    importances = booster.feature_importance(importance_type="gain")
    sorted_feat = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 15 Most Informative Features (by Gain):")
    for fname, imp in sorted_feat[:15]:
        print(f"  {fname:<28} = {imp:>10.1f}")

    # 6. Save Model and Configuration
    model_path = os.path.join(model_dir, "lgbm_matcher.bin")
    booster.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    config = {
        "feature_names": FEATURE_NAMES,
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "num_features": len(FEATURE_NAMES),
        "train_pairs_count": len(X),
        "seed": seed,
    }
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    total_time = round(time.time() - t0, 1)
    print(f"=== Matcher Training Completed in {total_time}s ===")
    return booster


if __name__ == "__main__":
    train_matcher()
