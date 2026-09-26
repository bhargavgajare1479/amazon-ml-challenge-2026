"""
Train the LightGBM matcher and calibrate its output into a real probability.
CPU-only - no GPU required for this model.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config


def train_model(train_part: pd.DataFrame, val_part: pd.DataFrame,
                 feature_cols=config.FEATURE_COLS_FULL):
    lgb_train_set = lgb.Dataset(train_part[feature_cols], label=train_part["label"])
    lgb_val_set = lgb.Dataset(val_part[feature_cols], label=val_part["label"], reference=lgb_train_set)

    model = lgb.train(
        config.LGB_PARAMS,
        lgb_train_set,
        num_boost_round=config.LGB_NUM_BOOST_ROUND,
        valid_sets=[lgb_train_set, lgb_val_set],
        callbacks=[
            lgb.early_stopping(config.LGB_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(50),
        ],
    )
    return model


def calibrate(model, val_part: pd.DataFrame, feature_cols=config.FEATURE_COLS_FULL):
    """Fits isotonic regression so raw LightGBM scores become real match probabilities.
    NOTE: this reuses the validation set for calibration; for a less optimistic estimate,
    split validation further into a calibration half and a threshold-tuning half."""
    val_raw_pred = model.predict(val_part[feature_cols], num_iteration=model.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw_pred, val_part["label"])
    return iso


def predict_proba(model, iso: IsotonicRegression, df: pd.DataFrame,
                   feature_cols=config.FEATURE_COLS_FULL) -> np.ndarray:
    raw = model.predict(df[feature_cols], num_iteration=model.best_iteration)
    return iso.predict(raw)


def feature_importance_report(model, feature_cols=config.FEATURE_COLS_FULL) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
