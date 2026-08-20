"""Feature-group ablation for the canonical 44-feature pipeline (lgbm_63,
seed-42, 5-fold), on GitHub Actions. Quantifies the contribution of each
feature family to the EXP-023 lgbm_63 baseline (OOF 0.96380).

Usage:
    python state/exp_ablate.py
"""
import json
import os
import sys
import time

# Make the repo root importable so this script can import the `source`
# package (sibling of `state/`) regardless of the invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from source.train_pipeline import (
    COMMON_PARAMS,
    MODEL_CONFIGS,
    N_SPLITS,
    TARGET,
    build_features,
    feature_columns,
)

GROUPS = {
    "other_screen_time": ["other_screen_time", "other_screen_time_isna"],
    "isna_indicators": [c for c in feature_columns() if c.endswith("_isna")],
    "missingness_counts": ["missing_count_total", "missing_count_numeric", "missing_count_categorical"],
    "domain_ratios": [
        "social_media_share", "gaming_share", "work_study_share", "entertainment_share",
        "weekend_weekday_ratio", "weekend_extra_screen", "screen_sleep_ratio", "sleep_awake_share",
        "notifications_per_screen_hour", "app_opens_per_screen_hour", "notifications_per_app_open",
        "engagement_intensity", "work_sleep_ratio", "app_opens_per_notification", "total_active_hours",
    ],
    "categoricals": ["gender", "stress_level", "academic_work_impact"],
    "age": ["age"],
    "notifications": ["notifications_per_day"],
    "app_opens": ["app_opens_per_day"],
    "sleep": ["sleep_hours"],
}


def run_features(X, y, feats):
    cfg = dict(MODEL_CONFIGS["lgbm_63"])
    params = dict(COMMON_PARAMS)
    params.update(cfg)
    params["random_state"] = 42
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X[feats], y):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X[feats].iloc[tr_idx], y[tr_idx],
            eval_X=X[feats].iloc[va_idx], eval_y=y[va_idx],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = m.predict_proba(X[feats].iloc[va_idx])[:, 1]
    return roc_auc_score(y, oof)


def main():
    t0 = time.time()
    train = pd.read_csv("train.csv")
    train = build_features(train)
    X = train[feature_columns()]
    y = train[TARGET].values

    results = {}
    full = run_features(X, y, feature_columns())
    results["full"] = {"oof_auc": full, "delta_vs_full": 0.0}
    print(f"full (44 features): OOF AUC = {full:.5f}", flush=True)

    for name, feats in GROUPS.items():
        keep = [c for c in feature_columns() if c not in feats]
        auc = run_features(X, y, keep)
        results[name] = {"oof_auc": auc, "delta_vs_full": auc - full}
        print(f"-{name:20s}: OOF AUC = {auc:.5f}  (delta {auc - full:+.5f})", flush=True)

    with open("ablate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
