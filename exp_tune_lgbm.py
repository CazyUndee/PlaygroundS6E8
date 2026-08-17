"""Small hyperparameter scan for the canonical LightGBM config (lgbm_63),
single-seed (42) 5-fold, compared against the EXP-023 lgbm_63 baseline
(0.96380). Runs on GitHub Actions.

Usage:
    python exp_tune_lgbm.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from agent.train_pipeline import COMMON_PARAMS, N_SPLITS, TARGET, build_features, feature_columns

# Candidate configs: vary one or two knobs around the canonical lgbm_63
# (num_leaves=63, lr=0.12, reg_alpha=0.5, reg_lambda=5, colsample=0.8,
#  subsample=0.8, min_child_samples=20).
CANDIDATES = {
    "canonical_63": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20},
    "lr_010": {"num_leaves": 63, "learning_rate": 0.10, "min_child_samples": 20},
    "lr_015": {"num_leaves": 63, "learning_rate": 0.15, "min_child_samples": 20},
    "leaves_95": {"num_leaves": 95, "learning_rate": 0.12, "min_child_samples": 20},
    "leaves_31": {"num_leaves": 31, "learning_rate": 0.12, "min_child_samples": 20},
    "mcs_50": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 50},
    "mcs_10": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 10},
    "colsample_095": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20, "colsample_bytree": 0.95},
    "colsample_070": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20, "colsample_bytree": 0.7},
    "subsample_095": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20, "subsample": 0.95},
    "lambda_100": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20, "reg_lambda": 10.0},
    "lambda_20": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20, "reg_lambda": 2.0},
    "alpha_10": {"num_leaves": 63, "learning_rate": 0.12, "min_child_samples": 20, "reg_alpha": 1.0},
}


def run_config(X, y, cfg, seed=42):
    params = dict(COMMON_PARAMS)
    params.update(cfg)
    params["random_state"] = seed
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_X=X.iloc[va_idx], eval_y=y[va_idx],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))
    return roc_auc_score(y, oof), fold_aucs


def main():
    t0 = time.time()
    train = pd.read_csv("train.csv")
    train = build_features(train)
    feats = feature_columns()
    X = train[feats]
    y = train[TARGET].values
    print(f"features: {len(feats)}", flush=True)

    results = {}
    for name, cfg in CANDIDATES.items():
        t1 = time.time()
        auc, folds = run_config(X, y, cfg)
        results[name] = {"oof_auc": auc, "fold_aucs": [round(a, 5) for a in folds]}
        print(f"{name:16s} OOF AUC = {auc:.5f}  folds={results[name]['fold_aucs']}  "
              f"({time.time()-t1:.1f}s)", flush=True)

    with open("tune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
