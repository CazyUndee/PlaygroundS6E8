"""
Reproducible canonical pipeline for Kaggle Playground Series S6E8
(Predicting Smartphone Addiction).

This is the rebuilt, fully-documented replacement for the original (lost)
train_competition.py. It implements:

  * 44 features (see build_features)
      - 9 raw numeric
      - 3 categorical (native LightGBM category dtype)
      - 12 single-column missingness indicators (_isna)
      - 3 missingness-count aggregates
      - 15 domain ratio/interaction features
      - other_screen_time + other_screen_time_isna
        (the promoted hard-constraint residual discovered in EXP-022)
  * 3 regularized LightGBM configs (num_leaves 63/45/127)
  * 5-Fold Stratified CV, seeds {42, 100}
  * rank-average within each seed, then rank-average across seeds
    (dual-seed super-ensemble, matching EXP-014's architecture)

Usage:
    python agent/train_pipeline.py [--seeds 42 100] [--single-seed]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

RAW_NUMERIC = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

RAW_CATEGORICAL = [
    "gender",
    "stress_level",
    "academic_work_impact",
]

TARGET = "addicted_label"
ID = "id"

# --- Model configs (from HISTORY.md / EXP-014 architecture description) ---
MODEL_CONFIGS = {
    "lgbm_63": {
        "num_leaves": 63,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    "lgbm_45": {
        "num_leaves": 45,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    "lgbm_127": {
        "num_leaves": 127,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    },
}

COMMON_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.12,
    "n_estimators": 2000,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
}

N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 100


def _safe_ratio(num, den, eps=1e-5):
    """Elementwise num/den with NaN propagation, guarding against div-by-0."""
    den = den.replace(0, np.nan)
    return num / den


def build_features(df):
    """
    Build the canonical 44-feature set.

    Returns a copy of `df` with feature columns added (target and id columns
    are preserved untouched). NaN is left as NaN for LightGBM's native
    missing-value handling.
    """
    out = df.copy()

    # --- 1. Raw numeric columns are used directly (9). ---

    # --- 2. Categorical columns: native LightGBM category dtype (3). ---
    for col in RAW_CATEGORICAL:
        out[col] = out[col].astype("category")

    # --- 3. Single-column missingness indicators (12). ---
    for col in RAW_NUMERIC + RAW_CATEGORICAL:
        out[f"{col}_isna"] = out[col].isna().astype(np.int8)

    # --- 4. Missingness-count aggregates (3). ---
    out["missing_count_total"] = (
        out[[f"{c}_isna" for c in RAW_NUMERIC + RAW_CATEGORICAL]].sum(axis=1)
    )
    out["missing_count_numeric"] = (
        out[[f"{c}_isna" for c in RAW_NUMERIC]].sum(axis=1)
    )
    out["missing_count_categorical"] = (
        out[[f"{c}_isna" for c in RAW_CATEGORICAL]].sum(axis=1)
    )

    # --- 5. Domain ratio / interaction features (15). ---
    d = out  # shorthand

    # Shares of daily screen time (4)
    d["social_media_share"] = d["social_media_hours"] / d["daily_screen_time_hours"]
    d["gaming_share"] = d["gaming_hours"] / d["daily_screen_time_hours"]
    d["work_study_share"] = d["work_study_hours"] / d["daily_screen_time_hours"]
    d["entertainment_share"] = (
        (d["social_media_hours"] + d["gaming_hours"]) / d["daily_screen_time_hours"]
    )

    # Screen-time structure (4)
    d["weekend_weekday_ratio"] = d["weekend_screen_time"] / d["daily_screen_time_hours"]
    d["weekend_extra_screen"] = d["weekend_screen_time"] - d["daily_screen_time_hours"]
    d["screen_sleep_ratio"] = d["daily_screen_time_hours"] / d["sleep_hours"]
    d["sleep_awake_share"] = d["sleep_hours"] / (d["sleep_hours"] + d["daily_screen_time_hours"])

    # Engagement rates (4)
    d["notifications_per_screen_hour"] = (
        d["notifications_per_day"] / d["daily_screen_time_hours"]
    )
    d["app_opens_per_screen_hour"] = (
        d["app_opens_per_day"] / d["daily_screen_time_hours"]
    )
    d["notifications_per_app_open"] = d["notifications_per_day"] / d["app_opens_per_day"]
    d["engagement_intensity"] = (
        (d["social_media_hours"] + d["gaming_hours"] + d["work_study_hours"])
        / d["daily_screen_time_hours"]
    )

    # Work/sleep and misc (3)
    d["work_sleep_ratio"] = d["work_study_hours"] / d["sleep_hours"]
    d["app_opens_per_notification"] = d["app_opens_per_day"] / d["notifications_per_day"]
    d["total_active_hours"] = (
        d["social_media_hours"] + d["gaming_hours"] + d["work_study_hours"]
    )

    # --- 6. Promoted hard-constraint residual (EXP-022). ---
    # daily_screen_time_hours >= social_media + gaming + work_study holds with
    # zero violations; the leftover "other" usage carries real signal.
    d["other_screen_time"] = (
        d["daily_screen_time_hours"]
        - d["social_media_hours"]
        - d["gaming_hours"]
        - d["work_study_hours"]
    )
    d["other_screen_time_isna"] = d["other_screen_time"].isna().astype(np.int8)

    return out


def feature_columns():
    """Return the ordered list of feature column names (44)."""
    feats = list(RAW_NUMERIC) + list(RAW_CATEGORICAL)
    for col in RAW_NUMERIC + RAW_CATEGORICAL:
        feats.append(f"{col}_isna")
    feats += [
        "missing_count_total",
        "missing_count_numeric",
        "missing_count_categorical",
    ]
    feats += [
        "social_media_share",
        "gaming_share",
        "work_study_share",
        "entertainment_share",
        "weekend_weekday_ratio",
        "weekend_extra_screen",
        "screen_sleep_ratio",
        "sleep_awake_share",
        "notifications_per_screen_hour",
        "app_opens_per_screen_hour",
        "notifications_per_app_open",
        "engagement_intensity",
        "work_sleep_ratio",
        "app_opens_per_notification",
        "total_active_hours",
        "other_screen_time",
        "other_screen_time_isna",
    ]
    return feats


def _rank01(preds):
    return rankdata(preds) / len(preds)


def run_seed(X_train, y, X_test, seed, n_splits=N_SPLITS):
    """Train all 3 models under one fold-partition seed; return seed ensemble.

    Returns dict with oof (rank-averaged), test (rank-averaged), per-model
    oof/test arrays, and fold-level AUCs.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oofs = {name: np.zeros(len(y)) for name in MODEL_CONFIGS}
    test_preds = {name: np.zeros(len(X_test)) for name in MODEL_CONFIGS}
    fold_aucs = {name: [] for name in MODEL_CONFIGS}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        for name, cfg in MODEL_CONFIGS.items():
            params = dict(COMMON_PARAMS)
            params.update(cfg)
            params["random_state"] = seed
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_X=X_va,
                eval_y=y_va,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oofs[name][va_idx] = model.predict_proba(X_va)[:, 1]
            test_preds[name] += model.predict_proba(X_test)[:, 1] / n_splits
            fold_aucs[name].append(roc_auc_score(y_va, oofs[name][va_idx]))
        print(
            f"    [seed={seed}] fold {fold + 1}/{n_splits} done: "
            + " ".join(f"{n}={fold_aucs[n][-1]:.5f}" for n in MODEL_CONFIGS),
            flush=True,
        )

    # Rank-average the 3 models within this seed.
    seed_oof = np.mean([_rank01(oofs[n]) for n in MODEL_CONFIGS], axis=0)
    seed_test = np.mean([_rank01(test_preds[n]) for n in MODEL_CONFIGS], axis=0)
    return {
        "seed": seed,
        "oof": seed_oof,
        "test": seed_test,
        "per_model_oof": oofs,
        "per_model_test": test_preds,
        "fold_aucs": fold_aucs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100])
    ap.add_argument("--single-seed", action="store_true", help="only seed 42")
    ap.add_argument("--out", default="submission_matched_super.csv")
    args = ap.parse_args()
    seeds = [42] if args.single_seed else args.seeds

    t0 = time.time()
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"Loaded train {train.shape}, test {test.shape}", flush=True)

    train = build_features(train)
    test = build_features(test)
    feats = feature_columns()
    print(f"Feature count: {len(feats)}", flush=True)

    X_train = train[feats]
    y = train[TARGET].values
    X_test = test[feats]

    results = {}
    for seed in seeds:
        print(f"\n=== Seed {seed} ({N_SPLITS}-Fold) ===", flush=True)
        res = run_seed(X_train, y, X_test, seed)
        results[str(seed)] = {
            "oof_auc": roc_auc_score(y, res["oof"]),
            "fold_aucs": res["fold_aucs"],
            "per_model_oof_aucs": {
                n: roc_auc_score(y, res["per_model_oof"][n]) for n in MODEL_CONFIGS
            },
        }
        print(
            f"  seed {seed} ensemble OOF AUC: {results[str(seed)]['oof_auc']:.5f}",
            flush=True,
        )

    # Dual-seed super-ensemble: rank-average the per-seed ensembles.
    if len(seeds) >= 2:
        super_oof = np.mean([_rank01(results[str(s)]["oof"]) for s in seeds], axis=0)
        super_test = np.mean([_rank01(results[str(s)]["test"]) for s in seeds], axis=0)
        super_auc = roc_auc_score(y, super_oof)
        print(f"\n=== Dual-seed super-ensemble OOF AUC: {super_auc:.5f} ===", flush=True)
    else:
        super_oof = results[str(seeds[0])]["oof"]
        super_test = results[str(seeds[0])]["test"]
        super_auc = results[str(seeds[0])]["oof_auc"]

    # Submission (AUC metric -> rank-only, but clip to [0,1] for cleanliness).
    sub = pd.DataFrame(
        {"id": test[ID].values, TARGET: np.clip(super_test, 0.0, 1.0)}
    )
    sub.to_csv(args.out, index=False)
    print(f"Submission saved: {args.out} ({len(sub)} rows)", flush=True)

    summary = {
        "seeds": seeds,
        "n_splits": N_SPLITS,
        "n_features": len(feats),
        "features": feats,
        "seed_results": results,
        "super_ensemble_oof_auc": super_auc,
        "wallclock_seconds": time.time() - t0,
    }
    with open("pipeline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: pipeline_results.json", flush=True)


if __name__ == "__main__":
    main()
