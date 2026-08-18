"""Fast feature screen: train lgbm_63 only (seed-42, 5-fold) with the
canonical 44 features plus candidate extras, and compare fold-level AUCs to
the EXP-023 lgbm_63 baseline (fold 1-4 already logged; fold 5 TBD).

Protocol matches train_pipeline.run_seed exactly (same folds, same config),
so fold-by-fold comparison against EXP-023's lgbm_63 is clean. Runs on
GitHub Actions runners (isolated, so parallel runs are safe — see
source/DECISIONS.md D19).

Usage:
    python state/exp_screen_features.py --feature total_screen
    python state/exp_screen_features.py --feature age
"""
import argparse
import os
import sys
import time

# Make the repo root importable so this script can import the `source`
# package (sibling of `state/`) regardless of the invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from source.train_pipeline import (
    COMMON_PARAMS,
    MODEL_CONFIGS,
    N_SPLITS,
    TARGET,
    ID,
    build_features,
    feature_columns,
)


def add_total_screen(df):
    out = df.copy()
    out["total_screen"] = (
        out["daily_screen_time_hours"] + out["weekend_screen_time"]
    )
    out["total_screen_isna"] = out["total_screen"].isna().astype(np.int8)
    return out


def add_sm_weekend(df):
    out = df.copy()
    out["sm_weekend"] = out["social_media_hours"] + out["weekend_screen_time"]
    out["sm_weekend_isna"] = out["sm_weekend"].isna().astype(np.int8)
    return out


def add_all3(df):
    out = df.copy()
    out["all3_screen"] = (
        out["daily_screen_time_hours"]
        + out["social_media_hours"]
        + out["weekend_screen_time"]
    )
    out["all3_screen_isna"] = out["all3_screen"].isna().astype(np.int8)
    return out


def add_age(df):
    out = df.copy()
    out["age_cat"] = out["age"].astype("category")
    out["age_even"] = (out["age"] % 2 == 0).astype(np.int8)
    out["age_high_band"] = out["age"].isin([22, 24, 26, 28, 32]).astype(np.int8)
    return out


FEATURES = {
    "total_screen": ("total_screen", "total_screen_isna"),
    "sm_weekend": ("sm_weekend", "sm_weekend_isna"),
    "all3": ("all3_screen", "all3_screen_isna"),
    "age": ("age_cat", "age_even", "age_high_band"),
    "both": ("total_screen", "total_screen_isna", "age_cat", "age_even", "age_high_band"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=list(FEATURES), required=True)
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    base = build_features(train)
    base_test = build_features(test)
    feats = feature_columns() + list(FEATURES[args.feature])

    if args.feature in ("total_screen", "both"):
        base = add_total_screen(base)
        base_test = add_total_screen(base_test)
    if args.feature in ("age", "both"):
        base = add_age(base)
        base_test = add_age(base_test)
    if args.feature == "sm_weekend":
        base = add_sm_weekend(base)
        base_test = add_sm_weekend(base_test)
    if args.feature == "all3":
        base = add_all3(base)
        base_test = add_all3(base_test)

    X = base[feats]
    y = base[TARGET].values
    X_test = base_test[feats]
    print(f"feature={args.feature}  n_features={len(feats)}", flush=True)

    cfg = dict(MODEL_CONFIGS["lgbm_63"])
    params = dict(COMMON_PARAMS)
    params.update(cfg)
    params["random_state"] = 42

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_X=X.iloc[va_idx], eval_y=y[va_idx],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        fold_aucs.append(auc)
        print(f"  fold {fold + 1}/{N_SPLITS} lgbm_63 AUC: {auc:.5f}", flush=True)

    oof_auc = roc_auc_score(y, oof)
    print(f"lgbm_63 OOF AUC (seed-42, +{args.feature}): {oof_auc:.5f}", flush=True)
    print(f"fold AUCs: {[round(a, 5) for a in fold_aucs]}", flush=True)
    print(f"wallclock: {time.time() - t0:.1f}s", flush=True)

    np.save(f"oof_screen_{args.feature}.npy", oof)
    np.save(f"test_screen_{args.feature}.npy", test_pred)
    with open(f"screen_{args.feature}.json", "w") as f:
        import json
        json.dump({"feature": args.feature, "oof_auc": oof_auc,
                   "fold_aucs": fold_aucs}, f, indent=2)

    # Auto-compare against the canonical EXP-023 lgbm_63 baseline if its
    # results are present (state/results/exp023/pipeline_results.json,
    # committed by the agent after the canonical run). Same folds/config ->
    # clean compare.
    import json
    baseline_path = os.path.join("state", "results", "exp023", "pipeline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            base = json.load(f)
        base_folds = base["seed_results"]["42"]["fold_aucs"]["lgbm_63"]
        base_oof = base["seed_results"]["42"]["per_model_oof_aucs"]["lgbm_63"]
        print("\n=== vs EXP-023 lgbm_63 baseline (seed-42) ===", flush=True)
        print(f"  baseline OOF AUC: {base_oof:.5f}  |  screen OOF AUC: {oof_auc:.5f}"
              f"  |  delta: {oof_auc - base_oof:+.5f}", flush=True)
        for i, (b, s) in enumerate(zip(base_folds, fold_aucs), 1):
            print(f"  fold {i}: baseline {b:.5f} -> screen {s:.5f} (delta {s - b:+.5f})", flush=True)
    else:
        print("\n(baseline state/results/exp023/pipeline_results.json not found; "
              "fold AUCs printed above for manual comparison)", flush=True)


if __name__ == "__main__":
    main()
