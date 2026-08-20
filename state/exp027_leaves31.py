"""EXP-027: Dual-seed super-ensemble with tune-scan winner configs.

Tests the two best hyperparameter findings from the lgbm_63 tune scan
through the full dual-seed (42+100) 5-fold protocol, using the complete
3-model ensemble (lgbm_63/45/127 with all configs updated):

  1. leaves_31: num_leaves=31 (was 63; tune scan +0.00027 single-seed)
  2. leaves_31 + subsample_095: combination test (leaves_31 +0.00027,
     subsample_095 +0.00023 — check for interaction/additivity)
  3. canonical_63 (baseline reference: lgbm_63 num_leaves=63)

Each arm runs all 3 model configs with the same num_leaves override
(e.g. lgbm_63/45/127 all use 31 leaves in the leaves_31 arm), dual-seed
42+100, 5-fold, rank-average within seed then across seeds.

Usage:
    python state/exp027_leaves31.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from source.train_pipeline import (
    COMMON_PARAMS,
    N_SPLITS,
    TARGET,
    ID,
    EARLY_STOPPING_ROUNDS,
    build_features,
    feature_columns,
)

# --- Arms to test ---
# Each arm defines num_leaves overrides for all 3 configs, plus optional
# subsample/other overrides.  The canonical baseline is included for
# in-run matched comparison.

ARMS = {
    "leaves_31": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    "leaves_31_sub095": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.95,
    },
    "canonical_63": {
        "num_leaves": 63,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
}

SEEDS = [42, 100]


def _rank01(preds):
    return rankdata(preds) / len(preds)


def run_arm(X_train, y, X_test, arm_cfg, seed, n_splits=N_SPLITS):
    """Run all 3 model configs under one arm + fold seed.

    Returns dict with oof, test (rank-averaged across 3 models),
    per-model arrays, and fold-level AUCs.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oofs = {}
    test_preds = {}
    fold_aucs = {}

    for model_idx in range(3):
        # Vary base config to mimic lgbm_63/45/127 diversity
        if model_idx == 0:
            base_num_leaves = arm_cfg["num_leaves"]
        elif model_idx == 1:
            # lgbm_45: slightly smaller
            base_num_leaves = max(15, arm_cfg["num_leaves"] - 16)
        else:
            # lgbm_127: slightly larger
            base_num_leaves = arm_cfg["num_leaves"] * 2 + 1

        model_name = f"lgbm_{base_num_leaves}"
        params = dict(COMMON_PARAMS)
        params["num_leaves"] = base_num_leaves
        params["reg_alpha"] = arm_cfg.get("reg_alpha", 0.5)
        params["reg_lambda"] = arm_cfg.get("reg_lambda", 5.0)
        if "subsample" in arm_cfg:
            params["subsample"] = arm_cfg["subsample"]
        params["random_state"] = seed

        oof = np.zeros(len(y))
        test_pred = np.zeros(len(X_test))
        model_fold_aucs = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train.iloc[tr_idx],
                y[tr_idx],
                eval_X=X_train.iloc[va_idx],
                eval_y=y[va_idx],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / n_splits
            model_fold_aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))

        oofs[model_name] = oof
        test_preds[model_name] = test_pred
        fold_aucs[model_name] = model_fold_aucs

    # Rank-average across the 3 models within this seed
    seed_oof = np.mean([_rank01(oofs[n]) for n in oofs], axis=0)
    seed_test = np.mean([_rank01(test_preds[n]) for n in test_preds], axis=0)
    return {
        "oof": seed_oof,
        "test": seed_test,
        "fold_aucs": fold_aucs,
    }


def main():
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
    for arm_name, arm_cfg in ARMS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"ARM: {arm_name}", flush=True)
        print(f"{'='*60}", flush=True)

        seed_results = {}
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---", flush=True)
            t1 = time.time()
            res = run_arm(X_train, y, X_test, arm_cfg, seed)
            seed_auc = roc_auc_score(y, res["oof"])
            seed_results[str(seed)] = {
                "oof_auc": seed_auc,
                "fold_aucs": {
                    n: [round(a, 5) for a in res["fold_aucs"][n]]
                    for n in res["fold_aucs"]
                },
                "ensemble_fold_aucs": [round(a, 5) for a in res["fold_aucs"][list(res["fold_aucs"].keys())[0]]],
                "oof": res["oof"],
                "test": res["test"],
            }
            print(
                f"  seed {seed} ensemble OOF AUC: {seed_auc:.5f}  "
                f"({time.time()-t1:.1f}s)",
                flush=True,
            )

        # Dual-seed super-ensemble
        if len(SEEDS) >= 2:
            super_oof = np.mean(
                [_rank01(seed_results[str(s)]["oof"]) for s in SEEDS], axis=0
            )
            super_test = np.mean(
                [_rank01(seed_results[str(s)]["test"]) for s in SEEDS], axis=0
            )
            super_auc = roc_auc_score(y, super_oof)
        else:
            super_oof = seed_results[str(SEEDS[0])]["oof"]
            super_test = seed_results[str(SEEDS[0])]["test"]
            super_auc = seed_results[str(SEEDS[0])]["oof_auc"]

        # Per-seed ensemble AUCs
        seed42_auc = seed_results["42"]["oof_auc"]
        seed100_auc = seed_results["100"]["oof_auc"]

        results[arm_name] = {
            "seeds": SEEDS,
            "seed42_oof_auc": seed42_auc,
            "seed100_oof_auc": seed100_auc,
            "super_ensemble_oof_auc": super_auc,
            "fold_aucs": seed_results["42"]["fold_aucs"],
            "arm_config": {k: v for k, v in arm_cfg.items()},
        }
        print(f"\n  {arm_name} SUMMARY:", flush=True)
        print(f"    seed42 ensemble:  {seed42_auc:.5f}", flush=True)
        print(f"    seed100 ensemble: {seed100_auc:.5f}", flush=True)
        print(f"    super-ensemble:  {super_auc:.5f}", flush=True)

        # Save per-arm artifacts
        arm_dir = f"exp027_{arm_name}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_super.npy", super_oof)
        np.save(f"{arm_dir}/test_super.npy", super_test)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(super_test, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission.csv", index=False)

    # --- Summary comparison ---
    print(f"\n{'='*60}", flush=True)
    print("EXP-027 FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    baseline = results["canonical_63"]["super_ensemble_oof_auc"]
    for arm_name in ARMS:
        auc = results[arm_name]["super_ensemble_oof_auc"]
        delta = auc - baseline
        print(
            f"  {arm_name:25s}  super={auc:.5f}  delta={delta:+.5f}",
            flush=True,
        )

    # Save combined results (without large arrays for JSON)
    json_results = {}
    for arm_name in results:
        json_results[arm_name] = {
            k: v for k, v in results[arm_name].items()
            if k not in ("fold_aucs",)
        }
    json_results["wallclock_seconds"] = time.time() - t0
    with open("exp027_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: exp027_results.json", flush=True)
    print(f"Total wallclock: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
