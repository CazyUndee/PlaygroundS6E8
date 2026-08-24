"""EXP-029: Triple fold-partition seed super-ensemble.

Tests whether adding a 3rd independent fold partition (seed=2026) to the
super-ensemble gives additional improvement over the dual-seed (42+100)
protocol. Two arms:
1. leaves_31 (best from tune scan) with 3 seeds
2. canonical_63 with 3 seeds (baseline)

Each arm: 3 independent 5-fold Stratified CV runs, rank-averaged within
each seed, then rank-averaged across all 3 seeds.

This is the definitive test of whether more seed diversity keeps helping
or has plateaued (D9 predicts diminishing returns given 0.994-0.997
model correlation).

Usage:
    python state/exp029_triple_seed.py
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

ARMS = {
    "leaves_31": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    "canonical_63": {
        "num_leaves": 63,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
}

SEEDS_2 = [42, 100]
SEEDS_3 = [42, 100, 2026]


def _rank01(preds):
    return rankdata(preds) / len(preds)


def run_single_seed(X_train, y, X_test, arm_cfg, seed, n_splits=N_SPLITS):
    """Run 3-model ensemble under one arm + one fold seed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oofs = {}
    test_preds = {}

    for model_idx in range(3):
        if model_idx == 0:
            base_leaves = arm_cfg["num_leaves"]
        elif model_idx == 1:
            base_leaves = max(15, arm_cfg["num_leaves"] - 16)
        else:
            base_leaves = arm_cfg["num_leaves"] * 2 + 1

        model_name = f"lgbm_{base_leaves}"
        params = dict(COMMON_PARAMS)
        params["num_leaves"] = base_leaves
        params["reg_alpha"] = arm_cfg.get("reg_alpha", 0.5)
        params["reg_lambda"] = arm_cfg.get("reg_lambda", 5.0)
        params["random_state"] = seed

        oof = np.zeros(len(y))
        test_pred = np.zeros(len(X_test))

        for tr_idx, va_idx in skf.split(X_train, y):
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

        oofs[model_name] = oof
        test_preds[model_name] = test_pred

    seed_oof = np.mean([_rank01(oofs[n]) for n in oofs], axis=0)
    seed_test = np.mean([_rank01(test_preds[n]) for n in test_preds], axis=0)
    return seed_oof, seed_test


def super_ensemble(seed_oofs, seed_tests):
    """Rank-average across seeds."""
    super_oof = np.mean([_rank01(s) for s in seed_oofs], axis=0)
    super_test = np.mean([_rank01(s) for s in seed_tests], axis=0)
    return super_oof, super_test


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

    all_results = {}

    for arm_name, arm_cfg in ARMS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"ARM: {arm_name}", flush=True)
        print(f"{'='*60}", flush=True)

        # --- 3-seed run ---
        seed_oofs_3 = []
        seed_tests_3 = []
        seed_aucs_3 = {}

        for seed in SEEDS_3:
            t1 = time.time()
            oof, test_pred = run_single_seed(X_train, y, X_test, arm_cfg, seed)
            auc = roc_auc_score(y, oof)
            seed_oofs_3.append(oof)
            seed_tests_3.append(test_pred)
            seed_aucs_3[str(seed)] = auc
            print(f"  seed {seed} OOF AUC: {auc:.5f}  ({time.time()-t1:.1f}s)", flush=True)

        # Dual-seed (seeds 42+100)
        dual_oof, dual_test = super_ensemble(seed_oofs_3[:2], seed_tests_3[:2])
        dual_auc = roc_auc_score(y, dual_oof)

        # Triple-seed (seeds 42+100+2026)
        tri_oof, tri_test = super_ensemble(seed_oofs_3, seed_tests_3)
        tri_auc = roc_auc_score(y, tri_oof)

        delta_tri_vs_dual = tri_auc - dual_auc

        all_results[arm_name] = {
            "seed_aucs": seed_aucs_3,
            "dual_seed_oof_auc": dual_auc,
            "triple_seed_oof_auc": tri_auc,
            "delta_tri_vs_dual": delta_tri_vs_dual,
            "arm_config": arm_cfg,
        }
        print(f"\n  {arm_name} RESULTS:", flush=True)
        print(f"    Dual (42+100):   {dual_auc:.5f}", flush=True)
        print(f"    Triple (42+100+2026): {tri_auc:.5f}", flush=True)
        print(f"    Delta (tri-dual):     {delta_tri_vs_dual:+.5f}", flush=True)

        # Save triple-seed artifacts
        arm_dir = f"exp029_{arm_name}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_triple.npy", tri_oof)
        np.save(f"{arm_dir}/test_triple.npy", tri_test)
        np.save(f"{arm_dir}/oof_dual.npy", dual_oof)
        np.save(f"{arm_dir}/test_dual.npy", dual_test)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(tri_test, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission_triple.csv", index=False)

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("EXP-029 FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    champion = 0.96466
    for arm_name in ARMS:
        r = all_results[arm_name]
        dual = r["dual_seed_oof_auc"]
        tri = r["triple_seed_oof_auc"]
        print(
            f"  {arm_name:20s}  dual={dual:.5f}  tri={tri:.5f}  "
            f"delta={r['delta_tri_vs_dual']:+.5f}  "
            f"vs_champ={tri - champion:+.5f}",
            flush=True,
        )

    json_results = {}
    for arm_name in all_results:
        json_results[arm_name] = {k: v for k, v in all_results[arm_name].items()
                                   if k not in ("oof", "test")}
    json_results["wallclock_seconds"] = time.time() - t0
    with open("exp029_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: exp029_results.json", flush=True)
    print(f"Total wallclock: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
