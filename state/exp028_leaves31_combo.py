"""EXP-028: Second-order combination sweep on top of leaves_31 base.

Assuming EXP-027 confirms leaves_31 as the winner, this tests:
1. leaves_31 + lr_010 (combine the two best single-knob improvements)
2. leaves_31 + mcs_50 (more conservative leaf min samples)
3. leaves_31 + lr_010 + mcs_50 (triple combo)
4. leaves_31 alone (reference from EXP-027)
5. Canonical_63 (baseline reference)

Plus a refined num_leaves grid: 20/25/31/35/45 (single-seed quick scan
to find the exact optimum around 31).

All arms use the full 44-feature pipeline, dual-seed (42+100), 5-fold.

Usage:
    python state/exp028_leaves31_combo.py
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

# --- Dual-seed arms (full protocol) ---
DUAL_ARMS = {
    "leaves_31": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    "leaves_31_lr010": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "learning_rate": 0.10,
    },
    "leaves_31_mcs50": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "min_child_samples": 50,
    },
    "leaves_31_lr010_mcs50": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "learning_rate": 0.10,
        "min_child_samples": 50,
    },
    "canonical_63": {
        "num_leaves": 63,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
}

# --- Quick single-seed leaves grid ---
LEAVES_GRID = {
    "leaves_20": {"num_leaves": 20, "reg_alpha": 0.5, "reg_lambda": 5.0},
    "leaves_25": {"num_leaves": 25, "reg_alpha": 0.5, "reg_lambda": 5.0},
    "leaves_31": {"num_leaves": 31, "reg_alpha": 0.5, "reg_lambda": 5.0},
    "leaves_35": {"num_leaves": 35, "reg_alpha": 0.5, "reg_lambda": 5.0},
    "leaves_45": {"num_leaves": 45, "reg_alpha": 0.5, "reg_lambda": 5.0},
}

SEEDS = [42, 100]


def _rank01(preds):
    return rankdata(preds) / len(preds)


def run_arm(X_train, y, X_test, arm_cfg, seed, n_splits=N_SPLITS):
    """Run all 3 model configs under one arm + fold seed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oofs = {}
    test_preds = {}
    fold_aucs = {}

    for model_idx in range(3):
        if model_idx == 0:
            base_num_leaves = arm_cfg["num_leaves"]
        elif model_idx == 1:
            base_num_leaves = max(15, arm_cfg["num_leaves"] - 16)
        else:
            base_num_leaves = arm_cfg["num_leaves"] * 2 + 1

        model_name = f"lgbm_{base_num_leaves}"
        params = dict(COMMON_PARAMS)
        params["num_leaves"] = base_num_leaves
        params["reg_alpha"] = arm_cfg.get("reg_alpha", 0.5)
        params["reg_lambda"] = arm_cfg.get("reg_lambda", 5.0)
        for key in ("learning_rate", "min_child_samples", "subsample",
                     "colsample_bytree"):
            if key in arm_cfg:
                params[key] = arm_cfg[key]
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

    seed_oof = np.mean([_rank01(oofs[n]) for n in oofs], axis=0)
    seed_test = np.mean([_rank01(test_preds[n]) for n in test_preds], axis=0)
    return {"oof": seed_oof, "test": seed_test, "fold_aucs": fold_aucs}


def run_single_seed_grid(X, y, grid, seed=42):
    """Quick single-seed scan for the leaves grid."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    results = {}
    for name, cfg in grid.items():
        params = dict(COMMON_PARAMS)
        params.update(cfg)
        params["random_state"] = seed
        oof = np.zeros(len(y))
        fold_aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_X=X.iloc[va_idx], eval_y=y[va_idx],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))
        auc = roc_auc_score(y, oof)
        results[name] = {"oof_auc": auc, "fold_aucs": [round(a, 5) for a in fold_aucs]}
        print(f"  {name:20s} OOF AUC = {auc:.5f}", flush=True)
    return results


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

    # --- Part 1: Quick single-seed leaves grid ---
    print(f"\n{'='*60}", flush=True)
    print("PART 1: Single-seed leaves grid (seed-42, lgbm_63 only)", flush=True)
    print(f"{'='*60}", flush=True)
    grid_results = run_single_seed_grid(X_train, y, LEAVES_GRID)

    # --- Part 2: Full dual-seed arms ---
    all_results = {}
    for arm_name, arm_cfg in DUAL_ARMS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"DUAL-SEED ARM: {arm_name}", flush=True)
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
                "oof": res["oof"],
                "test": res["test"],
            }
            print(
                f"  seed {seed} ensemble OOF AUC: {seed_auc:.5f}  "
                f"({time.time()-t1:.1f}s)",
                flush=True,
            )

        # Dual-seed super-ensemble
        super_oof = np.mean(
            [_rank01(seed_results[str(s)]["oof"]) for s in SEEDS], axis=0
        )
        super_test = np.mean(
            [_rank01(seed_results[str(s)]["test"]) for s in SEEDS], axis=0
        )
        super_auc = roc_auc_score(y, super_oof)

        results_entry = {
            "seeds": SEEDS,
            "seed42_oof_auc": seed_results["42"]["oof_auc"],
            "seed100_oof_auc": seed_results["100"]["oof_auc"],
            "super_ensemble_oof_auc": super_auc,
            "fold_aucs": seed_results["42"]["fold_aucs"],
            "arm_config": {k: v for k, v in arm_cfg.items()},
        }
        all_results[arm_name] = results_entry
        print(f"\n  {arm_name} SUMMARY:", flush=True)
        print(f"    seed42:  {seed_results['42']['oof_auc']:.5f}", flush=True)
        print(f"    seed100: {seed_results['100']['oof_auc']:.5f}", flush=True)
        print(f"    super:   {super_auc:.5f}", flush=True)

        # Save artifacts
        arm_dir = f"exp028_{arm_name}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_super.npy", super_oof)
        np.save(f"{arm_dir}/test_super.npy", super_test)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(super_test, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission.csv", index=False)

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("EXP-028 FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    baseline = all_results["canonical_63"]["super_ensemble_oof_auc"]
    champion = 0.96466  # EXP-023
    for arm_name in DUAL_ARMS:
        auc = all_results[arm_name]["super_ensemble_oof_auc"]
        delta_vs_canon = auc - baseline
        delta_vs_champ = auc - champion
        tag = " *** NEW CHAMPION ***" if auc > champion else ""
        print(
            f"  {arm_name:30s}  super={auc:.5f}  "
            f"vs_canon={delta_vs_canon:+.5f}  vs_champ={delta_vs_champ:+.5f}{tag}",
            flush=True,
        )

    # Save JSON
    json_results = {"grid": grid_results, "dual_seed": {}}
    for arm_name in all_results:
        json_results["dual_seed"][arm_name] = {
            k: v for k, v in all_results[arm_name].items()
            if k not in ("oof", "test", "fold_aucs")
        }
    json_results["wallclock_seconds"] = time.time() - t0
    with open("exp028_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: exp028_results.json", flush=True)
    print(f"Total wallclock: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
