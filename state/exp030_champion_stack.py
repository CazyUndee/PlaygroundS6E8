"""EXP-030: Champion full-stack combos (sub095 x lr010 x mcs50).

EXP-027 proved leaves_31 + subsample_095 = 0.96498 (champion).
EXP-028 proved lr010 and mcs50 each add a small dual-seed gain on top of
plain leaves_31 (+0.00005 / +0.00003; combined +0.00006), but did NOT test
them on top of subsample_095. This experiment stacks all proven knobs:

Arms (dual-seed 42+100, 5-fold, full 44-feature pipeline):
1. champ_sub095              (EXP-027 champion reference, expect ~0.96498)
2. champ_sub095_lr010        (add lr 0.10)
3. champ_sub095_mcs50        (add min_child_samples 50)
4. champ_sub095_lr010_mcs50  (full stack, leaves_31)
5. leaves25_sub095_lr010_mcs50 (full stack, leaves_25 — grid hinted 25>=31)
6. canonical_63              (baseline reference, expect ~0.96465)

If gains are additive, the full stack could reach ~0.96503+. A new champion
above 0.96498 gets promoted to the canonical pipeline.

Usage:
    python state/exp030_champion_stack.py
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

DUAL_ARMS = {
    "champ_sub095": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.95,
    },
    "champ_sub095_lr010": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.95,
        "learning_rate": 0.10,
    },
    "champ_sub095_mcs50": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.95,
        "min_child_samples": 50,
    },
    "champ_sub095_lr010_mcs50": {
        "num_leaves": 31,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.95,
        "learning_rate": 0.10,
        "min_child_samples": 50,
    },
    "leaves25_sub095_lr010_mcs50": {
        "num_leaves": 25,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.95,
        "learning_rate": 0.10,
        "min_child_samples": 50,
    },
    "canonical_63": {
        "num_leaves": 63,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
}

SEEDS = [42, 100]
# EXP-023 / EXP-027 / EXP-028 dual-seed super-ensemble numbers (matched protocol)
CHAMPION_REF = 0.96498  # EXP-027 leaves_31_sub095
BASELINE_REF = 0.96466  # EXP-023 canonical_63


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

        arm_dir = f"exp030_{arm_name}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_super.npy", super_oof)
        np.save(f"{arm_dir}/test_super.npy", super_test)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(super_test, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission.csv", index=False)

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("EXP-030 FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    baseline = all_results["canonical_63"]["super_ensemble_oof_auc"]
    for arm_name in DUAL_ARMS:
        auc = all_results[arm_name]["super_ensemble_oof_auc"]
        delta_vs_champ = auc - CHAMPION_REF
        delta_vs_baseline = auc - BASELINE_REF
        tag = " *** NEW CHAMPION ***" if auc > CHAMPION_REF else ""
        print(
            f"  {arm_name:34s}  super={auc:.5f}  "
            f"vs_exp027={delta_vs_champ:+.5f}  vs_exp023={delta_vs_baseline:+.5f}{tag}",
            flush=True,
        )

    json_results = {}
    for arm_name in all_results:
        json_results[arm_name] = {
            k: v for k, v in all_results[arm_name].items()
            if k not in ("oof", "test", "fold_aucs")
        }
    json_results["references"] = {
        "exp027_champion": CHAMPION_REF,
        "exp023_baseline": BASELINE_REF,
    }
    json_results["wallclock_seconds"] = time.time() - t0
    with open("exp030_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: exp030_results.json", flush=True)
    print(f"Total wallclock: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
