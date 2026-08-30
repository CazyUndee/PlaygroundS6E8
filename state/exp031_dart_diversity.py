"""EXP-031: DART boosting diversity probe + model correlation diagnostics.

Questions:
1. Does DART (dropout-based boosting) give a competitive-but-diverse model
   vs the champion gbdt config? DART trains with dropout, producing
   genuinely different trees — a real diversity probe (uncertainty #4,
   TODO P1 "DART boosting mode").
2. What is the actual OOF prediction correlation across the 3 model
   configs (lgbm_31/15/63) within a seed, and across seeds (42 vs 100)?
   If correlations are ~0.994-0.997 as suspected, the ensemble's value is
   near-duplicate averaging and cross-family blending (XGB/CatBoost) is the
   only path to real diversity.

Arms (dual-seed 42+100, 5-fold, full 44-feature pipeline):
1. dart_champ      — champion config (leaves_31+sub095) but boosting_type=dart
2. gbdt_champ      — champion config reference (expect ~0.96498)

Plus a printed correlation matrix of OOF predictions.

Usage:
    python state/exp031_dart_diversity.py
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

CHAMP_CFG = {
    "num_leaves": 31,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "subsample": 0.95,
}

ARMS = {
    "gbdt_champ": dict(CHAMP_CFG),                       # boosting_type=gbdt (default)
    "dart_champ": {**CHAMP_CFG, "boosting_type": "dart", "drop_rate": 0.1},
}

SEEDS = [42, 100]
CHAMPION_REF = 0.96498  # EXP-027 leaves_31_sub095


def _rank01(preds):
    return rankdata(preds) / len(preds)


def run_arm(X_train, y, X_test, arm_cfg, seed, n_splits=N_SPLITS):
    """Run all 3 model configs under one arm + fold seed; return per-model
    OOF preds too (for correlation diagnostics)."""
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
                    "colsample_bytree", "boosting_type", "drop_rate"):
            if key in arm_cfg:
                params[key] = arm_cfg[key]
        params["random_state"] = seed

        oof = np.zeros(len(y))
        test_pred = np.zeros(len(X_test))
        model_fold_aucs = []

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
            model_fold_aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))

        oofs[model_name] = oof
        test_preds[model_name] = test_pred
        fold_aucs[model_name] = model_fold_aucs

    seed_oof = np.mean([_rank01(oofs[n]) for n in oofs], axis=0)
    seed_test = np.mean([_rank01(test_preds[n]) for n in test_preds], axis=0)
    return {
        "oof": seed_oof,
        "test": seed_test,
        "fold_aucs": fold_aucs,
        "model_oofs": oofs,
    }


def print_corr_matrix(oofs_by_name, label):
    """Print Pearson correlation matrix of OOF preds (on rank-transformed)."""
    names = list(oofs_by_name.keys())
    ranks = {n: _rank01(oofs_by_name[n]) for n in names}
    print(f"\n--- {label} (Pearson corr of rank-transformed OOF) ---", flush=True)
    header = "".join(f"{n:>16s}" for n in names)
    print(f"{'':>16s}{header}", flush=True)
    for a in names:
        row = f"{a:>16s}"
        for b in names:
            row += f"{np.corrcoef(ranks[a], ranks[b])[0, 1]:>16.4f}"
        print(row, flush=True)


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
    diagnostics = {}

    for arm_name, arm_cfg in ARMS.items():
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
                "model_oofs": res["model_oofs"],
            }
            print(
                f"  seed {seed} ensemble OOF AUC: {seed_auc:.5f}  "
                f"({time.time()-t1:.1f}s)",
                flush=True,
            )

        # Per-seed correlation diagnostics (within-seed across model configs)
        for seed in SEEDS:
            print_corr_matrix(
                seed_results[str(seed)]["model_oofs"],
                f"{arm_name} seed {seed} across lgbm configs",
            )

        super_oof = np.mean(
            [_rank01(seed_results[str(s)]["oof"]) for s in SEEDS], axis=0
        )
        super_test = np.mean(
            [_rank01(seed_results[str(s)]["test"]) for s in SEEDS], axis=0
        )
        super_auc = roc_auc_score(y, super_oof)

        # Cross-seed correlation for this arm's ensemble preds
        cross_seed_corr = np.corrcoef(
            _rank01(seed_results["42"]["oof"]),
            _rank01(seed_results["100"]["oof"]),
        )[0, 1]

        all_results[arm_name] = {
            "seeds": SEEDS,
            "seed42_oof_auc": seed_results["42"]["oof_auc"],
            "seed100_oof_auc": seed_results["100"]["oof_auc"],
            "super_ensemble_oof_auc": super_auc,
            "cross_seed_corr": float(cross_seed_corr),
            "arm_config": {k: v for k, v in arm_cfg.items()},
        }
        diagnostics[arm_name] = {
            "cross_seed_corr": float(cross_seed_corr),
            "within_seed_model_corrs": {},
        }
        for seed in SEEDS:
            oofs = seed_results[str(seed)]["model_oofs"]
            names = list(oofs.keys())
            corrs = {}
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    corrs[f"{a}_vs_{b}"] = float(
                        np.corrcoef(_rank01(oofs[a]), _rank01(oofs[b]))[0, 1]
                    )
            diagnostics[arm_name]["within_seed_model_corrs"][str(seed)] = corrs
        print(f"\n  {arm_name} SUMMARY:", flush=True)
        print(f"    seed42:  {seed_results['42']['oof_auc']:.5f}", flush=True)
        print(f"    seed100: {seed_results['100']['oof_auc']:.5f}", flush=True)
        print(f"    super:   {super_auc:.5f}", flush=True)
        print(f"    cross_seed_corr: {cross_seed_corr:.4f}", flush=True)

        arm_dir = f"exp031_{arm_name}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_super.npy", super_oof)
        np.save(f"{arm_dir}/test_super.npy", super_test)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(super_test, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission.csv", index=False)

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("EXP-031 FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    for arm_name in ARMS:
        auc = all_results[arm_name]["super_ensemble_oof_auc"]
        delta = auc - CHAMPION_REF
        tag = " *** BEATS EXP-027 CHAMPION ***" if auc > CHAMPION_REF else ""
        print(
            f"  {arm_name:14s}  super={auc:.5f}  vs_exp027={delta:+.5f}{tag}",
            flush=True,
        )

    # Cross-arm blend: rank-average the two arms' super-ensemble OOF preds.
    print(f"\n--- Cross-arm blend (dart + gbdt) ---", flush=True)
    arm_oofs = {}
    for arm_name in ARMS:
        arm_dir = f"exp031_{arm_name}"
        arm_oofs[arm_name] = np.load(f"{arm_dir}/oof_super.npy")
    blend_oof = np.mean([_rank01(o) for o in arm_oofs.values()], axis=0)
    blend_auc = roc_auc_score(y, blend_oof)
    print(f"  blend(dart+gbdt) super OOF AUC: {blend_auc:.5f}  "
          f"vs_exp027={blend_auc - CHAMPION_REF:+.5f}", flush=True)
    json_results["blend_dart_gbdt_oof_auc"] = blend_auc

    json_results = {"arms": {}, "diagnostics": diagnostics}
    for arm_name in all_results:
        json_results["arms"][arm_name] = {
            k: v for k, v in all_results[arm_name].items()
            if k not in ("oof", "test", "fold_aucs")
        }
    json_results["references"] = {"exp027_champion": CHAMPION_REF}
    json_results["wallclock_seconds"] = time.time() - t0
    with open("exp031_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: exp031_results.json", flush=True)
    print(f"Total wallclock: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
