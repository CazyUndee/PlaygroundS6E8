"""EXP-032: Cross-family diversity probe (XGBoost + CatBoost).

EXP-031 measures whether LightGBM's 3-config ensemble is near-duplicate
averaging (predicted corr 0.994-0.997). If so, the only remaining diversity
source is genuinely different model families. This probes XGBoost and
CatBoost with champion-equivalent hyperparameters:

Arms (dual-seed 42+100, 5-fold, full 44-feature pipeline):
1. lgbm_champ — LightGBM champion config reference (expect ~0.96498)
2. xgb_champ  — XGBoost (max_depth=5 ~ num_leaves 31; reg alpha/lambda,
                subsample 0.95, colsample 0.8, native categoricals)
3. cat_champ  — CatBoost (depth=5, l2_leaf_reg 5, subsample 0.95,
                native categoricals)

Outputs: per-arm super-ensemble OOF/test preds, pairwise OOF correlation
matrix across families, and 2-way/3-way rank-average blend scores.

Usage:
    python state/exp032_cross_family.py
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
import xgboost as xgb
from catboost import CatBoostClassifier

from source.train_pipeline import (
    COMMON_PARAMS,
    N_SPLITS,
    TARGET,
    ID,
    EARLY_STOPPING_ROUNDS,
    build_features,
    feature_columns,
)

SEEDS = [42, 100]
CHAMPION_REF = 0.96498  # EXP-027 leaves_31_sub095

# Model configs per family (champion-equivalent hyperparameters)
LGBM_CFG = {
    "num_leaves": 31,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "subsample": 0.95,
}
XGB_CFG = {
    "max_depth": 5,           # ~ 2^5 = 32 leaves, close to num_leaves 31
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "subsample": 0.95,
    "colsample_bytree": 0.8,
    "learning_rate": 0.12,
    "min_child_weight": 20,
    "enable_categorical": True,
    "tree_method": "hist",
}
CAT_CFG = {
    "depth": 5,
    "l2_leaf_reg": 5.0,
    "subsample": 0.95,
    "colsample_bylevel": 0.8,
    "learning_rate": 0.12,
    "min_data_in_leaf": 20,
    "verbose": 0,
}


def _rank01(preds):
    return rankdata(preds) / len(preds)


def _lgbm_models(seed):
    """Champion 3-config LightGBM family (leaves 31/15/63)."""
    out = []
    for base in (31, 15, 63):
        params = dict(COMMON_PARAMS)
        params["num_leaves"] = base
        params["reg_alpha"] = LGBM_CFG["reg_alpha"]
        params["reg_lambda"] = LGBM_CFG["reg_lambda"]
        params["subsample"] = LGBM_CFG["subsample"]
        params["random_state"] = seed
        out.append(("lgbm", params))
    return out


def _xgb_models(seed):
    """XGB family: depth 5/3/6 to mirror leaves 31/15/63."""
    out = []
    for depth in (5, 3, 6):
        params = dict(XGB_CFG)
        params["max_depth"] = depth
        params["random_state"] = seed
        out.append(("xgb", params))
    return out


def _cat_models(seed):
    """CatBoost family: depth 5/4/6."""
    out = []
    for depth in (5, 4, 6):
        params = dict(CAT_CFG)
        params["depth"] = depth
        params["random_seed"] = seed
        out.append(("cat", params))
    return out


def run_family(X_train, y, X_test, family, seed, cat_features, n_splits=N_SPLITS):
    """Run one family's 3 configs under one fold seed; return model OOF preds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if family == "lgbm":
        models = _lgbm_models(seed)
    elif family == "xgb":
        models = _xgb_models(seed)
    else:
        models = _cat_models(seed)

    oofs = {}
    test_preds = {}
    fold_aucs = {}

    for idx, (kind, params) in enumerate(models):
        model_name = f"{kind}_{idx}"
        oof = np.zeros(len(y))
        test_pred = np.zeros(len(X_test))
        model_fold_aucs = []

        for tr_idx, va_idx in skf.split(X_train, y):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            if kind == "lgbm":
                m = lgb.LGBMClassifier(**params)
                m.fit(
                    Xtr, ytr, eval_X=Xva, eval_y=yva,
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                oof[va_idx] = m.predict_proba(Xva)[:, 1]
                test_pred += m.predict_proba(X_test)[:, 1] / n_splits
            elif kind == "xgb":
                m = xgb.XGBClassifier(**params, n_estimators=2000,
                                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                oof[va_idx] = m.predict_proba(Xva)[:, 1]
                test_pred += m.predict_proba(X_test)[:, 1] / n_splits
            else:
                m = CatBoostClassifier(**params, iterations=2000,
                                       early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cat_features,
                      verbose=False)
                oof[va_idx] = m.predict_proba(Xva)[:, 1]
                test_pred += m.predict_proba(X_test)[:, 1] / n_splits
            model_fold_aucs.append(roc_auc_score(yva, oof[va_idx]))

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


def main():
    t0 = time.time()
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"Loaded train {train.shape}, test {test.shape}", flush=True)

    train = build_features(train)
    test = build_features(test)
    feats = feature_columns()
    print(f"Feature count: {len(feats)}", flush=True)

    # Identify categorical columns for native handling
    cat_cols = [c for c in feats if str(train[c].dtype) == "category"]
    cat_features = [feats.index(c) for c in cat_cols]
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}", flush=True)
    print(f"CatBoost cat feature indices: {cat_features}", flush=True)

    X_train = train[feats]
    y = train[TARGET].values
    X_test = test[feats]

    families = ["lgbm", "xgb", "cat"]
    all_results = {}

    for family in families:
        print(f"\n{'='*60}", flush=True)
        print(f"DUAL-SEED FAMILY: {family}", flush=True)
        print(f"{'='*60}", flush=True)

        seed_results = {}
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---", flush=True)
            t1 = time.time()
            res = run_family(X_train, y, X_test, family, seed, cat_features)
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

        all_results[family] = {
            "seeds": SEEDS,
            "seed42_oof_auc": seed_results["42"]["oof_auc"],
            "seed100_oof_auc": seed_results["100"]["oof_auc"],
            "super_ensemble_oof_auc": super_auc,
            "fold_aucs": seed_results["42"]["fold_aucs"],
            "oof": super_oof,
            "test": super_test,
        }
        print(f"\n  {family} SUMMARY:", flush=True)
        print(f"    seed42:  {seed_results['42']['oof_auc']:.5f}", flush=True)
        print(f"    seed100: {seed_results['100']['oof_auc']:.5f}", flush=True)
        print(f"    super:   {super_auc:.5f}", flush=True)

        arm_dir = f"exp032_{family}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_super.npy", super_oof)
        np.save(f"{arm_dir}/test_super.npy", super_test)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(super_test, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission.csv", index=False)

    # --- Correlation matrix across families ---
    print(f"\n{'='*60}", flush=True)
    print("CROSS-FAMILY OOF CORRELATION (rank-transformed)", flush=True)
    print(f"{'='*60}", flush=True)
    names = families
    ranks = {n: _rank01(all_results[n]["oof"]) for n in names}
    header = "".join(f"{n:>12s}" for n in names)
    print(f"{'':>12s}{header}", flush=True)
    corr_matrix = {}
    for a in names:
        row = f"{a:>12s}"
        corr_matrix[a] = {}
        for b in names:
            c = np.corrcoef(ranks[a], ranks[b])[0, 1]
            corr_matrix[a][b] = float(c)
            row += f"{c:>12.4f}"
        print(row, flush=True)

    # --- Blends ---
    print(f"\n{'='*60}", flush=True)
    print("CROSS-FAMILY BLENDS (rank-average)", flush=True)
    print(f"{'='*60}", flush=True)
    blend_results = {}
    blend_sets = {
        "lgbm_xgb": ["lgbm", "xgb"],
        "lgbm_cat": ["lgbm", "cat"],
        "lgbm_xgb_cat": ["lgbm", "xgb", "cat"],
    }
    for bname, fams in blend_sets.items():
        boof = np.mean([_rank01(all_results[f]["oof"]) for f in fams], axis=0)
        btest = np.mean([_rank01(all_results[f]["test"]) for f in fams], axis=0)
        b_auc = roc_auc_score(y, boof)
        blend_results[bname] = {
            "families": fams,
            "oof_auc": b_auc,
            "delta_vs_champ": b_auc - CHAMPION_REF,
        }
        tag = " *** NEW CHAMPION ***" if b_auc > CHAMPION_REF else ""
        print(
            f"  {bname:16s}  blend={b_auc:.5f}  vs_exp027={b_auc - CHAMPION_REF:+.5f}{tag}",
            flush=True,
        )
        arm_dir = f"exp032_blend_{bname}"
        os.makedirs(arm_dir, exist_ok=True)
        np.save(f"{arm_dir}/oof_super.npy", boof)
        np.save(f"{arm_dir}/test_super.npy", btest)
        sub = pd.DataFrame({ID: test[ID].values, TARGET: np.clip(btest, 0, 1)})
        sub.to_csv(f"{arm_dir}/submission.csv", index=False)

    # --- Final summary ---
    print(f"\n{'='*60}", flush=True)
    print("EXP-032 FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    for family in families:
        auc = all_results[family]["super_ensemble_oof_auc"]
        print(f"  {family:8s}  super={auc:.5f}  vs_exp027={auc - CHAMPION_REF:+.5f}",
              flush=True)

    json_results = {
        "families": {},
        "corr_matrix": corr_matrix,
        "blends": blend_results,
        "references": {"exp027_champion": CHAMPION_REF},
        "wallclock_seconds": time.time() - t0,
    }
    for family in families:
        json_results["families"][family] = {
            k: v for k, v in all_results[family].items()
            if k not in ("oof", "test", "fold_aucs")
        }
    with open("exp032_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: exp032_results.json", flush=True)
    print(f"Total wallclock: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
