"""Analyze the OOF/test arrays saved by train_pipeline.py (EXP-023).

Usage (from repo root, after downloading the exp023-results artifact):
    python analysis/analyze_oof.py [path_to_oof_dir]

Reads oof_super.npy, oof_seed{42,100}*.npy, test arrays; prints:
  - per-model and per-seed OOF AUC
  - prediction correlation (model diversity, seed diversity)
  - super-ensemble AUC
  - subgroup AUC by missingness count (reproduces the difficulty gradient)
  - test-prediction correlations
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = ["lgbm_63", "lgbm_45", "lgbm_127"]
SEEDS = ["42", "100"]


def load(dirpath, name):
    p = os.path.join(dirpath, name + ".npy")
    return np.load(p) if os.path.exists(p) else None


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else "."
    y = pd.read_csv("train.csv")["addicted_label"].values
    print(f"=== OOF analysis from {dirpath} ===\n")

    # 1. per-model / per-seed AUCs
    for seed in SEEDS:
        for m in MODELS:
            arr = load(dirpath, f"oof_seed{seed}_{m}")
            if arr is not None:
                print(f"seed {seed} {m}: OOF AUC = {roc_auc_score(y, arr):.5f}")
        ens = load(dirpath, f"oof_seed{seed}")
        if ens is not None:
            print(f"seed {seed} ensemble (rank-avg): OOF AUC = {roc_auc_score(y, ens):.5f}")

    super_oof = load(dirpath, "oof_super")
    if super_oof is not None:
        print(f"super-ensemble: OOF AUC = {roc_auc_score(y, super_oof):.5f}")

    # 2. diversity: correlation between models within a seed, and between seeds
    print("\n=== prediction correlations (diversity) ===")
    for seed in SEEDS:
        arrs = {m: load(dirpath, f"oof_seed{seed}_{m}") for m in MODELS}
        if all(a is not None for a in arrs.values()):
            corr = np.corrcoef([arrs[m] for m in MODELS])
            for i, mi in enumerate(MODELS):
                for j, mj in enumerate(MODELS):
                    if j > i:
                        print(f"  seed {seed}: corr({mi}, {mj}) = {corr[i, j]:.4f}")
    if all(load(dirpath, f"oof_seed{s}") is not None for s in SEEDS):
        c = np.corrcoef([load(dirpath, f"oof_seed{s}") for s in SEEDS])
        print(f"  corr(seed42_ens, seed100_ens) = {c[0, 1]:.4f}")

    # 3. subgroup AUC by missingness count on the super-ensemble
    if super_oof is not None:
        tr = pd.read_csv("train.csv")
        raw = ["age", "daily_screen_time_hours", "social_media_hours",
               "gaming_hours", "work_study_hours", "sleep_hours",
               "notifications_per_day", "app_opens_per_day",
               "weekend_screen_time", "gender", "stress_level",
               "academic_work_impact"]
        tr["miss_n"] = tr[raw].isnull().sum(axis=1)
        print("\n=== subgroup OOF AUC by missingness count (super-ensemble) ===")
        for k in sorted(tr["miss_n"].unique()):
            m = tr["miss_n"] == k
            if m.sum() > 100:
                print(f"  missing={k}: n={m.sum():7d} AUC={roc_auc_score(y[m], super_oof[m]):.5f}")

    # 4. test-side correlation (do seeds agree on test?)
    print("\n=== test prediction correlations ===")
    t42 = load(dirpath, "test_seed42")
    t100 = load(dirpath, "test_seed100")
    if t42 is not None and t100 is not None:
        print(f"  corr(test_seed42, test_seed100) = {np.corrcoef([t42, t100])[0, 1]:.4f}")
    print("\ndone.")


if __name__ == "__main__":
    main()
