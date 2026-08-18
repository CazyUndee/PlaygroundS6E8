"""The label is a sharp threshold function of screen time. What exactly is the
threshold on, and what distinguishes addicted vs not near the boundary?"""
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
y = train["addicted_label"]

# narrow the screen-time → rate curve
print("=== target rate by fine daily_screen_time bins ===")
bins = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
train["dst_bin"] = pd.cut(train["daily_screen_time_hours"], bins=bins)
print(train.groupby("dst_bin", observed=True)["addicted_label"].agg(["mean", "count"]).round(4).to_string())

# is the threshold better described by social+gaming+work sum or by daily total?
train["sum3"] = train["social_media_hours"] + train["gaming_hours"] + train["work_study_hours"]
print("\n=== target rate by (sum3 = social+gaming+work) bins ===")
train["sum3_bin"] = pd.cut(train["sum3"], bins=bins)
print(train.groupby("sum3_bin", observed=True)["addicted_label"].agg(["mean", "count"]).round(4).to_string())

# other_screen_time (residual) near the boundary: does it separate addicted vs not?
print("\n=== within mid screen-time (5-9h), other_screen_time vs label ===")
mid = train[(train["daily_screen_time_hours"] >= 5) & (train["daily_screen_time_hours"] < 9)].copy()
mid["other"] = mid["daily_screen_time_hours"] - mid["sum3"]
mid["other_q"] = pd.qcut(mid["other"], 5, labels=False)
print(mid.groupby("other_q", observed=True)["addicted_label"].agg(["mean", "count"]).round(4).to_string())

# AUC of screen-time features individually and combined
from sklearn.metrics import roc_auc_score
print("\n=== single-feature AUC (rank) ===")
for c in ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
          "gaming_hours", "work_study_hours", "sum3"]:
    m = train[c].notnull()
    print(f"  {c:24s} {roc_auc_score(y[m], train.loc[m, c]):.5f}")

# is weekend_screen_time more discriminative than daily?
m = train[["daily_screen_time_hours", "weekend_screen_time"]].notnull().all(axis=1)
print(f"\n  daily AUC (both-nonnull subset): {roc_auc_score(y[m], train.loc[m,'daily_screen_time_hours']):.5f}")
print(f"  weekend AUC (both-nonnull subset): {roc_auc_score(y[m], train.loc[m,'weekend_screen_time']):.5f}")
# average of the two as a "combined" screen measure
comb = train.loc[m, "daily_screen_time_hours"] + train.loc[m, "weekend_screen_time"]
print(f"  daily+weekend sum AUC: {roc_auc_score(y[m], comb):.5f}")
