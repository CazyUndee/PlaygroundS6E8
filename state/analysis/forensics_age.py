"""Is the age target-rate pattern real (not missingness-confounded) and
exploitable?"""
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
y = train["addicted_label"]

# 1. Complete rows only (no missingness confound)
sc = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
      "work_study_hours", "weekend_screen_time"]
complete = train[sc].notnull().all(axis=1)
print("complete rows:", complete.sum(), "/", len(train))

print("\n=== target rate by age: all rows vs complete rows ===")
all_tab = train.groupby("age")["addicted_label"].agg(["mean", "count"])
comp_tab = train[complete].groupby("age")["addicted_label"].agg(["mean", "count"])
tab = pd.DataFrame({"all_mean": all_tab["mean"], "complete_mean": comp_tab["mean"],
                    "n_complete": comp_tab["count"]})
print(tab.to_string())

# 2. Within screen-time deciles, does age still shift the rate?
train = train.copy()
train["dst_q"] = pd.qcut(train["daily_screen_time_hours"], 4, labels=False)
print("\n=== target rate by (dst quartile, age parity) ===")
train["age_even"] = (train["age"] % 2 == 0).astype(int)
g = train.groupby(["dst_q", "age_even"])["addicted_label"].agg(["mean", "count"])
print(g.to_string())

# 3. Does age interact with screen time in the label? Logistic-style check:
#    within narrow screen-time bins, is age informative?
train["dst_bin"] = pd.cut(train["daily_screen_time_hours"], 10)
print("\n=== age effect within a mid screen-time bin (5.5-7.0h) ===")
mid = train[(train["daily_screen_time_hours"] >= 5.5) & (train["daily_screen_time_hours"] < 7.0)]
print(mid.groupby("age")["addicted_label"].agg(["mean", "count"]).to_string())

# 4. AUC of a few cheap age encodings (single-feature discriminative power)
from sklearn.metrics import roc_auc_score
print("\n=== single-feature AUC of age encodings ===")
m = train["age"].notnull()
print(f"  age (raw, as rank): {roc_auc_score(y[m], train.loc[m,'age']):.5f}")
print(f"  age==24 flag:       {roc_auc_score(y[m], (train.loc[m,'age']==24).astype(int)):.5f}")
print(f"  age in {{24,26,28}}: {roc_auc_score(y[m], train.loc[m,'age'].isin([24,26,28]).astype(int)):.5f}")
print(f"  age parity (even):  {roc_auc_score(y[m], (train.loc[m,'age']%2==0).astype(int)):.5f}")
print(f"  dst (raw, as rank): {roc_auc_score(y[train['daily_screen_time_hours'].notnull()], train['daily_screen_time_hours'].dropna()):.5f}")
