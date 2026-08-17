"""Do the integer count features (notifications, app_opens) and sleep_hours
carry hidden nonlinear signal like age does?"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
y = train["addicted_label"]

def single_auc(col, m=None):
    m = train[col].notnull() if m is None else (m & train[col].notnull())
    return roc_auc_score(y[m], train.loc[m, col])

print("=== single-feature AUC (raw, rank) ===")
for c in ["notifications_per_day", "app_opens_per_day", "sleep_hours", "age"]:
    print(f"  {c:22s} {single_auc(c):.5f}")

# target rate by notifications quantile (deciles)
print("\n=== target rate by notifications_per_day decile ===")
train["n_q"] = pd.qcut(train["notifications_per_day"], 10, labels=False)
print(train.groupby("n_q", observed=True)["addicted_label"].agg(["mean", "count"]).to_string())

print("\n=== target rate by app_opens_per_day decile ===")
train["ao_q"] = pd.qcut(train["app_opens_per_day"], 10, labels=False)
print(train.groupby("ao_q", observed=True)["addicted_label"].agg(["mean", "count"]).to_string())

# does the count signal persist within fixed screen time?
print("\n=== notifications effect within fixed screen-time bin (5.5-7.0h) ===")
mid = train[(train["daily_screen_time_hours"] >= 5.5) & (train["daily_screen_time_hours"] < 7.0)]
mid = mid.copy()
mid["n_q2"] = pd.qcut(mid["notifications_per_day"], 5, labels=False)
print(mid.groupby("n_q2", observed=True)["addicted_label"].agg(["mean", "count"]).to_string())

print("\n=== app_opens effect within fixed screen-time bin ===")
mid["ao_q2"] = pd.qcut(mid["app_opens_per_day"], 5, labels=False)
print(mid.groupby("ao_q2", observed=True)["addicted_label"].agg(["mean", "count"]).to_string())

# ratio features already in pipeline: check their single AUC vs raw
print("\n=== single-feature AUC of derived ratio features (rank) ===")
train2 = train.copy()
train2["n_per_screen"] = train2["notifications_per_day"] / train2["daily_screen_time_hours"]
train2["ao_per_screen"] = train2["app_opens_per_day"] / train2["daily_screen_time_hours"]
train2["n_per_ao"] = train2["notifications_per_day"] / train2["app_opens_per_day"]
for c in ["n_per_screen", "ao_per_screen", "n_per_ao"]:
    m = train2[c].notnull()
    print(f"  {c:22s} {roc_auc_score(y[m], train2.loc[m,c]):.5f}")

# sleep hours: target rate by sleep value (it's 2-decimal, bin it)
print("\n=== target rate by sleep_hours bins ===")
train["sl_bin"] = pd.cut(train["sleep_hours"], bins=np.arange(4.5, 9.25, 0.5))
print(train.groupby("sl_bin", observed=True)["addicted_label"].agg(["mean", "count"]).to_string())
