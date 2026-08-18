"""Do the categoricals moderate the dominant screen-time signal (interaction)?
If slopes are uniform, interaction features are unlikely to help (confirming
EXP-018)."""
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")

print("=== corr(daily_screen_time, target) within each categorical group ===")
for cat in ["gender", "stress_level", "academic_work_impact"]:
    for g, sub in train.groupby(cat, observed=True):
        r = sub["daily_screen_time_hours"].corr(sub["addicted_label"])
        print(f"  {cat}={g:8s} corr={r:.4f} n={len(sub)}")

print("\n=== mean daily_screen_time within each categorical group ===")
for cat in ["gender", "stress_level", "academic_work_impact"]:
    print(f"  {cat}:")
    print(train.groupby(cat, observed=True)["daily_screen_time_hours"].agg(["mean", "count"]).round(3).to_string())

print("\n=== target rate vs dst decile, split by gender (interaction check) ===")
train = train.copy()
train["dst_q"] = pd.qcut(train["daily_screen_time_hours"], 5, labels=False)
piv = train.pivot_table(index="dst_q", columns="gender", values="addicted_label", aggfunc="mean", observed=True)
print(piv.round(3).to_string())

print("\n=== target rate vs dst decile, split by stress_level ===")
piv2 = train.pivot_table(index="dst_q", columns="stress_level", values="addicted_label", aggfunc="mean", observed=True)
print(piv2.round(3).to_string())
