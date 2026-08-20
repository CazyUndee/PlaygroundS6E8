"""Is the age effect confounded with categoricals or missingness?"""
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
y = train["addicted_label"]

# age vs missingness counts
print("=== mean missing_count by age ===")
train["miss_n"] = train[["age","daily_screen_time_hours","social_media_hours","gaming_hours",
    "work_study_hours","sleep_hours","notifications_per_day","app_opens_per_day",
    "weekend_screen_time","gender","stress_level","academic_work_impact"]].isnull().sum(axis=1)
print(train.groupby("age")["miss_n"].mean().round(3).to_string())

# age vs each categorical rate (does the age effect differ by gender etc.?)
print("\n=== target rate by (age, gender) — pivot ===")
piv = train.pivot_table(index="age", columns="gender", values="addicted_label", aggfunc="mean", observed=True)
print(piv.round(3).to_string())

print("\n=== target rate by (age, stress_level) — pivot ===")
piv2 = train.pivot_table(index="age", columns="stress_level", values="addicted_label", aggfunc="mean", observed=True)
print(piv2.round(3).to_string())

print("\n=== target rate by (age, academic_work_impact) — pivot ===")
piv3 = train.pivot_table(index="age", columns="academic_work_impact", values="addicted_label", aggfunc="mean", observed=True)
print(piv3.round(3).to_string())

# Is the age effect present within each gender? (test independence)
print("\n=== age 24/26/28 vs others, within gender ===")
train["age_high"] = train["age"].isin([22,24,26,28,32]).astype(int)
for g, sub in train.groupby("gender", observed=True):
    hi = sub[sub["age_high"]==1]["addicted_label"].mean()
    lo = sub[sub["age_high"]==0]["addicted_label"].mean()
    print(f"  {g}: high={hi:.4f} low={lo:.4f} diff={hi-lo:+.4f}")

# Does age interact with daily screen time?  corr(screen, target) within high vs low age
print("\n=== corr(daily_screen_time, target) within age_high groups ===")
for v, sub in train.groupby("age_high"):
    r = sub["daily_screen_time_hours"].corr(sub["addicted_label"])
    print(f"  age_high={v}: corr={r:.4f} n={len(sub)}")
