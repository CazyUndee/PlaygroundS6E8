"""Signal-distribution forensics: where does the target signal live, and is
the integer age feature / weekend structure being used well?"""
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
y = train["addicted_label"]

NUM = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

print("=== Pearson |r| with target (raw numeric) ===")
for c in NUM:
    r = train[c].corr(y)
    print(f"  {c:24s} r={r:+.4f}")

print("\n=== Categorical target rate ===")
for c in ["gender", "stress_level", "academic_work_impact"]:
    print(f"  {c}:")
    print(train.groupby(c, observed=True)["addicted_label"].agg(["mean", "count"]).to_string())

print("\n=== age detail (integer [18,35]) ===")
age_tab = train.groupby("age")["addicted_label"].agg(["mean", "count"])
print(age_tab.to_string())

print("\n=== age vs screen time ===")
for c in ["daily_screen_time_hours", "gaming_hours", "weekend_screen_time"]:
    r = train["age"].corr(train[c])
    print(f"  corr(age, {c}) = {r:+.4f}")

print("\n=== other_screen_time by age ===")
tr = train.copy()
tr["other"] = (
    tr["daily_screen_time_hours"] - tr["social_media_hours"]
    - tr["gaming_hours"] - tr["work_study_hours"]
)
print(tr.groupby("age")["other"].agg(["mean", "count"]).to_string())

print("\n=== weekend_screen_time vs daily (structure) ===")
tr["weekend_extra"] = tr["weekend_screen_time"] - tr["daily_screen_time_hours"]
print("weekend_extra describe:")
print(tr["weekend_extra"].describe().to_string())
print(f"corr(weekend_extra, target) = {tr['weekend_extra'].corr(y):+.4f}")
print(f"corr(weekend_screen_time, target) = {tr['weekend_screen_time'].corr(y):+.4f}")
print(f"corr(daily_screen_time, target) = {tr['daily_screen_time_hours'].corr(y):+.4f}")
