"""Systematic search for hard generator constraints (zero-violation
relationships) among numeric features, following the other_screen_time
discovery. Cheap pandas-only forensics; runs while EXP-023 trains.

Reads train.csv/test.csv (already downloaded).
"""
import itertools

import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

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


def jn(df, cols):
    """rows where all cols are non-null."""
    return df[cols].notnull().all(axis=1)


def check_constraint(name, lhs, rhs, df, eps=1e-6):
    """check lhs >= rhs with zero (or near-zero) violations."""
    m = lhs.notnull() & rhs.notnull()
    n = int(m.sum())
    if n == 0:
        return None
    viol = int((lhs[m] < rhs[m] - eps).sum())
    min_resid = float((lhs[m] - rhs[m]).min())
    return {"constraint": name, "n_joint_nonnull": n, "violations": viol, "min_residual": min_resid}


results = []

# 1. Pairwise a >= b (all ordered pairs of numeric features)
for a, b in itertools.permutations(NUM, 2):
    if a == b:
        continue
    r = check_constraint(f"{a} >= {b}", train[a], train[b], train)
    if r and r["violations"] == 0 and r["n_joint_nonnull"] > 100000:
        results.append(r)

# 2. Sum-of-two <= third (a + b <= c)
for a, b in itertools.combinations(NUM, 2):
    for c in NUM:
        if c in (a, b):
            continue
        lhs = train[c]
        rhs = train[a] + train[b]
        r = check_constraint(f"{c} >= {a} + {b}", lhs, rhs, train)
        if r and r["violations"] == 0 and r["n_joint_nonnull"] > 100000:
            results.append(r)

# 3. Integer/quantization check per feature
print("=== Precision / quantization ===")
for col in NUM:
    x = train[col].dropna()
    n = len(x)
    frac = float((x != np.floor(x)).mean())  # fraction not integer
    # count distinct values of the 2-decimal rounding
    n2 = x.round(2).nunique()
    print(f"{col:24s} n={n:8d} non-integer_frac={frac:.4f} uniq@2dp={n2} min={x.min():.3f} max={x.max():.3f}")

print("\n=== Zero-violation constraints (train) ===")
for r in sorted(results, key=lambda d: d["n_joint_nonnull"], reverse=True):
    print(f"  {r['constraint']:60s} n={r['n_joint_nonnull']:8d} viol={r['violations']} min_resid={r['min_residual']:.4f}")

# Cross-check the discovered zero-violation constraints on test
print("\n=== Cross-check on test (violation counts) ===")
for r in results:
    parts = r["constraint"].split(" >= ")
    if len(parts) != 2:
        continue
    lhs, rhs = parts
    if " + " in rhs:
        rcols = rhs.split(" + ")
        lhs_s = test[lhs]
        rhs_s = test[rcols[0]] + test[rcols[1]]
    else:
        lhs_s = test[lhs]
        rhs_s = test[rhs]
    m = lhs_s.notnull() & rhs_s.notnull()
    viol = int((lhs_s[m] < rhs_s[m] - 1e-6).sum())
    print(f"  {r['constraint']:60s} test_viol={viol} / {int(m.sum())}")

# 4. Residual correlation with target for promising new residuals
print("\n=== Residual-vs-target correlation for zero-violation sum constraints ===")
y = train["addicted_label"]
for a, b in itertools.combinations(NUM, 2):
    for c in NUM:
        if c in (a, b):
            continue
        m = train[[c, a, b]].notnull().all(axis=1)
        resid = train[c] - train[a] - train[b]
        viol = int((resid[m] < -1e-6).sum())
        if viol == 0 and m.sum() > 100000:
            r = float(resid[m].corr(y[m]))
            if abs(r) > 0.1:
                print(f"  {c} - {a} - {b}: n={int(m.sum())} corr={r:.4f}")
