# Durable Decisions

Conventions and conclusions strong enough to guide future research. Do not
silently erase; if overturned, record the new evidence.

---

## Evaluation & protocol

**D1 — Canonical local comparison = 5-fold Stratified CV OOF ROC-AUC.**
Fold definitions are kept consistent across experiments that are compared.
(3-fold is the historical baseline; 10-fold did not help.)

**D2 — OOF predictions are required for trustworthy comparisons.**
Rank/weight sweeps over saved test-only submissions are "proxy-only" and are
never promoted (see EXP-019/020).

## Model architecture

**D3 — Regularized LightGBM is the champion family.**
L1/L2 regularization (reg_alpha 0.5–1.0, reg_lambda 5.0–10.0) beats
unregularized LightGBM (+0.00022). GBDT dominates linear/neural models by a
wide margin (0.964 vs 0.920–0.924); those are rejected for this task.

**D4 — colsample_bytree must stay >= ~0.70.**
Subsampling to 0.60 or below cuts access to the dominant
notifications/app-opens features and costs ~-0.002 AUC.

**D5 — learning_rate 0.12 is canonical.**
0.14 was worse in 10-fold; keep 0.12 unless new evidence appears.

**D6 — Do not use monotone step calibration.**
Isotonic regression destroys fine-grained probability ranks (-0.00072).
Rank-based ensembling already handles scale differences.

**D7 — Extra-trees are rejected.**
Random split thresholds severely damage ranking on smooth continuous features
(-0.015 AUC).

**D8 — HistGradientBoosting is rejected.**
~5× slower and lower AUC (0.9613) than LightGBM.

## Ensembling

**D9 — Dual-seed super-ensembling is a real, robust effect.**
Rank-averaging two independent 5-fold LightGBM ensembles (seeds 42 and 100)
gives a consistent +0.00025–0.00027 over either single seed, replicated
across two different feature pipelines (original 36-feature and the rebuilt
42-feature one). It cancels single-partition boundary noise.

**D10 — A model that scores slightly worse individually may still add value
only if measured prediction diversity supports it.** XGBoost (0.96338) did
not: it was too correlated with LightGBM (r≈0.90) and dragged the blend down.

## Feature knowledge

**D11 — `other_screen_time` is a promoted canonical feature.**
The generator satisfies `daily_screen_time_hours >= social_media_hours +
gaming_hours + work_study_hours` with ZERO violations; the residual
"other usage" component carries real signal (Pearson r≈0.305 with the label)
and adding it (+ its `_isna` flag) gave the largest single feature-engineering
gain of the program (+0.00042 OOF AUC, EXP-022).

**D12 — Missingness is empirically MCAR w.r.t. the target.**
Every single-column `_isna` indicator correlates with the label only
[-0.0003, +0.0027]. Subgroup AUC drops with missingness are an
information/difficulty effect, not a "missingness pattern is itself
predictive" signal. Consequently pairwise missingness-conjunction features
are a LOW-prior hypothesis (cheap to test, modest expectations).

**D13 — Interaction/ratio features beyond the canonical set add negligible
value.** Automated pairwise screening (EXP-018) found only +0.00003 max.
The canonical domain ratios + the other_screen_time residual already expose
the sufficient structure.

## Process

**D14 — ALWAYS commit the exact script that produced a canonical score.**
The original train_competition.py behind EXP-014 (0.96448) was only ever
local scratch and was permanently lost. This is the single biggest process
failure in the program's history. Never repeat it.

**D15 — Persistent repo (GitHub/HF) is a clean research notebook, not a
filesystem dump.** Scratch files live locally only. A fresh agent must be
able to recover the full research from the repo alone.

**D16 — Statistical significance in 691k-row data must not be
over-interpreted.** Train/test missingness-rate differences reach p<1e-200
yet are only a few percentage points in absolute terms and, combined with
D12, are unlikely to materially bias OOF-vs-leaderboard comparisons.

**D17 — The label is a sharp threshold function of screen time, and the
categoricals are effectively non-informative.** dst-decile -> target rate is
~0.28/0.46/0.81/0.985/0.999, and corr(dst, target) is ~0.61 within every
categorical group. Categorical main effects and screen-time×categorical
interactions are therefore low-value; do not re-invest in them. This also
explains the linear/neural underfit (0.92) vs trees (~0.964).

**D18 — "Low linear correlation" does not mean "no signal" in this data.**
age (r≈0.004), app_opens (r≈0.064), and sleep (r≈0.042) all carry real
nonlinear per-value effects. Always inspect target-rate by value/bins before
concluding a feature is useless.
