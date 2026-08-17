# Research History
Long-term searchable memory. Append new sessions below; do not delete old
entries (correct/annotate instead). Search this file with grep/regex when
you need to recover a specific past detail — do not read it front-to-back
by default.
---
## Session 2026-08-14 (env reset) — Recovery, reproducibility fix, and EXP-021/022/023
### Context at session start
This was a fresh sandbox with no local state. The Hugging Face dataset
`cazyundee/PlaygroundS6E8` was the only source of truth. Retrieved via
`huggingface_hub.snapshot_download` (public dataset, no token required).
`RESEARCH_STATE.md` and `checkpoint.json` described a prior 22-day research
program that reached a "canonical" dual-seed LightGBM super-ensemble
(`EXP-014`) at **0.96448 OOF ROC-AUC**, with `EXP-015`–`EXP-020` exploring
calibration, stacking, linear/neural diversity, interaction screening, and
blend-weight sweeps on top of it — none of which beat EXP-014.
`GOALS.md`, `DECISIONS.md`, `TODO.md`, and `RUN_LOG.md` were all present but
essentially empty (a single blank line each). The prior session had recorded
almost everything in `RESEARCH_STATE.md` + `checkpoint.json` +
`experiments_registry.json` instead. This session backfilled GOALS/DECISIONS/
TODO from that material (see those files' initial commit for the
reconstructed content) so future sessions have the intended layered memory
structure.
### Critical reliability problem discovered: unrecoverable exact pipeline
`RESEARCH_STATE.md`'s own "Artifact Caveat" section already flagged this,
but it's worth restating clearly: the exact script that produced the
historical EXP-014 = 0.96448 score (`train_competition.py`) was **only ever
saved to the local /home/user sandbox**, never committed to the persistent
`agent/` directory on Hugging Face. `features.py`/`models.py`/`ensemble.py`
in the persistent repo are generic reusable infrastructure, NOT the actual
36-domain-feature pipeline described in `experiments_registry.json` (which
references specific named features like `notifications_stress_pct`,
`missing_key_engagement`, etc. that do not appear anywhere in the persisted
Python files). When the sandbox was recycled, this exact pipeline became
permanently unrecoverable. The saved `submission.csv` in the repo was
already flagged as "regenerated with a faster approximate configuration"
that does not provably reproduce 0.96448.
**Decision made this session (see DECISIONS.md D10/D11):** rather than
trying to guess-reconstruct the exact lost script, rebuild an equivalent,
fully-documented, reproducible pipeline from the architecture description
(3 regularized LightGBM configs x num_leaves {63,45,127}, 5-Fold Stratified
CV, dual-seed 42+100 super-ensemble via rank averaging), commit it to
`agent/train_pipeline.py` on Hugging Face this time, and treat its matched
OOF as the new ground truth (`EXP-021`).
### EXP-021: Reproducible canonical pipeline rebuild
- Built 42 features: 9 raw numeric, 3 categorical (native LightGBM category
  dtype, handles NaN natively), 12 missingness indicators (`_isna` per raw
  column), 3 missingness-count aggregates, and ~15 domain ratio/interaction
  features (shares, per-hour rates, work/sleep ratio, engagement intensity,
  etc.) — see `agent/train_pipeline.py::build_features` for the exact list.
- 3 LightGBM configs matching the historical description: num_leaves=63/45/127,
  reg_alpha=0.5/0.5/1.0, reg_lambda=5.0/5.0/10.0, learning_rate=0.12,
  n_estimators=2000 with early_stopping(100), colsample_bytree=0.8,
  subsample=0.8, min_child_samples=20.
- 5-Fold StratifiedKFold, seeds 42 and 100, rank-averaged within each seed
  and then across seeds (dual-seed super-ensemble, matching EXP-014's
  architecture).
- **Result:** seed42 ensemble = 0.96365, seed100 ensemble = 0.96377,
  dual-seed super-ensemble = **0.96402**. Total wall-clock runtime: 14.3 min
  on 4 vCPU / ~3.8GB RAM sandbox.
- **Interpretation:** this is close to but not identical to the historical
  EXP-006 (0.96421 single-seed) / EXP-014 (0.96448 dual-seed) numbers — a
  gap of roughly -0.0004 to -0.0005. This is expected and consistent with
  D10/D11: the exact historical feature set is lost, so an equivalent (not
  identical) 42-feature pipeline was built instead. The *relative* effect of
  dual-seed blending replicated well: +0.00025 to +0.00037 gain over either
  single seed here, vs. +0.00027 historically — same direction, same rough
  magnitude. This is good evidence that the dual-seed-super-ensemble finding
  (D9) is a real, robust effect and not an artifact of the exact historical
  feature set.
- Artifacts (local scratch, not all pushed to HF): `train_pipeline.py`,
  `oof_predictions_matched.csv.gz`, `submission_matched_super.csv`,
  `submission_matched_seed42.csv`, `pipeline_results.json`,
  `oof_seed{42,100}.npy`, `test_seed{42,100}.npy`.
### Dataset forensics performed this session
While EXP-021 trained in the background (CPU-bound, ~15 min), ran several
cheap forensic checks on `train.csv`/`test.csv`:
1. **Duplicate rows**: 0 exact duplicate feature-rows within train, 0 within
   test. 2 "duplicate" rows appear when combining train+test, but both are
   rows with only 2–3 non-null values out of 13 columns — coincidental
   matches on very sparse rows, not real leakage. **Conclusion: no
   duplicate-row leakage risk.**
2. **Quantization**: `notifications_per_day` and `app_opens_per_day` are
   always exact integers (100% of non-null values), bounded in [20, 250] and
   [15, 180] respectively. All continuous "*_hours" columns round to exactly
   2 decimal places (99.999%+ match), with hard-looking bounds (e.g.
   `daily_screen_time_hours` in [0.5, 15.0], `sleep_hours` in [4.5, 9.0]).
   Consistent with a synthetic generator using bounded/clipped continuous
   distributions plus integer count distributions. Not yet exploited beyond
   informing the "other_screen_time" discovery below.
3. **Missingness is (empirically) MCAR w.r.t. the target.** This is an
   important correction to a prior interpretation. Checked: correlation of
   every single-column `_isna` indicator with `addicted_label` — all are in
   [-0.0003, +0.0027], i.e. essentially zero. Checked: mean target rate by
   `missing_count_total` (0 through 11 missing columns per row) — flat
   around 0.708–0.719 for counts 0–8 (only the extreme tail with <500 rows
   deviates, which is sampling noise). **This means missingness itself
   carries almost no direct signal about the label.** The historical
   `EXP-009` finding ("AUC is 0.972 at 0-missing rows vs 0.914 at 5-missing
   rows") is still true and not contradicted by this — but it should be
   interpreted as "less input information → harder to predict the label
   accurately for that subgroup" (an information/difficulty effect), NOT as
   "missingness pattern is itself predictive of addiction" (a signal
   effect). These are different claims. **Practical implication: the old
   TODO item "test missingness conjunction flags because missingness might
   carry signal" is now a low-prior hypothesis** — individual missingness
   indicators showing ~0 correlation makes it unlikely (though not provably
   impossible — trees could still find nonlinear conjunctive effects) that
   pairwise AND-conjunctions of missingness will help. Downgraded in
   TODO.md from P1 to a lower-confidence P1 item; worth a quick, cheap test
   before investing more, but expectations should be modest.
4. **No evidence of train/test covariate shift on feature *values*.**
   KS-tests on every numeric column comparing train vs test distributions
   all had large p-values (0.14–0.997) — no significant shift in the
   observed value distributions.
5. **Missingness RATES differ significantly between train and test per
   column** (e.g. `daily_screen_time_hours` 13.9% missing in train vs 11.1%
   in test; `academic_work_impact` 6.4% vs 8.7%). With n=691k/296k, even
   small differences reach extreme statistical significance (z-tests,
   p-values as low as 1e-261), so **statistical significance here should
   not be over-interpreted** — practically these are small absolute
   differences (a few percentage points) and, combined with the MCAR
   finding above (missingness doesn't correlate with target), are unlikely
   to bias the OOF-AUC-vs-leaderboard relationship much. Still worth
   remembering as a documented, checked risk rather than an assumed
   non-issue.
6. **Categorical features (`gender`, `stress_level`, `academic_work_impact`)
   have very weak marginal relationships with the target** (group means
   differ by at most ~0.02 from the ~0.709 base rate). Consistent with
   EXP-018's finding that additional interaction features added negligible
   value — the signal in this dataset is concentrated in the continuous
   usage-intensity features, not the categoricals.
7. **Weekend vs weekday screen time is NOT a hard multiplicative constraint**
   — checked whether `weekend_screen_time >= daily_screen_time_hours`
   always holds; it does not (14.5% of rows have weekend < weekday). Ruled
   out a broader "time budget" constraint hypothesis (also checked
   `sleep_hours + work_study_hours + daily_screen_time_hours <= 24`: 1.8% of
   rows exceed 24, so no hard daily-hours-budget constraint either).
8. **MAJOR FINDING — hard generator constraint discovered:**
   `daily_screen_time_hours >= social_media_hours + gaming_hours +
   work_study_hours` holds with **zero violations** across every row where
   all four values are jointly non-null (421,427 train rows, 182,287 test
   rows checked; min residual across both = exactly 0.0). This strongly
   suggests the generator constructs `daily_screen_time_hours` as (at
   least) the sum of `social_media_hours + gaming_hours + work_study_hours`
   plus an additional, always-non-negative "other" usage component
   (messaging, calls, browsing, streaming, or similar — not explicitly
   broken out as its own column). The residual
   (`other_screen_time = daily_screen_time_hours - social_media_hours -
   gaming_hours - work_study_hours`) has a standalone Pearson correlation
   of **0.305** with `addicted_label` on its own (vs. 0.611 for
   `daily_screen_time_hours` itself, 0.548 for the 3-component sum) —
   i.e. it carries real, non-trivial, previously-unexploited signal that
   is NOT simply redundant with the existing ratio/share features (which
   only expressed each component as a fraction of the total, never the
   leftover un-modeled remainder).
### EXP-022: Testing the other_screen_time hypothesis
- **Hypothesis:** adding `other_screen_time` (+ its `_isna` flag) as an
  explicit feature will improve OOF ROC-AUC because it captures previously
  implicit signal (the residual "other usage" component) that the existing
  ratio features do not directly expose to the trees.
- **Method:** single-seed (42) 5-Fold ensemble, identical fold splits and
  model configs to EXP-021's seed-42 run, features = EXP-021's 42 features
  + `other_screen_time` + `other_screen_time_isna` (44 total). Single-seed
  chosen deliberately to get a fast (~7 min) directional read before
  committing to a full dual-seed re-run.
- **Result:** ensemble OOF AUC = **0.96407** vs EXP-021 seed-42 baseline of
  0.96365 under the identical protocol. **Delta: +0.00042.** Improvement
  was consistent across nearly every fold/model combination observed live
  (fold 1: all 3 models +0.0004 to +0.0004; fold 2: all 3 positive; fold 3:
  all 3 positive; fold 4: all 3 positive, largest fold-level gain
  ~+0.0006–0.0009; fold 5: mixed but ensemble still net positive).
- **Conclusion: PROMOTED.** This is the single largest feature-engineering
  gain found across the entire research program to date (larger than
  EXP-002's regularization gain of +0.00022, larger than EXP-006's 5-fold
  upgrade of +0.00032, and larger than EXP-014's dual-seed gain of
  +0.00027). `other_screen_time` and `other_screen_time_isna` were added to
  `agent/train_pipeline.py::build_features` as permanent canonical features
  (44 features total now). See DECISIONS.md for the durable record.
### EXP-023: Full dual-seed re-run with the promoted feature
Launched immediately after EXP-022 confirmed the win, using the now-44-
feature `train_pipeline.py`, same dual-seed (42+100) 5-Fold protocol as
EXP-021. (Result recorded separately once complete — check
`experiments_registry.json` / `RESEARCH_STATE.md` for the final number if
this paragraph predates that run finishing.)
### Lessons for future sessions
- **Always commit the exact script producing a "canonical" score to the
  persistent HF repo**, not just local scratch. This was the single biggest
  process failure of the previous research cycle (see D10).
- **"Signal in the missingness pattern" and "difficulty caused by missing
  information" are different claims** — check both explicitly (marginal
  correlation of `_isna` with target vs. subgroup AUC by missingness count)
  before concluding "missingness carries signal."
- **Deterministic/near-deterministic inter-feature constraints are worth
  actively searching for** in synthetic Playground-series data — the
  `other_screen_time` residual was the highest-value single discovery of
  this session and came directly from checking a simple sum-vs-total
  constraint across the 4 screen-time-related columns. Worth systematically
  checking other plausible sum/ratio constraints among remaining features
  (e.g. is there a hidden relationship between `notifications_per_day` and
  `app_opens_per_day` beyond the simple ratio already computed?).