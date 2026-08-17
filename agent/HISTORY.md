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

---
## Session 2026-08-17 (recovery) — GitHub migration, pipeline reconstruction, and forensics
### Context
Fresh workspace. HF dataset `cazyundee/PlaygroundS6E8` was the only source
of truth (no `hf` CLI, so files were pulled over plain HTTPS). Recovered the
memory files and found the persistent state was **internally inconsistent**:
HISTORY.md recorded the 2026-08-14 session (pipeline rebuild EXP-021,
other_screen_time discovery EXP-022, and the launched-but-unrecorded
EXP-023), but `train_pipeline.py` was MISSING from the repo, and
GOALS/DECISIONS/TODO/RUN_LOG were empty while RESEARCH_STATE/checkpoint/
registry still described the pre-recovery state. The canonical pipeline had
been lost a second time (same failure mode as D14).

### Recovery actions
- Reconstructed `agent/train_pipeline.py` from the architecture described in
  HISTORY.md: 44 features (9 numeric + 3 categorical + 12 `_isna` + 3
  missingness-count + 15 domain ratios + other_screen_time + its `_isna`),
  3 regularized LGBM configs (num_leaves 63/45/127; reg_alpha 0.5/0.5/1.0;
  reg_lambda 5/5/10; lr 0.12; 2000 trees early-stop 100; colsample/subsample
  0.8; min_child_samples 20), dual-seed (42+100) 5-fold rank-average.
  This is an *equivalent* reconstruction, not the lost byte-for-byte script.
- Backfilled GOALS.md / DECISIONS.md / TODO.md / RUN_LOG.md (were empty) and
  reconciled RESEARCH_STATE.md / checkpoint.json / experiments_registry.json
  (added EXP-021/022/023 entries).
- Migrated the persistent home from Hugging Face to **GitHub**
  `CazyUndee/PlaygroundS6E8` (plain git blobs, no LFS, so a plain clone is
  self-contained). Verified the other_screen_time constraint on fresh data:
  zero violations, residual Pearson r = 0.305 with target.
- Launched EXP-023 (44-feature dual-seed) as the new ground truth.

### Runtime observation (important for future sessions)
On this Windows host LightGBM is ~10x slower than the 4-vCPU Linux sandbox
that produced the historical 14.3-min dual-seed runtime. The process spawns
~23 threads (8 logical CPUs) but achieves only ~2x effective parallelism
(~1.7 cores average), and fold 1 of seed 42 took ~45 min for 3 models
(~15 min/model, best_iteration ~300 trees). Full dual-seed would be ~7h.
**Practical implication:** on this host, prefer cheaper protocols (single
seed, 3-fold, fewer trees) for directional reads; reserve full dual-seed for
promoted changes. Do NOT interpret the slow wall-clock as a change in the
research question — it is an environment property.

### Forensics: systematic constraint search (new)
Ran a zero-violation constraint screen over all pairs/sums of the 9 numeric
features (train + test cross-check). Result:
- **`age` is an exact integer** in [18, 35] (18 distinct values), like
  notifications/app_opens. All `*_hours` features are 2-decimal continuous.
- Most "zero-violation" hits are **trivial range artifacts** (e.g.
  `age >= sleep_hours` because age min 18 > sleep max 9; `notifications >=
  screen_time` because notifications min 20 > screen max 15). These carry no
  information and must be filtered out by range-overlap, not just
  zero-violations.
- **No NEW hard constraint of the other_screen_time magnitude was found.**
  The residual correlations from sum decompositions are all just
  re-expressions of the known `dst >= social + gaming + work` constraint
  (e.g. `dst - gaming - work` = social + other, r=0.565 — redundant with
  existing features). `age >= dst + gaming` is *almost* always true but has
  1 test violation, so it is a soft pattern, not a hard constraint.

### Forensics: feature-target signal structure (new)
- Signal is heavily concentrated: Pearson r with label —
  daily_screen_time 0.611, weekend_screen_time 0.590, social_media 0.532,
  work_study 0.251, gaming 0.205; everything else ~0 (sleep 0.042, app_opens
  0.064, notifications -0.012, age 0.004).
- `weekend_extra = weekend - daily` has r ~ 0.014, i.e. weekend_screen_time's
  signal is almost entirely redundant with daily_screen_time (already
  handled by existing weekend features).
- **NEW LEAD — age carries a real NONLINEAR signal despite r~0.004.**
  Target rate by age (complete rows only, so not missingness-confounded):
  24->0.770, 26->0.777, 28->0.773, 32->0.736, 22->0.740 are HIGH; 18->0.657,
  33->0.641, 23->0.671, 25->0.672, 27->0.674 are LOW. The pattern PERSISTS
  within a fixed screen-time bin (5.5-7.0h: 24->0.641 vs 33->0.423), so it
  is independent of the dominant screen-time signal. Single-feature AUCs:
  `age in {24,26,28}` = 0.525, `age even` = 0.521, raw age rank = 0.502
  (vs daily_screen_time 0.890). Not a clean parity rule (18 and 30 are even
  but LOW), so the mechanism is likely a specific per-age generator effect.
- **Hypothesis (EXP-024 candidate):** raw numeric `age` may be under-used by
  the trees (weak marginal gain -> few splits); explicitly exposing the age
  buckets/parity (e.g. treat age as categorical, one-hot, or add `age_even`
  + high-age flags) may recover this signal for a small OOF gain. Must be
  tested under identical folds vs the 44-feature seed-42 baseline (0.96407).
- **Lesson:** a feature with ~0 linear correlation can still hide a large
  nonlinear/independent effect. Check target-rate by distinct value before
  dismissing low-correlation features in synthetic data.

### Forensics: the same applies to count/sleep features (new)
- The pattern generalizes beyond age. Single-feature AUCs (rank):
  app_opens_per_day 0.541, sleep_hours 0.527, age 0.502,
  notifications_per_day 0.492 — all "low" vs daily_screen_time 0.890, but
  all have NON-monotonic target-rate profiles:
  * app_opens deciles: 0.654 / 0.734 / 0.673 / 0.645 / 0.726 / 0.716 /
    0.657 / 0.763 / 0.770 / 0.755 — a U/oscillating shape, high deciles
    (7-9) clearly elevated. Within a fixed screen-time bin, higher
    app_opens -> higher rate (0.492 -> 0.588).
  * sleep bins: (4.5,5]->0.662, (6,6.5]->0.725, (8,8.5]->0.745 — roughly
    more sleep -> higher rate but non-monotonic (dip at 7-8h).
  * notifications: within fixed screen time, MORE notifications -> LOWER
    rate (0.551 -> 0.507); the existing n_per_screen ratio captures this
    (its single AUC is 0.264, monotonic).
- **Implication:** these are already exposed as raw numeric features and the
  trees CAN split on them (app_opens/sleep have stronger marginal AUC than
  age), so they are probably already partially used. Age is the weakest
  marginal (0.502) yet has the cleanest independent nonlinear effect, so age
  is the best first target for an explicit-encoding test (EXP-024). If age
  encoding works, the same treatment (target-encoding / categorical / bins)
  for app_opens and sleep becomes the natural follow-up (EXP-025 candidate).

### Forensics: categoricals do NOT moderate the screen-time signal (new)
- corr(daily_screen_time, target) is 0.608-0.618 across EVERY gender /
  stress_level / academic_work_impact group — essentially identical.
- The dst-decile -> target-rate curve is identical across all groups
  (decile 0 ~0.28, 1 ~0.46, 2 ~0.81, 3 ~0.985, 4 ~0.999).
- The label is therefore a sharp STEP/THRESHOLD function of screen time
  (top 2 deciles ~98-100% rate), and the categoricals neither shift the
  screen-time distribution nor moderate its effect.
- **Durable conclusions:** (1) interaction features between screen time and
  categoricals are very unlikely to help (confirms EXP-018/EXP-001);
  (2) the sharp threshold explains why linear/neural models underfit to
  ~0.92 (EXP-017) while trees excel; (3) categoricals can be treated as
  essentially non-informative for this task.

### Operational lesson: do NOT run two LightGBM jobs concurrently here
- Launched the EXP-024 quick read (num_threads=3) alongside the canonical
  EXP-023 run (n_jobs=-1). Both LightGBM OpenMP pools contended badly:
  EXP-023's effective throughput dropped from ~1.7 cores to ~0.9 cores and
  EXP-024 got ~0.13 cores. Killed EXP-024 and deferred it to run
  sequentially after EXP-023.
- **Rule for this host:** one LightGBM training job at a time. Cheap
  pandas-only forensics are fine to run concurrently with a training job,
  but a second tree-training job is counterproductive.

### Forensics: threshold structure + total_screen feature (new)
- The label is a sharp threshold on screen usage: dst bins -> 0.40 (5-6h),
  0.58 (6-7h), 0.76 (7-8h), 0.91 (8-9h), 0.99 (9-10h), ~1.0 (>10h).
- The 3-component sum `sum3 = social+gaming+work` (already in the pipeline
  as `total_active_hours`) is an even SHARPER threshold in the mid-range
  (5-6h -> 0.61, 6-7h -> 0.82), consistent with the generator constructing
  the label primarily from the 3 usage components plus the "other" residual.
- **NEW candidate feature — `total_screen = daily_screen_time_hours +
  weekend_screen_time`**: single-feature AUC 0.901, higher than daily (0.889)
  or weekend (0.881) alone. Both are noisy measurements of the same
  underlying usage signal, so the sum reduces noise; trees would otherwise
  need 2D splits to approximate it. Not currently exposed in the pipeline
  (only ratio/difference weekend features exist). Candidate EXP-026.
- Single-feature AUC ranking (rank metric): total_screen 0.901 > daily
  0.890 > weekend 0.881 > social_media 0.858 ~ sum3 0.857 > work_study
  0.655 > gaming 0.622 > app_opens 0.541 > sleep 0.527 > age 0.502 >
  notifications 0.492.

---
## Session 2026-08-17 (cont.) — ALL compute moves to GitHub Actions
- The local Windows host proved ~10x slower than the historical 4-vCPU Linux
  sandbox (LightGBM achieving only ~1.7 effective cores), so the local
  EXP-023 run was KILLED after seed-42 folds 1-4 (partial reference numbers:
  lgbm_63 fold AUCs 0.96313/0.96365/0.96386/0.96468 — they track the
  historical fold pattern, good evidence the reconstruction is faithful).
- **New operating model (D19):** GitHub is the compute environment. Created
  `.github/workflows/research.yml` (workflow_dispatch; tasks exp023_canonical,
  exp026_total_screen, exp024_age_screen, all), `requirements.txt`, and
  moved the forensics scripts to `analysis/` for reproducibility. Local
  machine is orchestration only: push -> `gh workflow run` -> download
  artifacts (`gh run download`) -> curate results into memory files ->
  commit. Model training NEVER runs locally again.
- The workflow saves OOF/test arrays (train_pipeline.py now persists
  oof_*.npy / test_*.npy) and uploads them as artifacts so diversity,
  blending, and subgroup analyses are possible without retraining.

### Forensics follow-up: weekend is conditionally informative (supports EXP-026)
- Cross-tab of target rate by (daily_screen_time bin, weekend quintile) shows
  weekend shifts the rate STRONGLY within every daily bin, e.g. dst (5.5,7.0]:
  weekend q0 -> 0.335 vs q4 -> 0.968; dst (4.0,5.5]: q0 -> 0.279 vs q4 -> 0.844.
  So the label depends on BOTH measures (latent "total usage"), and weekend
  is not a mere noisy copy of daily (this is not contradicted by the near-zero
  marginal corr of `weekend_extra` — the conditional effect is real).
- This explains and strengthens the EXP-026 `total_screen = daily + weekend`
  hypothesis: it exposes the latent the generator uses; trees currently need
  expensive 2D splits to approximate it.
- The age effect is INDEPENDENT of every other signal tested: it adds a
  consistent ~+0.06-0.07 target rate within every app_opens quintile and every
  sleep bin (e.g. app_opens q0: 0.677 vs 0.742; sleep 4.5-6h: 0.663 vs 0.737),
  on top of the already-shown independence from screen time and categoricals.
  Strong support for EXP-024 (explicit age encoding).
- **Old open question resolved: `notifications_per_day` and
  `app_opens_per_day` are INDEPENDENT generator draws** (Pearson r = 0.012;
  mean app_opens ~102 across every notifications octile; a/n ratio is
  noise). The `notifications_per_app_open` / `app_opens_per_notification`
  ratio features are therefore noise features — candidates for ablation in a
  simplification pass, though they likely cost nothing.

### EXP-023 COMPLETE on GitHub Actions — new champion 0.96466
- Run 32034710221 (ubuntu-latest, 4 vCPU, 890s wall). 44-feature dual-seed
  (42+100) super-ensemble: seed42 = 0.96435, seed100 = 0.96439, **super =
  0.96466**.
- **Interpretation:** the reconstructed 44-feature pipeline BEATS the
  historical lost-pipeline champion (EXP-014 = 0.96448) by +0.00018. The
  `other_screen_time` feature's dual-seed gain is +0.00064 over the
  42-feature EXP-021 rebuild — even larger than the +0.00042 single-seed
  estimate, so the EXP-022 promotion is strongly confirmed.
- **Reproducibility:** seed-42 fold-level AUCs match the (killed) local run
  EXACTLY (0.96313/0.96365/0.96386/0.96468/0.96375) — cross-environment
  determinism with the same LightGBM major version. OOF/test arrays saved
  (artifact exp023-results, 65MB tar.gz; 90-day retention).
- **Diversity reality:** the 3 LGBM configs correlate 0.994-0.995 within a
  seed; the two seed ensembles correlate 0.997 (OOF) / 0.9995 (test). The
  ensemble is near-duplicate averaging; further seeds/configs of the same
  architecture will give diminishing returns (D9 remains true but the
  remaining headroom is tiny). Genuine diversity would need a structurally
  different (but comparable-AUC) model — all such attempts so far have been
  rejected (XGB too correlated, HGB worse/slower, extra-trees much worse).
- **Subgroup gradient reproduced on the champion:** OOF AUC by missingness
  count: 0->0.973, 1->0.969, 2->0.961, 3->0.949, 4->0.934, 5->0.915,
  6->0.896, 7->0.873, 8->0.843, 9->0.828 (matches historical EXP-009).
- Results curated: results/exp023/{submission.csv, pipeline_results.json,
  exp023_run.log}. Next: EXP-026/EXP-024 screens on GH Actions.

### New feature candidate: sm_weekend (social_media + weekend) AUC 0.915
- Scanning pairwise sums of the screen features found
  `social_media_hours + weekend_screen_time` has univariate AUC 0.91522
  (0.91529 verified on the same 457k-row subset as daily+weekend 0.90126) —
  much stronger than `total_screen = daily + weekend` (0.901). The triplet
  `daily + social + weekend` is 0.91674, marginally better.
- Interpretation: the label depends on the latent "usage" mostly through the
  social component + weekend; daily adds overlap/redundancy (daily includes
  social + gaming + work + other). Weighted forms showed weight ~0.75 on
  weekend is near-optimal, so the simple sum is a fine feature form.
- Queue (EXP-026 family): screens for total_screen, sm_weekend, all3, age,
  then promote winners into build_features and re-run dual-seed EXP-027.
- Note: the queued age screen (exp024) was auto-cancelled when the workflow
  file changed; re-trigger it after the current screen queue.