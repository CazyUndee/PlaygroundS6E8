# Research State — Smartphone Addiction (Kaggle Playground S6E8)

**Last updated**: 2026-08-17 (recovery session)
**Metric**: OOF ROC-AUC (5-fold stratified CV; dual-seed rank-average super-ensemble)

---

## Current position (after recovery)

The persistent repo was recovered from Hugging Face, but its state was
**internally inconsistent**:

- `HISTORY.md` recorded the 2026-08-14 session's work (pipeline rebuild +
  the `other_screen_time` discovery), but
- `train_pipeline.py` (the reproducible pipeline) was **missing** from the
  repo, and GOALS/DECISIONS/TODO/RUN_LOG were empty, while
  RESEARCH_STATE/checkpoint/registry still described the *pre-recovery* state.

So the canonical pipeline was lost a second time, and this session
reconstructed it (`source/train_pipeline.py`) from the architecture described
in HISTORY.md. It is now committed to the persistent repo (GitHub).

## Champion (reproducible, current)

| Experiment | Pipeline | OOF ROC-AUC | Status |
| :--- | :--- | :---: | :--- |
| EXP-021 | 42-feature rebuild, dual-seed 5-fold | 0.96402 | reproducible baseline |
| EXP-022 | + `other_screen_time` (+_isna), **single-seed** | 0.96407 (+0.00042) | **PROMOTED** |
| EXP-023 | 44-feature, **dual-seed** (42+100) | **0.96466** (+0.00064 vs EXP-021) | **CANONICAL WINNER** |

**Current champion = EXP-023: 0.96466 OOF ROC-AUC** (seed 42: 0.96435,
seed 100: 0.96439), computed on GitHub Actions (15 min, fully reproducible;
fold-level AUCs verified identical to the partial local run). This **beats
the historical (unreproducible) EXP-014 = 0.96448**, so the reconstructed
44-feature pipeline with `other_screen_time` is the best result of the whole
program. Subgroup difficulty gradient reproduced on the champion
(0-missing AUC 0.973 -> 9-missing 0.828); ensemble models are near-duplicates
(corr 0.994-0.997), so extra seeds/configs add little.

## Feature importance (ablation, 2026-08-17) — important correction
A feature-group ablation (lgbm_63, seed-42, 5-fold; full = 0.96382) shows
**`notifications_per_day` (-0.0082) and `app_opens_per_day` (-0.0064) are
the most important features conditionally** despite near-zero marginal
correlation; `age` (-0.0002) and `sleep` (-0.0003) also matter. Removing the
domain-ratio group (+0.0002), categoricals (+0.0001), or `other_screen_time`
(+0.00004) is neutral (within fold noise ~0.0016). The 44-feature set is
near-optimal; keep the count features. (This corrects the earlier "counts
are weak" framing — true marginally, false conditionally. See D20.)

## Feature-engineering direction is EXHAUSTED (as of 2026-08-17)
All four screens on the champion (lgbm_63, seed-42, 5-fold, vs 0.96380
baseline) were neutral or negative: total_screen -0.00003, sm_weekend
-0.00003, all3 0.00000, age encodings -0.00029. Even large univariate AUC
advantages (+0.026) and a real independent per-value signal (age) do not
help — the 44 canonical features already capture the sufficient statistics.
Do not add more features without a mechanism genuinely missing from the raw
inputs. Remaining value: hyperparameter tuning (running on GH Actions),
validation/robustness work, or accepting EXP-023 as the champion.

## Canonical pipeline (44 features)

`source/train_pipeline.py::build_features`:

1. 9 raw numeric (`age`, 4 screen-time cols, `sleep_hours`, 2 engagement counts, `weekend_screen_time`)
2. 3 categorical (`gender`, `stress_level`, `academic_work_impact`) — native LightGBM `category` dtype
3. 12 per-column `_isna` indicators
4. 3 missingness-count aggregates (total / numeric / categorical)
5. 15 domain ratio/interaction features (shares of screen time, weekend structure, engagement rates, work/sleep ratio, etc.)
6. `other_screen_time` = `daily_screen_time - social - gaming - work` (+ its `_isna` flag) — **the promoted hard-constraint residual**

Models: 3 regularized LightGBM configs (num_leaves 63/45/127; reg_alpha
0.5/0.5/1.0; reg_lambda 5/5/10; lr 0.12; 2000 trees early-stop 100;
colsample/subsample 0.8; min_child_samples 20). 5-fold stratified CV, seeds
{42, 100}, rank-average within seed then across seeds.

## Strongest findings

1. **`other_screen_time` is real, non-redundant signal.** The generator obeys
   `daily_screen_time >= social + gaming + work` with zero violations; the
   leftover component correlates r≈0.305 with the label and gave +0.00042
   OOF AUC — the largest single feature gain of the whole program.
2. **Dual-seed super-ensembling is robust** (D9): +0.00025–0.00027 across
   two different feature pipelines. Fold-partition boundary noise is real.
3. **Regularized LightGBM dominates** everything else tried (linear/neural
   0.92, extra-trees 0.949, HGB 0.961, XGB 0.9634).
4. **Missingness is MCAR w.r.t. target** (D12): indicators correlate ~0 with
   the label; the subgroup-AUC drop is an information/difficulty effect.
5. **No duplicate-row leakage; no significant train/test covariate shift on
   values** (though missingness *rates* differ a few points — see D16).
6. **`age` hides a real nonlinear signal despite r≈0 with the label**
   (integer [18,35]; ages 24/26/28 ≈ 0.77 target rate vs 18/33 ≈ 0.64,
   independent of screen time). Raw numeric age may be under-used by the
   trees — EXP-024 candidate (explicit categorical/parity encoding).

## Rejected / not-promoted (durable)

- Subgroup quantile ranks, proportional imputation, interaction screening
  (all <= +0.00003) — canonical features already suffice.
- Isotonic calibration, extra-trees, linear/neural hyperplanes, HGB, XGB blend.
- EXP-019/020 blend weight sweeps: 100:0 EXP-014 control remained best on
  proxy evidence; no blend promoted.

## Key uncertainties / next questions

1. Does the EXP-022 `other_screen_time` gain survive the **full dual-seed**
   protocol? (EXP-023 — running.)
2. Does explicitly encoding the **nonlinear `age` signal** beat raw numeric
   age? (EXP-024 candidate.)
3. Are there **other undiscovered hard generator constraints** with non-trivial
   residuals? *(systematic search this session found none beyond the known
   `dst >= social+gaming+work`; see HISTORY.)*
4. Does **more seed diversity** (3rd partition seed) keep helping or plateau?
5. Is the ensemble's value real **prediction diversity** or near-duplicate
   averaging (correlation re-measurement)?
6. Is the new champion's subgroup behavior (by missingness/usage decile)
   consistent with the documented difficulty gradient?

See `TODO.md` for the live queue and `DECISIONS.md` for durable conclusions.
