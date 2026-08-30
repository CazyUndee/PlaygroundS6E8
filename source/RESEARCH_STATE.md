# Research State — Smartphone Addiction (Kaggle Playground S6E8)

**Last updated**: 2026-08-30 (EXP-028/029 complete — champion still EXP-027 0.96498; EXP-030/031 running)
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
| EXP-023 | 44-feature, **dual-seed** (42+100) | 0.96466 (+0.00064 vs EXP-021) | previous champion |
| **EXP-027** | leaves_31 + sub095, **dual-seed** (42+100) | **0.96498** (+0.00032 vs EXP-023) | **CANONICAL WINNER** |
| EXP-028 | leaves_31+lr010+mcs50, **dual-seed** | 0.96491 (+0.00025 vs EXP-023) | below champion — not promoted |
| EXP-029 | triple seed (42+100+2026) | +0.00006 vs dual-seed | **dual-seed is the cost/performance optimum** |

**Current champion = EXP-027 leaves_31_sub095: 0.96498 OOF ROC-AUC** (seed 42: 0.96435,
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

## Hyperparameter tuning — COMPLETE (EXP-027 confirmed)

A 17-config single-seed scan found leaves_31 (+0.00027) and subsample_095
(+0.00023) as winners. EXP-027 validated both through the full dual-seed
(42+100) 5-fold protocol (run 32733942935, 43 min, GitHub Actions):

| Arm | seed42 | seed100 | **super** | vs EXP-023 |
| :--- | :---: | :---: | :---: | :--- |
| canonical_63 (baseline) | 0.96431 | 0.96440 | 0.96465 | — |
| leaves_31 | 0.96453 | 0.96466 | **0.96486** | **+0.00021** |
| **leaves_31_sub095** | **0.96470** | **0.96473** | **0.96498** | **+0.00034** |

**The gains SURVIVED dual-seed.** leaves_31_sub095 is the new champion at
0.96498, beating EXP-023 (0.96466) by +0.00032. The two effects combine
nearly additively (single-seed: +0.00027 + +0.00023 ≈ +0.0005; dual-seed:
+0.00034, reflecting expected ~50% attenuation from fold noise).

Canonical_63 replicated at 0.96465 (vs EXP-023 0.96466), confirming the
matched comparison is valid. The fold pattern is stable (seed42 lgbm_63
fold AUCs in EXP-027 match EXP-023 to 3 decimal places).

**Key insight: num_leaves=31 is the strongest single-knob improvement.**
Fewer leaves reduces overfitting to the many correlated ratio features in
the 44-feature pipeline. Subsample=0.95 provides a clean additional win.
Both promote to the canonical pipeline.

## Second-order combos and seed diversity — RESOLVED (EXP-028/029)

**EXP-028 (second-order combos, run 32739428150, 124 min, GH Actions):**

| Arm (dual-seed super) | super OOF | vs EXP-023 | vs EXP-027 |
| :--- | :---: | :---: | :---: |
| canonical_63 (baseline) | 0.96464 | — | — |
| leaves_31 | 0.96486 | +0.00020 | — |
| leaves_31_lr010 | 0.96491 | +0.00024 | — |
| leaves_31_mcs50 | 0.96489 | +0.00023 | — |
| **leaves_31_lr010_mcs50** | **0.96491** | +0.00025 | — |

leaves grid (single-seed): 20=0.96418, 25=0.96419, 31=0.96409, 35=0.96410,
45=0.96397 — **flat 20-35, 25 marginally best; no grid point beats the
champion**. lr010/mcs50 add small gains (+0.00003 to +0.00006) but were
**tested WITHOUT subsample_095** — the stacking question is EXP-030's job.

**EXP-029 (triple fold-partition seed, run 32739432326, 58 min):** adding
seed 2026 to the super-ensemble adds only **+0.00006** (leaves_31: dual
0.96486 → triple 0.96491; canonical: 0.96464 → 0.96470). **Dual-seed is
the cost/performance optimum** — D9's diminishing-returns prediction
confirmed. More seed diversity is not the path forward.

## Key uncertainties / next questions

1. **Does stacking sub095 with lr010/mcs50 beat 0.96498?** EXP-030 running
   (champion stack arms: sub095+lr010, sub095+mcs50, full stack, leaves_25
   full stack). If gains are additive, full stack could reach ~0.96503.
2. **Is DART a competitive diverse model for blending?** EXP-031 running —
   DART vs gbdt champion + OOF correlation matrix (within-seed across the
   3 configs, and cross-seed). Directly tests the near-duplicate-averaging
   hypothesis (corr 0.994-0.997).
3. **Are there other undiscovered hard generator constraints?** *(systematic
   search found none beyond `dst >= social+gaming+work`.)*
4. **Does the new champion's subgroup gradient change?** (0-missing ~0.973
   vs 5-missing ~0.915 — verify on EXP-027 predictions.)
5. **Should the canonical pipeline now be updated to the EXP-027 config?**
   Memory files record num_leaves=31/sub095 (D24) but `train_pipeline.py` /
   `models.py` still hardcode 63/0.8 — **pending EXP-030 confirmation of the
   final champion config before touching shared code** (EXP-030 imports
   COMMON_PARAMS).

See `TODO.md` for the live queue and `DECISIONS.md` for durable conclusions.
