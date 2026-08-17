# Research Queue

Promising questions and investigations. Priorities are guidance, not a
command sequence — let evidence decide.

---

## P0 (active / high value)

- [ ] **EXP-023**: Record the dual-seed (42+100) 5-fold result of the 44-feature
      pipeline with `other_screen_time` (reconstructed `train_pipeline.py`).
      Confirm the EXP-022 gain (+0.00042) survives the full dual-seed protocol
      and produce a fresh matched submission. *(running — see RUN_LOG)*

- [ ] **EXP-024 — age nonlinearity**: `age` (integer [18,35]) has ~0 linear
      correlation with the label but a large nonlinear per-age effect that
      survives within fixed screen-time bins (24/26/28 high ~0.77; 18/33 low
      ~0.64). Hypothesis: raw numeric age is under-split by the trees.
      Test, under identical seed-42 folds vs the 44-feature baseline
      (0.96407): (a) age as native categorical, (b) one-hot age, (c) add
      `age_even` + high-age flags. *(quick subsample A/B running — EXP-024)*

- [ ] **EXP-025 — count/sleep nonlinearity (follow-up if EXP-024 works)**:
      `app_opens_per_day` (AUC 0.541) and `sleep_hours` (0.527) also have
      non-monotonic target-rate profiles. If explicit age encoding helps,
      apply the same treatment (OOF target-encode / categorical / bins) to
      app_opens and sleep, and possibly a `notifications` inverse-rate
      encoding.

## P1 (strong hypotheses, quick reads)

- [ ] **Systematic generator-constraint search.** The `other_screen_time`
      discovery came from checking one sum constraint. Systematically test
      other plausible hard constraints among remaining features (e.g.
      relationships between `notifications_per_day` / `app_opens_per_day` /
      screen time / `weekend_screen_time`, and bounds/quantization of
      `age`, `sleep_hours`). Any *zero-violation* constraint with a
      non-trivial residual is worth evaluating as a feature.
- [ ] **DART boosting mode** (LightGBM) for prediction diversity — only after
      the reproducible matched baseline (EXP-023) exists, with per-model
      artifacts saved.
- [ ] **Third fold-partition seed** (e.g. 2026) added to the super-ensemble,
      if memory/compute allows, to test whether additional seed diversity
      keeps helping or plateaus.
- [ ] **Missingness conjunction flags** (cheap, low prior per D12): test
      `screen_time_isna * notifications_isna`, `social_media_isna *
      gaming_isna`, etc. Confirm D12 rather than assume it.

## P2 (medium value / robustness)

- [ ] **Feature ablation of `other_screen_time`** to quantify its exact
      contribution in the 44-feature pipeline (leave-one-feature OOF) and
      check it is not merely duplicating `engagement_intensity`.
- [ ] **Model diversity re-measurement**: correlation of the 3 LGBM configs'
      OOF predictions and of the two seeds, to confirm the ensemble's value
      is real diversity and not near-duplicate averaging.
- [ ] **Validation stability**: repeat the seed-42 run with a second
      `random_state` for model bagging to estimate variance of the OOF number
      (the fold seed is fixed but model randomness is a confound).
- [ ] **Subgroup robustness**: measure OOF AUC by missingness count and by
      usage intensity deciles on the new champion; verify the documented
      0-missing ≈ 0.97 vs 5-missing ≈ 0.91 difficulty gradient still holds.

## P3 (exploratory / lower prior)

- [ ] Quantization-aware features (e.g. integer-ness of notifications/app_opens
      already known; test whether rounding screen-time features to 2 decimals
      exposes generator buckets that help).
- [ ] CatBoost (native categorical) as a diversity probe — only if the
      library is available and a quick single-seed read is cheap.
- [ ] Reduced-feature / interpretability check: confirm the top-ranked
      features and document which are doing the work (guides future feature
      hypotheses).
