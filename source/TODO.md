# Research Queue

Promising questions and investigations. Priorities are guidance, not a
command sequence — let evidence decide.

---

## P0 (active / high value)

- [ ] **EXP-023**: Record the dual-seed (42+100) 5-fold result of the 44-feature
      pipeline with `other_screen_time` (reconstructed `train_pipeline.py`).
      Confirm the EXP-022 gain (+0.00042) survives the full dual-seed protocol
      and produce a fresh matched submission. *(running on GitHub Actions —
      `gh workflow run research.yml -f task=exp023_canonical`)*

- [ ] **EXP-024 — age nonlinearity**: `age` (integer [18,35]) has ~0 linear
      correlation with the label but a large nonlinear per-age effect that
      survives within fixed screen-time bins (24/26/28 high ~0.77; 18/33 low
      ~0.64). Hypothesis: raw numeric age is under-split by the trees.
      Test, under identical seed-42 folds vs the 44-feature baseline
      (0.96407): (a) age as native categorical, (b) one-hot age, (c) add
      `age_even` + high-age flags. *(quick subsample A/B running — EXP-024)*

- [x] **EXP-026 — combination features (total_screen/sm_weekend/all3)**:
      ALL NEUTRAL (deltas -0.00003 / -0.00003 / 0.00000). Trees already
      capture combinations; direction exhausted. Do not revisit without a
      new mechanism.

- [x] **EXP-024 — age nonlinearity**: REJECTED (OOF 0.96352 vs 0.96380,
      -0.00029). Raw numeric age already lets trees use the per-value signal;
      explicit encodings add overfitting capacity.

- [ ] **EXP-025 — count/sleep nonlinearity (LOW PRIORITY now)**: app_opens
      and sleep have nonlinear profiles, but given the age encoding failed,
      explicit encodings are unlikely to help. Only worth a quick screen if
      the tune scan reveals headroom.

- [ ] **Hyperparameter tuning (active)**: lgbm_63 scan (lr 0.10/0.12/0.15,
      leaves 31/63/95/127/255, min_child_samples 10/20/50, colsample
      0.7/0.8/0.95, subsample 0.8/0.95, reg variants, capacity probes). If a
      config clearly beats 0.96380, promote and re-run the dual-seed
      protocol (EXP-027) on GH Actions.

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
