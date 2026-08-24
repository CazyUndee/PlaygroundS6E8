# Research Queue

Promising questions and investigations. Priorities are guidance, not a
command sequence — let evidence decide.

---

## P0 (active / high value)

- [x] **EXP-023**: COMPLETED — 0.96466 OOF ROC-AUC (dual-seed 44-feature
      super-ensemble). New champion, beats historical EXP-014 (0.96448).

- [x] **EXP-024 — age nonlinearity**: REJECTED (OOF 0.96352 vs 0.96380,
      -0.00029). Raw numeric age already lets trees use the per-value signal;
      explicit encodings add overfitting capacity.

- [x] **EXP-026 — combination features (total_screen/sm_weekend/all3)**:
      ALL NEUTRAL (deltas -0.00003 / -0.00003 / 0.00000). Trees already
      capture combinations; direction exhausted.

- [ ] **EXP-027 — dual-seed tune winners**: leaves_31 (+0.00027 single-seed)
      and subsample_095 (+0.00023) through full dual-seed (42+100) 5-fold
      protocol, with canonical baseline for matched comparison. Script:
      `state/exp027_leaves31.py`. *(launched on GitHub Actions)*
      If any arm beats 0.96466, promote to canonical and submit.

- [ ] **EXP-025 — count/sleep nonlinearity (LOW PRIORITY now)**: app_opens
      and sleep have nonlinear profiles, but given the age encoding failed,
      explicit encodings are unlikely to help. Only worth a quick screen if
      the tune scan reveals headroom.

- [x] **Hyperparameter tuning scan**: COMPLETED. 17 configs scanned.
      Winners: leaves_31 (+0.00027), subsample_095 (+0.00023), lr_010
      (+0.00014). Regularization is critical (reg_none = -0.0017).
      EXP-027 tests top 2 findings through dual-seed protocol.

## P1 (strong hypotheses, quick reads)

- [x] **Systematic generator-constraint search**: COMPLETED. Zero-violation
      screen over all pairs/sums of 9 numeric features found no new hard
      constraint of the other_screen_time magnitude. Most hits are trivial
      range artifacts. No new feature candidate.
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
