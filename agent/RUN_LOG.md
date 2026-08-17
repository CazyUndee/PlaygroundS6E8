# Run Log

Execution facts only (not narrative — see HISTORY.md).

---

## 2026-08-17 (recovery session)

- 13:04 UTC — Retrieved Hugging Face dataset `cazyundee/PlaygroundS6E8`
  (sha b6ec49f) into local workspace via HTTPS (no `hf` CLI available).
- 13:06 — Verified persistent state is INCONSISTENT: `HISTORY.md` describes
  the 2026-08-14 session (pipeline rebuild + `other_screen_time` discovery)
  but `train_pipeline.py` is missing from the repo and GOALS/DECISIONS/TODO/
  RUN_LOG are empty; RESEARCH_STATE/checkpoint/registry are stale.
- 13:08 — Verified the `other_screen_time` constraint on fresh data:
  zero violations of `daily_screen_time >= social+gaming+work` (421,427
  jointly-non-null train rows), residual Pearson r = 0.305 with target.
- 13:10 — Reconstructed `agent/train_pipeline.py` (44 features, 3 LGBM
  configs, dual-seed 5-fold). Smoke test on 20k rows: OK (AUC 0.937, 100
  trees).
- 13:12 — Launched full dual-seed pipeline (EXP-023) in background.
- 13:56 — EXP-023 seed-42 fold 1 complete: lgbm_63=0.96313, lgbm_45=0.96334,
  lgbm_127=0.96302 (Windows host ~10x slower than the Linux sandbox).
- ~14:00 — Forensics (constraint search, signal structure, age nonlinearity,
  count/sleep nonlinearity) completed; findings committed to GitHub.
- ~14:45 — Launched EXP-024 age-encoding quick A/B read (15% subsample,
  3-fold, 3 models, 2 arms) in parallel with EXP-023.
- ~15:45 — KILLED EXP-024 (PID 5644) after it starved EXP-023: two
  concurrent LightGBM jobs on this host collapse effective throughput
  (EXP-023 fell from ~1.7 to ~0.9 cores). EXP-024 deferred to run after
  EXP-023. Lesson recorded.
- ~17:00 — Local EXP-023 reached seed-42 folds 1-4 (lgbm_63 fold AUCs
  0.96313/0.96365/0.96386/0.96468; tracks the historical fold pattern),
  then was KILLED (PID 30592) when the decision was made to run ALL compute
  on GitHub Actions. Fold 5 + seed 100 were never completed locally.
- ~17:00 — Created `.github/workflows/research.yml` (workflow_dispatch tasks:
  exp023_canonical, exp026_total_screen, exp024_age_screen, all) +
  `requirements.txt`. Local machine is now orchestration only: push →
  `gh workflow run` → download artifacts → curate results → commit.
- ~14:37 — **GH Actions EXP-023 COMPLETE (run 32034710221, 890s, success).**
  Seed42 ensemble = 0.96435, Seed100 = 0.96439, **super-ensemble =
  0.96466** (new reproducible champion, beats historical 0.96448). Fold
  AUCs match the local partial run exactly (cross-env reproducibility).
  Artifacts downloaded (submission.csv, pipeline_results.json,
  oof_arrays.tar.gz) and curated into results/exp023/.
- ~14:40 — OOF analysis: models are near-duplicates (corr 0.994-0.997;
  test seeds 0.9995); subgroup gradient reproduced (0-missing 0.973 ->
  9-missing 0.828).
- (pending) — GH Actions EXP-026 (total_screen) and EXP-024 (age) screens.
