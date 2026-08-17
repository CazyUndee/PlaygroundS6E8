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
- (pending) — EXP-023 completion + result.
