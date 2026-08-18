# Playground S6E8 — Persistent ML Research Environment

Kaggle Playground Series **S6E8** (*Predicting Smartphone Addiction*).
Binary classification, metric **ROC-AUC**.

This repository is the persistent home of an autonomous ML research program.
It was migrated from the Hugging Face dataset
[`cazyundee/PlaygroundS6E8`](https://huggingface.co/datasets/cazyundee/PlaygroundS6E8).

All model training and heavy compute runs on **GitHub Actions** (see
`.github/workflows/research.yml`). The local machine is orchestration only:
edit code and research state, trigger jobs with `gh workflow run`, download
artifacts, curate results, and commit.

## Layout

```text
train.csv               # 691,369 rows × 14 cols (features + addicted_label)
test.csv                # 296,302 rows × 13 cols
sample_submission.csv   # id, addicted_label
requirements.txt        # python dependencies
source/                 # research source of truth: memory files + reusable code
state/                  # research run state / artifacts (per-experiment results, scripts)
.github/workflows/      # GitHub Actions compute jobs
```

### `source/` — the research source

| File | Purpose |
| :--- | :--- |
| `AGENT.md` | operating rules |
| `GOALS.md` | strategic objectives |
| `RESEARCH_STATE.md` | current understanding / champion |
| `HISTORY.md` | long-term searchable memory |
| `DECISIONS.md` | durable conclusions |
| `TODO.md` | research queue |
| `RUN_LOG.md` | execution facts |
| `experiments_registry.json` | structured experiment database |
| `checkpoint.json` | machine-readable resume state |
| `train_pipeline.py` | **canonical reproducible pipeline** (44 features, dual-seed 5-fold LightGBM) |
| `features.py` / `models.py` / `ensemble.py` / `utils.py` | reusable infrastructure |

### `state/` — research run state / artifacts

Committed per-run results and artifacts recovered from GitHub Actions runs:
per-experiment results (`state/results/<exp>/`, e.g. the EXP-023 champion
submission + results JSON), forensics scripts (`state/analysis/`), and ad-hoc
feature screens (`state/exp_screen_features.py`). A fresh agent can recover
the latest run from here without re-downloading workflow artifacts. It is
allowed to be looser than `source/`, but keep it organized per experiment and
avoid accumulating duplicate submissions.

## Run the research (GitHub Actions)

All model compute runs on GitHub Actions (D19) — the local machine is
orchestration only. Trigger experiments with the `gh` CLI:

```bash
# Canonical dual-seed 5-fold pipeline (the champion, EXP-023)
gh workflow run research-compute -f task=exp023_canonical
gh run watch

# Feature screen (lgbm_63, seed-42, 5-fold) vs the committed baseline
gh workflow run research-compute -f task=screen -f feature=total_screen
# feature options: total_screen, sm_weekend, all3, age, both

# Hyperparameter scan and feature ablation
gh workflow run research-compute -f task=tune
gh workflow run research-compute -f task=ablate

# Import/feature smoke test (fast sanity check after code moves)
gh workflow run research-compute -f task=smoke

# Reproduce the pandas forensics
gh workflow run research-compute -f task=forensics
```

Collect results:

```bash
gh run list --workflow=research-compute
gh run download <run_id> -n <artifact-name> -D artifacts/
```

Then curate small results into `state/results/<exp>/`, update the memory
files in `source/`, and commit. Large OOF arrays live as workflow artifacts
(90-day retention); the exact scripts are always in the repo (D14).

Read `source/RESEARCH_STATE.md` and `source/DECISIONS.md` first to recover
the current understanding; search `source/HISTORY.md` for historical detail.
The entry prompt for a fresh agent lives in `PROMPT.md`.

## Data

Synthetic tabular data from the Kaggle Playground Series. Features cover
demographics, screen-time usage, sleep, engagement counts, and three
categorical fields. See `source/RESEARCH_STATE.md` for the discovered
generator structure (including the zero-violation constraint behind the
promoted `other_screen_time` feature).

## License

Refer to the original [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e8)
for data terms.
