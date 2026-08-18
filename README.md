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
state/                  # scratch from the last agent run (analysis, results, ad-hoc scripts)
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

### `state/` — scratch / leftovers

Disposable artifacts from previous runs: throwaway forensics scripts
(`state/analysis/`), per-experiment results (`state/results/`), and ad-hoc
feature-screen scripts (`state/exp_screen_features.py`). It is allowed to be
messy and is not part of the clean research notebook.

## Run the research (GitHub Actions)

```bash
# regenerate the canonical champion (EXP-023)
gh workflow run research.yml -f task=exp023_canonical
gh run watch
gh run download <run-id>
```

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
