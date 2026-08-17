# Playground S6E8 — Persistent ML Research Environment

Kaggle Playground Series **S6E8** (*Predicting Smartphone Addiction*).
Binary classification, metric **ROC-AUC**.

This repository is the persistent home of an autonomous ML research program.
It was migrated from the Hugging Face dataset
[`cazyundee/PlaygroundS6E8`](https://huggingface.co/datasets/cazyundee/PlaygroundS6E8).

## Layout

```text
train.csv               # 691,369 rows × 14 cols (features + addicted_label)
test.csv                # 296,302 rows × 13 cols
sample_submission.csv   # id, addicted_label
agent/                  # persistent research state + reusable code
```

The `agent/` directory is the research system:

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

## Resume the research

```bash
pip install lightgbm scikit-learn pandas numpy scipy
python -u agent/train_pipeline.py --out submission_matched_super.csv
```

Read `agent/RESEARCH_STATE.md` and `agent/DECISIONS.md` first to recover the
current understanding; search `agent/HISTORY.md` for historical detail.

## Data

Synthetic tabular data from the Kaggle Playground Series. Features cover
demographics, screen-time usage, sleep, engagement counts, and three
categorical fields. See `agent/RESEARCH_STATE.md` for the discovered
generator structure (including the zero-violation constraint behind the
promoted `other_screen_time` feature).

## License

Refer to the original [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e8)
for data terms.
