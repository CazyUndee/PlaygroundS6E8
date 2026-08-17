# Research Goals

**Competition**: Kaggle Playground Series S6E8 — *Predicting Smartphone Addiction*
**Task**: Binary classification; metric = ROC-AUC.
**Data**: 691,369 train rows × 13 features, 296,302 test rows (no public leaderboard labels in repo).

## Overall objective
Maximize the ROC-AUC of the final prediction for the smartphone-addiction
classification task, while continuously discovering, testing, and preserving
reliable knowledge about the problem.

## Strategic priorities (in order)
1. **Reproducibility of the champion.** A previous research cycle permanently
   lost the exact script behind its best score. Every canonical score must now
   have its exact pipeline committed to the persistent repository.
2. **Feature knowledge.** Understand which features capture the underlying
   data-generation structure. The `other_screen_time` residual (from the
   hard constraint `daily_screen_time >= social + gaming + work`) was the
   single highest-value feature discovery to date.
3. **Model architecture.** Confirm GBDT (LightGBM) as the champion family and
   exploit ensemble diversity only where it is measured to be real.
4. **Validation discipline.** Trust only out-of-fold ROC-AUC under consistent
   fold partitions; distinguish OOF from leaderboard performance.
5. **Dataset forensics.** Synthetic Playground data hides generator structure
   (quantization, bounds, hard constraints, missingness). Search for and then
   *test* such structure rather than assuming an artifact is useful.

## Constraints
- No Kaggle API credentials available: leaderboard AUC cannot be measured;
  rely on OOF estimates and document the caveat.
- **All compute runs on GitHub Actions** (ubuntu-latest, 4 vCPU) via
  `.github/workflows/research.yml`; the local machine is orchestration only
  (push, trigger, download artifacts, curate results, commit). One training
  job at a time; experiments are triggered explicitly with `gh workflow run`.
- GitHub is the persistent home of the research (migrated from Hugging Face
  `cazyundee/PlaygroundS6E8`). The repo must stay a clean, understandable
  representation of the research — not a scratch dump; large artifacts live
  as workflow artifacts (90-day retention), small results are curated into
  `results/` and the memory files.
