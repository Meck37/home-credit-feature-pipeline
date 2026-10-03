# DECISIONS.md — Home Credit Feature Pipeline

Running log of *why*. Code lives in `src/`; reasoning lives here.

## Fixed settings
- Seed: 47 (defined once in `src/config.py`)
- Python version: 3.13.14
- Split: stratified 80/20 on TARGET; ID files set in Stage 2
- CV: 5-fold StratifiedKFold
- Model: HistGradientBoostingClassifier, default settings

---

## Stage 0 — Repo setup and reproducibility

**What I did**
- Set up the repo structure: `src/`, `tests/`, `notebooks/`, `docs/`, `reports/`.
- Wrote the README with the exact steps to reproduce the environment and get the data.
- Pinned dependencies with a full `pip freeze` into `requirements.txt`.

**Decisions and why**
- **Seed 47:** arbitrary, fixed before any results were produced, and defined once in `src/config.py` so the split, CV folds and model all use the same value.
- **Python 3.13:** a mature release that is still supported, with full support from pandas, NumPy and scikit-learn.
- **Full `pip freeze`:** pins every package, including sub-dependencies, so a fresh install matches my environment exactly.
- **Data not committed:** mainly licensing, because Kaggle's competition rules don't allow redistribution. The files are also several GB, too large for a git repo.
- **Logic in `src/`, not notebooks:** notebooks are for exploration and reporting only. Code in `src/` can be imported and unit-tested (a notebook cell can't be), it keeps notebooks from drifting into different versions of the same feature, and the same function can serve both training and production.