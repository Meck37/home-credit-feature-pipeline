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


## Stage 1 — Frame the feature problem

**What I did:**
- Read `HomeCredit_columns_description.csv` for the three satellite tables (`bureau`, `previous_application`, `installments_payments`).
- Inspected each table with `head(10)` in `notebooks/00_data_exploration.ipynb` to confirm what one row represents and which ID columns link the tables.
- Defined the prediction moment, the grain, and the three feature families before designing any features.

**What I decided, and why:**

*Prediction moment.* At the moment an applicant submits their loan application, predict the probability that they default, using only information available at that moment. This is the leakage rule every feature is tested against in Stage 5.

*Grain.* One row per applicant (`SK_ID_CURR`). The model predicts default per applicant and `TARGET` sits at that level, so every feature must describe one applicant on one row. The satellite tables are one-to-many, so each must be aggregated down to `SK_ID_CURR` before joining. Joining raw rows would duplicate applicants and inflate the training data.

| Table | One row = | Row ID | Link to applicant |
|---|---|---|---|
| `application_train` | One current applicant / application | `SK_ID_CURR` | (main table) |
| `bureau` | One credit the applicant held with another lender, as reported by the credit bureau | `SK_ID_BUREAU` | `SK_ID_CURR` |
| `previous_application` | One earlier application the applicant made to Home Credit (approved, refused, cancelled or unused) | `SK_ID_PREV` | `SK_ID_CURR` |
| `installments_payments` | One payment toward an installment on a past Home Credit loan; one installment can span several rows if paid in parts | (none; installment = `SK_ID_PREV` + `NUM_INSTALMENT_NUMBER`) | `SK_ID_PREV` → `SK_ID_CURR` |

Some applicants have zero rows in a satellite table. How they are treated (no history vs. a value of 0) is a Stage 3 decision.

*Feature families.*

| Family | Business reasoning | Expected direction | Source |
|---|---|---|---|
| Repayment track record | How a person has repaid before is direct evidence of how they repay. | More / worse lateness → higher risk | `installments_payments`; overdue fields in `bureau` |
| Current debt burden | Owing money is normal; what matters is how much they owe relative to their credit limits and income. A stretched borrower has less room to absorb a new payment. | Higher debt relative to limits / income → higher risk | Active credits and debt in `bureau`; credit-to-income from `application_train` |
| Credit-seeking behaviour | Many recent applications, or refusals by lenders, may signal financial distress or that others already judged the person risky. | More applications / refusals → higher risk | `previous_application`; recency of new credits in `bureau` |

**What I checked, and the result (evidence):**
- Confirmed the row meaning and linking IDs for each table from `head(10)` and the column descriptions (table above).
- Confirmed from the column descriptions that `DAYS_*` columns are counted relative to the current application date: 0 = application day, negative = before, positive = after. The `DAYS_CREDIT` values seen were negative (credits opened before the application).
- Not yet checked: the min/max of every `DAYS_*` column and the number of applicants with no history in each table. Deferred to Stage 3 exploration.

**What surprised me / open questions:**
- `DAYS_*` sign convention was initially unclear; resolved (see above).
- Positive `DAYS_*` values (e.g. `DAYS_CREDIT_ENDDATE`): are they *planned* dates known at application time, or events that *happened* after it? Needed for the Stage 5 temporal-leakage check.
- `DAYS_CREDIT_ENDDATE` vs `DAYS_ENDDATE_FACT`: planned end vs actual end. Confirm which one is safe to use.
- The 365243 placeholder value (in `DAYS_EMPLOYED` and some `previous_application` date columns): what it means and how to treat it.
- `installments_payments`: an installment paid in parts appears on several rows. Decide in Stage 3/4 whether lateness and underpayment are counted per row or per installment.

**Gate answers:**
- **Q1 — Why should past payment lateness predict future default, and why might it fail?**
  It should predict default because lateness shows the person has already struggled to pay on time, and repeated struggle can escalate into default.
  It might fail because: (a) lateness can have causes unrelated to ability to pay, such as a one-off shock, payment-channel delays, or paying a few days after the due date out of habit; (b) the past may not represent the present, since income or circumstances may have improved or worsened since those loans; (c) applicants with only a few past installments give little evidence, so one late payment can swing a lateness share heavily; (d) the history exists only for people Home Credit previously approved, so it doesn't describe applicants who were refused.
- **Q2 — How are time columns like `DAYS_CREDIT` measured?**
  In days relative to the current application date. 0 = the day of application, negative = days before it (the past), positive = days after it (the future). A positive value is a leakage risk only if it records something that *happened* after the application; a date that was *planned* and known at application time (such as a scheduled end date) can be acceptable. Each such column is checked in Stage 5.
- **Q3 — Who uses these features, and how does that change naming and documentation?**
  Both data scientists (who build and reuse them) and credit/model-risk reviewers (who approve them). So each feature gets a consistent, descriptive code name (`table_prefix_measure`, e.g. `bureau_active_debt_ratio`) that a data scientist can read, plus a catalog entry with a one-sentence business meaning, source, transformation, and edge rules that a credit officer can read without opening the code. The name alone can't carry all that; the catalog is where the reviewer gets the full definition.


## Stage 2 — Baseline model

**What I did:**
- Wrote `src/split.py`, which creates the train / hold-out split once, saves the ID lists, and provides loaders (`load_train_ids`, `load_holdout_ids`) so later stages never re-split.
- Ran it twice to confirm the split is reproducible and that re-running cannot silently replace it.
- *Pending:* `src/evaluate.py`, baseline run in `01_baseline.ipynb`.

**What I decided, and why:**

*Split ratio — 80/20.* Total defaulters: ~24,825 (8.07% of 307,511). The 20% hold-out holds ~4,965 defaulters, enough for a stable AUC estimate; the 80% keeps ~19,860 defaulters for development. With a rare class, the number of positives, not rows, decides whether each part is large enough.

*Stratified on TARGET.* Only ~8% of applicants default, so a purely random split could shift the default rate between parts. Stratifying keeps both parts at the overall rate, so the hold-out represents the same population I develop on.

*Hold-out frozen, used once.* The hold-out is scored exactly once, at the end of Stage 6, for the final baseline and engineered models. It is not used for any feature design, selection or tuning decision before then. Every decision informed by it would leak it into development, and its score would stop being an honest estimate. Because the split is random, the hold-out estimates performance on unseen applicants from the same population and period, not on future applicants (see Stage 9 limitations: no out-of-time validation).

*ID files stored in `data/splits/` (gitignored, not committed).* The split is reproducible from: the same `application_train.csv` (same content and row order), seed 47 (`src/config.py`), and the same scikit-learn version (pinned in `requirements.txt`, version: ___). Two controls make this safe without committing the files:
1. **Verify, don't overwrite:** if the files exist, `split.py` regenerates the split and compares; it stops with an error if they differ and never replaces the saved files.
2. **Fingerprints:** SHA-256 hashes (below) let anyone confirm they hold the identical split. Files are written with `\n` line endings, so the hashes match on Windows, Mac and Linux.
These controls *detect* a changed split; they don't prevent one. Any mismatch must be investigated before the split is touched.

*Pending:* baseline feature set, model choice, CV results.

**What I checked, and the result (evidence):**

| Check | Result |
|---|---|
| Input: row count = 307,511 | Pass |
| Input: `SK_ID_CURR` unique | Pass |
| Input: `TARGET` only 0/1 | Pass |
| Counts: 246,008 + 61,503 = 307,511 | Pass |
| No ID in both parts | Pass |
| Default rate: overall 8.0729%, train 8.0729%, hold-out 8.0728% (within one applicant) | Pass |
| Reproducibility: second run regenerated an identical split ("Split matches the saved files. Nothing written."), same hashes | Pass |

Hashes:
- `train_ids.csv`: `853b604ce03a8940ac9e318c9c464f6569375b7a55c0cce7fc886cba730bbfae`
- `holdout_ids.csv`: `e3a1ee37f2ec813232aafa992814314807b7c50ee58d8b3445e6bd529a9e4b21`

**What surprised me / open questions:**
- pandas writes `\r\n` line endings on Windows by default, which would give different file hashes on different operating systems for identical IDs. Fixed by forcing `\n`.
- Open: how to guarantee identical CV folds across feature sets in Stage 6 (row order after `build.py` merges). To resolve in `evaluate.py`.

**Gate answers:** *Pending (after the baseline run).*
