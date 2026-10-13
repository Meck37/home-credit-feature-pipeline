"""Create, verify and load the frozen train / hold-out split.

Run once from the repo root:
    python -m src.split

Later stages never re-split. They import the loaders:
    from src.split import load_train_ids, load_holdout_ids
"""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# --- Settings ---------------------------------------------------------------
# Paths are built from this file's location, so the script works on any
# machine and from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_FILE = SPLIT_DIR / "train_ids.csv"
HOLDOUT_FILE = SPLIT_DIR / "holdout_ids.csv"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
EXPECTED_ROWS = 307_511
HOLDOUT_SIZE = 0.2
RATE_TOLERANCE = 0.001  # max gap in default rate: 0.1 percentage points


# --- Loaders (used by later stages) -----------------------------------------
def read_ids(path: Path) -> pd.Series:
    """Read one saved ID file as a sorted Series of SK_ID_CURR."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.split` from the repo root first."
        )
    return pd.read_csv(path)[ID_COL]


def load_train_ids() -> pd.Series:
    return read_ids(TRAIN_FILE)


def load_holdout_ids() -> pd.Series:
    return read_ids(HOLDOUT_FILE)


# --- Split steps -------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    """Read only the ID and TARGET columns of application_train."""
    return pd.read_csv(RAW_FILE, usecols=[ID_COL, TARGET_COL])


def check_input(data: pd.DataFrame) -> None:
    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows but found {len(data):,}. "
            "Is this the correct application_train.csv?"
        )
    if not data[ID_COL].is_unique:
        n_dupes = data[ID_COL].duplicated().sum()
        raise ValueError(f"{ID_COL} should be unique; found {n_dupes:,} duplicates.")
    target_values = set(data[TARGET_COL].unique())
    if target_values != {0, 1}:
        raise ValueError(f"{TARGET_COL} should be 0/1 only; found {sorted(target_values)}.")
    print(f"Input checks passed: {len(data):,} rows, unique IDs, binary TARGET.")


def make_split(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Stratified 80/20 split. Returns sorted ID Series for train and hold-out."""
    train_ids, holdout_ids = train_test_split(
        data[ID_COL],
        test_size=HOLDOUT_SIZE,
        random_state=config.SEED,
        stratify=data[TARGET_COL],
    )
    # Sorting makes the saved files identical on every run.
    return (
        train_ids.sort_values().reset_index(drop=True),
        holdout_ids.sort_values().reset_index(drop=True),
    )


def check_split(data: pd.DataFrame, train_ids: pd.Series, holdout_ids: pd.Series) -> None:
    # 1. Counts add up
    if len(train_ids) + len(holdout_ids) != len(data):
        raise ValueError(
            f"Train ({len(train_ids):,}) + hold-out ({len(holdout_ids):,}) "
            f"!= total ({len(data):,})."
        )

    # 2. No ID in both parts
    overlap = set(train_ids) & set(holdout_ids)
    if overlap:
        raise ValueError(f"{len(overlap):,} IDs in both parts, e.g. {sorted(overlap)[:5]}.")

    # 3. Default rate preserved in each part
    overall_rate = data[TARGET_COL].mean()
    print(f"Overall:  {len(data):>7,} rows, default rate {overall_rate:.4%}")
    for name, ids in [("Train", train_ids), ("Hold-out", holdout_ids)]:
        rate = data.loc[data[ID_COL].isin(ids), TARGET_COL].mean()
        if abs(rate - overall_rate) > RATE_TOLERANCE:
            raise ValueError(f"{name} default rate {rate:.4%} vs overall {overall_rate:.4%}.")
        print(f"{name + ':':<9} {len(ids):>7,} rows, default rate {rate:.4%}")

    print("Output checks passed: counts add up, no overlap, default rate preserved.")


def save_or_verify(train_ids: pd.Series, holdout_ids: pd.Series) -> None:
    """Write the split once. On later runs, verify it instead of overwriting."""
    exists = [TRAIN_FILE.exists(), HOLDOUT_FILE.exists()]

    if all(exists):
        same = read_ids(TRAIN_FILE).equals(train_ids) and read_ids(HOLDOUT_FILE).equals(holdout_ids)
        if not same:
            raise RuntimeError(
                "New split differs from the saved split. Saved files left untouched. "
                "Investigate (data file? seed? library version?) before changing the frozen split."
            )
        print("Split matches the saved files. Nothing written.")
    elif any(exists):
        raise RuntimeError("Only one split file exists. Investigate before regenerating.")
    else:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        # lineterminator="\n" gives byte-identical files on Windows, Mac and Linux.
        train_ids.to_csv(TRAIN_FILE, index=False, lineterminator="\n")
        holdout_ids.to_csv(HOLDOUT_FILE, index=False, lineterminator="\n")
        print(f"Wrote {TRAIN_FILE.relative_to(PROJECT_ROOT)} and {HOLDOUT_FILE.relative_to(PROJECT_ROOT)}")


def file_hash(path: Path) -> str:
    """SHA-256 fingerprint of a file, to record in DECISIONS.md."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = load_labels()
    check_input(data)
    train_ids, holdout_ids = make_split(data)
    check_split(data, train_ids, holdout_ids)
    save_or_verify(train_ids, holdout_ids)
    for path in (TRAIN_FILE, HOLDOUT_FILE):
        print(f"SHA-256 {path.name}: {file_hash(path)}")


if __name__ == "__main__":
    main()