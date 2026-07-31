"""
Clean + split the Normal dataset (data/Normal/labels.csv) into
train / validation / test CSVs.

Steps:
    1. Load labels.csv (columns: FILENAME, IDENTITY).
    2. Delete rows whose IDENTITY is null OR contains characters
       outside the 70-class dictionary (e.g. the apostrophe ').
       The matching image file is removed from disk too.
    3. Split what's left into train 70% / val 10% / test 20%.

Output (next to labels.csv):
    data/Normal/train.csv
    data/Normal/val.csv
    data/Normal/test.csv
"""

import os
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_DIR = "data"
NORMAL_FOLDER = os.path.join(DATA_DIR, "Normal")
IMAGES_DIR = os.path.join(NORMAL_FOLDER, "images", "test")
LABELS_CSV_PATH = os.path.join(NORMAL_FOLDER, "labels.csv")

TRAIN_CSV_PATH = os.path.join(NORMAL_FOLDER, "train.csv")
VAL_CSV_PATH = os.path.join(NORMAL_FOLDER, "val.csv")
TEST_CSV_PATH = os.path.join(NORMAL_FOLDER, "test.csv")

FILENAME_COL = "FILENAME"
TEXT_COL = "IDENTITY"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

RANDOM_SEED = 42

# The 70-class dictionary. Any label containing a character outside this
# set is considered invalid and will be dropped (with its image deleted).
ALLOWED_CHARS = set(
    " %()-./?0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)


def _is_valid_label(s):
    """True only if s is a non-empty string using only allowed characters."""
    return isinstance(s, str) and len(s) > 0 and all(c in ALLOWED_CHARS for c in s)


def drop_bad_rows(df, images_dir=IMAGES_DIR,
                  filename_col=FILENAME_COL, text_col=TEXT_COL):
    """
    Delete the image file + drop the row for every entry whose label is
    null OR contains characters outside the 70-class dictionary.
    """
    # A row is bad if the label is NaN, or not a valid (in-dictionary) string.
    def _bad(v):
        if pd.isna(v):
            return True
        return not _is_valid_label(str(v))

    bad_mask = df[text_col].apply(_bad)
    bad_rows = df[bad_mask]

    deleted, missing = 0, 0
    for filename in bad_rows[filename_col]:
        image_path = os.path.join(images_dir, filename)
        if os.path.isfile(image_path):
            os.remove(image_path)
            deleted += 1
        else:
            missing += 1

    print(f"[INFO] Bad '{text_col}' rows (null or invalid chars): {len(bad_rows)} "
          f"(images deleted: {deleted}, already missing: {missing})")

    return df[~bad_mask].reset_index(drop=True)


def split_dataset(labels_csv_path=LABELS_CSV_PATH,
                  images_dir=IMAGES_DIR,
                  train_csv_path=TRAIN_CSV_PATH,
                  val_csv_path=VAL_CSV_PATH,
                  test_csv_path=TEST_CSV_PATH,
                  train_ratio=TRAIN_RATIO,
                  val_ratio=VAL_RATIO,
                  test_ratio=TEST_RATIO,
                  random_seed=RANDOM_SEED):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, \
        "train_ratio + val_ratio + test_ratio must add up to 1.0"

    df = pd.read_csv(labels_csv_path)
    before = len(df)

    df = drop_bad_rows(df, images_dir=images_dir)
    print(f"[INFO] Rows after cleaning: {len(df)}/{before}")

    if len(df) != before:
        df.to_csv(labels_csv_path, index=False)
        print(f"[INFO] Rewrote {labels_csv_path} without the dropped rows.")

    # Shuffle once, then cut into three contiguous blocks.
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"[INFO] Total rows: {n}")
    print(f"[INFO] train: {len(train_df)} ({len(train_df) / n:.1%}) -> {train_csv_path}")
    print(f"[INFO] val:   {len(val_df)} ({len(val_df) / n:.1%}) -> {val_csv_path}")
    print(f"[INFO] test:  {len(test_df)} ({len(test_df) / n:.1%}) -> {test_csv_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    split_dataset()