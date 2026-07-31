"""
Split the combined Doctor dataset (data/Doctor/labels.csv) into
train / validation / test CSVs.

Run this AFTER combineDataset.py.

Ratios: train 70%, validation 10%, test 20%.

Images stay in one place (data/Doctor/images/); only the label CSV is split,
so train/val/test each just reference a subset of filenames.

Output (next to labels.csv):
    data/Doctor/train.csv
    data/Doctor/val.csv
    data/Doctor/test.csv
"""

import os
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_DIR = "data"
DOCTOR_FOLDER = os.path.join(DATA_DIR, "Doctor")
LABELS_CSV_PATH = os.path.join(DOCTOR_FOLDER, "labels.csv")

TRAIN_CSV_PATH = os.path.join(DOCTOR_FOLDER, "train.csv")
VAL_CSV_PATH = os.path.join(DOCTOR_FOLDER, "val.csv")
TEST_CSV_PATH = os.path.join(DOCTOR_FOLDER, "test.csv")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

RANDOM_SEED = 42


def split_dataset(labels_csv_path=LABELS_CSV_PATH,
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
