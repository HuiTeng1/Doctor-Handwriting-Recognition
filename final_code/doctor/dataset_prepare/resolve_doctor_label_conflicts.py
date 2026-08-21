"""
6 label conflicts that were manually verified, with which image/label to keep hardcoded
(confirmed by eye against the image, not an algorithmic guess). Writes the result back to
data/Doctor_clean/labels.csv and removes the resolved entries from
duplicate_label_conflicts.csv. Re-run split_doctor_clean.py after this.
"""
import os
import shutil

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..")
SRC_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "Doctor", "images")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "Doctor_clean")
CLEAN_IMG_DIR = os.path.join(CLEAN_DIR, "images")

# (duplicate_group_id, filename to keep, label to keep) - verified by eye against the image, not automatic
RESOLUTIONS = [
    ("group_0005", "img_690.jpg", "Calbo-D"),
    ("group_0008", "img_889.jpg", "Carulcal D"),
    ("group_0023", "img_5593.jpg", "Xyril"),
    ("group_0047", "img_6296.png", "Opton"),
    ("group_0009", "img_7688.png", "Fexo"),
    ("group_0030", "img_6386.png", "Sergel"),
]


def main():
    labels_path = os.path.join(CLEAN_DIR, "labels.csv")
    conflicts_path = os.path.join(CLEAN_DIR, "duplicate_label_conflicts.csv")

    labels_df = pd.read_csv(labels_path)
    conflicts_df = pd.read_csv(conflicts_path)

    # source column: looked up from the original data/Doctor/labels.csv to keep the info complete
    orig_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "Doctor", "labels.csv"))
    source_map = dict(zip(orig_df["filename"], orig_df["source"]))

    resolved_group_ids = set()
    for group_id, filename, label in RESOLUTIONS:
        shutil.copy2(os.path.join(SRC_IMG_DIR, filename), os.path.join(CLEAN_IMG_DIR, filename))
        # Drop any existing row for this filename first, so re-running this script is
        # idempotent instead of appending a duplicate row each time.
        labels_df = labels_df[labels_df["filename"] != filename]
        labels_df = pd.concat([labels_df, pd.DataFrame([{
            "filename": filename, "text": label, "source": source_map.get(filename, "unknown"),
        }])], ignore_index=True)
        resolved_group_ids.add(group_id)

    labels_df.to_csv(labels_path, index=False)

    remaining_conflicts = conflicts_df[~conflicts_df["duplicate_group_id"].isin(resolved_group_ids)]
    remaining_conflicts.to_csv(conflicts_path, index=False)

    print(f"[resolve] Resolved {len(RESOLUTIONS)} conflicts, labels.csv now has {len(labels_df)} rows")
    print(f"[resolve] {len(remaining_conflicts)} conflicts remain unresolved in duplicate_label_conflicts.csv")
    print("[resolve] Please re-run dataset_prepare/split_doctor_clean.py to update train/val/test")


if __name__ == "__main__":
    main()
