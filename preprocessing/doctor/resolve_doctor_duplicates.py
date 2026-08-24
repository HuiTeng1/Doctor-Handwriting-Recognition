"""
Doctor dataset duplicate resolution: reads discover_doctor_duplicates.py's output
(duplicate_groups.csv, duplicate_label_conflicts.csv) and actually writes
doctor_clean/{images/, labels.csv}. Two separate passes, since the two cases resolve
differently and shouldn't be forced into one generic loop:

  Pass 1 (automatic) - every row that ISN'T part of a "conflict" group: singletons, plus
    the first row of each "equivalent" group (same image, same label just formatted
    differently). Copied straight across from doctor_raw/ - no human judgment involved.

  Pass 2 (manual) - the handful of "conflict" groups (same image, genuinely different
    labels) discover_doctor_duplicates.py couldn't auto-resolve. RESOLUTIONS below is a
    hardcoded list of which image/label to keep for each, confirmed by eye against the
    actual image - not an algorithmic guess.

Both passes' rows get written into the same labels.csv in one go. Re-running this script
is idempotent: it doesn't re-copy/duplicate anything, and rewrites
duplicate_label_conflicts.csv to drop whichever conflicts RESOLUTIONS covers.

Input: doctor_raw/{labels.csv, images/} + doctor_clean/{duplicate_groups.csv,
       duplicate_label_conflicts.csv} (from discover_doctor_duplicates.py)
Output: doctor_clean/{images/, labels.csv}, and a rewritten duplicate_label_conflicts.csv
"""
import os
import shutil

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "..", "data", "doctor", "doctor_raw")
SRC_IMG_DIR = os.path.join(SRC_DIR, "images")
OUT_DIR = os.path.join(PROJECT_ROOT, "..", "data", "doctor", "doctor_clean")
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")

# (duplicate_group_id, filename to keep, label to keep) - verified by eye against the
# image, not automatic. Keyed by group_id + filename, not row position, so it still
# applies correctly even if discover_doctor_duplicates.py is re-run from scratch.
RESOLUTIONS = [
    ("group_0005", "img_690.jpg", "Calbo-D"),
    ("group_0008", "img_889.jpg", "Carulcal D"),
    ("group_0023", "img_5593.jpg", "Xyril"),
    ("group_0047", "img_6296.png", "Opton"),
    ("group_0009", "img_7688.png", "Fexo"),
    ("group_0030", "img_6386.png", "Sergel"),
]


def resolve_equivalent_and_singletons(raw_df, groups_df):
    """Pass 1: every row except the ones belonging to a "conflict" group.
    Returns (auto_df, conflict_images)."""
    equivalent = groups_df[groups_df["label_status"] == "equivalent"]
    removed_equivalent = set()
    for cell in equivalent["removed_images"]:
        removed_equivalent.update(str(cell).split(";"))

    conflict = groups_df[groups_df["label_status"] == "conflict"]
    conflict_images = set()
    for cell in conflict["removed_images"]:
        conflict_images.update(str(cell).split(";"))

    kept = raw_df[~raw_df["filename"].isin(removed_equivalent) & ~raw_df["filename"].isin(conflict_images)]
    return kept[["filename", "text", "source"]].reset_index(drop=True), conflict_images


def resolve_conflicts(raw_df, conflicts_path):
    """Pass 2: apply the manually-verified RESOLUTIONS list.
    Returns (manual_df, remaining_conflicts_df)."""
    source_map = dict(zip(raw_df["filename"], raw_df["source"]))
    rows = []
    resolved_group_ids = set()
    for group_id, filename, label in RESOLUTIONS:
        rows.append({"filename": filename, "text": label, "source": source_map.get(filename, "unknown")})
        resolved_group_ids.add(group_id)
    manual_df = pd.DataFrame(rows, columns=["filename", "text", "source"])

    conflicts_df = pd.read_csv(conflicts_path)
    remaining = conflicts_df[~conflicts_df["duplicate_group_id"].isin(resolved_group_ids)]
    return manual_df, remaining


def main():
    groups_path = os.path.join(OUT_DIR, "duplicate_groups.csv")
    conflicts_path = os.path.join(OUT_DIR, "duplicate_label_conflicts.csv")
    if not os.path.isfile(groups_path):
        raise FileNotFoundError(f"{groups_path} not found - run discover_doctor_duplicates.py first")

    raw_df = pd.read_csv(os.path.join(SRC_DIR, "labels.csv"))
    groups_df = pd.read_csv(groups_path)

    print("[resolve] Pass 1/2: singletons + equivalent-group survivors...")
    auto_df, conflict_images = resolve_equivalent_and_singletons(raw_df, groups_df)
    print(f"[resolve] {len(auto_df)} rows resolved automatically, "
          f"{len(conflict_images)} images held back pending manual conflict resolution")

    print("[resolve] Pass 2/2: manually-verified conflict resolutions...")
    manual_df, remaining_conflicts = resolve_conflicts(raw_df, conflicts_path)
    print(f"[resolve] {len(manual_df)} rows resolved manually")

    final_df = pd.concat([auto_df, manual_df], ignore_index=True)
    # Idempotency guard: harmless no-op in the normal case (pass 1 already excludes every
    # conflict-group image, so pass 2's filenames never legitimately collide with pass 1's).
    final_df = final_df.drop_duplicates(subset="filename", keep="last").reset_index(drop=True)

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    for fn in final_df["filename"]:
        shutil.copy2(os.path.join(SRC_IMG_DIR, fn), os.path.join(OUT_IMG_DIR, fn))
    final_df.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)

    remaining_conflicts.to_csv(conflicts_path, index=False)

    print(f"[resolve] Clean images: {len(final_df)}")
    print(f"[resolve] Conflicts still unresolved: {len(remaining_conflicts)}")
    print(f"[resolve] Wrote images + labels.csv to: {OUT_DIR}")


if __name__ == "__main__":
    main()
