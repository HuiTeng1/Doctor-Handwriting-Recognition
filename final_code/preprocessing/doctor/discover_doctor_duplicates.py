"""
Doctor dataset duplicate discovery: within-source dedup -> cross-source dedup ->
classify each duplicate group as "equivalent" (same image, labels agree once normalized -
auto-resolvable) or "conflict" (same image, labels genuinely disagree - needs a human to
pick). Read-only against doctor_raw/ - doesn't write anything to doctor_clean/images/ or
labels.csv, doesn't decide what the final dataset contains. That's
resolve_doctor_duplicates.py's job, reading this script's output.

Input: doctor_raw/labels.csv + images/ (combineDataset.py's merged-but-not-deduped output)
Output: doctor_clean/{duplicate_groups.csv, duplicate_label_conflicts.csv, near_duplicates.csv}
"""
import hashlib
import os
import re

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "..", "data", "doctor", "doctor_raw")
SRC_IMG_DIR = os.path.join(SRC_DIR, "images")
OUT_DIR = os.path.join(PROJECT_ROOT, "..", "data", "doctor", "doctor_clean")


def content_hash(path):
    """Decode to a pixel array before hashing, so it catches "same image, different
    encoding/file format" duplicates too."""
    arr = np.array(Image.open(path).convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def normalize_label(text):
    """Only used to judge whether two labels are "semantically equivalent" - doesn't
    change the original label text that gets kept."""
    t = str(text).strip().lower()
    t = re.sub(r"[\s\-_]+", " ", t)  # collapse whitespace/hyphens/underscores into a single space
    t = re.sub(r"[^\w\s]", "", t)     # strip minor punctuation
    return t.strip()


def dedup_group(df, group_col, hash_col, label_col, filename_col):
    """Group df by group_col x hash_col: singletons pass straight through; a group with
    equivalent labels keeps its first row and logs the rest as "equivalent"; a group with
    genuinely conflicting labels contributes nothing to kept_df and is logged as
    "conflict" instead - resolving those is left to resolve_doctor_duplicates.py.
    Returns (kept_df, duplicate_groups_rows, conflict_rows)."""
    kept_rows = []
    dup_group_rows = []
    conflict_rows = []
    group_counter = 0

    for (group_key, h), sub in df.groupby([group_col, hash_col], sort=False):
        if len(sub) == 1:
            kept_rows.append(sub.iloc[0])
            continue

        group_counter += 1
        group_id = f"group_{group_counter:04d}"
        norm_labels = sub[label_col].map(normalize_label)

        if norm_labels.nunique() == 1:
            # Case A: same image, semantically equivalent labels (different formatting) - keep the first row (original row order)
            first = sub.iloc[0]
            kept_rows.append(first)
            dup_group_rows.append({
                "group_id": group_id,
                "kept_image": first[filename_col],
                "removed_images": ";".join(sub.iloc[1:][filename_col]),
                "label_status": "equivalent",
                "original_labels": ";".join(sub[label_col].astype(str)),
            })
        else:
            # Case B: same image, different semantic labels - exclude the whole group, report as a conflict, don't guess
            dup_group_rows.append({
                "group_id": group_id,
                "kept_image": "",
                "removed_images": ";".join(sub[filename_col]),
                "label_status": "conflict",
                "original_labels": ";".join(sub[label_col].astype(str)),
            })
            rows = sub.to_dict("records")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    conflict_rows.append({
                        "duplicate_group_id": group_id,
                        "image_1": rows[i][filename_col],
                        "image_2": rows[j][filename_col],
                        "label_1": rows[i][label_col],
                        "label_2": rows[j][label_col],
                        "reason": "conflicting labels for identical image",
                    })

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    return kept_df, dup_group_rows, conflict_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(SRC_DIR, "labels.csv"))
    original_count = len(df)
    per_source_original = df["source"].value_counts().to_dict()

    print("[discover] Computing image content hashes...")
    df["content_hash"] = [content_hash(os.path.join(SRC_IMG_DIR, fn)) for fn in df["filename"]]

    # Stage 1: within-source dedup
    print("[discover] Grouping within each source...")
    df["_within_key"] = df["source"]
    stage1_df, dup_rows_1, conflict_rows_1 = dedup_group(
        df, "_within_key", "content_hash", "text", "filename"
    )

    # Stage 2: cross-source dedup (all data shares one key, compared across sources)
    print("[discover] Grouping across sources...")
    stage1_df["_cross_key"] = "ALL"
    final_df, dup_rows_2, conflict_rows_2 = dedup_group(
        stage1_df, "_cross_key", "content_hash", "text", "filename"
    )

    kept_filenames = final_df["filename"].tolist()

    all_dup_groups = dup_rows_1 + dup_rows_2
    pd.DataFrame(all_dup_groups).to_csv(os.path.join(OUT_DIR, "duplicate_groups.csv"), index=False)

    all_conflicts = conflict_rows_1 + conflict_rows_2
    pd.DataFrame(all_conflicts).to_csv(os.path.join(OUT_DIR, "duplicate_label_conflicts.csv"), index=False)

    # Near-duplicates (perceptual hash): report only, don't remove. Computed against the
    # kept (post-equivalent-dedup, pre-conflict-resolution) set, read straight from
    # doctor_raw/ - resolve_doctor_duplicates.py hasn't copied anything into
    # doctor_clean/images/ yet at this point.
    print("[discover] Computing perceptual hashes (near-duplicate detection, report only)...")
    phashes = {fn: imagehash.phash(Image.open(os.path.join(SRC_IMG_DIR, fn)).convert("RGB")) for fn in kept_filenames}

    near_dup_rows = []
    # A simple hex-distance-<=-threshold nearest-neighbor comparison only within a shared
    # bucket prefix; at scale, degenerating to a full pairwise comparison is infeasible.
    # Coarse filter here: only pairs sharing the same first-4-hex-digit prefix get a real
    # Hamming distance computed, which keeps the scale manageable.
    prefix_buckets = {}
    for fn in kept_filenames:
        prefix_buckets.setdefault(str(phashes[fn])[:4], []).append(fn)
    for _, group in prefix_buckets.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dist = phashes[group[i]] - phashes[group[j]]
                if 0 < dist <= 5:
                    near_dup_rows.append({
                        "image_1": group[i], "image_2": group[j], "hamming_distance": dist,
                    })
    pd.DataFrame(near_dup_rows).to_csv(os.path.join(OUT_DIR, "near_duplicates.csv"), index=False)

    label_conflict_count = len(all_conflicts)
    equivalent_count = sum(1 for r in all_dup_groups if r["label_status"] == "equivalent")
    conflict_group_count = sum(1 for r in all_dup_groups if r["label_status"] == "conflict")
    report_lines = [
        f"Original images: {original_count}",
        f"Per-source original counts: {per_source_original}",
        f"Duplicate groups (within-source): {len(dup_rows_1)}",
        f"Duplicate groups (cross-source): {len(dup_rows_2)}",
        f"Duplicate groups total: {len(all_dup_groups)} ({equivalent_count} equivalent, {conflict_group_count} conflict)",
        f"Label conflict pairs: {label_conflict_count}",
        f"Would keep automatically (pending manual conflict resolution): {len(kept_filenames)}",
        f"Near-duplicate pairs reported (not removed): {len(near_dup_rows)}",
    ]
    print("\n" + "\n".join(report_lines))
    print(f"\n[discover] Wrote duplicate_groups.csv, duplicate_label_conflicts.csv, near_duplicates.csv to: {OUT_DIR}")
    print("[discover] Run resolve_doctor_duplicates.py next to actually write doctor_clean/images/ + labels.csv")


if __name__ == "__main__":
    main()
