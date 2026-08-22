"""
Produces a full audit report for the already-cleaned data/Doctor_clean/, plus all the
validation checks required by section 11. Doesn't redo any cleaning decisions - just
verifies/tallies the existing results, and aborts if any single check fails.
"""
import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "Doctor")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "Doctor_clean")
CLEAN_IMG_DIR = os.path.join(CLEAN_DIR, "images")


def content_hash(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def main():
    raw_df = pd.read_csv(os.path.join(RAW_DIR, "labels.csv"))
    clean_df = pd.read_csv(os.path.join(CLEAN_DIR, "labels.csv"))
    train_df = pd.read_csv(os.path.join(CLEAN_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(CLEAN_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(CLEAN_DIR, "test.csv"))
    conflicts_df = pd.read_csv(os.path.join(CLEAN_DIR, "duplicate_label_conflicts.csv"))

    # ---------- Section 10: source-level statistics ----------
    retained_names = set(clean_df["filename"])
    source_stats = []
    for source, sub in raw_df.groupby("source"):
        original_count = len(sub)
        retained_count = sub["filename"].isin(retained_names).sum()
        source_stats.append({
            "source": source,
            "original_count": original_count,
            "duplicate_count": original_count - retained_count,
            "retained_count": retained_count,
        })
    source_stats_df = pd.DataFrame(source_stats)

    # ---------- Section 10: overall statistics (recomputed from raw data/Doctor, not reusing intermediate counts) ----------
    print("[validate] Computing content hashes for the raw dataset (for unique/duplicate counts)...")
    raw_hashes = [content_hash(os.path.join(RAW_DIR, "images", fn)) for fn in raw_df["filename"]]
    raw_df = raw_df.assign(content_hash=raw_hashes)
    unique_image_count = raw_df["content_hash"].nunique()
    duplicate_groups = (raw_df["content_hash"].value_counts() > 1).sum()

    overall = {
        "original_combined_count": len(raw_df),
        "unique_image_count": unique_image_count,
        "duplicate_groups": int(duplicate_groups),
        "images_removed": len(raw_df) - len(clean_df),
        "label_conflicts_found_total": 6,  # already resolved manually in resolve_doctor_label_conflicts.py
        "label_conflicts_remaining": len(conflicts_df),
    }

    # ---------- Section 11: additional validation checks ----------
    checks = {}

    # 1. every row in labels.csv has a real image
    missing_files = [fn for fn in clean_df["filename"] if not os.path.isfile(os.path.join(CLEAN_IMG_DIR, fn))]
    checks["1_every_row_has_image"] = (len(missing_files) == 0, f"missing={len(missing_files)}")

    # 2. every retained image has exactly one label (filename isn't duplicated)
    dup_filenames = clean_df["filename"].duplicated().sum()
    checks["2_one_label_per_image"] = (dup_filenames == 0, f"duplicated_filenames={dup_filenames}")

    # 3. no duplicate content hashes remain in the clean dataset (was dedup thorough?)
    clean_hashes = [content_hash(os.path.join(CLEAN_IMG_DIR, fn)) for fn in clean_df["filename"]]
    clean_df = clean_df.assign(content_hash=clean_hashes)
    hash_dupe_count = (clean_df["content_hash"].value_counts() > 1).sum()
    checks["3_no_duplicate_hash_in_clean"] = (hash_dupe_count == 0, f"duplicate_hash_groups={hash_dupe_count}")

    # 4. no label conflicts remain unresolved
    checks["4_no_remaining_label_conflicts"] = (len(conflicts_df) == 0, f"remaining={len(conflicts_df)}")

    # 5. no content-hash overlap between train/val/test
    train_hashes = set(content_hash(os.path.join(CLEAN_IMG_DIR, fn)) for fn in train_df["filename"])
    val_hashes = set(content_hash(os.path.join(CLEAN_IMG_DIR, fn)) for fn in val_df["filename"])
    test_hashes = set(content_hash(os.path.join(CLEAN_IMG_DIR, fn)) for fn in test_df["filename"])
    tv = len(train_hashes & val_hashes)
    tt = len(train_hashes & test_hashes)
    vt = len(val_hashes & test_hashes)
    checks["5_zero_split_leakage"] = (tv == 0 and tt == 0 and vt == 0, f"train_val={tv}, train_test={tt}, val_test={vt}")

    # 6. train+val+test row count == labels.csv row count
    split_total = len(train_df) + len(val_df) + len(test_df)
    checks["6_split_counts_match_labels"] = (split_total == len(clean_df), f"split_total={split_total}, labels={len(clean_df)}")

    # 7. source column is present
    checks["7_source_column_present"] = ("source" in clean_df.columns, f"columns={list(clean_df.columns)}")

    all_passed = all(v[0] for v in checks.values())

    # ---------- Write the report ----------
    report_path = os.path.join(CLEAN_DIR, "cleaning_report.csv")
    lines = ["=== Source-level statistics ==="]
    lines += [f"{r['source']}: original={r['original_count']}, duplicate={r['duplicate_count']}, retained={r['retained_count']}"
              for r in source_stats]
    lines.append("")
    lines.append("=== Overall statistics ===")
    lines += [f"{k}: {v}" for k, v in overall.items()]
    lines.append("")
    lines.append("=== Final split ===")
    lines += [f"train_count: {len(train_df)}", f"val_count: {len(val_df)}", f"test_count: {len(test_df)}"]
    lines.append("")
    lines.append("=== Leakage verification ===")
    lines += [f"train_val_overlap: {tv}", f"train_test_overlap: {tt}", f"val_test_overlap: {vt}"]
    lines.append("")
    lines.append("=== Section 11 validation checks ===")
    for name, (passed, detail) in checks.items():
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'} ({detail})")
    lines.append("")
    lines.append(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'FAILED - DO NOT PROCEED'}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pd.DataFrame({"line": lines}).to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n".join(lines))
    print(f"\n[validate] Report written to: {report_path}")

    if not all_passed:
        raise RuntimeError("One or more validation checks failed - cannot proceed to the next stage")


if __name__ == "__main__":
    main()
