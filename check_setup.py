"""
Checks this machine against every gitignored / hand-off-required path the project
depends on (see RUN_ORDER.md Stage 0) and reports what's missing. Read-only - doesn't
generate, download, or move anything. Run this right after `git clone` + whatever data
you were handed, before starting Stage 1, to see exactly what's still needed.

"Doctor raw zips", "Doctor raw source folders", and "Doctor derived image folders" are
ALTERNATIVES to each other, not all required - having any one complete group covers
Stage 2's needs even if the others are entirely missing (see combineDataset.py's own
comment for which path applies).

Usage:
    python check_setup.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(_HERE, "preprocessing", "doctor"))
from combineDataset import (  # noqa: E402
    SOURCE_FOLDERS, OUTPUT_FOLDER,
    _DOCTOR_ZIP_SEARCH_NAME, _RXHAND_ZIP_NAME, _BD_ZIP_NAME, _find_zip,
)


def count_files(path):
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


# (relative label for display, absolute path, kind, expected file count if a dir)
def rel(path):
    return os.path.relpath(path, _HERE)


CHECKS = [
    ("Raw datasets - IAM (uncleaned - place at data/normal/I_Am_Dataset/, don't pre-extract)", [
        (os.path.join(_HERE, "data", "normal", "I_Am_Dataset", "iam_words.zip"), "file", None),
        (os.path.join(_HERE, "data", "normal", "I_Am_Dataset", "words_new.txt"), "file", None),
    ]),
    ("Raw datasets - CRNN's own IAM split (no generator exists for this one - must be handed over as-is)", [
        (os.path.join(_HERE, "data", "normal", "crnn", "train.csv"), "file", None),
        (os.path.join(_HERE, "data", "normal", "crnn", "val.csv"), "file", None),
        (os.path.join(_HERE, "data", "normal", "crnn", "test.csv"), "file", None),
    ]),
    ("Raw datasets - TrOCR's own IAM split (regenerable for real via generate_trocr_split.py)", [
        (os.path.join(_HERE, "data", "normal", "trocr", "train.csv"), "file", None),
        (os.path.join(_HERE, "data", "normal", "trocr", "val.csv"), "file", None),
    ]),
    ("Doctor raw zips (ALTERNATIVE A - simplest: place these 3 zips directly in doctor_raw/, "
     "named after each dataset - combineDataset.py's ensure_source_folders_extracted() auto-unzips "
     "them on next run, verified to reproduce the checked-in labels.csv byte-for-byte)", [
        (_DOCTOR_ZIP_SEARCH_NAME, "zip", None),
        (_RXHAND_ZIP_NAME, "zip", None),
        (_BD_ZIP_NAME, "zip", None),
    ]),
    ("Doctor raw source folders (ALTERNATIVE B - already-extracted, e.g. from a prior run of "
     "ensure_source_folders_extracted() - combineDataset.py will use these directly)", [
        (f, "dir", None) for f in SOURCE_FOLDERS
    ]),
    ("Doctor derived image folders (ALTERNATIVE C - gitignored outputs, hand these over directly to skip Stage 2/3 regeneration)", [
        (os.path.join(OUTPUT_FOLDER, "images"), "dir", 10347),
        (os.path.join(_HERE, "data", "doctor", "doctor_clean", "images"), "dir", 10050),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "baseline"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "clhaa"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "fajardo"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "benitez"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor_test", "baseline"), "dir", 2010),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor_test", "clhaa"), "dir", 2010),
    ]),
    ("Checkpoint files over GitHub's 100MB limit (gitignored - needs a separate transfer, not git)", [
        (os.path.join(_HERE, "checkpoint", "normal", "trocr", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "baseline", "final", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "clhaa", "final", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "baseline", "final", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "clhaa", "final", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "fajardo", "final", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "benitez", "final", "model.safetensors"), "file", None),
    ]),
    ("Training completion markers (DONE.txt - written once a stage's training loop finishes; "
     "confirms both machines actually finished training, not just that git pull succeeded)", [
        (os.path.join(_HERE, "checkpoint", "normal", "crnn", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn", "baseline", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn", "clhaa", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn_pilot", "baseline", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn_pilot", "clhaa", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn_pilot", "fajardo", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn_pilot", "benitez", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "baseline", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "clhaa", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "baseline", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "clhaa", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "fajardo", "DONE.txt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "benitez", "DONE.txt"), "file", None),
    ]),
    ("CRNN checkpoints (git-tracked, ~21MB each - should come through git pull automatically; "
     "checked here anyway to catch an incomplete/partial pull)", [
        (os.path.join(_HERE, "checkpoint", "normal", "crnn", "last.ckpt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn", "baseline", "last.ckpt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "crnn", "clhaa", "last.ckpt"), "file", None),
    ]),
    ("Final result CSVs (git-tracked - the numbers actually cited in the report; "
     "presence here confirms both machines are looking at the same finished results)", [
        (os.path.join(_HERE, "data", "doctor", "results", "crnn_finetune_summary.csv"), "file", None),
        (os.path.join(_HERE, "data", "doctor", "results", "trocr_finetune_summary.csv"), "file", None),
        (os.path.join(_HERE, "data", "doctor", "results", "crnn_pilot_summary.csv"), "file", None),
        (os.path.join(_HERE, "data", "doctor", "results", "trocr_pilot_summary.csv"), "file", None),
        (os.path.join(_HERE, "data", "doctor", "results", "pilot_combined_ranking.csv"), "file", None),
        (os.path.join(_HERE, "data", "doctor", "results", "final_test_scores_full.csv"), "file", None),
        (os.path.join(_HERE, "data", "doctor", "results", "inference_speed_benchmark_cpu.csv"), "file", None),
    ]),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    missing = []

    for section, items in CHECKS:
        print(f"\n[{section}]")
        for abs_path, kind, expected_count in items:
            if kind == "zip":
                found = _find_zip(abs_path)  # abs_path is just the dataset name here, not a real path
                label = f"{abs_path}.zip (in {rel(OUTPUT_FOLDER)}/)"
                ok = found is not None
                detail = f" (found: {os.path.basename(found)})" if found else ""
                print(f"  [{'OK  ' if ok else 'MISS'}] {label}{detail}")
                if not ok:
                    missing.append(label)
                continue
            label = rel(abs_path)
            if kind == "file":
                ok = os.path.isfile(abs_path)
                detail = ""
            else:
                ok = os.path.isdir(abs_path)
                if ok and expected_count is not None:
                    actual = count_files(abs_path)
                    if actual != expected_count:
                        ok = False
                        detail = f" (found {actual} files, expected {expected_count})"
                    else:
                        detail = f" ({actual} files)"
                elif ok:
                    detail = f" ({count_files(abs_path)} files)"
                else:
                    detail = ""
            print(f"  [{'OK  ' if ok else 'MISS'}] {label}{detail}")
            if not ok:
                missing.append(label)

    print("\n" + "=" * 70)
    if missing:
        print(f"{len(missing)} item(s) missing or incomplete:")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nNote: 'Doctor raw source folders' and 'Doctor derived image folders' are "
            "alternatives to each other - if one group is fully OK, the other being "
            "MISS is not a blocker for Stage 2/3."
        )
    else:
        print("Everything checked is present. Good to go - see RUN_ORDER.md for what to run next.")


if __name__ == "__main__":
    main()
