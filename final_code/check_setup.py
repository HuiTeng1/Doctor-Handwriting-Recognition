"""
Checks this machine against every gitignored / hand-off-required path the project
depends on (see RUN_ORDER.md Stage 0) and reports what's missing. Read-only - doesn't
generate, download, or move anything. Run this right after `git clone` + whatever data
you were handed, before starting Stage 1, to see exactly what's still needed.

"Doctor raw source folders" and "Doctor derived image folders" are ALTERNATIVES to each
other, not both required - having one complete group covers Stage 2's needs even if the
other is entirely missing (see combineDataset.py's own comment for which path applies).

Usage:
    python check_setup.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(_HERE, "preprocessing", "doctor"))
from combineDataset import SOURCE_FOLDERS, OUTPUT_FOLDER  # noqa: E402


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
    ("Doctor raw source folders (ALTERNATIVE A - extract the 5 original zips here, then run combineDataset.py for real)", [
        (f, "dir", None) for f in SOURCE_FOLDERS
    ]),
    ("Doctor derived image folders (ALTERNATIVE B - gitignored outputs, hand these over directly to skip Stage 2/3 regeneration)", [
        (os.path.join(OUTPUT_FOLDER, "images"), "dir", 10347),
        (os.path.join(_HERE, "data", "doctor", "doctor_clean", "images"), "dir", 10050),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "baseline"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "clhaa"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "fajardo"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor", "benitez"), "dir", 8040),
        (os.path.join(_HERE, "data", "doctor", "preprocessed_doctor_test"), "dir", 2010),
    ]),
    ("Checkpoint files over GitHub's 100MB limit (gitignored - needs a separate transfer, not git)", [
        (os.path.join(_HERE, "checkpoint", "normal", "trocr", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "baseline", "checkpoint-3954", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "baseline", "checkpoint-3954", "optimizer.pt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "baseline", "checkpoint-4833", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr", "baseline", "checkpoint-4833", "optimizer.pt"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "baseline", "final", "model.safetensors"), "file", None),
        (os.path.join(_HERE, "checkpoint", "doctor", "trocr_pilot", "clhaa", "final", "model.safetensors"), "file", None),
    ]),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    missing = []

    for section, items in CHECKS:
        print(f"\n[{section}]")
        for abs_path, kind, expected_count in items:
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
