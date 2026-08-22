"""
Final 70/10/20 split of doctor_clean/labels.csv, grouped by image content hash
(a group can't be split across sets). Automatically verifies the hash sets of
train/val/test don't overlap once done.
"""
import hashlib
import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image

# Windows consoles default to cp1252, which can't encode the "∩" used in the leakage
# check output below - force utf-8 so a plain `python split_doctor_clean.py` doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "..", "data", "doctor", "doctor_clean")
IMG_DIR = os.path.join(CLEAN_DIR, "images")

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
SEED = 42


def content_hash(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def main():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "labels.csv"))
    df["content_hash"] = [content_hash(os.path.join(IMG_DIR, fn)) for fn in df["filename"]]

    groups = df.groupby("content_hash")["filename"].apply(list).to_dict()
    group_ids = list(groups.keys())
    random.Random(SEED).shuffle(group_ids)

    total = len(df)
    train_target, val_target = total * TRAIN_RATIO, total * VAL_RATIO

    assigned = {}
    train_n, val_n = 0, 0
    for gid in group_ids:
        size = len(groups[gid])
        if train_n < train_target:
            assigned[gid] = "train"
            train_n += size
        elif val_n < val_target:
            assigned[gid] = "val"
            val_n += size
        else:
            assigned[gid] = "test"

    df["split"] = df["content_hash"].map(assigned)

    out = {}
    for split_name in ("train", "val", "test"):
        sub = df[df["split"] == split_name][["filename", "text", "source"]].reset_index(drop=True)
        out[split_name] = sub
        sub.to_csv(os.path.join(CLEAN_DIR, f"{split_name}.csv"), index=False)

    # Leakage check: pairwise intersection by content hash, must all be 0
    hashes = {s: set(df[df["split"] == s]["content_hash"]) for s in ("train", "val", "test")}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    leak_rows = []
    all_clean = True
    for a, b in pairs:
        inter = hashes[a] & hashes[b]
        leak_rows.append({"split_pair": f"{a}∩{b}", "overlap_count": len(inter)})
        if inter:
            all_clean = False
    pd.DataFrame(leak_rows).to_csv(os.path.join(CLEAN_DIR, "split_leakage_check.csv"), index=False)

    print(f"Total: {total}")
    for s in ("train", "val", "test"):
        n = len(out[s])
        print(f"{s}: {n} ({n/total:.1%})")
    print("\nLeakage check (must all be 0):")
    for row in leak_rows:
        print(f"  {row['split_pair']}: {row['overlap_count']}")
    print(f"\n{'PASS' if all_clean else 'FAIL'}: split_leakage_check.csv -> {CLEAN_DIR}")

    if not all_clean:
        raise RuntimeError("Content hash overlap remains across splits after splitting - cannot continue")


if __name__ == "__main__":
    main()
