"""
Runs P0-P3 on the Train+Val images from data/Doctor_clean (Test is untouched), writing
output to preprocessed_doctor/<pipeline>/. Reuses preprocessing/pipelines.py rather than
reimplementing the preprocessing logic.
"""
import os
import sys

import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCTOR_CLEAN_DIR = os.path.join(_HERE, "data", "Doctor_clean")
DOCTOR_IMG_DIR = os.path.join(DOCTOR_CLEAN_DIR, "images")
OUTPUT_ROOT = os.path.join(_HERE, "preprocessed_doctor")

sys.path.insert(0, os.path.join(_HERE, "preprocessing"))
from pipelines import PIPELINES  # noqa: E402


def main():
    train_df = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "val.csv"))
    filenames = pd.concat([train_df["filename"], val_df["filename"]]).tolist()
    print(f"[generate] Train+Val: {len(filenames)} images total (Test not processed), "
          f"{len(PIPELINES)} pipelines")

    for pipeline_name, fn in PIPELINES.items():
        out_dir = os.path.join(OUTPUT_ROOT, pipeline_name)
        os.makedirs(out_dir, exist_ok=True)
        existing = set(os.listdir(out_dir))
        todo = [f for f in filenames if f not in existing]
        if not todo:
            print(f"[generate] {pipeline_name}: all {len(filenames)} images already present, skipping")
            continue

        print(f"[generate] {pipeline_name}: processing {len(todo)}/{len(filenames)} images...")
        for i, filename in enumerate(todo):
            img = Image.open(os.path.join(DOCTOR_IMG_DIR, filename))
            processed = fn(img)
            processed.save(os.path.join(out_dir, filename))
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(todo)}")

        produced = len(os.listdir(out_dir))
        assert produced == len(filenames), f"{pipeline_name} produced {produced} images, expected {len(filenames)}"
        print(f"[generate] {pipeline_name}: done, {produced} images")

    print("[generate] All done")


if __name__ == "__main__":
    main()
