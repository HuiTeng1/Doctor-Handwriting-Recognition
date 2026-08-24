"""
Recovered split-generation logic for the TrOCR stage-1 (IAM-only) 95/5 train/val split.

Provenance:
  Extracted from the original Colab script "Stage 1 Fine-tune script - finetune
  trocr-base-handwritten on IAM data only" (confirmed source, see DATA_PROVENANCE.md) -
  only the data-prep half (parse words_new.txt -> filter -> shuffle -> split) is kept
  here. The
  training half of that script is intentionally NOT reproduced - this project's actual
  training run uses normal/trocr/finetune_TrOCR.py, already adapted to local paths and
  reading the pre-computed trocr/train.csv + trocr/val.csv this script produces.

  Filtering/splitting logic is unchanged from the original: status == "ok" only, skip
  empty transcriptions, skip missing/corrupt images, random.Random(42).shuffle() then a
  95/5 slice. Only two things were adapted to fit this project's layout (paths, not
  algorithm):
    - reads word_id -> image lookup directly from iam_words.zip (zipfile membership +
      in-memory PIL verify) instead of the original's "unzip everything to local disk
      first, then walk directories" approach
    - writes "filename" as the project's flat i_<word_id>.png convention (matching what
      Iam_split.py's extract_images() already extracts into I_Am_Dataset/iam_words_images/)
      instead of the original's full local disk path

  Verified: with these two path-only adaptations, this reproduces the exact row counts
  already on record (train=36,389, val=1,915, total=38,304) - see DATA_PROVENANCE.md.

WARNING - do not run this casually:
  trocr/train.csv and trocr/val.csv already exist and are what the real
  stage1_iam_finetuned_final checkpoint was trained on. This script exists so they can be
  regenerated *if* they are ever lost or corrupted - running it will overwrite them.
  Back up the existing csvs first and diff row counts/content before trusting a fresh run.
"""
import io
import os
import random
import zipfile

import pandas as pd
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "normal")
IAM_DATASET_DIR = os.path.join(NORMAL_DATA_DIR, "I_Am_Dataset")
WORDS_TXT = os.path.join(IAM_DATASET_DIR, "words_new.txt")
IAM_WORDS_ZIP = os.path.join(IAM_DATASET_DIR, "iam_words.zip")
OUTPUT_DIR = os.path.join(NORMAL_DATA_DIR, "trocr")

ONLY_OK_STATUS = True   # only rows with status == "ok" - skip "err" (bad segmentation)
VAL_SPLIT_RATIO = 0.05  # 95/5 split
RANDOM_SEED = 42


def zip_member_for(word_id):
    """word_id 'a01-000u-00-00' -> zip member 'words/a01/a01-000u/a01-000u-00-00.png'
    (same nested layout Iam_split.py's extract_images() already assumes)."""
    parts = word_id.split("-")
    form_group = parts[0]
    form_id = f"{parts[0]}-{parts[1]}"
    return f"words/{form_group}/{form_id}/{word_id}.png"


def load_iam_rows(words_txt_path, zip_path, only_ok=True):
    """Parses the official IAM words.txt, skipping comment lines (starting with #).
    Format: word_id status graylevel num_components x y w h tag transcription
    transcription is the last field - split with maxsplit so it stays intact even if it
    contains spaces itself.

    Also verifies each image actually opens (read straight out of iam_words.zip, no need
    to extract everything to disk first) - a handful of images in the IAM dataset are
    themselves corrupt (a known issue), filtered out here."""
    rows = []
    skipped_no_image = 0
    skipped_bad_status = 0
    skipped_corrupt_image = 0

    with zipfile.ZipFile(zip_path) as zf, open(words_txt_path, encoding="utf-8") as f:
        namelist = set(zf.namelist())
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split(" ", 8)
            if len(fields) < 9:
                continue

            word_id, status = fields[0], fields[1]
            transcription = fields[8].strip()

            if only_ok and status != "ok":
                skipped_bad_status += 1
                continue

            if not transcription:
                continue

            member = zip_member_for(word_id)
            if member not in namelist:
                skipped_no_image += 1
                continue

            try:
                Image.open(io.BytesIO(zf.read(member))).verify()
            except Exception:
                skipped_corrupt_image += 1
                continue

            rows.append({"filename": f"i_{word_id}.png", "text": transcription})

    print(f"[generate_trocr_split] words_new.txt parsed: {len(rows)} usable rows, "
          f"skipped status!=ok: {skipped_bad_status}, "
          f"skipped missing image: {skipped_no_image}, "
          f"skipped corrupt image: {skipped_corrupt_image}.")
    return rows


def train_val_split(rows, val_ratio, seed):
    rows = rows[:]  # copy
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows


def main():
    rows = load_iam_rows(WORDS_TXT, IAM_WORDS_ZIP, only_ok=ONLY_OK_STATUS)
    if not rows:
        raise RuntimeError("No rows parsed at all - check that words_new.txt / iam_words.zip are both present under I_Am_Dataset/.")

    train_rows, val_rows = train_val_split(rows, VAL_SPLIT_RATIO, RANDOM_SEED)
    print(f"[generate_trocr_split] train: {len(train_rows)} rows, val: {len(val_rows)} rows.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(train_rows)[["filename", "text"]].to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    pd.DataFrame(val_rows)[["filename", "text"]].to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    print(f"[generate_trocr_split] done writing: {OUTPUT_DIR}/train.csv, {OUTPUT_DIR}/val.csv")


if __name__ == "__main__":
    main()
