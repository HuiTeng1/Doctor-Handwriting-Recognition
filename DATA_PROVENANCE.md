# Data Provenance

`preprocessing/normal/Iam_split.py` only *reads* pre-computed IAM splits
(`data/normal/crnn/{train,val,test}.csv`, `data/normal/trocr/{train,val}.csv`) -
it never generates them from scratch. The actual generation happened in separate Google
Colab notebooks that live outside this repo. This file records which of those notebooks
have been located and verified against the current data, and which are still missing.

**Why this matters**: if any of the `crnn/*.csv` or `trocr/*.csv` files under
`data/normal/` are ever lost or corrupted, this project cannot regenerate them on its own - the
generating notebook(s) would need to be found again.

## Verification method

For each candidate script, checked whether its exact split arithmetic (fraction, seed,
truncation behavior) reproduces the row counts actually present in the current
`data/normal/crnn/` or `data/normal/trocr` CSVs. An exact match is strong evidence (not
conclusive proof) that a script is the real source, since split ratios + `int()`
truncation rarely collide by chance.

## Confirmed: TrOCR Stage-1 IAM fine-tune script

**Match confirmed.** The user located a Colab script titled "阶段一 Fine-tune 脚本 — 只用
IAM 数据微调 trocr-base-handwritten" (Chinese original; English: "Stage 1 Fine-tune Script
- IAM-only finetune of trocr-base-handwritten") with:
- `WORDS_TXT` = `words_new.txt`, `IAM_WORDS_ZIP` = `iam_words.zip` (matches this project's files)
- Parses IAM's `words.txt` format, keeps only `status == "ok"` rows
- `VAL_SPLIT_RATIO = 0.05` (95/5 split), `RANDOM_SEED = 42`
- Output dir named `stage1_iam_finetuned` (this project's checkpoint folder was originally
  `stage1_iam_finetuned_final` - a very close naming match, "_final" likely appended once
  this training run was chosen as the keeper)

Arithmetic check:
```
data/normal/trocr/train.csv = 36,389 rows
data/normal/trocr/val.csv   =  1,915 rows
total                   = 38,304 rows

script's train_val_split(): n_val = max(1, int(38304 * 0.05)) = 1915
                             n_train = 38304 - 1915 = 36389
```
Exact match. This is the TrOCR Normal-stage source script.

**2026-08-22 update - recovered into the repo, verified byte-exact, gap closed.** The
user provided the original Colab script's full source. Its data-prep half (parse
`words_new.txt` -> filter `status=="ok"` -> shuffle with seed 42 -> 95/5 slice) is now in
this repo at `preprocessing/normal/generate_trocr_split.py`, with only two path-level
adaptations (image existence/corruption check reads directly from `iam_words.zip` instead
of a pre-extracted local folder; writes the project's flat `i_<word_id>.png` filename
convention instead of a full local disk path) - the filtering/splitting algorithm itself
is untouched. The training half of that Colab script was deliberately not brought in,
since `train_model/normal/trocr/finetune_TrOCR.py` already covers that, adapted to local paths.

Verified stronger than the arithmetic check above: ran it against the real
`I_Am_Dataset/` on disk and compared the resulting train/val **filename sets** (not just
counts) against the existing `data/normal/trocr/train.csv` and `val.csv` - **exact match,
every filename**. This is no longer a documented gap - the split is fully reproducible
from `words_new.txt` + `iam_words.zip` again.

## Not a match: three other located scripts

The user located three more Colab scripts while searching; none of them match the
current `data/normal/crnn/*.csv` data. All three turned out to belong to a **different,
abandoned lineage** that merged the old Kaggle "Normal" handwriting dataset together
with IAM into one combined dataset (filenames prefixed `n_` for Normal, `i_` for IAM),
rather than training CRNN on IAM alone the way this project's `crnn/train_iam_only.py`
does.

| Script | What it is | Why it doesn't match |
|---|---|---|
| "全量数据准备 (Normal + IAM, 不抽样)" (Chinese original; English: "Full Dataset Preparation (Normal + IAM, No Sampling)") | Merges Normal `labels.csv` + IAM `words_new.txt` into one 70/10/20 split (`full_dataset/{train,val,test}.csv`), filenames prefixed `n_`/`i_` | Current `data/normal/crnn/train.csv` and `data/normal/trocr/train.csv` are **100% `i_`-prefixed** (verified: 0 rows with any other prefix) - no Normal data mixed in, so this can't be the source |
| CRNN training script using `train_df_full` / `IMG_DIR_FULL` / `DRIVE_CKPT_DIR = f"{DRIVE_DIR}/full_dataset/checkpoints"` | Training counterpart to the script above | Variable names (`train_df_full`, `IMG_DIR_FULL`, `full_dataset` in paths) tie back exactly to the merged-dataset script's own output naming - trains on the merged data, not pure IAM |
| `split_analysis.py` (reads `val_predictions_large.csv`, splits stats by `i_`/`n_` prefix) | Post-hoc analysis of an already-trained model's predictions | Also handles both `i_` and `n_` prefixes together - evidence of a model trained on the same merged dataset, not a split-generation script at all (doesn't touch train/val/test creation) |

## Still missing: CRNN's pure-IAM 70/10/20 split script

**Not found**, despite checking `previous_code/` (the full archive of every superseded
script in this project's history) and three candidate Colab scripts the user located.
`previous_code/B_superseded_pipeline/code/` has `split_Normal.py` and `split_Doctor.py`
but no IAM equivalent.

Reverse-engineered how close a guess gets, working only from `words_new.txt` and the
extracted `iam_words_images/` (read-only, no project files modified):

| Filter applied | Rows remaining | Diff from actual (34,056) |
|---|---|---|
| `status == "ok"` only | 38,304 | 4,248 |
| + restricted to the CRNN 70-class character dictionary (`CHARS` in `modeling.py`) | 35,925 | 1,869 |
| + image file must actually exist in `iam_words_images/` | 34,140 | 84 |

The remaining gap of 84 is most likely a small number of corrupted/unopenable IAM images
being filtered out (the TrOCR script above has an explicit `img.verify()` check for
exactly this; the CRNN script probably had the same step) - not independently confirmed,
since that would require actually opening all 34,140 candidate images with PIL.

**Conclusion**: the real generating script very likely applied `status=="ok"` +
CRNN-dictionary filtering + image-existence/corruption checks + a 70/10/20 split, and is
a sibling of the confirmed TrOCR script above (same `words_new.txt`/`iam_words.zip`
source, same era) - but the actual file has not been located. The data itself
(`data/normal/crnn/*.csv`) is intact and working; only the generating script is missing.
