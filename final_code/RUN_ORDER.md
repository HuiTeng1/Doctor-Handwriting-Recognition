# Run Order

`final_code/` has two stages after setup: `normal/` (IAM pretraining) runs first and
produces the base checkpoints that `doctor/` (medicine-name finetuning) builds on top of.

## Stage 0 - Setup (new machine / new developer)

None of the large data or checkpoint files are in git (too big, or over GitHub's 100MB
push limit) - `git clone` alone does **not** give you a working project. Everything below
must be handed to you directly (cloud storage, external drive, direct transfer - not git)
and placed at the exact path shown. Once all of it is in place, everything from Stage 1
onward behaves exactly as described in this document.

### Data folders (gitignored - not in git)

| # | What | Size | Files | Place at |
|---|---|---|---|---|
| 1 | IAM raw data (`iam_words.zip`, `words_new.txt`, extracted `iam_words_images/`) | 1,490.8 MB | 34,248 | `final_code/normal/dataset/I_Am_Dataset/` |
| 2 | Doctor raw images (already merged, `combineDataset.py`'s output) | 58.2 MB | 10,347 | `final_code/doctor/data/Doctor/images/` |
| 3 | Doctor cleaned images (`clean_doctor_dataset.py`'s output) | 56.9 MB | 10,050 | `final_code/doctor/data/Doctor_clean/images/` |
| 4 | Preprocessed doctor images, train+val, all 4 pipelines | 90.2 MB | 32,160 | `final_code/doctor/preprocessed_doctor/` |
| 5 | Preprocessed doctor images, test set | 6.3 MB | 2,010 | `final_code/doctor/preprocessed_doctor_test/` |

Items 2-5 are each the *output* of one pipeline stage (`combineDataset.py`,
`clean_doctor_dataset.py`, `generate_doctor_preprocessed.py`) - having them in place lets
you skip re-running those stages entirely and jump straight to whichever step you
actually need.

### Checkpoint files over GitHub's 100MB limit (gitignored - not in git)

Everything else under `checkpoint/`, `checkpoints/`, `checkpoints_pilot/`,
`checkpoints_trocr/`, `checkpoints_trocr_pilot/` (configs, tokenizers, small `.ckpt`
files) **is** in git already - only these large files need to be handed over separately,
into the same folder structure git already gave you:

| # | File | Size | Place at |
|---|---|---|---|
| 6 | TrOCR Stage-1 (IAM) model weights | 1,273.9 MB | `final_code/normal/checkpoint/trocr/model.safetensors` |
| 7 | TrOCR doctor finetune, checkpoint-3954 weights | 1,273.9 MB | `final_code/doctor/checkpoints_trocr/baseline/checkpoint-3954/model.safetensors` |
| 8 | TrOCR doctor finetune, checkpoint-3954 optimizer state | 2,543.5 MB | `final_code/doctor/checkpoints_trocr/baseline/checkpoint-3954/optimizer.pt` |
| 9 | TrOCR doctor finetune, checkpoint-4833 weights | 1,273.9 MB | `final_code/doctor/checkpoints_trocr/baseline/checkpoint-4833/model.safetensors` |
| 10 | TrOCR doctor finetune, checkpoint-4833 optimizer state | 2,543.5 MB | `final_code/doctor/checkpoints_trocr/baseline/checkpoint-4833/optimizer.pt` |
| 11 | TrOCR doctor pilot (baseline) weights | 1,273.9 MB | `final_code/doctor/checkpoints_trocr_pilot/baseline/final/model.safetensors` |
| 12 | TrOCR doctor pilot (clhaa) weights | 1,273.9 MB | `final_code/doctor/checkpoints_trocr_pilot/clhaa/final/model.safetensors` |

Total to transfer: **~1.7 GB of images + ~11.4 GB of checkpoint weights.**

### What the new developer actually needs to do

1. `git clone` the repo as normal - this gives you all code, all small config/checkpoint
   files, all CSVs (splits, results, labels).
2. Get the 12 items above from whoever's handing off the project, and copy each one to
   the exact path listed (paths are relative to the repo root).
3. That's it - no script needs to be run to "set up" the data. Everything from Stage 1
   below will find its cached results/checkpoints already in place and skip straight to
   whatever hasn't been done yet (see the per-stage tables for what's already done vs.
   still runnable).

## Stage 1 - `normal/` (IAM pretraining)

Both CRNN and TrOCR are pretrained independently on IAM handwriting data before ever
seeing Doctor data. **Already done** - the checkpoints these produce already exist under
`normal/checkpoint/`. Listed here for reference / in case they ever need to be rerun.

| # | Script | Purpose |
|---|---|---|
| 1 | `crnn/train_iam_only.py` | Trains the CRNN (CTC+GRU) model on the IAM 70/10/20 split. Produces `normal/checkpoint/crnn/`. Resumable. |
| 2 | `crnn/test_CrnnNormal.py` | Evaluates the trained CRNN checkpoint on the IAM held-out test set. |
| 3 | `trocr/TROcrFineTune.py` | Fine-tunes `microsoft/trocr-base-handwritten` on the IAM 95/5 split. Produces `normal/checkpoint/trocr/`. |
| 4 | `trocr/Test_TrOcr.py` | Evaluates the TrOCR checkpoint on the IAM val set. |

`dataset/Iam_split.py` isn't run directly - it's a shared library imported by all four
scripts above for loading/extracting the IAM splits.

## Stage 2 - `doctor/dataset_prepare/` (data cleaning)

Turns the raw scraped prescription images into a clean, deduplicated, leak-free dataset.
**Already done** - the output already exists at `doctor/data/Doctor_clean/`. Run in this
exact order if it ever needs to be redone from scratch.

| # | Script | Purpose |
|---|---|---|
| 1 | `combineDataset.py` | Merges 5 raw source folders into `data/Doctor/`. The 5 raw folders still exist (kept outside this repo) but `SOURCE_FOLDERS` points at the author's old machine-specific path - **edit `SOURCE_FOLDERS` to wherever those 5 folders actually are on the current machine before running.** If you don't have those 5 raw folders handy, easier to **skip this step**: get `data/Doctor/images/` (~58MB, 10,347 files, gitignored) handed to you directly instead and place it at `final_code/doctor/data/Doctor/images/` - `labels.csv` is already in git. Then start at step 2. |
| 2 | `clean_doctor_dataset.py` | Deduplicates by image content hash (within-source, then cross-source) and flags label conflicts. Produces `data/Doctor_clean/`. **Needs `data/Doctor/images/` in place first (see step 1).** |
| 3 | `resolve_doctor_label_conflicts.py` | Applies 6 manually-verified resolutions to the label conflicts found in step 2. |
| 4 | `split_doctor_clean.py` | Final 70/10/20 train/val/test split, grouped by content hash so no image leaks across splits. |
| 5 | `validate_doctor_clean.py` | Full audit report + validation checks on the cleaned dataset. Fails loudly if anything is wrong. |

## Stage 3 - `doctor/` (preprocessing decision + finetune)

The actual "which preprocessing method wins, and what's the final doctor model" pipeline.
**This is the part that's actually runnable/rerunnable** - everything above it is already
complete.

| # | Script | Purpose |
|---|---|---|
| 1 | `generate_doctor_preprocessed.py` | Applies all 4 preprocessing pipelines (baseline/clhaa/fajardo/benitez, defined in `preprocessing/pipelines.py`) to the train+val images, caching results to `preprocessed_doctor/<pipeline>/`. **Needs `data/Doctor_clean/images/` in place** (gitignored - see Stage 0). |
| 2 | `pilot_compare.py` | Fast decision step: finetunes both CRNN and TrOCR on a 15% stratified sample, checking `PILOT_PIPELINES = ["baseline", "clhaa", "fajardo", "benitez"]` (all four - `baseline`/`clhaa` already have results and will skip; `fajardo`/`benitez` will run for real). |
| 3 | `finetune_all.py` | The full finetune, using only the pipeline that won step 2 (`PIPELINES = ["baseline"]`). Contains the actual `run_crnn_all()`/`run_trocr_all()` training logic that `pilot_compare.py` also calls. Produces the real doctor-finetuned checkpoints under `checkpoints/` and `checkpoints_trocr/`. CRNN half is done; TrOCR half is not (see `KNOWN_ISSUES.md`). |
| 4 | `final_test_eval.py` | Calls the shared `run_eval()` twice: **1/2** validates the pilot decision on `test.csv` using the pilot (15%-trained) checkpoints -> `results/final_test_scores_pilot.csv`; **2/2** reports the true production score using the full-finetune checkpoints -> `results/final_test_scores_full.csv` (currently skips - TrOCR's full finetune summary doesn't exist yet). |

`zeroshot_normal_check.py` is a standalone sanity check, not part of this sequence -
it checks whether ranking preprocessing pipelines via a zero-shot (non-finetuned) CRNN is
a trustworthy shortcut. Informational only.

## Summary

```
Stage 0: hand over data/checkpoints, place at the paths listed above (once, per machine)
        │
        ▼
normal/crnn/train_iam_only.py  ─┐
normal/crnn/test_CrnnNormal.py ─┤ (done)
normal/trocr/TROcrFineTune.py  ─┤
normal/trocr/Test_TrOcr.py     ─┘
        │
        ▼
doctor/dataset_prepare/combineDataset.py (needs SOURCE_FOLDERS updated, or skip - see Stage 0) ─┐
doctor/dataset_prepare/clean_doctor_dataset.py                                │
doctor/dataset_prepare/resolve_doctor_label_conflicts.py                     ├─ (done)
doctor/dataset_prepare/split_doctor_clean.py                                  │
doctor/dataset_prepare/validate_doctor_clean.py                              ─┘
        │
        ▼
doctor/generate_doctor_preprocessed.py
        │
        ▼
doctor/pilot_compare.py   (picks a preprocessing winner per model)
        │
        ▼
doctor/finetune_all.py    (full finetune on the winning pipeline)
        │
        ▼
doctor/final_test_eval.py (1/2 pilot check, 2/2 full-finetune final score)
```
