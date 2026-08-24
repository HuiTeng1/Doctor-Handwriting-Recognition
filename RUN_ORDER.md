# Run Order

This project has two stages after setup: `train_model/normal/` (IAM pretraining) runs
first and produces the base checkpoints that `train_model/doctor/` (medicine-name
finetuning) builds on top of. `preprocessing/{normal,doctor}/` sits alongside `train_model/`
- it's not a training stage, just the "raw data -> clean, split dataset" prep code each
domain's Stage 1/Stage 2 depends on.

## Stage 0 - Setup (new machine / new developer, starting completely from raw data)

None of the large data or checkpoint files are in git (too big, or over GitHub's 100MB
push limit) - `git clone` alone does **not** give you a working project. This section
assumes the extreme case: **nothing** is handed over except the raw datasets below - no
checkpoints (not even the small ones), no results CSVs, no `doctor_raw/`/`doctor_clean/`/
`preprocessed_doctor/`. Every checkpoint and every CSV gets produced by actually running
the pipeline, from real training and real preprocessing, start to finish.

Run `python check_setup.py` any time to see exactly which of the items below
are present on the current machine and which are still missing - it doesn't generate or
move anything, just reports.

### What has to be handed over

| # | What | Extract/place at | Why |
|---|---|---|---|
| 1 | `iam_words.zip` + `words_new.txt` (IAM raw archive + label file, uncleaned) | `data/normal/I_Am_Dataset/` - don't pre-extract | `Iam_split.py`'s `extract_images()` pulls out whichever images each script needs, on demand |
| 2 | The 3 raw Doctor prescription dataset zips (`Doctor Handwriting Recognition Dataset.zip`, `RxHand Original.zip`, `Doctor's Handwritten Prescription BD dataset.zip` - the last one unpacks into `Testing`/`Training`/`Validation` subfolders, giving `SOURCE_FOLDERS` 5 paths from these 3 zips), uncombined | `data/doctor/doctor_raw/`, place the 3 zips there (named after the dataset itself, e.g. `RxHand Original.zip`) | `combineDataset.py`'s `ensure_source_folders_extracted()` auto-unzips these on run (handles each zip's different internal packaging) - see its docstring. One exception: the "Doctor Handwriting Recognition Dataset" zip extracts into a folder literally named `89`, not the zip's own name - that's the `source` value already baked into the live `labels.csv` (from years ago) and `pilot_compare.py`'s `FAMILY_MAP` hardcodes `"89"` for stratified sampling, so the folder name can't be the readable one without also updating `FAMILY_MAP`. `SOURCE_FOLDERS` already points at these exact paths under `doctor_raw/` (no per-machine editing needed anymore - see Stage 2 below) |
| 3 | `data/normal/crnn/{train,val,test}.csv` (CRNN's own IAM 70/10/20 split) | `data/normal/crnn/` | **Exception - must be handed over, no way around it.** TrOCR's split can be regenerated from raw IAM data (`generate_trocr_split.py` does this for real - see item below), but the original script that generated *CRNN's* split was lost years ago (see `DATA_PROVENANCE.md`) and nothing in this codebase can rebuild it from `words_new.txt`. Without this file, `train_iam_only.py`'s first line (`load_crnn_splits()`) throws `FileNotFoundError` immediately. |

Everything downstream of these three - every checkpoint, every other CSV, `doctor_raw/`,
`doctor_clean/`, `preprocessed_doctor/` - is produced by the run order below. Nothing is
cached, so nothing gets skipped.

### Run order

1. **CRNN, Stage 1**: `train_iam_only.py` (works now that item 3 above is in place) - real
   IAM pretrain, real GPU time. Then `test_CrnnNormal.py` - no cached `test_results.csv`
   exists, so this really loads the checkpoint and runs inference on the test split
   (extracting the needed test images from `iam_words.zip` on demand).
2. **TrOCR, Stage 1**: `preprocessing/normal/generate_trocr_split.py` first - this is the
   one IAM split that *does* rebuild itself from raw data, producing
   `data/normal/trocr/{train,val}.csv` from `words_new.txt`. Then `finetune_TrOCR.py` -
   its "already trained" check finds no `model.safetensors`, so it downloads
   `microsoft/trocr-base-handwritten` from Hugging Face (needs internet) and really
   pretrains on IAM from scratch - the single most expensive step in this whole list.
   Then `test_TrOCR.py` for eval - same "no cache, does a real pass" story as
   `test_CrnnNormal.py` above.
3. **Doctor data prep (Stage 2)**, in this exact order: `combineDataset.py` (place the 3
   raw zips from item 2 above directly in `doctor_raw/` and run - it auto-unzips them via
   `ensure_source_folders_extracted()`, no manual extraction or code changes needed) ->
   `discover_doctor_duplicates.py` ->
   `resolve_doctor_duplicates.py` -> `split_doctor_clean.py` -> `validate_doctor_clean.py`.
   All five run for real - CPU only, hashing/copying ~10k images, a few minutes - producing
   `doctor_raw/` and `doctor_clean/` from nothing.
4. **Doctor preprocessing (Stage 3, step 1)**: `generate_doctor_preprocessed.py` - real
   work for all 4 pipelines (`baseline`/`clhaa`/`fajardo`/`benitez`), producing
   `preprocessed_doctor/` from `doctor_clean/images/`.
5. **Pilot + full finetune (Stage 3, steps 2-3)**: `pilot_compare.py` and
   `finetune_all.py` each check a `DONE.txt`/results-CSV cache per pipeline before
   training - since none of that exists yet, every pipeline they're configured to check
   (`pilot_compare.py`'s `PILOT_PIPELINES` = all 4; `finetune_all.py`'s `PIPELINES` =
   `["baseline", "clhaa"]`) trains for real, for both CRNN and TrOCR. This is the second
   big compute step, on top of Stage 1's IAM pretraining.
6. **Final eval (Stage 3, step 4)**: `final_test_eval.py` - real eval on `test.csv`, using
   whatever checkpoints/results steps 1-5 actually produced.

This is an independent run from raw data, not a reproduction of this repo's own numbers -
IAM pretrain and the pilot/finetune steps both involve randomness (seeded, but a fresh
run is still a fresh run), so expect checkpoints and CER numbers close to, but not
necessarily identical to, the ones already on record elsewhere in this repo.

## Stage 1 - `train_model/normal/` (IAM pretraining)

Both CRNN and TrOCR are pretrained independently on IAM handwriting data before ever
seeing Doctor data. **Already done** - the checkpoints these produce already exist under
`checkpoint/`. Listed here for reference / in case they ever need to be rerun.

| # | Script | Purpose |
|---|---|---|
| 1 | `crnn/train_iam_only.py` | Trains the CRNN (CTC+GRU) model on the IAM 70/10/20 split. Produces `checkpoint/normal/crnn/`. Resumable. |
| 2 | `crnn/test_CrnnNormal.py` | Evaluates the trained CRNN checkpoint on the IAM held-out test set. |
| 3 | `trocr/finetune_TrOCR.py` | Fine-tunes `microsoft/trocr-base-handwritten` on the IAM 95/5 split. Produces `checkpoint/normal/trocr/`. |
| 4 | `trocr/test_TrOCR.py` | Evaluates the TrOCR checkpoint on the IAM val set. |

`preprocessing/normal/Iam_split.py` isn't run directly - it's a shared library
imported by all four scripts above for loading/extracting the IAM splits.

## Stage 2 - `preprocessing/doctor/` (data cleaning)

Turns the raw scraped prescription images into a clean, deduplicated, leak-free dataset.
Reads/writes its data under `data/doctor/` (only `doctor_raw/` and `doctor_clean/`
live there - the scripts themselves live in `preprocessing/doctor/`, alongside
`preprocessing/normal/`'s IAM split scripts since both are "raw data -> clean, split
dataset" prep code, not model code). **Already done** - the output already exists at
`data/doctor/doctor_clean/`. Run in this exact order if it ever needs to be redone from
scratch.

| # | Script | Purpose |
|---|---|---|
| 1 | `combineDataset.py` | Merges 5 raw source folders into `data/doctor/doctor_raw/{images/, labels.csv}`. `SOURCE_FOLDERS` points at 5 fixed paths right under `doctor_raw/` itself (`doctor_raw/89/`, `doctor_raw/RxHand Original/`, `doctor_raw/Doctor's Handwritten Prescription BD dataset/{Testing,Training,Validation}/`) - **place the 3 raw zips directly in `doctor_raw/` (named after each dataset, e.g. `RxHand Original.zip`) and running the script auto-unzips them via `ensure_source_folders_extracted()`** (verified to reproduce the live `labels.csv` byte-for-byte). These 5 folders are gitignored (see `.gitignore`), so they won't get committed. If you don't have those 3 raw zips handy, easier to **skip this step**: get `doctor_raw/images/` (~58MB, 10,347 files, gitignored) handed to you directly instead and place it at `data/doctor/doctor_raw/images/` - `labels.csv` is already in git. Then start at step 2. |
| 2 | `discover_doctor_duplicates.py` | Deduplicates by image content hash (within-source, then cross-source), classifies each duplicate group as "equivalent" (labels agree, auto-resolvable) or "conflict" (labels disagree, needs a human). Read-only against `doctor_raw/` - writes `duplicate_groups.csv`/`duplicate_label_conflicts.csv`/`near_duplicates.csv` to `doctor_clean/`, but no images or `labels.csv` yet. **Needs `data/doctor/doctor_raw/images/` in place first (see step 1).** |
| 3 | `resolve_doctor_duplicates.py` | Reads step 2's output and actually writes `doctor_clean/{images/, labels.csv}`, in two passes: Pass 1 auto-resolves every non-conflict row (singletons + equivalent-group survivors); Pass 2 applies the 6 manually-verified `RESOLUTIONS` for the conflict groups. Idempotent - safe to re-run. |
| 4 | `split_doctor_clean.py` | Final 70/10/20 train/val/test split, grouped by content hash so no image leaks across splits. |
| 5 | `validate_doctor_clean.py` | Full audit report + validation checks on the cleaned dataset. Fails loudly if anything is wrong. |

## Stage 3 - `train_model/doctor/` (preprocessing decision + finetune)

The actual "which preprocessing method wins, and what's the final doctor model" pipeline.
**This is the part that's actually runnable/rerunnable** - everything above it is already
complete.

| # | Script | Purpose |
|---|---|---|
| 1 | `generate_doctor_preprocessed.py` | Applies all 4 preprocessing pipelines (baseline/clhaa/fajardo/benitez, defined in `train_model/doctor/preprocessing/pipelines.py` - doctor's own image-transform algorithms, not to be confused with the top-level `preprocessing/` data-prep folder) to the train+val images, caching results to `../../data/doctor/preprocessed_doctor/<pipeline>/`. **Needs `data/doctor/doctor_clean/images/` in place** (gitignored - see Stage 0). |
| 2 | `pilot_compare.py` | Fast decision step: finetunes both CRNN and TrOCR on a 15% stratified sample, checking `PILOT_PIPELINES = ["baseline", "clhaa", "fajardo", "benitez"]` (all four - `baseline`/`clhaa` already have results and will skip; `fajardo`/`benitez` will run for real). |
| 3 | `finetune_all.py` | The full finetune - `PIPELINES = ["baseline", "clhaa"]`, both trained fully for both CRNN and TrOCR rather than betting everything on step 2's pilot call alone (the baseline/clhaa gap there is close). Contains the actual `run_crnn_all()`/`run_trocr_all()` training logic that `pilot_compare.py` also calls. Produces the real doctor-finetuned checkpoints under `checkpoint/doctor/crnn/` and `checkpoint/doctor/trocr/`. **Done** - `crnn_finetune_summary.csv`/`trocr_finetune_summary.csv` each have both rows; TrOCR's `baseline` was manually finalized at epoch 9 (see `KNOWN_ISSUES.md`), `clhaa` ran to its epoch-11 cap. |
| 4 | `final_test_eval.py` | Reports the true production score on `test.csv` using the full-finetune checkpoints -> `data/doctor/results/final_test_scores_full.csv`. **Done** - it has all 4 rows (CRNN baseline/clhaa, TrOCR baseline/clhaa), `is_winner` picked per model by validation CER (CRNN: `baseline`; TrOCR: `clhaa`). Used to also run a "pilot-decision check" (evaluating the 15%-trained pilot winner on `test.csv` too) - removed, since that checkpoint is never the production model and nothing reads that number; see `previous_code/C_decided_not_used/run_pilot_eval_only.py`. |

`zeroshot_normal_check.py` (standalone sanity check - ranking preprocessing pipelines via a
zero-shot, non-finetuned CRNN) was moved to `previous_code/C_decided_not_used/` - decided
not to keep any zero-shot "Normal checkpoint on doctor data" testing in the active pipeline.

## Summary

```
Stage 0: git clone + hand over the raw datasets + CRNN's split csv, place at the paths listed above (once, per machine)
        │
        ▼
train_model/normal/crnn/train_iam_only.py  ─┐
train_model/normal/crnn/test_CrnnNormal.py ─┤ (done, using preprocessing/normal/Iam_split.py)
train_model/normal/trocr/finetune_TrOCR.py  ─┤
train_model/normal/trocr/test_TrOCR.py     ─┘
        │
        ▼
preprocessing/doctor/combineDataset.py (place the 3 raw zips in doctor_raw/ and run - auto-unzips, or skip - see Stage 0) ─┐
preprocessing/doctor/discover_doctor_duplicates.py                          │
preprocessing/doctor/resolve_doctor_duplicates.py                          ├─ (done)
preprocessing/doctor/split_doctor_clean.py                                  │
preprocessing/doctor/validate_doctor_clean.py                              ─┘
        │
        ▼
train_model/doctor/generate_doctor_preprocessed.py
        │
        ▼
train_model/doctor/pilot_compare.py   (picks a preprocessing winner per model)
        │
        ▼
train_model/doctor/finetune_all.py    (full finetune on the winning pipeline)
        │
        ▼
train_model/doctor/final_test_eval.py (full-finetune final score on test.csv)
```
