"""
Preprocessing pipeline comparison (baseline/clhaa/fajardo/benitez, see PILOT_PIPELINES
below), run on a stratified-sampled 15% subset of train+val, for both CRNN and TrOCR.

======================================================================
Background (full rationale is in finetune_all.py's top docstring, only the conclusions
are here):
======================================================================
- fajardo/benitez were already ruled out as the winner by three independent data points -
  CRNN full finetune, CRNN 15% pilot (old version, before the letterbox fix), and
  zero-shot (CRNN+TrOCR, Normal model not finetuned) - all show them clearly worse than
  baseline/clhaa. They're still included in PILOT_PIPELINES below and get a real
  finetune here anyway, so that conclusion has a reproducible checked-in checkpoint
  behind it instead of resting on documentation alone.
- baseline vs clhaa is the real open question - a small gap that's hard to measure
  precisely; it has to be tested with a real finetune, not guessed from existing
  data/zero-shot/theory.
- CRNN's ranking can't be assumed to answer the question for TrOCR - the two run
  independently, each picking its own winner.
- 15% (not 5%): at 5%, CRNN gets only 5-6 batches per epoch (train_batch_size=64) and
  learns nothing - CER gets stuck at 0.88~0.91 and just collapses; 15% is the only scale
  verified to work for CRNN.
- Decision layering: this step only uses train(15%)+val(15%), test.csv is untouched. val
  is only used for early stopping + picking the best checkpoint + ranking the pipelines
  against each other; test.csv is reserved until after this step picks a winner, at
  which point a single inference pass on the winner (in a later script, not yet run)
  produces the actual final score.
======================================================================

Stratification is based on the 3 source families this file's own FAMILY_MAP recognizes:
    RxHand Original (square + vignetting) / the BD dataset (Training+Validation+Testing,
    clean small images) / 89 (phone photos, high resolution, wide aspect ratio)
Sampled proportionally by TRAIN_FRACTION/VAL_FRACTION so the three families keep the same
ratio in the pilot set as in the full set, avoiding a "winner" that only looks good
because it happened to do well on one particular source.

Reuses run_crnn_all/run_trocr_all from finetune_all.py, passing this script's own
PILOT_PIPELINES (independent of finetune_all.py's own PIPELINES, see the comment above
that constant) plus this script's own train_df/val_df and separate checkpoint/results
directories (checkpoint/doctor/crnn_pilot/ / checkpoint/doctor/trocr_pilot/) so they
don't overwrite other historical results. The early-stopping rule (patience=5) matches
finetune_all.py. The letterbox fix (preventing the aspect ratio from being hard-stretched)
is already in effect - this is the first time this comparison has been rerun since the fix.

Usage:
    cd final_code/train_model/doctor
    PYTHONIOENCODING=utf-8 py -3.12 -u pilot_compare.py
"""
import os

import pandas as pd

from finetune_all import DOCTOR_CLEAN_DIR, SEED, run_crnn_all, run_trocr_all

_HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_FRACTION = 0.15
VAL_FRACTION = 0.15

FAMILY_MAP = {
    "RxHand Original": "rxhand",
    "Training": "bd", "Validation": "bd", "Testing": "bd",
    "89": "phone",
}


def stratified_sample(df, frac, seed):
    family = df["source"].map(FAMILY_MAP)
    assert family.isna().sum() == 0, f"Unknown source value(s): {sorted(df.loc[family.isna(), 'source'].unique())}"

    parts = []
    for name, group in df.groupby(family):
        parts.append(group.sample(frac=frac, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_pilot_splits():
    train_full = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "train.csv"))
    val_full = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "val.csv"))

    pilot_train = stratified_sample(train_full, TRAIN_FRACTION, SEED)
    pilot_val = stratified_sample(val_full, VAL_FRACTION, SEED)

    print(f"[pilot] train: {len(train_full)} -> {len(pilot_train)} (frac={TRAIN_FRACTION})  "
          f"val: {len(val_full)} -> {len(pilot_val)} (frac={VAL_FRACTION})")
    for name, df in [("train", pilot_train), ("val", pilot_val)]:
        family = df["source"].map(FAMILY_MAP)
        print(f"[pilot] {name} family distribution: {family.value_counts().to_dict()}")

    pilot_train.to_csv(os.path.join(_HERE, "..", "..", "data", "doctor", "pilot_train.csv"), index=False)
    pilot_val.to_csv(os.path.join(_HERE, "..", "..", "data", "doctor", "pilot_val.csv"), index=False)
    return pilot_train, pilot_val


# The pilot's own pipeline list - deliberately independent of finetune_all.py's PIPELINES
# (which gets narrowed to the winner once the decision is made), so re-running this file
# always re-verifies all four candidates rather than silently only checking whichever
# one(s) finetune_all.py currently points at. fajardo/benitez are included here (even
# though earlier evidence already ruled them out) so this pilot run produces real,
# checked-in checkpoints for all four pipelines instead of leaving fajardo/benitez as
# a documentation-only claim with no reproducible artifact.
PILOT_PIPELINES = ["baseline", "clhaa", "fajardo", "benitez"]


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    pilot_train, pilot_val = build_pilot_splits()

    print("=" * 60)
    print(f"Stage 1/2: CRNN pilot - {' vs '.join(PILOT_PIPELINES)}")
    print("=" * 60)
    run_crnn_all(
        pilot_train, pilot_val,
        ckpt_root=os.path.join(_HERE, "..", "..", "checkpoint", "doctor", "crnn_pilot"),
        results_path=os.path.join(_HERE, "..", "..", "data", "doctor", "results", "crnn_pilot_summary.csv"),
        pipelines=PILOT_PIPELINES,
    )

    print("\n" + "=" * 60)
    print(f"Stage 2/2: TrOCR pilot - {' vs '.join(PILOT_PIPELINES)}")
    print("=" * 60)
    run_trocr_all(
        pilot_train, pilot_val,
        ckpt_root=os.path.join(_HERE, "..", "..", "checkpoint", "doctor", "trocr_pilot"),
        results_path=os.path.join(_HERE, "..", "..", "data", "doctor", "results", "trocr_pilot_summary.csv"),
        eval_num_beams=1,  # the pilot only needs a relative ranking, so greedy decoding trades accuracy for speed; all 4 pipelines get the same slowdown, so the ranking stays fair
        pipelines=PILOT_PIPELINES,
    )
