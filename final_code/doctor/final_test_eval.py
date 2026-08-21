"""
Runs the same test.csv evaluation for two different purposes, sharing all the logic and
only swapping which results table (and therefore which checkpoints) get evaluated:

  1. Pilot-decision check - after pilot_compare.py finishes baseline vs clhaa (15% data),
     validates that decision against held-out test.csv, using the pilot (15%-trained)
     checkpoints. Written to results/final_test_scores_pilot.csv.
  2. Full-finetune final score - after finetune_all.py finishes the full (100% data)
     finetune of the winning pipeline, reports the true production score on held-out
     test.csv, using the full-finetune checkpoints. Written to results/final_test_scores_full.csv.
     Skipped automatically (with a message, not an error) if the full finetune hasn't
     produced its summary csv yet - e.g. currently true for TrOCR.

These two were previously conflated into a single hardcoded run against the pilot
summary only - see the project discussion log for why that was wrong: the "final" number
on record was silently the pilot (15%-trained) model's score, not the full-finetune
model's.

======================================================================
Why this evaluation exists (methodology summary; same underlying decision as the detailed
rationale at the top of finetune_all.py):
======================================================================
train.csv/val.csv are only responsible for "choosing": train produces gradients, val
handles early stopping + picking the checkpoint + comparing baseline vs clhaa. But val
gets reused repeatedly for early stopping/selection, so its numbers are naturally
optimistic (the winner that gets picked is, by construction, the one that did well on
val - this "optimism" has nothing to do with how good the pipeline actually is, it's an
artifact of the selection process itself).

test.csv has never been touched since the start of the project, held back specifically
for this: each of CRNN and TrOCR's chosen winners gets exactly one inference pass here
(no training, no gradients, no further selection of any kind) - this number is the one
that can actually be trusted and used to draw conclusions. After this step, the choice
should never be revisited based on this number - doing so would mean test.csv has been
"contaminated by selection" too, leaving no clean data left to validate against in the
future. (This still applies separately to each of the two runs above - the pilot-decision
number shouldn't be used to re-litigate baseline vs clhaa, and the full-finetune number
shouldn't be used to re-litigate anything either.)

======================================================================
What run_eval() does, for whichever pair of results files it's given:
======================================================================
1. Reads the two results csvs, and for each finds the row with the lowest best_val_cer -
   i.e. the winning pipeline that CRNN and TrOCR each picked.
2. test.csv's images have never been touched by any pipeline (generate_doctor_preprocessed.py
   only processed train+val), so this script processes test.csv on the spot using the
   winner's pipeline, writing output to preprocessed_doctor_test/<pipeline>/.
3. Loads the corresponding winner's checkpoint, runs one inference pass on this freshly
   processed test set, computes CER, and prints + saves it to the given output csv.

Usage:
    cd final_code/doctor
    PYTHONIOENCODING=utf-8 py -3.12 -u final_test_eval.py
"""
import os
import sys

import pandas as pd
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCTOR_CLEAN_DIR = os.path.join(_HERE, "data", "Doctor_clean")
TEST_PREPROCESSED_ROOT = os.path.join(_HERE, "preprocessed_doctor_test")

sys.path.insert(0, os.path.join(_HERE, "preprocessing"))
from pipelines import PIPELINES as PREPROCESS_FNS  # noqa: E402



def pick_winner(results_path, path_col):
    df = pd.read_csv(results_path)
    best_row = df.loc[df["best_val_cer"].idxmin()]
    return best_row["pipeline"], best_row["best_val_cer"], best_row[path_col]


def load_and_filter_test(chars_filter=None):
    df = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "test.csv"))
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if chars_filter is not None:
        allowed = set(chars_filter)
        mask = df["text"].apply(lambda v: all(c in allowed for c in v))
        dropped = (~mask).sum()
        if dropped:
            print(f"[final_test] test.csv: dropping {dropped}/{len(df)} rows (chars outside the dictionary, needed for CRNN)")
        df = df[mask].reset_index(drop=True)
    return df


def ensure_test_preprocessed(pipeline_name, filenames):
    doctor_img_dir = os.path.join(DOCTOR_CLEAN_DIR, "images")
    out_dir = os.path.join(TEST_PREPROCESSED_ROOT, pipeline_name)
    os.makedirs(out_dir, exist_ok=True)
    existing = set(os.listdir(out_dir))
    todo = [f for f in filenames if f not in existing]
    if not todo:
        print(f"[final_test] {pipeline_name}: all test images already processed, skipping")
        return out_dir
    fn = PREPROCESS_FNS[pipeline_name]
    print(f"[final_test] {pipeline_name}: processing {len(todo)}/{len(filenames)} test images...")
    for filename in todo:
        img = Image.open(os.path.join(doctor_img_dir, filename))
        fn(img).save(os.path.join(out_dir, filename))
    return out_dir


def eval_crnn(pipeline_name, ckpt_path):
    crnn_dir = os.path.join(_HERE, "..", "normal", "crnn")
    sys.path.insert(0, crnn_dir)
    from hw_datasets import KaggleHandwritingDataModule, KaggleHandwrittenNames
    from training_modules import HandwritingRecogTrainModule
    from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS
    from ctc_decoder import best_path
    from torchmetrics.text import CharErrorRate

    hparams = {
        "lr": 1e-4, "gru_input_size": 256,
        "train_batch_size": 64, "val_batch_size": 256,
        "input_height": 36, "input_width": 324,
        "gru_hidden_size": 128, "gru_num_layers": 2, "num_classes": NUM_CLASSES,
        "filename_col": "filename", "label_col": "text",
        "train_img_path": TEST_PREPROCESSED_ROOT, "val_img_path": TEST_PREPROCESSED_ROOT,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_df = load_and_filter_test(chars_filter=CHARS)
    img_dir = ensure_test_preprocessed(pipeline_name, test_df["filename"].tolist())

    module = HandwritingRecogTrainModule.load_from_checkpoint(
        ckpt_path, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
        map_location=torch.device("cpu") if device == "cpu" else None,
    )
    module.to(device).eval()

    # letterbox not used for now, to stay consistent with training (see the note in finetune_all.py)
    dummy_dm = KaggleHandwritingDataModule(test_df, test_df, hparams, LABEL_TO_INDEX)
    eval_transforms = dummy_dm.transforms

    dataset = KaggleHandwrittenNames(test_df, eval_transforms, LABEL_TO_INDEX, img_dir + os.sep,
                                      filename_col="filename", label_col="text")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0,
        collate_fn=KaggleHandwritingDataModule.custom_collate)

    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["transformed_images"].to(device)
            labels = batch["labels"].cpu().numpy()
            target_lens = batch["target_lens"].cpu().numpy()
            output = module.model(images).permute(1, 0, 2)
            output = torch.exp(output).cpu().numpy()
            for i, pred in enumerate(output):
                predicted_string = best_path(pred, module.chars)
                gt_string = "".join(INDEX_TO_LABELS[idx] for idx in labels[i][0:target_lens[i]])
                all_preds.append(predicted_string)
                all_gts.append(gt_string)

    cer = CharErrorRate()(all_preds, all_gts).item()
    acc = sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_gts) * 100
    return cer, acc, len(all_gts)


def eval_trocr(pipeline_name, final_model_dir):
    from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizerFast, VisionEncoderDecoderModel
    from torchmetrics.text import CharErrorRate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_df = load_and_filter_test(chars_filter=None)
    img_dir = ensure_test_preprocessed(pipeline_name, test_df["filename"].tolist())

    image_processor = ViTImageProcessor.from_pretrained(final_model_dir, local_files_only=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(final_model_dir, local_files_only=True)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(final_model_dir, local_files_only=True).to(device)
    model.eval()

    all_preds, all_gts = [], list(test_df["text"])
    batch_size = 16
    filenames = test_df["filename"].tolist()
    with torch.no_grad():
        for i in range(0, len(filenames), batch_size):
            chunk_fn = filenames[i:i + batch_size]
            # letterbox_square not used for now, to stay consistent with training
            imgs = [Image.open(os.path.join(img_dir, fn)).convert("RGB") for fn in chunk_fn]
            pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            all_preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))

    cer = CharErrorRate()(all_preds, all_gts).item()
    acc = sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_gts) * 100
    return cer, acc, len(all_gts)


def run_eval(crnn_results_path, trocr_results_path, out_filename):
    """Shared evaluation logic. crnn_results_path/trocr_results_path pick which summary
    table (and therefore which checkpoints) get evaluated on test.csv - the only thing
    that differs between the pilot-decision check and the full-finetune final score.

    CRNN and TrOCR are evaluated independently: whichever one's results file exists gets
    evaluated and written out now; whichever doesn't is skipped (not an error) and can be
    filled in by a later run once it's ready. This lets e.g. CRNN's full-finetune score
    be reported today even though TrOCR's full finetune hasn't finished yet."""
    print("[final_test] Note: each model's score below is a one-and-only evaluation on test.csv - it should not be used to revisit that model's pipeline choice afterward\n")

    records = []

    if os.path.isfile(crnn_results_path):
        crnn_winner, crnn_val_cer, crnn_ckpt = pick_winner(crnn_results_path, "best_ckpt")
        print(f"[final_test] CRNN winner : {crnn_winner} (val CER={crnn_val_cer:.4f})")
        crnn_cer, crnn_acc, crnn_n = eval_crnn(crnn_winner, crnn_ckpt)
        print(f"[final_test] CRNN  ({crnn_winner}) test CER={crnn_cer:.4f}  exact-match={crnn_acc:.2f}%  n={crnn_n}")
        records.append({"model": "CRNN", "winner_pipeline": crnn_winner, "val_cer": crnn_val_cer,
                         "test_cer": crnn_cer, "test_exact_match_pct": crnn_acc, "n": crnn_n})
    else:
        print(f"[final_test] Skipping CRNN - {crnn_results_path} doesn't exist yet")

    if os.path.isfile(trocr_results_path):
        trocr_winner, trocr_val_cer, trocr_ckpt = pick_winner(trocr_results_path, "final_model_dir")
        print(f"[final_test] TrOCR winner: {trocr_winner} (val CER={trocr_val_cer:.4f})")
        trocr_cer, trocr_acc, trocr_n = eval_trocr(trocr_winner, trocr_ckpt)
        print(f"[final_test] TrOCR ({trocr_winner}) test CER={trocr_cer:.4f}  exact-match={trocr_acc:.2f}%  n={trocr_n}")
        records.append({"model": "TrOCR", "winner_pipeline": trocr_winner, "val_cer": trocr_val_cer,
                         "test_cer": trocr_cer, "test_exact_match_pct": trocr_acc, "n": trocr_n})
    else:
        print(f"[final_test] Skipping TrOCR - {trocr_results_path} doesn't exist yet")

    if not records:
        print("[final_test] Nothing to evaluate - both results files are missing.")
        return

    out = pd.DataFrame(records)
    out_path = os.path.join(_HERE, "results", out_filename)
    out.to_csv(out_path, index=False)
    print(f"\n[final_test] Saved to: {out_path}")


def main():
    results_dir = os.path.join(_HERE, "results")

    print("=" * 60)
    print("[final_test] 1/2: pilot-decision check (baseline vs clhaa, 15%-trained checkpoints)")
    print("=" * 60)
    run_eval(
        os.path.join(results_dir, "crnn_pilot_summary.csv"),
        os.path.join(results_dir, "trocr_pilot_summary.csv"),
        "final_test_scores_pilot.csv",
    )

    print("\n" + "=" * 60)
    print("[final_test] 2/2: full-finetune final score (production checkpoints)")
    print("=" * 60)
    run_eval(
        os.path.join(results_dir, "crnn_finetune_summary.csv"),
        os.path.join(results_dir, "trocr_finetune_summary.csv"),
        "final_test_scores_full.csv",
    )


if __name__ == "__main__":
    main()
