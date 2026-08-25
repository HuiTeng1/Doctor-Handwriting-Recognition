"""
Reports the true production score on held-out test.csv, using the full-finetune
checkpoints (after finetune_all.py finishes the full, 100%-data finetune of a pipeline).
Written to results/final_test_scores_full.csv. Skipped automatically (with a message,
not an error) if the full finetune hasn't produced its summary csv yet.

This used to also run a "pilot-decision check" (evaluating pilot_compare.py's 15%-data
winner on test.csv too, sharing run_eval() with a winners_only=True, include_wer=False
call) - removed: that pilot checkpoint is never the production model, the decision it
would be "checking" was already made and executed via val CER alone, and test.csv
should be touched as rarely as possible. See
previous_code/C_decided_not_used/run_pilot_eval_only.py for the removed call.

Before that, these two were conflated into a single hardcoded run against the pilot
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
    cd train_model/doctor
    PYTHONIOENCODING=utf-8 py -3.12 -u final_test_eval.py
"""
import os
import sys

import pandas as pd
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCTOR_CLEAN_DIR = os.path.join(_HERE, "..", "..", "data", "doctor", "doctor_clean")
TEST_PREPROCESSED_ROOT = os.path.join(_HERE, "..", "..", "data", "doctor", "preprocessed_doctor_test")

# Same anchor finetune_all.py's to_relative_path() stores checkpoint paths relative to -
# resolve them back to an absolute path usable on this machine, wherever this project
# happens to be cloned/copied to.
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def to_absolute_path(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_PROJECT_ROOT, path))

sys.path.insert(0, os.path.join(_HERE, "preprocessing"))
from pipelines import PIPELINES as PREPROCESS_FNS  # noqa: E402



FINAL_CANDIDATES = {"baseline", "clhaa"}  # fajardo/benitez are pilot-only reference points,
# already ruled out by design (see finetune_all.py's top docstring) - excluded here so a
# stronger number from them (they can be trained under a different epoch budget than
# baseline/clhaa, e.g. the pilot's per-pipeline epoch caps) never gets picked as "the"
# winner by accident just because it has the lowest recorded CER in the summary table.


def pick_candidates(results_path, path_col):
    """Every row that's still a live final candidate (baseline/clhaa - see
    FINAL_CANDIDATES), each flagged with whether it's the one the validation-set CER
    picked as the winner. Winner selection uses best_val_cer (Lightning's per-batch-averaged
    training-time metric) - val_cer_jiwer, where present (CRNN only), is a second CER over
    the same val set computed the same way as test_cer (jiwer, pooled over every prediction),
    carried through purely for reporting since it's the number actually comparable to
    test_cer - it does not affect which pipeline is picked as the winner.
    Returns a list of (pipeline, val_cer, val_cer_jiwer, ckpt_path, is_winner)."""
    try:
        df = pd.read_csv(results_path)
    except (OSError, pd.errors.ParserError) as e:
        raise RuntimeError(f"Could not read results summary at {results_path}: {e}") from e
    df = df[df["pipeline"].isin(FINAL_CANDIDATES)]
    if df.empty:
        raise ValueError(f"{results_path}: none of {FINAL_CANDIDATES} found - nothing eligible to evaluate")
    winner_pipeline = df.loc[df["best_val_cer"].idxmin(), "pipeline"]
    has_jiwer_col = "best_val_cer_jiwer" in df.columns
    return [
        (row["pipeline"], row["best_val_cer"], row["best_val_cer_jiwer"] if has_jiwer_col else None,
         to_absolute_path(row[path_col]), row["pipeline"] == winner_pipeline)
        for _, row in df.iterrows()
    ]


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


def ensure_winner_images(model_name, pipeline_name):
    """Regenerate preprocessed_doctor_test/<pipeline>/ for the current winner, even when
    that model/pipeline's score is already recorded and the scoring loop below will skip
    it entirely. Without this, a machine that received final_test_scores_full.csv via git
    (but not the gitignored preprocessed_doctor_test/ folders themselves) has no way to
    get those folders back short of deleting a row from the CSV to force a full rerun.
    Cheap - just filters test.csv and writes any missing images, no checkpoint loading."""
    if model_name == "CRNN":
        crnn_dir = os.path.join(_HERE, "..", "normal", "crnn")
        sys.path.insert(0, crnn_dir)
        from modeling import CHARS
        test_df = load_and_filter_test(chars_filter=CHARS)
    else:
        test_df = load_and_filter_test(chars_filter=None)
    ensure_test_preprocessed(pipeline_name, test_df["filename"].tolist())


def eval_crnn(pipeline_name, ckpt_path, device="cpu"):
    crnn_dir = os.path.join(_HERE, "..", "normal", "crnn")
    sys.path.insert(0, crnn_dir)
    from hw_datasets import KaggleHandwritingDataModule, KaggleHandwrittenNames
    from training_modules import HandwritingRecogTrainModule
    from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS
    from ctc_decoder import best_path
    from jiwer import cer as jiwer_cer, wer as jiwer_wer

    hparams = {
        "lr": 1e-4, "gru_input_size": 256,
        "train_batch_size": 64, "val_batch_size": 256,
        "input_height": 36, "input_width": 324,
        "gru_hidden_size": 128, "gru_num_layers": 2, "num_classes": NUM_CLASSES,
        "filename_col": "filename", "label_col": "text",
        "train_img_path": TEST_PREPROCESSED_ROOT, "val_img_path": TEST_PREPROCESSED_ROOT,
    }

    test_df = load_and_filter_test(chars_filter=CHARS)
    img_dir = ensure_test_preprocessed(pipeline_name, test_df["filename"].tolist())

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"CRNN checkpoint not found at {ckpt_path} (summary CSV row is stale - checkpoint may have moved or been deleted)")
    try:
        module = HandwritingRecogTrainModule.load_from_checkpoint(
            ckpt_path, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
            map_location=torch.device("cpu") if device == "cpu" else None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CRNN checkpoint at {ckpt_path}: {e}") from e
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

    cer = jiwer_cer(all_gts, all_preds)
    wer = jiwer_wer(all_gts, all_preds)
    acc = sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_gts) * 100
    return cer, wer, acc, len(all_gts)


def eval_trocr(pipeline_name, final_model_dir, device="cpu"):
    from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizerFast, VisionEncoderDecoderModel
    from jiwer import cer as jiwer_cer, wer as jiwer_wer

    test_df = load_and_filter_test(chars_filter=None)
    img_dir = ensure_test_preprocessed(pipeline_name, test_df["filename"].tolist())

    if not os.path.isdir(final_model_dir):
        raise FileNotFoundError(f"TrOCR checkpoint dir not found at {final_model_dir} (summary CSV row is stale - checkpoint may have moved or been deleted)")
    try:
        image_processor = ViTImageProcessor.from_pretrained(final_model_dir, local_files_only=True)
        tokenizer = RobertaTokenizerFast.from_pretrained(final_model_dir, local_files_only=True)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(final_model_dir, local_files_only=True).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load TrOCR checkpoint at {final_model_dir}: {e}") from e
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

    cer = jiwer_cer(all_gts, all_preds)
    wer = jiwer_wer(all_gts, all_preds)
    acc = sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_gts) * 100
    return cer, wer, acc, len(all_gts)


def run_eval(crnn_results_path, trocr_results_path, out_filename, include_wer=True, winners_only=False):
    """Shared evaluation logic. crnn_results_path/trocr_results_path pick which summary
    table (and therefore which checkpoints) get evaluated on test.csv - the only thing
    that differs between the pilot-decision check and the full-finetune final score.
    include_wer=False drops the test_wer column entirely (used for the pilot-decision
    check, which was never meant to track WER - only the full-finetune final score is).
    winners_only=True evaluates only the pipeline currently flagged is_winner, skipping
    every other candidate entirely - this file's own docstring says "each of CRNN and
    TrOCR's chosen winners gets exactly one inference pass", singular, and the
    pilot-decision check has no further use for the runner-up's test.csv number: that
    pipeline's real production checkpoint (the full-finetune one, not this 15%-trained
    one) already gets its own honest test.csv pass in the full-finetune final score, so
    testing the pilot checkpoint here would just be an extra, purposeless look at
    test.csv for a model nothing downstream depends on.

    CRNN and TrOCR are evaluated independently: whichever one's results file exists gets
    evaluated and written out now; whichever doesn't is skipped (not an error) and can be
    filled in by a later run once it's ready. This lets e.g. CRNN's full-finetune score
    be reported today even though TrOCR's full finetune hasn't finished yet."""
    print("[final_test] Note: every candidate below gets exactly one evaluation on test.csv - none of these numbers should be used to revisit the baseline/clhaa pipeline choice afterward\n")

    out_path = os.path.join(_HERE, "..", "..", "data", "doctor", "results", out_filename)
    existing = pd.read_csv(out_path) if os.path.isfile(out_path) else pd.DataFrame()
    records = existing.to_dict("records")
    done_keys = {(r["model"], r["pipeline"]) for r in records}

    def save():
        pd.DataFrame(records).to_csv(out_path, index=False)

    def refresh_winners(model_name, candidates):
        """Keep is_winner correct on rows already written in a previous run too, not
        just newly-evaluated ones. Without this, a row written back when it was the
        only candidate (e.g. baseline before clhaa's training finished) keeps its
        is_winner=True forever, even after a later candidate with a lower val_cer
        (e.g. clhaa) is added - two rows would then both read is_winner=True."""
        current_winner = {pipeline: is_winner for pipeline, _, _, _, is_winner in candidates}
        changed = False
        for r in records:
            if r["model"] == model_name and r["pipeline"] in current_winner:
                if r["is_winner"] != current_winner[r["pipeline"]]:
                    r["is_winner"] = current_winner[r["pipeline"]]
                    changed = True
        return changed

    # Forced to CPU deliberately: this is a one-off inference pass, not worth fighting
    # a concurrently-running GPU training job for (see evaluate_cer_wer's note in
    # finetune_all.py for the same reasoning). Written to out_path after every single
    # candidate (not just once at the end) so a crash/interrupt mid-run - the slow TrOCR
    # passes make this a real risk, not a hypothetical one - only costs re-running
    # whichever candidate hadn't finished yet, not everything already evaluated so far.
    if os.path.isfile(crnn_results_path):
        crnn_candidates = pick_candidates(crnn_results_path, "best_ckpt")
        if refresh_winners("CRNN", crnn_candidates):
            save()
        if winners_only:
            crnn_candidates = [c for c in crnn_candidates if c[4]]
        for crnn_pipeline, crnn_val_cer, crnn_val_cer_jiwer, crnn_ckpt, is_winner in crnn_candidates:
            if is_winner:
                ensure_winner_images("CRNN", crnn_pipeline)
            if ("CRNN", crnn_pipeline) in done_keys:
                print(f"[final_test] CRNN {crnn_pipeline}: already in {out_filename}, skipping")
                continue
            print(f"[final_test] CRNN {crnn_pipeline} (val CER={crnn_val_cer:.4f}){' [winner]' if is_winner else ''}")
            crnn_cer, crnn_wer, crnn_acc, crnn_n = eval_crnn(crnn_pipeline, crnn_ckpt, device="cpu")
            print(f"[final_test] CRNN  ({crnn_pipeline}) test CER={crnn_cer:.4f}  test WER={crnn_wer:.4f}  exact-match={crnn_acc:.2f}%  n={crnn_n}")
            record = {"model": "CRNN", "pipeline": crnn_pipeline, "is_winner": is_winner, "val_cer": crnn_val_cer,
                      "val_cer_jiwer": crnn_val_cer_jiwer,
                      "test_cer": crnn_cer, "test_exact_match_pct": crnn_acc, "n": crnn_n}
            if include_wer:
                record["test_wer"] = crnn_wer
            records.append(record)
            save()
    else:
        print(f"[final_test] Skipping CRNN - {crnn_results_path} doesn't exist yet")

    if os.path.isfile(trocr_results_path):
        trocr_candidates = pick_candidates(trocr_results_path, "final_model_dir")
        if refresh_winners("TrOCR", trocr_candidates):
            save()
        if winners_only:
            trocr_candidates = [c for c in trocr_candidates if c[4]]
        for trocr_pipeline, trocr_val_cer, trocr_val_cer_jiwer, trocr_ckpt, is_winner in trocr_candidates:
            if is_winner:
                ensure_winner_images("TrOCR", trocr_pipeline)
            if ("TrOCR", trocr_pipeline) in done_keys:
                print(f"[final_test] TrOCR {trocr_pipeline}: already in {out_filename}, skipping")
                continue
            print(f"[final_test] TrOCR {trocr_pipeline} (val CER={trocr_val_cer:.4f}){' [winner]' if is_winner else ''}")
            trocr_cer, trocr_wer, trocr_acc, trocr_n = eval_trocr(trocr_pipeline, trocr_ckpt, device="cpu")
            print(f"[final_test] TrOCR ({trocr_pipeline}) test CER={trocr_cer:.4f}  test WER={trocr_wer:.4f}  exact-match={trocr_acc:.2f}%  n={trocr_n}")
            record = {"model": "TrOCR", "pipeline": trocr_pipeline, "is_winner": is_winner, "val_cer": trocr_val_cer,
                      "val_cer_jiwer": trocr_val_cer_jiwer,
                      "test_cer": trocr_cer, "test_exact_match_pct": trocr_acc, "n": trocr_n}
            if include_wer:
                record["test_wer"] = trocr_wer
            records.append(record)
            save()
    else:
        print(f"[final_test] Skipping TrOCR - {trocr_results_path} doesn't exist yet")

    if not records:
        print("[final_test] Nothing to evaluate - both results files are missing.")
        return

    print(f"\n[final_test] Saved to: {out_path}")


def main():
    # Capped deliberately: this runs on CPU so it can coexist with a GPU training job
    # elsewhere - left uncapped, PyTorch grabs every logical core for its BLAS/attention
    # kernels and starves that job's (CPU-side) data loading, slowing it down further.
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    results_dir = os.path.join(_HERE, "..", "..", "data", "doctor", "results")

    # The pilot-decision check (evaluating the 15%-trained pilot winner on test.csv) used
    # to run here too - removed: the pilot checkpoint is never the production model (the
    # full-finetune one is), the decision it would "check" was already made and executed
    # via val CER alone, and test.csv should be touched as rarely as possible. See
    # previous_code/C_decided_not_used/run_pilot_eval_only.py for the removed call and
    # final_test_scores_pilot.csv (also archived there) for the two numbers it produced
    # before this was decided against.
    print("=" * 60)
    print("[final_test] full-finetune final score (production checkpoints)")
    print("=" * 60)
    run_eval(
        os.path.join(results_dir, "crnn_finetune_summary.csv"),
        os.path.join(results_dir, "trocr_finetune_summary.csv"),
        "final_test_scores_full.csv",
    )


if __name__ == "__main__":
    main()
