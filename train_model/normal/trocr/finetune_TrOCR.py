"""
Stage 1 fine-tune script - fine-tune trocr-base-handwritten on IAM data only

Purpose:
  The first step of a two-stage transfer learning setup: give the model a solid
  foundation on clean, single-line handwritten text (IAM) first, then in stage 2
  take this checkpoint and fine-tune it further on the Doctor Handwriting
  (medicine name) data.

Data source:
  train/val use the 95/5 split from load_trocr_splits() in Iam_split.py (preprocessing/normal/) -
  this split reproduces the logic of the original Google Colab script (words_new.txt +
  iam_words.zip + seed=42, status==ok and the image opens fine), and is the same data
  used to train stage1_iam_finetuned_final. Note this is a different split from the
  70/10/20 one CRNN uses (load_crnn_splits()) - the two sides' CER numbers are not
  directly comparable, see the project discussion log for why.
  Image extraction and split logic don't live in this script - go to Iam_split.py to
  change either of those.

Local (VSCode) usage notes:
  If this machine has an NVIDIA GPU, make sure the matching CUDA build of PyTorch is
  installed; otherwise it falls back to CPU automatically and training will be very slow.

Output:
  - Training checkpoints saved to local disk (resumable)
  - Final model (processor + model) saved to local disk, for stage 2 to load
"""

# pip install transformers jiwer accelerate torch pillow

import os
import sys
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from jiwer import cer, wer

from transformers import (
    RobertaTokenizer,
    ViTImageProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'preprocessing', 'normal'))
from Iam_split import load_trocr_splits, extract_images, IMG_DIR

# ----------------------------------------------------------------------
# PATHS - edit as needed
# ----------------------------------------------------------------------
# Stage 1 model output dir - stage1_iam_finetuned_final was already trained on this
# 95/5 split; this script is currently kept for reference / for re-running later,
# it doesn't need to be run again.
STAGE1_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'checkpoint', 'normal', 'trocr')

MODEL_NAME = "microsoft/trocr-base-handwritten"
MAX_TARGET_LENGTH = 32      # IAM is sentences/phrases, a bit longer than names/medicine names, so leave some headroom
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 4              # 4GB VRAM (RTX3050) can't handle 16, dropped to 4 to avoid CUDA OOM
GRAD_ACCUM_STEPS = 4        # 4*4=16, use gradient accumulation to get the effective batch size back to 16
LEARNING_RATE = 5e-5
RANDOM_SEED = 42
SAVE_STEPS = 200            # checkpoint every 200 steps
                            # on CPU, consider a smaller value (e.g. 50-100) for denser checkpointing

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# Data prep - split/extraction logic lives in preprocessing/normal/Iam_split.py, shared with CRNN
# ----------------------------------------------------------------------
def rows_from_split(df):
    return [{"image_path": os.path.join(IMG_DIR, row.filename), "text": row.text}
            for row in df.itertuples()]


class OCRDataset(Dataset):
    """Turns (image path, text label) pairs into the pixel_values / labels TrOCR needs."""

    def __init__(self, rows, processor, max_target_length):
        self.rows = rows
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        # swap the pad token for -100 so it's ignored during training and doesn't count toward loss
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# ----------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------
def build_compute_metrics(processor):
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # swap -100 back to the pad token so decoding works correctly
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # filter out empty strings to avoid jiwer errors
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"cer": float("nan"), "wer": float("nan")}
        preds, labels = zip(*pairs)

        return {
            "cer": cer(list(labels), list(preds)),
            "wer": wer(list(labels), list(preds)),
        }

    return compute_metrics


# ----------------------------------------------------------------------
# Model loading and configuration
# ----------------------------------------------------------------------
def load_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("Warning: no GPU detected, training will be very slow.")

    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load/download '{MODEL_NAME}' from Hugging Face: {e}. "
            "Needs internet access on first run (cached locally after that)."
        ) from e

    # required config for generation on the decoder side
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # newer transformers requires generation-related params (max_length/num_beams etc.)
    # to be set on model.generation_config, not directly on model.config anymore
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    return processor, model, device


# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Checked against the model file directly rather than a "final/" subfolder: the
    # checkpoint currently on disk was moved in flat (files sit directly under
    # STAGE1_OUTPUT_DIR, e.g. STAGE1_OUTPUT_DIR/model.safetensors), not nested under an
    # extra "final/" folder the way this script's own save step below produces.
    existing_model_file = os.path.join(STAGE1_OUTPUT_DIR, "model.safetensors")
    if os.path.isfile(existing_model_file):
        raise FileExistsError(
            f"{existing_model_file} already exists (a trained Stage 1 model is already there). "
            "This script takes a long time to run and writes to the same directory by default. "
            "If you really want to retrain, move/rename that directory first, or change STAGE1_OUTPUT_DIR."
        )

    train_df, val_df = load_trocr_splits()
    extract_images([train_df, val_df])

    train_rows = rows_from_split(train_df)
    val_rows = rows_from_split(val_df)
    print(f"Train set: {len(train_rows)} rows, val set: {len(val_rows)} rows.")

    processor, model, device = load_model_and_processor()
    model.to(device)

    train_dataset = OCRDataset(train_rows, processor, MAX_TARGET_LENGTH)
    val_dataset = OCRDataset(val_rows, processor, MAX_TARGET_LENGTH)

    training_args = Seq2SeqTrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, BATCH_SIZE // 2),  # eval uses predict_with_generate, which is more VRAM-hungry than training, so halve the batch
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        predict_with_generate=True,
        eval_strategy="steps",       # evaluate by step, don't wait for a full epoch
        eval_steps=SAVE_STEPS,
        save_strategy="steps",       # checkpoint by step
        save_steps=SAVE_STEPS,
        save_total_limit=3,          # checkpoint more often and keep a few extra, in case one lands on a bad point
        logging_steps=50,
        fp16=(device == "cuda"),     # mixed precision on GPU for speed
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,     # lower CER is better
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics(processor),
    )

    print("=" * 60)
    print("Starting Stage 1 training (IAM only)...")
    print("=" * 60)

    # Auto-detect whether to resume: passing resume_from_checkpoint=True on a genuine first
    # run (no checkpoint-* yet) makes the Trainer raise "No valid checkpoint found" instead
    # of just training from scratch, so only pass True once a checkpoint actually exists -
    # that way re-running this same script after a disconnect/interruption "just works"
    # without anyone needing to hand-edit a flag first.
    resume_from_checkpoint = os.path.isdir(STAGE1_OUTPUT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(STAGE1_OUTPUT_DIR)
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("=" * 60)
    print("Training complete, saving final model...")
    print("=" * 60)

    final_model_dir = os.path.join(STAGE1_OUTPUT_DIR, "final")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)

    print(f"Stage 1 model saved to: {final_model_dir}")
    print("For the Stage 2 (Doctor Handwriting) fine-tune, just point MODEL_NAME at this path to continue training.")

    # print final validation set metrics
    metrics = trainer.evaluate()
    print("=" * 60)
    print("Final validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # also save a log for later reference, without needing to rerun
    log_path = os.path.join(STAGE1_OUTPUT_DIR, "stage1_final_metrics.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Stage: 1 (IAM only fine-tune)\n")
        f.write(f"Train size: {len(train_rows)}\n")
        f.write(f"Val size: {len(val_rows)}\n")
        f.write(f"Epochs: {NUM_TRAIN_EPOCHS}\n")
        f.write("-" * 40 + "\n")
        f.write("Final validation metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
    print(f"Final metrics saved to: {log_path}")

    # The full per-epoch history (including loss/cer/wer from every intermediate eval)
    # is already saved automatically in trainer_state.json, located at:
    print(f"Full per-epoch train/eval history is in trainer_state.json under each checkpoint dir "
          f"(e.g. {STAGE1_OUTPUT_DIR}\\checkpoint-XXX\\trainer_state.json)")
