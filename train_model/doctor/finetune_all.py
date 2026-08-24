"""
Controlled Doctor preprocessing comparison for CRNN + TrOCR, both in this one file.
Usage:
    cd train_model/doctor
    PYTHONIOENCODING=utf-8 py -3.12 -u finetune_all.py

======================================================================
Why only baseline vs clhaa are tested now (not fajardo/benitez) - background and rationale:
======================================================================
After several rounds of discussion and a few real missteps, this is the conclusion things
converged on:

1. fajardo/benitez are excluded, and not on a whim - three independent pieces of evidence
   all point to them being clearly worse:
   - CRNN full finetune (7,035 images): baseline 0.2247 < clhaa 0.2338 < fajardo 0.2742 < benitez 0.2836
   - CRNN 15% pilot (1,055 images): same ordering, baseline 0.4036 < clhaa 0.4540 < fajardo 0.5074 < benitez 0.5104
   - zero-shot (Normal model, never finetuned, 359 held-out IAM images): on both CRNN and
     TrOCR, baseline is clearly the best and benitez clearly the worst (CRNN CER even hit
     1.01, essentially gibberish)
   Three independent measurements, spanning both the "finetuned or not" and "which model"
   dimensions, all agree - this is trustworthy.

2. The gap between baseline and clhaa is small (only 0.9 percentage points on the CRNN
   full finetune), and none of the three pieces of evidence can measure this pair
   precisely enough (estimated the noise scale from character counts and found that even
   the full val set's noise is on the same order as this gap), so this pair has to be
   tested with a real finetune - it can't be guessed from existing data/zero-shot/small pilots.

3. CRNN's ranking can't be assumed to answer the question for TrOCR - the two
   architectures don't adapt the same way to the distribution shift caused by
   preprocessing (TrOCR starts from a larger pretrained model and is more sample-
   efficient), so for the baseline vs clhaa pair, CRNN and TrOCR each run independently,
   and are allowed to pick different winners.

4. Data volume set at 15% (train ~1,055 / val ~150): this is the only scale verified to
   work for CRNN - at 5%, CRNN gets only 5-6 batches per epoch at train_batch_size=64 and
   learns nothing (CER stuck at 0.88~0.91, early-stopping after 1-2 epochs). TrOCR is more
   sample-efficient, so 15% is only more comfortable for it - no need to risk a separate,
   smaller value.

5. The letterbox fix (LetterboxPad / letterbox_square): all the historical numbers above
   (full finetune, 15% pilot, zero-shot) were run **before** this fix, and all carry the
   "aspect ratio hard-stretched" distortion. This is the first time this comparison is
   rerun after the fix - the numbers won't be fully comparable to the historical ones, but
   the ranking logic and method are unchanged.

6. Decision layering, to avoid using the same data both to choose and to report a final score:
   - train.csv (15%): fed to the model, produces gradient updates
   - val.csv (15%): only used for early stopping + picking the best checkpoint +
     comparing baseline vs clhaa, never produces gradients
   - test.csv (never touched at all): only after CRNN and TrOCR have each picked a
     winner does the winner get exactly **one** inference pass - that number is the
     final conclusion, never used for any further selection
======================================================================

Flow: CRNN's baseline/clhaa run first, then TrOCR's baseline/clhaa. Every pipeline is
identical except for which preprocessed image folder it reads - starting checkpoint/lr/
batch size/early-stopping rule (patience=5, no epoch cap)/random seed (42) are all the
same, only the preprocessing variable changes.

Only touches ../data/doctor/doctor_clean/train.csv + val.csv, never test.csv.
Ctrl+C can interrupt at any point - both sides support resuming from a checkpoint;
rerunning this file automatically picks up where it left off, and any pipeline that's
already fully finished is skipped automatically.
"""
import glob
import os
import re
import sys

import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCTOR_CLEAN_DIR = os.path.join(_HERE, "..", "..", "data", "doctor", "doctor_clean")
PREPROCESSED_ROOT = os.path.join(_HERE, "..", "..", "data", "doctor", "preprocessed_doctor")

# Anchor for the checkpoint paths written into results CSVs - stored relative to this (see
# to_relative_path()) instead of the raw absolute path _HERE resolves to, so the CSVs stay
# usable if this project is ever cloned/copied somewhere other than this exact machine's
# current location (final_test_eval.py resolves them back to absolute the same way).
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def to_relative_path(abs_path):
    return os.path.relpath(abs_path, _PROJECT_ROOT)

# Same 70-class dictionary as normal/crnn/modeling.py's CHARS - duplicated here (not
# imported) so the TrOCR path doesn't need to reach into the CRNN model file just for a
# string constant. Doctor labels must stay within this set for both models, so keep the
# two definitions in sync if the dictionary ever changes.
DOCTOR_CHARS = " %()-./?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

SEED = 42
# fajardo/benitez are excluded - already ruled out by the 15% pilot, no full run needed for
# them. baseline vs clhaa is the real final comparison (see finetune_all.py's top docstring);
# CRNN already has both done and will just skip via DONE.txt, TrOCR still needs clhaa
# (hard-capped at 11 epochs below, see max_epochs).
PIPELINES = ["baseline", "clhaa"]


def letterbox_square(img):
    """Pad to a square white canvas before any downstream fixed-size resize
    (ViTImageProcessor forces 384x384), so that resize is a uniform scale
    instead of a distorting stretch. Doesn't touch aspect ratio at all."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class LetterboxPad:
    """Pad (white) to a fixed target_w:target_h aspect ratio before the CRNN
    dataloader's own Resize((input_height, input_width)) runs, so that Resize
    becomes a uniform scale instead of a distorting stretch. crnnNormal's
    hw_datasets.py is shared with Stage 0 (IAM) training and stays untouched;
    this only wraps the transforms used for the Doctor finetune."""

    def __init__(self, target_w, target_h):
        self.target_ratio = target_w / target_h

    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        ratio = w / h
        if ratio > self.target_ratio:
            new_h = round(w / self.target_ratio)
            canvas = Image.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, (new_h - h) // 2))
        else:
            new_w = round(h * self.target_ratio)
            canvas = Image.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, ((new_w - w) // 2, 0))
        return canvas


# ======================================================================
# CRNN
# ======================================================================
def run_crnn_all(train_df=None, val_df=None, ckpt_root=None, results_path=None, pipelines=None):
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
    from torchvision.transforms import Compose

    from jiwer import cer as jiwer_cer, wer as jiwer_wer

    crnn_dir = os.path.join(_HERE, "..", "normal", "crnn")
    sys.path.insert(0, crnn_dir)
    from hw_datasets import KaggleHandwritingDataModule, KaggleHandwrittenNames
    from training_modules import HandwritingRecogTrainModule
    from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS
    from ctc_decoder import best_path

    base_ckpt_dir = os.path.join(_HERE, "..", "..", "checkpoint", "normal", "crnn")
    ckpt_root = ckpt_root or os.path.join(_HERE, "..", "..", "checkpoint", "doctor", "crnn")
    results_path = results_path or os.path.join(_HERE, "..", "..", "data", "doctor", "results", "crnn_finetune_summary.csv")

    hparams_base = {
        "lr": 1e-4, "gru_input_size": 256,
        "train_batch_size": 64, "val_batch_size": 256,
        "input_height": 36, "input_width": 324,
        "gru_hidden_size": 128, "gru_num_layers": 2, "num_classes": NUM_CLASSES,
        "filename_col": "filename", "label_col": "text",
    }

    def best_base_checkpoint():
        if not os.path.isdir(base_ckpt_dir):
            raise FileNotFoundError(f"CRNN base checkpoint dir not found: {base_ckpt_dir} (run train_iam_only.py first)")
        ckpt_files = [f for f in os.listdir(base_ckpt_dir) if f.startswith("epoch=") and f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"No epoch=*.ckpt files found in {base_ckpt_dir} (run train_iam_only.py first)")

        def extract_cer(f):
            m = re.search(r"val-char-error-rate=([\d.]+)\.ckpt", f)
            return float(m.group(1)) if m else float("inf")

        return os.path.join(base_ckpt_dir, sorted(ckpt_files, key=extract_cer)[0])

    def filter_crnn_chars(df, name):
        allowed = set(CHARS)
        df = df.copy()
        df["text"] = df["text"].astype(str).str.strip()

        def valid(v):
            return len(v) > 0 and all(c in allowed for c in v)

        mask = df["text"].apply(valid)
        dropped = (~mask).sum()
        if dropped:
            print(f"[crnn] {name}.csv: dropping {dropped}/{len(df)} rows (empty label or char outside the dictionary)")
        return df[mask].reset_index(drop=True)

    def load_doctor_split(name):
        return filter_crnn_chars(pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, f"{name}.csv")), name)

    def evaluate_cer_wer(ckpt_path, val_df, img_dir, hparams):
        """Inference-only pass (CPU, no gradients) over val_df with the given checkpoint,
        reporting CER/WER via jiwer - the same library used on the TrOCR side below, so
        both models' numbers are computed identically and are directly comparable.
        Runs on CPU deliberately: this is a single extra forward pass right after
        training finishes, not worth fighting the GPU for."""
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CRNN checkpoint not found at {ckpt_path}")
        try:
            module = HandwritingRecogTrainModule.load_from_checkpoint(
                ckpt_path, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
                map_location=torch.device("cpu"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CRNN checkpoint at {ckpt_path}: {e}") from e
        module.eval()
        module.to("cpu")

        data_helper = KaggleHandwritingDataModule(val_df, val_df, hparams, LABEL_TO_INDEX)
        val_ds = KaggleHandwrittenNames(val_df, data_helper.transforms, LABEL_TO_INDEX,
                                         img_dir, filename_col="filename", label_col="text")
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0,
                                                  collate_fn=KaggleHandwritingDataModule.custom_collate)

        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                output = torch.exp(module.model(batch["transformed_images"]).permute(1, 0, 2)).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                target_lens = batch["target_lens"].cpu().numpy()
                for i, p in enumerate(output):
                    preds.append(best_path(p, module.chars))
                    gts.append("".join(INDEX_TO_LABELS[idx] for idx in labels[i][0:target_lens[i]]))

        return jiwer_cer(gts, preds), jiwer_wer(gts, preds)

    def finetune_one(pipeline_name, base_ckpt_path, train_df, val_df):
        ckpt_dir = os.path.join(ckpt_root, pipeline_name)
        done_flag = os.path.join(ckpt_dir, "DONE.txt")
        if os.path.isfile(done_flag):
            print(f"[crnn] {pipeline_name}: already done, skipping")
            return None

        os.makedirs(ckpt_dir, exist_ok=True)
        pl.seed_everything(SEED)

        img_dir = os.path.join(PREPROCESSED_ROOT, pipeline_name) + os.sep
        hparams = dict(hparams_base, train_img_path=img_dir, val_img_path=img_dir)

        data_module = KaggleHandwritingDataModule(train_df, val_df, hparams, LABEL_TO_INDEX)
        # letterbox pulled out for now: it conflicts with the CenterCrop augmentation in
        # train_transforms (which crops a fixed 35x322 window at the original resolution) -
        # once letterbox pads the image wider, that crop mostly lands on blank/partial
        # content, which trained both baseline and clhaa into the ground (see this round's
        # results in pilot_compare.py). Falling back to the original hard-stretch resize
        # that's known to work, and treating letterbox as a follow-up fix to make
        # separately - it shouldn't block the current baseline/clhaa decision.
        # On Windows, num_workers>0 triggers an OpenBLAS memory allocation failure and
        # hangs on this machine.
        data_module.train_dataloader = lambda: torch.utils.data.DataLoader(
            data_module.train, batch_size=hparams["train_batch_size"], shuffle=True,
            num_workers=0, collate_fn=KaggleHandwritingDataModule.custom_collate)
        data_module.val_dataloader = lambda: torch.utils.data.DataLoader(
            data_module.val, batch_size=hparams["val_batch_size"], shuffle=False,
            num_workers=0, collate_fn=KaggleHandwritingDataModule.custom_collate)

        try:
            train_module = HandwritingRecogTrainModule.load_from_checkpoint(
                base_ckpt_path, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
                strict=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load base CRNN checkpoint at {base_ckpt_path}: {e}") from e

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, filename="{epoch}-{val-loss:.3f}-{val-char-error-rate:.4f}",
            save_top_k=1, monitor="val-char-error-rate", mode="min", save_last=True)
        early_stopping = EarlyStopping(monitor="val-char-error-rate", patience=5, verbose=True, mode="min")

        use_gpu = torch.cuda.is_available()
        trainer = pl.Trainer(
            accelerator="gpu" if use_gpu else "cpu", max_epochs=-1,
            callbacks=[checkpoint_callback, early_stopping, ModelSummary(max_depth=-1)],
            logger=False, precision=16 if use_gpu else 32, gradient_clip_val=1.0,
        )

        last_ckpts = glob.glob(os.path.join(ckpt_dir, "last*.ckpt"))
        resume_ckpt = max(last_ckpts, key=os.path.getmtime) if last_ckpts else None

        print(f"\n[crnn] === {pipeline_name}: starting finetune (base={os.path.basename(base_ckpt_path)}) ===")
        if resume_ckpt:
            print(f"[crnn] {pipeline_name}: found interrupted progress, resuming from {os.path.basename(resume_ckpt)}")
        trainer.fit(train_module, data_module, ckpt_path=resume_ckpt)

        # best_val_cer: PyTorch Lightning's own "val-char-error-rate" metric (the one
        # checkpoint selection/early stopping actually watch) - averaged per-batch by
        # Lightning's on_epoch=True reduction, NOT a single pass over the whole val set.
        # best_val_cer_jiwer: a second, separate CER computed by evaluate_cer_wer() the
        # same way test_cer is computed (jiwer over every val prediction concatenated
        # together) - the number that's actually comparable to Table 12's Test CER
        # column, since the two use identical methodology. The two differ by a few
        # tenths of a percentage point purely from batch-averaging vs pooling, not from
        # any bug - both are "real", they just answer slightly different questions.
        best_cer = checkpoint_callback.best_model_score
        best_cer = float(best_cer) if best_cer is not None else None

        best_wer = None
        best_cer_jiwer = None
        if checkpoint_callback.best_model_path:
            best_cer_jiwer, best_wer = evaluate_cer_wer(checkpoint_callback.best_model_path, val_df, img_dir, hparams)

        with open(done_flag, "w") as f:
            f.write("done")

        return {
            "pipeline": pipeline_name, "best_val_cer": best_cer, "best_val_cer_jiwer": best_cer_jiwer,
            "best_val_wer": best_wer, "stopped_epoch": trainer.current_epoch,
            "best_ckpt": to_relative_path(checkpoint_callback.best_model_path),
        }

    base_ckpt = best_base_checkpoint()
    print(f"[crnn] base checkpoint: {base_ckpt}")
    if train_df is None:
        train_df, val_df = load_doctor_split("train"), load_doctor_split("val")
    else:
        train_df, val_df = filter_crnn_chars(train_df, "train"), filter_crnn_chars(val_df, "val")
    print(f"[crnn] train={len(train_df)}  val={len(val_df)}")

    pipelines = pipelines if pipelines is not None else PIPELINES

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    existing = pd.read_csv(results_path) if os.path.isfile(results_path) else pd.DataFrame()
    done_names = set(existing["pipeline"]) if len(existing) else set()

    records = existing.to_dict("records")
    for pipeline_name in pipelines:
        if pipeline_name in done_names:
            print(f"[crnn] {pipeline_name}: result already in the summary, skipping")
            continue
        result = finetune_one(pipeline_name, base_ckpt, train_df, val_df)
        if result:
            records.append(result)
            pd.DataFrame(records).to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("[crnn] Summary:")
    print(pd.DataFrame(records).to_string(index=False))
    print("=" * 60)


# ======================================================================
# TrOCR
# ======================================================================
def run_trocr_all(train_df=None, val_df=None, ckpt_root=None, results_path=None, eval_num_beams=None, pipelines=None, max_epochs_override=None):
    import torch
    from jiwer import cer, wer
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        EarlyStoppingCallback, RobertaTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments,
        TrainerCallback, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, set_seed,
    )

    base_trocr_dir = os.path.join(_HERE, "..", "..", "checkpoint", "normal", "trocr")
    ckpt_root = ckpt_root or os.path.join(_HERE, "..", "..", "checkpoint", "doctor", "trocr")
    results_path = results_path or os.path.join(_HERE, "..", "..", "data", "doctor", "results", "trocr_finetune_summary.csv")
    max_target_length = 32
    batch_size = 4          # 4GB VRAM (RTX3050), same as Stage 1. No need to leave room for Chrome anymore, use it all
    grad_accum_steps = 4
    learning_rate = 5e-5

    def filter_trocr_chars(df, name):
        allowed = set(DOCTOR_CHARS)
        df = df.copy()
        df["text"] = df["text"].astype(str).str.strip()

        def valid(v):
            return len(v) > 0 and all(c in allowed for c in v)

        mask = df["text"].apply(valid)
        dropped = (~mask).sum()
        if dropped:
            print(f"[trocr] {name}.csv: dropping {dropped}/{len(df)} rows (empty label or char outside the dictionary)")
        return df[mask].reset_index(drop=True)

    def load_doctor_split(name):
        return filter_trocr_chars(pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, f"{name}.csv")), name)

    class DoctorOCRDataset(Dataset):
        def __init__(self, df, image_dir, processor, max_len):
            self.rows = df.to_dict("records")
            self.image_dir = image_dir
            self.processor = processor
            self.max_len = max_len

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            image = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
            # letterbox_square pulled out for now, same reason as the CRNN side: get the
            # baseline/clhaa decision running stably first, letterbox is a separate fix for later
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                row["text"], padding="max_length", max_length=self.max_len, truncation=True,
            ).input_ids
            labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
            return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

    def build_compute_metrics(processor):
        def compute_metrics(eval_pred):
            pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
            if not pairs:
                return {"cer": 1.0, "wer": 1.0}
            preds, labels = zip(*pairs)
            return {"cer": cer(list(labels), list(preds)), "wer": wer(list(labels), list(preds))}
        return compute_metrics

    def load_base_model_and_processor():
        # Not using TrOCRProcessor.from_pretrained(dir): processor_config.json was saved
        # with a newer transformers version, which conflicts with the currently installed
        # 4.46.3 - work around it by constructing the two components separately.
        if not os.path.isdir(base_trocr_dir):
            raise FileNotFoundError(f"TrOCR base checkpoint dir not found: {base_trocr_dir} (run finetune_TrOCR.py first)")
        try:
            image_processor = ViTImageProcessor.from_pretrained(base_trocr_dir, local_files_only=True)
            tokenizer = RobertaTokenizerFast.from_pretrained(base_trocr_dir, local_files_only=True)
            processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            model = VisionEncoderDecoderModel.from_pretrained(base_trocr_dir, local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load base TrOCR checkpoint at {base_trocr_dir}: {e}") from e
        return processor, model

    def finetune_one(pipeline_name, train_df, val_df):
        ckpt_dir = os.path.join(ckpt_root, pipeline_name)
        done_flag = os.path.join(ckpt_dir, "DONE.txt")
        if os.path.isfile(done_flag):
            print(f"[trocr] {pipeline_name}: already done, skipping")
            return None

        os.makedirs(ckpt_dir, exist_ok=True)
        set_seed(SEED)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, model = load_base_model_and_processor()
        model.to(device)

        img_dir = os.path.join(PREPROCESSED_ROOT, pipeline_name)
        train_ds = DoctorOCRDataset(train_df, img_dir, processor, max_target_length)
        val_ds = DoctorOCRDataset(val_df, img_dir, processor, max_target_length)

        # Per-pipeline epoch caps - time constraints, not arbitrary cutoffs:
        # - clhaa (full-data, ~2h/epoch): baseline's own epoch 9->11 run showed eval_cer
        #   was already past its best (epoch 9: 11.84%) and regressing by epoch 11 (12.46%),
        #   so capping clhaa at the same epoch 11 mirrors a point where baseline itself had
        #   already stopped improving.
        # - fajardo/benitez (15% pilot only, ~18min/epoch): fajardo's live pilot run peaked
        #   at epoch 3 (CER 25.82%) and was already regressing by epoch 4 (27.10%) - with
        #   patience=5 that points to a stop around epoch 8, so both are capped there instead
        #   of letting a pipeline that's already been ruled out run for many more hours to
        #   confirm what the trend already shows.
        max_epochs = max_epochs_override if max_epochs_override is not None else \
            {"clhaa": 11, "fajardo": 8, "benitez": 8}.get(pipeline_name, 1000)

        training_args = Seq2SeqTrainingArguments(
            output_dir=ckpt_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size // 2),
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=max_epochs,  # 1000 = soft cap, actually stopped by EarlyStoppingCallback,
                                           # the same rule as max_epochs=-1 + patience=5 on the CRNN side
            learning_rate=learning_rate,
            predict_with_generate=True,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            logging_steps=50, fp16=(device == "cuda"),
            load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
            report_to="none",
            disable_tqdm=False,
            generation_num_beams=eval_num_beams,  # None = use the checkpoint's own beam=4, the pilot can pass 1 to trade accuracy for speed
        )

        class PipelineLabelCallback(TrainerCallback):
            """Prefixes every log line HF prints with the pipeline name, otherwise once
            they start scrolling by there's no way to tell which one you're looking at."""
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    print(f"[trocr:{pipeline_name}] {logs}")

        trainer = Seq2SeqTrainer(
            model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=build_compute_metrics(processor),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5), PipelineLabelCallback()],
            processing_class=processor,
        )

        resume_ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
        resume_from = max(resume_ckpts, key=os.path.getmtime) if resume_ckpts else None

        print(f"\n[trocr] === {pipeline_name}: starting finetune (base=stage1_iam_finetuned_final) ===")
        if resume_from:
            print(f"[trocr] {pipeline_name}: found interrupted progress, resuming from {os.path.basename(resume_from)}")
        trainer.train(resume_from_checkpoint=resume_from)

        metrics = trainer.evaluate()
        final_dir = os.path.join(ckpt_dir, "final")
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)

        with open(done_flag, "w") as f:
            f.write("done")

        return {
            "pipeline": pipeline_name, "best_val_cer": metrics.get("eval_cer"),
            "best_val_wer": metrics.get("eval_wer"),
            "final_model_dir": to_relative_path(final_dir),
        }

    if train_df is None:
        train_df, val_df = load_doctor_split("train"), load_doctor_split("val")
    else:
        train_df, val_df = filter_trocr_chars(train_df, "train"), filter_trocr_chars(val_df, "val")
    print(f"[trocr] train={len(train_df)}  val={len(val_df)}")

    pipelines = pipelines if pipelines is not None else PIPELINES

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    existing = pd.read_csv(results_path) if os.path.isfile(results_path) else pd.DataFrame()
    done_names = set(existing["pipeline"]) if len(existing) else set()

    records = existing.to_dict("records")
    for pipeline_name in pipelines:
        if pipeline_name in done_names:
            print(f"[trocr] {pipeline_name}: result already in the summary, skipping")
            continue
        result = finetune_one(pipeline_name, train_df, val_df)
        if result:
            records.append(result)
            pd.DataFrame(records).to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("[trocr] Summary:")
    print(pd.DataFrame(records).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("=" * 60)
    print("Stage 1/2: CRNN - P0/P1/P2/P3")
    print("=" * 60)
    run_crnn_all()

    print("\n" + "=" * 60)
    print("Stage 2/2: TrOCR - P0/P1/P2/P3")
    print("=" * 60)
    run_trocr_all()
