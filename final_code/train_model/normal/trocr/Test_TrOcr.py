import os
import sys
import evaluate
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizerFast, VisionEncoderDecoderModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'preprocessing', 'normal'))
from Iam_split import load_trocr_splits, extract_images, IMG_DIR

# ----------------------------------------------------------------------
# 1. Path configuration
# ----------------------------------------------------------------------
# Evaluation uses the val half of the 95/5 split from load_trocr_splits() in
# Iam_split.py (final_code/preprocessing/normal/, 1,915 rows) - the same data and val set used to train
# stage1_iam_finetuned_final, reproduced from the original Colab script's split logic.
# This CER is not directly comparable to CRNN's test CER (the two use different splits) -
# see the project discussion log for why.

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'checkpoint', 'normal', 'trocr')

CSV_SAVE_PATH = os.path.join(MODEL_PATH, "test_results.csv")

BATCH_SIZE = 4  # bump this up (e.g. 16) if you have a GPU


# ----------------------------------------------------------------------
# 2. PyTorch Dataset loader (unchanged)
# ----------------------------------------------------------------------
class FastIAMDataset(Dataset):

  def __init__(self, data_rows, processor):
    self.data_rows = data_rows
    self.processor = processor

  def __len__(self):
    return len(self.data_rows)

  def __getitem__(self, idx):
    item = self.data_rows[idx]
    image = Image.open(item["image_path"]).convert("RGB")
    pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
    return {
        "pixel_values": pixel_values,
        "text": item["text"],
        "image_path": item["image_path"],
    }


# ----------------------------------------------------------------------
# 4. Prepare test data (the val half of TrOCR's own 95/5 split, same one used to
#    train stage1_iam_finetuned_final)
# ----------------------------------------------------------------------
train_df, val_df = load_trocr_splits()
extract_images([val_df])

val_rows = [{"image_path": os.path.join(IMG_DIR, row.filename), "text": row.text}
            for row in val_df.itertuples()]
print(f"[OK] Loaded trocr/val.csv: {len(val_rows)} rows for testing.")


def report(df):
  preds_list = df["prediction"].tolist()
  gt_list = df["ground_truth"].tolist()
  cer_val = cer_metric.compute(predictions=preds_list, references=gt_list)
  wer_val = wer_metric.compute(predictions=preds_list, references=gt_list)
  char_acc = (1.0 - cer_val) * 100.0
  word_acc = (1.0 - wer_val) * 100.0
  print("\n" + "=" * 50)
  print(f"[Stage 1 IAM] evaluation results:")
  print(f"  Char accuracy : {char_acc:.2f}% (CER: {cer_val:.4f})")
  print(f"  Word accuracy : {word_acc:.2f}% (WER: {wer_val:.4f})")
  print("=" * 50)


cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

# If the CSV already covers every val image, just read it and compute results -
# no need to load the model or build a dataloader
needed_names = {os.path.basename(r["image_path"]) for r in val_rows}
if os.path.exists(CSV_SAVE_PATH):
  existing_df = pd.read_csv(CSV_SAVE_PATH)
  if "image_name" in existing_df.columns and needed_names <= set(existing_df["image_name"].astype(str)):
    print(f"[trocr] All {len(needed_names)} images already tested, reading existing results "
          f"(delete {CSV_SAVE_PATH} to force a re-test)")
    report(existing_df[existing_df["image_name"].isin(needed_names)])
    sys.exit(0)

# ----------------------------------------------------------------------
# 5. Load model and config (auto-detects CPU/GPU)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Running on: [{device.upper()}]")

# Not using TrOCRProcessor.from_pretrained(dir): processor_config.json was saved with an
# older transformers version and isn't compatible with the parsing logic in the version
# currently installed (TypeError: got multiple values for argument 'image_processor').
# Load the two components separately and assemble them by hand instead, matching the
# approach used in doctor/finetune_all.py.
image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_PATH, local_files_only=True
).to(device)
model.eval()

model.generation_config.max_new_tokens = 32

dataset = FastIAMDataset(val_rows, processor)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    # num_workers=0: this script's main flow runs at module level, not wrapped in
    # if __name__=="__main__", so on Windows the spawn start method would re-import
    # the whole file for each worker - num_workers>0 causes a recursive-spawn hang.
)

# ----------------------------------------------------------------------
# 6. Batched inference (with resume support, logic unchanged)
# ----------------------------------------------------------------------
results = []
processed_images = set()

if os.path.exists(CSV_SAVE_PATH):
  try:
    existing_df = pd.read_csv(CSV_SAVE_PATH)
    if "image_name" in existing_df.columns and len(existing_df) > 0:
      processed_images = set(existing_df["image_name"].astype(str).tolist())
      drop_cols = ["overall_char_accuracy", "overall_word_accuracy"]
      existing_df = existing_df.drop(
          columns=[c for c in drop_cols if c in existing_df.columns]
      )
      results = existing_df.to_dict(orient="records")
      print(
          f"[Resume] Found an existing CSV, skipping {len(processed_images)} already-inferred "
          f"images and resuming..."
      )
  except Exception as e:
    print(f"[Warn] Failed to read existing CSV ({e}), starting inference from scratch.")

print("\n[Start] Running inference and evaluation...")
with torch.no_grad():
  for batch in tqdm(dataloader):
    unprocessed_indices = [
        i
        for i, path in enumerate(batch["image_path"])
        if os.path.basename(path) not in processed_images
    ]

    if not unprocessed_indices:
      continue

    pixel_values = batch["pixel_values"][unprocessed_indices].to(device)
    generated_ids = model.generate(pixel_values)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for idx, pred in zip(unprocessed_indices, preds):
      path = batch["image_path"][idx]
      gt = batch["text"][idx]
      img_name = os.path.basename(path)

      results.append({
          "image_name": img_name,
          "ground_truth": gt,
          "prediction": pred,
          "is_exact_match": (gt == pred),
      })
      processed_images.add(img_name)

    temp_df = pd.DataFrame(results)
    temp_df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 7. Final metrics and CSV export (unchanged)
# ----------------------------------------------------------------------
df = pd.DataFrame(results)

if len(df) > 0:
  df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")
  report(df)
  print(f"[OK] Results fully written to CSV:")
  print(f"     -> {CSV_SAVE_PATH}")
  print("=" * 50)
else:
  print("[Warn] No valid data was collected.")
