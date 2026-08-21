"""
Sanity check (not the result used for the decision): take the Stage 0 (Normal, IAM-only)
CRNN checkpoint, run it zero-shot - no doctor-data finetuning at all - on doctor val
images processed by the 4 pipelines, and see whether the resulting CER ranking matches
the real, fully-finetuned ranking (baseline 0.2247 < clhaa 0.2338 < fajardo 0.2742 <
benitez 0.2836).

Purpose: just to check whether "test straight off the Normal checkpoint, no finetuning"
is a trustworthy method at all (or whether every pipeline would come out equally garbled
with no distinguishable ranking). Even if this matches the CRNN full-finetune result, it
doesn't prove the same method works for TrOCR - CRNN and TrOCR have different
architectures, this has been double-checked repeatedly on both sides, and this script
doesn't attempt to answer the TrOCR question.

Usage:
    cd final_code/doctor
    PYTHONIOENCODING=utf-8 py -3.12 -u zeroshot_normal_check.py
"""
import os
import sys

import pandas as pd
import torch
from ctc_decoder import best_path
from torchmetrics import CharErrorRate
from torchvision.transforms import Compose

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCTOR_CLEAN_DIR = os.path.join(_HERE, "data", "Doctor_clean")
PREPROCESSED_ROOT = os.path.join(_HERE, "preprocessed_doctor")

crnn_dir = os.path.join(_HERE, "..", "normal", "crnn")
sys.path.insert(0, crnn_dir)
from hw_datasets import KaggleHandwritingDataModule, KaggleHandwrittenNames
from training_modules import HandwritingRecogTrainModule
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS

from finetune_all import LetterboxPad, PIPELINES as _UNUSED  # reuse the same letterbox fix

CKPT_PATH = os.path.join(_HERE, "..", "normal", "checkpoint", "crnn",
                         "epoch=198-val-loss=0.529-val-char-error-rate=0.1439.ckpt")
PIPELINES_TO_TEST = ["baseline", "clhaa", "fajardo", "benitez"]

KNOWN_FINETUNED_RANKING = {
    "baseline": 0.2247378975152969,
    "clhaa": 0.23382216691970825,
    "fajardo": 0.27416494488716125,
    "benitez": 0.2836242914199829,
}


def load_doctor_val():
    df = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "val.csv"))
    allowed = set(CHARS)
    df["text"] = df["text"].astype(str).str.strip()
    mask = df["text"].apply(lambda v: len(v) > 0 and all(c in allowed for c in v))
    dropped = (~mask).sum()
    if dropped:
        print(f"[zeroshot] val.csv: dropping {dropped}/{len(df)} rows (empty label or char outside the dictionary)")
    return df[mask].reset_index(drop=True)


def main():
    hparams = {
        "lr": 1e-4, "gru_input_size": 256,
        "train_batch_size": 64, "val_batch_size": 256,
        "input_height": 36, "input_width": 324,
        "gru_hidden_size": 128, "gru_num_layers": 2, "num_classes": NUM_CLASSES,
        "filename_col": "filename", "label_col": "text",
        "train_img_path": PREPROCESSED_ROOT, "val_img_path": PREPROCESSED_ROOT,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[zeroshot] device: {device}")
    print(f"[zeroshot] loading frozen Normal checkpoint: {os.path.basename(CKPT_PATH)}")
    train_module = HandwritingRecogTrainModule.load_from_checkpoint(
        CKPT_PATH, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
        strict=True,
    )
    train_module.to(device).eval()

    val_df = load_doctor_val()
    print(f"[zeroshot] val n={len(val_df)}")

    letterbox = LetterboxPad(hparams["input_width"], hparams["input_height"])
    # Reuse the eval transforms defined inside KaggleHandwritingDataModule.__init__
    # (grayscale + autocontrast + normalize), just inserting letterbox in front,
    # consistent with finetune_all.py.
    dummy_dm = KaggleHandwritingDataModule(val_df, val_df, hparams, LABEL_TO_INDEX)
    eval_transforms = Compose([letterbox, dummy_dm.transforms])

    char_metric = CharErrorRate()
    results = {}

    for pipeline_name in PIPELINES_TO_TEST:
        img_dir = os.path.join(PREPROCESSED_ROOT, pipeline_name) + os.sep
        dataset = KaggleHandwrittenNames(val_df, eval_transforms, LABEL_TO_INDEX, img_dir,
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
                output = train_module.model(images).permute(1, 0, 2)
                output = torch.exp(output).cpu().numpy()
                for i, pred in enumerate(output):
                    predicted_string = best_path(pred, train_module.chars)
                    gt_string = "".join(INDEX_TO_LABELS[idx] for idx in labels[i][0:target_lens[i]])
                    all_preds.append(predicted_string)
                    all_gts.append(gt_string)

        cer = char_metric(all_preds, all_gts).item()
        results[pipeline_name] = cer
        print(f"[zeroshot] {pipeline_name}: CER={cer:.4f}")

    print("\n" + "=" * 60)
    print("[zeroshot] zero-shot ranking vs the real post-finetune ranking")
    print("=" * 60)
    zs_order = sorted(results, key=results.get)
    ft_order = sorted(KNOWN_FINETUNED_RANKING, key=KNOWN_FINETUNED_RANKING.get)
    print(f"zero-shot ranking : {' < '.join(zs_order)}")
    print(f"finetuned ranking : {' < '.join(ft_order)}")
    print(f"rankings match    : {zs_order == ft_order}")
    for p in PIPELINES_TO_TEST:
        print(f"  {p}: zero-shot={results[p]:.4f}  finetuned={KNOWN_FINETUNED_RANKING[p]:.4f}")


if __name__ == "__main__":
    main()
