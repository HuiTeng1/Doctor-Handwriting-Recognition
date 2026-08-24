"""
Objective 4 support: measures actual inference speed for CRNN and TrOCR (full-dataset
Baseline winners), instead of the "Faster" / "Slower" qualitative labels used in Table 3.8.
Loads the same production checkpoints final_test_eval.py evaluates, times a single-image
forward pass (batch=1, the realistic per-prescription-word latency) over N test images, and
reports mean/median ms per image and images/sec. Written to a csv so the numbers going
into the report's trade-off table are reproducible, not just quoted from a console log.

Forced to CPU by default so it can run without competing with a concurrent GPU training
job - pass --device cuda to benchmark GPU latency once the GPU is free.

Usage:
    cd train_model/doctor
    PYTHONIOENCODING=utf-8 py -3.12 -u inference_speed_benchmark.py [--device cpu|cuda] [--n 100]
"""
import argparse
import os
import statistics
import sys
import time

import pandas as pd
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data", "doctor")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
DOCTOR_CLEAN_DIR = os.path.join(DATA_DIR, "doctor_clean")
PREPROCESSED_TEST_ROOT = os.path.join(DATA_DIR, "preprocessed_doctor_test", "baseline")

# Same anchor finetune_all.py's to_relative_path() stores best_ckpt/final_model_dir
# relative to - resolve them back to an absolute path usable on this machine.
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def to_absolute_path(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_PROJECT_ROOT, path))
CKPT_ROOT = os.path.join(_HERE, "..", "..", "checkpoint", "doctor")


def load_test_filenames(n):
    df = pd.read_csv(os.path.join(DOCTOR_CLEAN_DIR, "test.csv"))
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df["filename"].tolist()[:n]


def benchmark_crnn(device, filenames):
    crnn_dir = os.path.join(_HERE, "..", "normal", "crnn")
    sys.path.insert(0, crnn_dir)
    from hw_datasets import KaggleHandwritingDataModule
    from training_modules import HandwritingRecogTrainModule
    from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES
    from ctc_decoder import best_path

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "crnn_finetune_summary.csv"))
    ckpt_path = to_absolute_path(summary.loc[summary["pipeline"] == "baseline", "best_ckpt"].iloc[0])

    hparams = {
        "lr": 1e-4, "gru_input_size": 256, "train_batch_size": 64, "val_batch_size": 256,
        "input_height": 36, "input_width": 324, "gru_hidden_size": 128, "gru_num_layers": 2,
        "num_classes": NUM_CLASSES, "filename_col": "filename", "label_col": "text",
        "train_img_path": PREPROCESSED_TEST_ROOT, "val_img_path": PREPROCESSED_TEST_ROOT,
    }
    module = HandwritingRecogTrainModule.load_from_checkpoint(
        ckpt_path, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
        map_location=torch.device(device),
    )
    module.eval().to(device)
    dummy_df = pd.DataFrame({"filename": filenames, "text": ["x"] * len(filenames)})
    transforms = KaggleHandwritingDataModule(dummy_df, dummy_df, hparams, LABEL_TO_INDEX).transforms

    times_ms = []
    with torch.no_grad():
        for i, fn in enumerate(filenames):
            img = Image.open(os.path.join(PREPROCESSED_TEST_ROOT, fn)).convert("L")
            tensor = transforms(img).unsqueeze(0).to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = torch.exp(module.model(tensor).permute(1, 0, 2)).cpu().numpy()
            best_path(output[0], module.chars)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= 5:  # skip first 5 as warmup
                times_ms.append((t1 - t0) * 1000)
    return times_ms


def benchmark_trocr(device, filenames):
    from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizerFast, VisionEncoderDecoderModel

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "trocr_finetune_summary.csv"))
    model_dir = to_absolute_path(summary.loc[summary["pipeline"] == "baseline", "final_model_dir"].iloc[0])

    image_processor = ViTImageProcessor.from_pretrained(model_dir, local_files_only=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()

    times_ms = []
    with torch.no_grad():
        for i, fn in enumerate(filenames):
            img = Image.open(os.path.join(PREPROCESSED_TEST_ROOT, fn)).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(pixel_values, num_beams=4, max_length=32)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= 5:
                times_ms.append((t1 - t0) * 1000)
    return times_ms


def summarize(name, times_ms):
    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    imgs_per_sec = 1000 / mean_ms
    print(f"[bench] {name}: mean={mean_ms:.2f}ms  median={median_ms:.2f}ms  throughput={imgs_per_sec:.2f} img/s  n={len(times_ms)}")
    return {"model": name, "mean_ms": mean_ms, "median_ms": median_ms, "images_per_sec": imgs_per_sec, "n": len(times_ms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))

    filenames = load_test_filenames(args.n + 5)  # +5 for warmup images consumed inside each benchmark
    print(f"[bench] device={args.device}  n={args.n}")

    records = []
    records.append(summarize("CRNN (baseline)", benchmark_crnn(args.device, filenames)))
    records.append(summarize("TrOCR (baseline)", benchmark_trocr(args.device, filenames)))

    out_path = os.path.join(RESULTS_DIR, f"inference_speed_benchmark_{args.device}.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[bench] Saved to: {out_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
