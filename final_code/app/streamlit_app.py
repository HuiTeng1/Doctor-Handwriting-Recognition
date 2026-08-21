"""
Streamlit demo: upload a doctor's handwritten prescription word image, pick which finetuned
model to run it through (CRNN or TrOCR, both using the baseline preprocessing pipeline -
the one that won on every metric across every comparison in this project, see
doctor/KNOWN_ISSUES.md and doctor/results/), and show both the raw OCR prediction and the
closest match in a mock drug-name database (see drug_lexicon.py - an independent,
externally-sourced Bangladesh medicine database, not derived from this project's own
training data).

No training happens here - this only loads already-finetuned checkpoints and runs
inference on whatever image is uploaded:
  - CRNN:  doctor/checkpoints/baseline/epoch=54-...-0.2247.ckpt
           full finetune, test CER 23.09% (doctor/results/final_test_scores_full.csv)
  - TrOCR: doctor/checkpoints_trocr_pilot/baseline/final
           NOT the full finetune - TrOCR's full finetune isn't done yet (see
           doctor/KNOWN_ISSUES.md), this is only the 15%-data pilot checkpoint,
           pilot-test CER 19.31% (doctor/results/final_test_scores_pilot.csv). Numbers
           will change once the real finetune finishes - swap the path below when it does.

Usage:
    cd final_code/app
    streamlit run streamlit_app.py
"""
import os
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, Grayscale, RandomAutocontrast, ToTensor, Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_CRNN_DIR = os.path.join(_HERE, "..", "normal", "crnn")
_CRNN_CKPT_PATH = os.path.join(
    _HERE, "..", "doctor", "checkpoints", "baseline",
    "epoch=54-val-loss=0.869-val-char-error-rate=0.2247.ckpt",
)
_TROCR_CKPT_DIR = os.path.join(_HERE, "..", "doctor", "checkpoints_trocr_pilot", "baseline", "final")

sys.path.insert(0, _CRNN_DIR)
from training_modules import HandwritingRecogTrainModule  # noqa: E402
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS  # noqa: E402
from ctc_decoder import best_path  # noqa: E402

sys.path.insert(0, _HERE)
from drug_lexicon import load_vocabulary, load_variants, suggest_corrections  # noqa: E402

INPUT_HEIGHT, INPUT_WIDTH = 36, 324

CRNN_TRANSFORMS = Compose([
    Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    Grayscale(),
    RandomAutocontrast(p=1.0),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
])


@st.cache_resource
def load_crnn():
    hparams = {
        "lr": 1e-4, "gru_input_size": 256,
        "gru_hidden_size": 128, "gru_num_layers": 2, "num_classes": NUM_CLASSES,
        "input_height": INPUT_HEIGHT, "input_width": INPUT_WIDTH,
        "train_batch_size": 1, "val_batch_size": 1,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = HandwritingRecogTrainModule.load_from_checkpoint(
        _CRNN_CKPT_PATH, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
        map_location=torch.device("cpu") if device == "cpu" else None,
    )
    module.to(device).eval()
    return module, device


@st.cache_resource
def load_trocr():
    from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizerFast, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = ViTImageProcessor.from_pretrained(_TROCR_CKPT_DIR, local_files_only=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(_TROCR_CKPT_DIR, local_files_only=True)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(_TROCR_CKPT_DIR, local_files_only=True).to(device)
    model.eval()
    return model, processor, device


@st.cache_resource
def load_lexicon():
    return load_vocabulary(), load_variants()


def run_crnn(module, device, image: Image.Image) -> str:
    tensor = CRNN_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = module(tensor)[0].cpu().numpy()
    return best_path(output, CHARS)


def run_trocr(model, processor, device, image: Image.Image) -> str:
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    st.set_page_config(page_title="Doctor Handwriting OCR", page_icon="\U0001FA7A")
    st.title("Doctor Handwriting Recognition")

    trocr_weights_path = os.path.join(_TROCR_CKPT_DIR, "model.safetensors")
    trocr_available = os.path.isfile(trocr_weights_path)

    options = ["CRNN (full finetune, test CER 23.09%)"]
    if trocr_available:
        options.append("TrOCR (pilot checkpoint only - full finetune not done yet, CER 19.31%)")
    else:
        st.info(
            "TrOCR is disabled in this deployment - its checkpoint (~1.3GB) is too large "
            "to ship here. CRNN's checkpoint (21MB) is included, so it runs normally.",
            icon="ℹ️",
        )

    model_choice = st.radio("Model", options, horizontal=True)

    vocabulary, variants = load_lexicon()

    uploaded = st.file_uploader(
        "Upload a single word image (e.g. one medicine name)",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    if uploaded is None:
        return

    image = Image.open(uploaded)
    st.image(image, caption="Input", width=300)

    if model_choice.startswith("CRNN"):
        module, device = load_crnn()
        raw_prediction = run_crnn(module, device, image)
    else:
        model, processor, device = load_trocr()
        raw_prediction = run_trocr(model, processor, device, image)

    st.subheader("Raw OCR output")
    st.code(raw_prediction or "(empty)")

    st.subheader("Closest match in mock drug database")
    candidates = suggest_corrections(raw_prediction, vocabulary)
    if not candidates:
        st.warning("No close match found in the mock database - showing raw OCR output only.")
    else:
        best_name, best_score = candidates[0]
        st.success(f"**{best_name}**  (similarity {best_score:.0%})")
        if len(candidates) > 1:
            others = ", ".join(f"{name} ({score:.0%})" for name, score in candidates[1:])
            st.caption(f"Other candidates: {others}")

        rows = variants.get(best_name, [])
        if rows:
            st.caption(f"{len(rows)} form(s)/strength(s) on record for {best_name}:")
            st.dataframe(
                pd.DataFrame(rows).rename(columns={
                    "category_name": "form", "generic_name": "generic",
                    "manufacturer_name": "manufacturer", "price": "price (BDT)",
                }),
                hide_index=True,
            )


if __name__ == "__main__":
    main()
