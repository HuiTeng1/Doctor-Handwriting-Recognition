"""
Streamlit demo: upload a doctor's handwritten prescription word image, pick which model
to run it through (CRNN or TrOCR), toggle between the `baseline` and `clhaa` preprocessing
pipelines (see `train_model/doctor/preprocessing/pipelines.py` - the toggle switches BOTH
the preprocessing function AND the checkpoint together, since a pipeline's checkpoint was
only ever finetuned on images that went through that same pipeline; running one pipeline's
preprocessing through the other's checkpoint would be a train/inference mismatch), type in
the correct word to score the prediction (character accuracy / CER), and show the closest
match in a mock drug-name database (see drug_lexicon.py - an independent, externally-sourced
Bangladesh medicine database, not derived from this project's own training data).

No training happens here - this only loads already-finetuned checkpoints and runs
inference on whatever image is uploaded. CRNN currently has both `baseline` and `clhaa`
as completed full finetunes (see `checkpoint/doctor/crnn/{baseline,clhaa}/DONE.txt`), so
its preprocessing toggle is fully live. TrOCR's `baseline` is now a completed full
finetune too (stopped at epoch 9, its recorded best - see `KNOWN_ISSUES.md`), but `clhaa`
isn't done yet, so TrOCR's preprocessing toggle stays disabled until
`checkpoint/doctor/trocr/clhaa/` also has its own `DONE.txt` - until then TrOCR always
runs its full-finetune `baseline` checkpoint (falling back further to the 15%-data pilot
checkpoint only if even that isn't present).

Usage:
    cd app
    streamlit run streamlit_app.py
"""
import html
import os
import re
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, Grayscale, RandomAutocontrast, ToTensor, Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_CRNN_DIR = os.path.join(_HERE, "..", "train_model", "normal", "crnn")
_PIPELINES_DIR = os.path.join(_HERE, "..", "train_model", "doctor", "preprocessing")
_CRNN_CKPT_ROOT = os.path.join(_HERE, "..", "checkpoint", "doctor", "crnn")
_TROCR_CKPT_ROOT = os.path.join(_HERE, "..", "checkpoint", "doctor", "trocr")
_TROCR_PILOT_CKPT_DIR = os.path.join(_HERE, "..", "checkpoint", "doctor", "trocr_pilot", "baseline", "final")
_CRNN_RESULTS_CSV = os.path.join(_HERE, "..", "data", "doctor", "results", "crnn_finetune_summary.csv")
_TROCR_RESULTS_CSV = os.path.join(_HERE, "..", "data", "doctor", "results", "trocr_finetune_summary.csv")

sys.path.insert(0, _CRNN_DIR)
from training_modules import HandwritingRecogTrainModule  # noqa: E402
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS  # noqa: E402
from ctc_decoder import best_path  # noqa: E402

sys.path.insert(0, _PIPELINES_DIR)
from pipelines import apply_baseline, apply_clhaa  # noqa: E402

sys.path.insert(0, _HERE)
from drug_lexicon import load_vocabulary, load_variants, suggest_corrections  # noqa: E402

INPUT_HEIGHT, INPUT_WIDTH = 36, 324

PREP_FUNCS = {"baseline": apply_baseline, "clhaa": apply_clhaa}

CRNN_TRANSFORMS = Compose([
    Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    Grayscale(),
    RandomAutocontrast(p=1.0),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
])

# Component-level styling only - the page background, inputs, toggle, etc. come from
# .streamlit/config.toml (primaryColor = baby blue). Keep both files together.
CUSTOM_CSS = """
<style>
/* Hide Streamlit's own chrome: top toolbar (Deploy button + menu), the sidebar
   collapse control (this app doesn't use a sidebar), and the little anchor-link
   icon Streamlit attaches to every heading. */
header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: none; }
[data-testid="stHeaderActionElements"] { display: none !important; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }

.app-header {
    padding: 1.4rem 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0d3a52 0%, #6fb8e0 100%);
    color: #ffffff;
    margin-bottom: 1.25rem;
    box-shadow: 0 6px 24px rgba(137, 207, 240, 0.25);
}
.app-header h1 { margin: 0; font-size: 1.7rem; }
.app-header p { margin: .35rem 0 0 0; opacity: .88; font-size: .9rem; }

/* Two side-by-side frames - left_frame (upload) and right_frame (result / mock-db /
   forms, stacked with dividers inside the ONE frame). Targets the container's own
   wrapper div directly via the "st-key-<key>" class Streamlit generates, so widgets
   placed inside via `with st.container(...):` are genuinely nested in the DOM (unlike
   an open-a-<div>-then-close-it-later approach, which silently fails to contain
   anything - each st.markdown call auto-closes its own HTML independently, it doesn't
   leak an open tag into sibling elements). */
.st-key-left_frame, .st-key-right_frame {
    border-radius: 20px !important;
    box-shadow: 0 4px 16px rgba(137, 207, 240, 0.08);
    border-color: #26313a !important;
}
.st-key-left_frame h4, .st-key-right_frame h4 {
    margin-top: 0;
    color: #89cff0;
}
/* Square off the upload frame specifically, and keep any preview image inside it tidy. */
.st-key-left_frame { aspect-ratio: 1 / 1; overflow-y: auto; }
.st-key-left_frame [data-testid="stImage"] img {
    max-height: 240px;
    object-fit: contain;
    border-radius: 12px;
}

.raw-output {
    font-family: 'Courier New', monospace;
    font-size: 1.3rem;
    color: #9ca3af;
    background: #0f1a20;
    border: 1px solid #26313a;
    border-radius: 14px;
    padding: .6rem 1rem;
    display: inline-block;
}
.corrected-output {
    font-family: 'Courier New', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f5f5f5;
    background: #0f1a20;
    border-radius: 14px;
    padding: .5rem 1.1rem;
    display: inline-block;
    border: 1px solid #26313a;
}
.pill {
    display: inline-block;
    background: #16222c;
    color: #89cff0;
    border: 1px solid #223544;
    border-radius: 999px;
    padding: .2rem .8rem;
    font-size: .8rem;
    margin: .15rem .25rem .15rem 0;
    transition: transform .15s ease, border-color .15s ease;
}
.pill:hover { transform: translateY(-1px); border-color: #89cff0; }

/* Style the model-select radio as soft rounded pill buttons */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: .5rem;
}
div[data-testid="stRadio"] label {
    background: #161616;
    border: 1px solid #26313a;
    border-radius: 999px;
    padding: .45rem 1.2rem;
    transition: transform .15s ease, border-color .15s ease;
}
div[data-testid="stRadio"] label:hover { transform: translateY(-1px); border-color: #89cff0; }
</style>
"""


def crnn_pipeline_checkpoint(pipeline: str):
    """Best full-finetune CRNN checkpoint for `pipeline`, or None if that pipeline's
    finetune hasn't finished (no DONE.txt yet)."""
    ckpt_dir = os.path.join(_CRNN_CKPT_ROOT, pipeline)
    if not os.path.isfile(os.path.join(ckpt_dir, "DONE.txt")):
        return None
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch=") and f.endswith(".ckpt")]
    if not ckpt_files:
        return None

    def extract_cer(f):
        m = re.search(r"val-char-error-rate=([\d.]+)\.ckpt", f)
        return float(m.group(1)) if m else float("inf")

    return os.path.join(ckpt_dir, sorted(ckpt_files, key=extract_cer)[0])


def trocr_pipeline_checkpoint(pipeline: str):
    """Full-finetune TrOCR checkpoint dir for `pipeline`, or None if that pipeline's
    finetune hasn't finished (no DONE.txt yet under checkpoint/doctor/trocr/<pipeline>/)."""
    ckpt_dir = os.path.join(_TROCR_CKPT_ROOT, pipeline)
    if not os.path.isfile(os.path.join(ckpt_dir, "DONE.txt")):
        return None
    return os.path.join(ckpt_dir, "final")


def trocr_baseline_dir():
    """The checkpoint dir the single (non-toggled) TrOCR path actually loads: the real
    full finetune once it's done, falling back to the 15%-data pilot checkpoint until
    then. Returns (dir, is_full_finetune)."""
    full = trocr_pipeline_checkpoint("baseline")
    if full is not None:
        return full, True
    return _TROCR_PILOT_CKPT_DIR, False


@st.cache_resource
def load_crnn_summary():
    if os.path.isfile(_CRNN_RESULTS_CSV):
        return pd.read_csv(_CRNN_RESULTS_CSV).set_index("pipeline")
    return pd.DataFrame()


@st.cache_resource
def load_trocr_summary():
    if os.path.isfile(_TROCR_RESULTS_CSV):
        return pd.read_csv(_TROCR_RESULTS_CSV).set_index("pipeline")
    return pd.DataFrame()


@st.cache_resource
def load_crnn(pipeline: str):
    ckpt_path = crnn_pipeline_checkpoint(pipeline)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No completed CRNN checkpoint found for pipeline '{pipeline}' under {_CRNN_CKPT_ROOT}")
    hparams = {
        "lr": 1e-4, "gru_input_size": 256,
        "gru_hidden_size": 128, "gru_num_layers": 2, "num_classes": NUM_CLASSES,
        "input_height": INPUT_HEIGHT, "input_width": INPUT_WIDTH,
        "train_batch_size": 1, "val_batch_size": 1,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        module = HandwritingRecogTrainModule.load_from_checkpoint(
            ckpt_path, hparams=hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
            map_location=torch.device("cpu") if device == "cpu" else None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CRNN checkpoint at {ckpt_path}: {e}") from e
    module.to(device).eval()
    return module, device


@st.cache_resource
def load_trocr(ckpt_dir: str):
    from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizerFast, VisionEncoderDecoderModel

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"TrOCR checkpoint directory not found: {ckpt_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        image_processor = ViTImageProcessor.from_pretrained(ckpt_dir, local_files_only=True)
        tokenizer = RobertaTokenizerFast.from_pretrained(ckpt_dir, local_files_only=True)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir, local_files_only=True).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load TrOCR checkpoint at {ckpt_dir}: {e}") from e
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


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(prediction: str, ground_truth: str):
    """Character Error Rate = edit distance / len(ground truth). None if no ground truth."""
    gt = ground_truth.strip()
    pred = prediction.strip()
    if len(gt) == 0:
        return None
    return _levenshtein(pred, gt) / len(gt)


def main():
    st.set_page_config(page_title="Doctor Handwriting OCR", page_icon="\U0001FA7A", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    trocr_ckpt_dir, trocr_is_full = trocr_baseline_dir()
    trocr_weights_path = os.path.join(trocr_ckpt_dir, "model.safetensors")
    trocr_available = os.path.isfile(trocr_weights_path)
    crnn_summary = load_crnn_summary()
    trocr_summary = load_trocr_summary()

    st.markdown(
        '<div class="app-header"><h1>\U0001F48A Prescription Handwriting Recognition</h1>'
        '<p>Upload a single handwritten word (e.g. one medicine name) to run OCR and match it against a known drug database.</p></div>',
        unsafe_allow_html=True,
    )

    # ---- Top controls: model select, then the preprocessing toggle right below it ----
    options = ["CRNN"]
    if trocr_available:
        options.append("TrOCR")
    model_choice = st.radio("Model", options, horizontal=True, label_visibility="collapsed")
    if not trocr_available:
        st.caption("TrOCR disabled in this deployment - its checkpoint (~1.3GB) is too large to ship here. CRNN (21MB) runs normally.")

    # The toggle switches preprocessing AND checkpoint together - a pipeline's checkpoint
    # was only ever finetuned on images that went through that same pipeline, so the two
    # always move as a pair. Only available once a model has *both* baseline and clhaa
    # as completed full finetunes.
    if model_choice == "CRNN":
        clhaa_ready = crnn_pipeline_checkpoint("clhaa") is not None
    else:
        clhaa_ready = trocr_pipeline_checkpoint("clhaa") is not None and trocr_pipeline_checkpoint("baseline") is not None

    if clhaa_ready:
        use_clhaa = st.toggle("clhaa preprocessing", value=False, help="ON = clhaa pipeline + clhaa-finetuned model. OFF = baseline (no preprocessing) + baseline-finetuned model.")
    else:
        use_clhaa = False
        st.toggle("clhaa preprocessing", value=False, disabled=True, help="Not available yet for this model - its clhaa full finetune isn't done.")

    pipeline_name = "clhaa" if use_clhaa else "baseline"

    if model_choice == "CRNN" and not crnn_summary.empty and pipeline_name in crnn_summary.index:
        val_cer = crnn_summary.loc[pipeline_name, "best_val_cer"]
        st.caption(f"CRNN / {pipeline_name} - full finetune, val CER {val_cer:.2%}")
    elif model_choice == "TrOCR" and trocr_is_full:
        if not trocr_summary.empty and "baseline" in trocr_summary.index:
            val_cer = trocr_summary.loc["baseline", "best_val_cer"]
            st.caption(f"TrOCR / baseline - full finetune, val CER {val_cer:.2%}")
        else:
            st.caption("TrOCR / baseline - full finetune")
    elif model_choice == "TrOCR":
        st.caption("TrOCR / baseline - pilot checkpoint (15% of training data), full finetune not done yet")

    vocabulary, variants = load_lexicon()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        with st.container(border=True, key="left_frame"):
            st.markdown("<h4>1. \U0001F4E4 Upload image</h4>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Single word image",
                type=["png", "jpg", "jpeg", "bmp"],
                label_visibility="collapsed",
            )

            image = None
            image_for_model = None
            if uploaded is not None:
                try:
                    image = Image.open(uploaded)
                    image.load()  # force-decode now, so a corrupted file fails here, not later mid-pipeline
                except Exception as e:
                    st.error(f"Could not read this file as an image: {e}")
                    st.stop()
                image_for_model = PREP_FUNCS[pipeline_name](image)
                if pipeline_name == "clhaa":
                    p1, p2 = st.columns(2)
                    with p1:
                        st.caption("Original")
                        st.image(image, use_container_width=True)
                    with p2:
                        st.caption("clhaa preprocessed")
                        st.image(image_for_model, use_container_width=True)
                else:
                    st.image(image, use_container_width=True)

            ground_truth = st.text_input(
                "Correct answer (optional)",
                placeholder="Type the true word here to score the prediction",
            )

    if uploaded is None:
        with col_right:
            with st.container(border=True, key="right_frame"):
                st.markdown("<h4>2. \U0001F50D Result</h4>", unsafe_allow_html=True)
                st.caption("Upload an image on the left to see the OCR result here.")
        return

    with st.spinner(f"Running {model_choice} inference..."):
        try:
            if model_choice == "CRNN":
                module, device = load_crnn(pipeline_name)
                raw_prediction = run_crnn(module, device, image_for_model)
            else:
                model, processor, device = load_trocr(trocr_ckpt_dir)
                raw_prediction = run_trocr(model, processor, device, image_for_model)
        except Exception as e:
            st.error(f"{model_choice} inference failed: {e}")
            st.stop()

    candidates = suggest_corrections(raw_prediction, vocabulary)

    with col_right:
        with st.container(border=True, key="right_frame"):
            # ---- 2. Result ----
            st.markdown("<h4>2. \U0001F50D Result</h4>", unsafe_allow_html=True)

            st.caption("Raw OCR output")
            st.markdown(f'<span class="raw-output">{html.escape(raw_prediction) or "(empty)"}</span>', unsafe_allow_html=True)

            st.write("")
            cer = compute_cer(raw_prediction, ground_truth) if ground_truth else None
            if cer is not None:
                accuracy = max(0.0, 1 - cer) * 100
                exact_match = raw_prediction.strip() == ground_truth.strip()
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy:.1f}%")
                m2.metric("CER", f"{cer:.2%}")
                m3.metric("Exact match", "Yes" if exact_match else "No")
            else:
                st.caption("Type the correct answer on the left to see accuracy / CER here.")

            st.divider()

            # ---- 3. Mock-database correction ----
            st.markdown("<h4>3. \U0001F499 Mock-database correction</h4>", unsafe_allow_html=True)
            if candidates:
                best_name, best_score = candidates[0]
                st.markdown(f'<span class="corrected-output">{html.escape(best_name)}</span>', unsafe_allow_html=True)
                st.caption(f"similarity {best_score:.0%}")
                if len(candidates) > 1:
                    st.write("")
                    st.caption("Other candidates")
                    st.markdown(
                        "".join(f'<span class="pill">{html.escape(name)} ({score:.0%})</span>' for name, score in candidates[1:]),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<span class="raw-output">no close match</span>', unsafe_allow_html=True)

            # ---- 4. Forms & strengths on record (only when there's a match) ----
            if candidates:
                best_name = candidates[0][0]
                rows = variants.get(best_name, [])
                if rows:
                    st.divider()
                    st.markdown(
                        f"<h4>4. \U0001F4CB {html.escape(best_name)} - forms &amp; strengths on record</h4>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(rows).rename(columns={
                            "category_name": "form", "generic_name": "generic",
                            "manufacturer_name": "manufacturer", "price": "price (BDT)",
                        }),
                        hide_index=True,
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
