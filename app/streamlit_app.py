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
inference on whatever image is uploaded. Both CRNN and TrOCR now have `baseline` and
`clhaa` as completed full finetunes (see `checkpoint/doctor/{crnn,trocr}/{baseline,clhaa}/
DONE.txt`), so the preprocessing toggle is fully live for both - each model reloads its
matching pipeline's checkpoint the moment the toggle changes, falling back to that same
pipeline's 15%-data pilot checkpoint only if a given full finetune isn't present.

Usage:
    cd app
    streamlit run streamlit_app.py
"""
import html
import math
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
_TROCR_PILOT_CKPT_ROOT = os.path.join(_HERE, "..", "checkpoint", "doctor", "trocr_pilot")
_CRNN_RESULTS_CSV = os.path.join(_HERE, "..", "data", "doctor", "results", "crnn_finetune_summary.csv")
_TROCR_RESULTS_CSV = os.path.join(_HERE, "..", "data", "doctor", "results", "trocr_finetune_summary.csv")

sys.path.insert(0, _CRNN_DIR)
from training_modules import HandwritingRecogTrainModule  # noqa: E402
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES, CHARS  # noqa: E402
from ctc_decoder import best_path  # noqa: E402

sys.path.insert(0, _PIPELINES_DIR)
from pipelines import apply_baseline, apply_clhaa  # noqa: E402

sys.path.insert(0, _HERE)
from drug_lexicon import load_vocabulary, load_variants, verify_against_database, levenshtein_distance  # noqa: E402

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
# .streamlit/config.toml (primaryColor = teal). Keep both files together.
# Palette/type follows the team's slide deck: navy background, teal accent dot, serif
# display headings (Playfair Display) over sans-serif body copy (Inter).
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

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

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4, .app-header h1 { font-family: 'Playfair Display', serif; }

/* Small teal dot used throughout the deck as an accent marker next to headings. */
.dot {
    display: inline-block;
    width: .5rem;
    height: .5rem;
    border-radius: 50%;
    background: #2dd4bf;
    margin-right: .5rem;
    vertical-align: middle;
}

.app-header {
    padding: 1.6rem 1.8rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0b1a33 0%, #1c6f6a 100%);
    color: #ffffff;
    margin-bottom: 1.25rem;
    box-shadow: 0 6px 24px rgba(45, 212, 191, 0.18);
}
.app-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.app-header p { margin: .45rem 0 0 0; opacity: .85; font-size: .92rem; font-family: 'Inter', sans-serif; }

/* Two side-by-side frames - left_frame (upload) and right_frame (result / mock-db /
   forms, stacked with dividers inside the ONE frame). Targets the container's own
   wrapper div directly via the "st-key-<key>" class Streamlit generates, so widgets
   placed inside via `with st.container(...):` are genuinely nested in the DOM (unlike
   an open-a-<div>-then-close-it-later approach, which silently fails to contain
   anything - each st.markdown call auto-closes its own HTML independently, it doesn't
   leak an open tag into sibling elements). */
.st-key-left_frame, .st-key-right_frame,
.st-key-compare_left_frame, .st-key-compare_right_frame {
    background: #132646 !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    border-color: rgba(45, 212, 191, 0.25) !important;
}
.st-key-left_frame h4, .st-key-right_frame h4,
.st-key-compare_left_frame h4, .st-key-compare_right_frame h4 {
    margin-top: 0;
    color: #ffffff;
}
/* Square off the upload frame specifically, and keep any preview image inside it tidy. */
.st-key-left_frame, .st-key-compare_left_frame { aspect-ratio: 1 / 1; overflow-y: auto; }
.st-key-left_frame [data-testid="stImage"] img,
.st-key-compare_left_frame [data-testid="stImage"] img {
    max-height: 240px;
    object-fit: contain;
    border-radius: 12px;
}

/* Per-model result cards in Compare mode - key is dynamic (one per model/pipeline
   combo), so match on the "st-key-compare_card_" prefix instead of an exact class. */
div[class*="st-key-compare_card_"] {
    background: #0b1a33 !important;
    border-radius: 14px !important;
    border-color: rgba(45, 212, 191, 0.3) !important;
}
.model-card-title {
    font-weight: 700;
    color: #f2f5f9;
    margin-bottom: .5rem;
    font-size: .95rem;
}

.raw-output {
    font-family: 'Courier New', monospace;
    font-size: 1.3rem;
    color: #9fb3c8;
    background: #0b1a33;
    border: 1px solid rgba(45, 212, 191, 0.3);
    border-radius: 14px;
    padding: .6rem 1rem;
    display: inline-block;
}
.corrected-output {
    font-family: 'Courier New', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f2f5f9;
    background: #0b1a33;
    border-radius: 14px;
    padding: .5rem 1.1rem;
    display: inline-block;
    border: 1px solid rgba(45, 212, 191, 0.5);
}
.pill {
    display: inline-block;
    background: #16294f;
    color: #2dd4bf;
    border: 1px solid rgba(45, 212, 191, 0.3);
    border-radius: 999px;
    padding: .2rem .8rem;
    font-size: .8rem;
    margin: .15rem .25rem .15rem 0;
    transition: transform .15s ease, border-color .15s ease;
}
.pill:hover { transform: translateY(-1px); border-color: #2dd4bf; }

.verify-box {
    border-radius: 14px;
    padding: .7rem 1rem;
    margin-top: .6rem;
    border: 1px solid;
}
.verify-box.match { background: rgba(45, 212, 191, 0.08); border-color: rgba(45, 212, 191, 0.5); }
.verify-box.suggested { background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.5); }
.verify-box.no-match { background: rgba(239, 68, 68, 0.08); border-color: rgba(239, 68, 68, 0.5); }
.verify-badge {
    font-weight: 700;
    font-size: .85rem;
    letter-spacing: .04em;
}
.verify-badge.match { color: #2dd4bf; }
.verify-badge.suggested { color: #f59e0b; }
.verify-badge.no-match { color: #ef4444; }
.verify-detail {
    font-family: 'Courier New', monospace;
    font-size: .85rem;
    color: #9fb3c8;
    margin-top: .35rem;
}

/* Style the model-select radio as soft rounded pill buttons */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: .5rem;
}
div[data-testid="stRadio"] label {
    background: #132646;
    border: 1px solid rgba(45, 212, 191, 0.25);
    border-radius: 999px;
    padding: .45rem 1.2rem;
    transition: transform .15s ease, border-color .15s ease;
}
div[data-testid="stRadio"] label:hover { transform: translateY(-1px); border-color: #2dd4bf; }

/* ---------- Micro-interactions: hover lift on the main frames and compare cards ---------- */
.st-key-left_frame, .st-key-right_frame,
.st-key-compare_left_frame, .st-key-compare_right_frame {
    transition: box-shadow .2s ease;
}
.st-key-left_frame:hover, .st-key-right_frame:hover,
.st-key-compare_left_frame:hover, .st-key-compare_right_frame:hover {
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.32);
}
div[class*="st-key-compare_card_"] {
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[class*="st-key-compare_card_"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
}

/* Upload dropzone - dashed teal border instead of Streamlit's default grey box, so it
   visually invites a drag-and-drop instead of looking like a disabled form field. */
[data-testid="stFileUploaderDropzone"] {
    background: #0b1a33 !important;
    border: 1.5px dashed rgba(45, 212, 191, 0.4) !important;
    border-radius: 14px !important;
    transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(45, 212, 191, 0.85) !important;
    background: #0e2140 !important;
}

/* Alerts (st.error) - rounded to match the rest of the deck instead of Streamlit's
   square-cornered default. */
div[data-testid="stAlert"] { border-radius: 14px !important; }

/* Winner badge + glow, used in Compare mode to call out the best-scoring model at a
   glance instead of making the reader scan four cards of numbers. */
.winner-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2dd4bf, #14b8a6);
    color: #05261f;
    font-weight: 700;
    font-size: .72rem;
    letter-spacing: .04em;
    border-radius: 999px;
    padding: .15rem .65rem;
    margin-bottom: .5rem;
}

/* Stepper - horizontal step indicator for the 4-step mock-database verification
   pipeline (normalize -> exact match -> fuzzy match -> decision), replacing a plain
   stack of caption lines with something scannable at a glance. */
.stepper { display: flex; align-items: center; margin: .8rem 0 .3rem 0; }
.step { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; }
.step-circle {
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .78rem; font-weight: 700;
    border: 2px solid rgba(45, 212, 191, 0.4);
    background: #0b1a33;
    color: #2dd4bf;
}
.step-done .step-circle { background: #2dd4bf; border-color: #2dd4bf; color: #05261f; }
.step-skip .step-circle { border-color: rgba(159, 179, 200, 0.3); color: #5a6d85; background: transparent; }
.step-label {
    font-size: .66rem;
    color: #9fb3c8;
    margin-top: .3rem;
    text-align: center;
    white-space: nowrap;
}
.step-done .step-label { color: #cbd5e1; }
.step-line { flex: 1 1 auto; height: 2px; margin: 0 .35rem; margin-bottom: 1.15rem; }
.step-line.done { background: rgba(45, 212, 191, 0.5); }
.step-line.skip { background: rgba(159, 179, 200, 0.2); }

/* Accuracy gauge - replaces a bare st.metric percentage with a ring so the OCR result
   panel has one clear visual focal point instead of three same-weight numbers. */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; }
.gauge-label { font-size: .72rem; color: #9fb3c8; margin-top: .25rem; letter-spacing: .03em; }

/* Gauge + stat chips side by side, in the Result panel and each Compare-mode card. */
.gauge-row { display: flex; align-items: center; gap: 1rem; margin-top: .5rem; flex-wrap: wrap; }
.gauge-row-stats { margin-top: 0; flex: 1 1 90px; }

/* Label/value chips laid out with flexbox instead of st.columns - used anywhere a
   metric pair sits inside an already-nested column (e.g. each Compare-mode card),
   so it wraps on its own via CSS instead of depending on a second level of Streamlit
   column nesting, which is unreliable on narrow screens. */
.stat-row { display: flex; gap: .7rem; margin-top: .5rem; flex-wrap: wrap; }
.stat-chip { flex: 1 1 80px; min-width: 80px; }
.stat-chip-label {
    font-size: .72rem;
    color: #9fb3c8;
    text-transform: uppercase;
    letter-spacing: .04em;
}
.stat-chip-value { font-size: 1.15rem; font-weight: 700; color: #f2f5f9; margin-top: .1rem; }

/* ---------- Responsive fallback for small screens ----------
   Reported: a teammate running this on a smaller display couldn't see the CER value
   at all - Streamlit's column layout doesn't reliably reflow narrow side-by-side
   columns on its own, so widen this to force every column row (top-level and nested)
   to stack into one column below this width instead of squeezing until something gets
   clipped out of view. */
@media (max-width: 900px) {
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .st-key-left_frame, .st-key-compare_left_frame { aspect-ratio: auto !important; }
}
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


def trocr_pilot_checkpoint(pipeline: str):
    """Pilot (15%-data) TrOCR checkpoint dir for `pipeline`, or None if that pilot run
    hasn't finished either (no DONE.txt under checkpoint/doctor/trocr_pilot/<pipeline>/)."""
    ckpt_dir = os.path.join(_TROCR_PILOT_CKPT_ROOT, pipeline)
    if not os.path.isfile(os.path.join(ckpt_dir, "DONE.txt")):
        return None
    return os.path.join(ckpt_dir, "final")


def trocr_pipeline_dir(pipeline: str):
    """The checkpoint dir to load for `pipeline`: its full finetune once done, falling
    back to that same pipeline's 15%-data pilot checkpoint until then. Returns
    (dir_or_None, is_full_finetune)."""
    full = trocr_pipeline_checkpoint(pipeline)
    if full is not None:
        return full, True
    pilot = trocr_pilot_checkpoint(pipeline)
    return pilot, False


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


def compute_cer(prediction: str, ground_truth: str):
    """Character Error Rate = edit distance / len(ground truth). None if no ground truth."""
    gt = ground_truth.strip()
    pred = prediction.strip()
    if len(gt) == 0:
        return None
    return levenshtein_distance(pred, gt) / len(gt)


def render_gauge(percent, label, size=104, stroke=9):
    """Inline SVG ring gauge for a 0-100 percentage - stands in for a bare st.metric so
    the result panel has one clear focal number instead of three same-weight ones.
    Colour tracks the same High/Medium/Low bands used by verify_against_database's
    confidence tiers, so a green ring and a MATCH badge always mean the same thing."""
    percent = max(0.0, min(100.0, percent))
    radius = (size - stroke) / 2
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - percent / 100)
    color = "#2dd4bf" if percent >= 70 else ("#f59e0b" if percent >= 40 else "#ef4444")
    center = size / 2
    return (
        f'<div class="gauge-wrap"><svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="#16294f" stroke-width="{stroke}" />'
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" '
        f'transform="rotate(-90 {center} {center})" />'
        f'<text x="{center}" y="{center + size * 0.07:.0f}" text-anchor="middle" font-size="{size * 0.22:.0f}" '
        f'font-weight="700" fill="#f2f5f9" font-family="Inter, sans-serif">{percent:.0f}%</text>'
        f'</svg><div class="gauge-label">{html.escape(label)}</div></div>'
    )


def render_stepper(result):
    """Horizontal step indicator mirroring verify_against_database's 4-step pipeline
    (normalize -> exact match -> fuzzy match -> decision). The fuzzy-match step renders
    as skipped when step 2 already resolved things via an exact match."""
    status = result["status"]
    if status == "empty":
        return ""

    fuzzy_skipped = status == "match"
    step_labels = ["Normalize", "Exact Match", "Fuzzy Match", "Decision"]
    step_states = ["done", "done", "skip" if fuzzy_skipped else "done", "done"]

    parts = ['<div class="stepper">']
    for i, (label, state) in enumerate(zip(step_labels, step_states)):
        mark = "✓" if state == "done" else "–"
        parts.append(
            f'<div class="step step-{state}"><div class="step-circle">{mark}</div>'
            f'<div class="step-label">{label}</div></div>'
        )
        if i < len(step_labels) - 1:
            line_state = "skip" if "skip" in (state, step_states[i + 1]) else "done"
            parts.append(f'<div class="step-line {line_state}"></div>')
    parts.append('</div>')
    return "".join(parts)


def render_stat_row(pairs):
    """Label/value pairs as a flexbox row that wraps on its own via CSS (see .stat-row)
    instead of via st.columns - used specifically where the pair would otherwise be a
    second level of nested columns (e.g. inside each Compare-mode card), since that
    nesting depth was the likely cause of a metric going invisible on a narrow screen."""
    chips = "".join(
        f'<div class="stat-chip"><div class="stat-chip-label">{html.escape(lbl)}</div>'
        f'<div class="stat-chip-value">{html.escape(str(val))}</div></div>'
        for lbl, val in pairs
    )
    return f'<div class="stat-row">{chips}</div>'


def render_gauge_row(accuracy, extra_pairs, size=104, stroke=9):
    """Accuracy gauge plus label/value chips side by side, as one HTML block - the same
    visual pairing used in both Single and Compare mode, kept as a single st.markdown
    call (no st.columns split) for the same small-screen-reflow reason render_stat_row
    replaced st.columns: Compare mode's cards already sit inside two levels of nested
    columns, and a third would be the least reliable to reflow on a narrow screen."""
    gauge_html = render_gauge(accuracy, "Accuracy", size=size, stroke=stroke)
    stats_html = render_stat_row(extra_pairs).replace('class="stat-row"', 'class="stat-row gauge-row-stats"')
    return f'<div class="gauge-row">{gauge_html}{stats_html}</div>'


def render_single_mode():
    trocr_ckpt_dir, trocr_is_full = trocr_pipeline_dir("baseline")
    trocr_available = trocr_ckpt_dir is not None and os.path.isfile(os.path.join(trocr_ckpt_dir, "model.safetensors"))
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

    if model_choice == "TrOCR":
        # Recompute for the toggled pipeline - the value from trocr_pipeline_dir("baseline")
        # above was only ever used to decide whether "TrOCR" appears as an option at all.
        trocr_ckpt_dir, trocr_is_full = trocr_pipeline_dir(pipeline_name)

    if model_choice == "CRNN" and not crnn_summary.empty and pipeline_name in crnn_summary.index:
        val_cer = crnn_summary.loc[pipeline_name, "best_val_cer"]
        st.caption(f"CRNN / {pipeline_name} - full finetune, val CER {val_cer:.2%}")
    elif model_choice == "TrOCR" and trocr_is_full:
        if not trocr_summary.empty and pipeline_name in trocr_summary.index:
            val_cer = trocr_summary.loc[pipeline_name, "best_val_cer"]
            st.caption(f"TrOCR / {pipeline_name} - full finetune, val CER {val_cer:.2%}")
        else:
            st.caption(f"TrOCR / {pipeline_name} - full finetune")
    elif model_choice == "TrOCR":
        st.caption(f"TrOCR / {pipeline_name} - pilot checkpoint (15% of training data), full finetune not done yet")

    vocabulary, variants = load_lexicon()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        with st.container(border=True, key="left_frame"):
            st.markdown('<h4><span class="dot"></span>1. \U0001F4E4 Upload image</h4>', unsafe_allow_html=True)
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
                st.markdown('<h4><span class="dot"></span>2. \U0001F50D Result</h4>', unsafe_allow_html=True)
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

    result = verify_against_database(raw_prediction, vocabulary)

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
                st.markdown(
                    render_gauge_row(accuracy, [
                        ("CER", f"{cer:.2%}"),
                        ("Exact match", "Yes" if exact_match else "No"),
                    ]),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Type the correct answer on the left to see accuracy / CER here.")

            st.divider()

            # ---- 3. Mock-database verification ----
            st.markdown('<h4><span class="dot"></span>3. \U0001F499 Mock-database verification</h4>', unsafe_allow_html=True)

            if result["status"] == "empty":
                st.markdown('<span class="raw-output">no input</span>', unsafe_allow_html=True)
            else:
                st.markdown(render_stepper(result), unsafe_allow_html=True)

                if result["status"] == "match":
                    st.markdown(f'<span class="corrected-output">{html.escape(result["matched_name"])}</span>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="verify-box match"><span class="verify-badge match">✅ MATCH</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f'<span class="corrected-output">{html.escape(result["matched_name"])}</span>', unsafe_allow_html=True)

                    if result["status"] == "suggested":
                        st.markdown(
                            f'<div class="verify-box suggested">'
                            f'<span class="verify-badge suggested">⚠️ SUGGESTED</span>'
                            f'<div class="verify-detail">"{html.escape(raw_prediction)}" → "{html.escape(result["matched_name"])}"<br>'
                            f'Confidence: {result["confidence"]} (distance {result["distance"]})</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="verify-box no-match">'
                            f'<span class="verify-badge no-match">❌ NO MATCH</span>'
                            f'<div class="verify-detail">Closest DB entry is too far '
                            f'(distance {result["distance"]}) to suggest with confidence.</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    if result["alternatives"]:
                        st.write("")
                        st.caption("Other candidates")
                        st.markdown(
                            "".join(
                                f'<span class="pill">{html.escape(name)} (d={d})</span>'
                                for name, d in result["alternatives"]
                            ),
                            unsafe_allow_html=True,
                        )

                with st.expander("Show verification steps"):
                    st.caption(f'Step 1 - Normalize: "{raw_prediction}" → "{result["normalized"]}"')
                    if result["status"] == "match":
                        st.caption(f'Step 2 - Exact match search: DB contains "{result["normalized"]}"? → YES')
                    else:
                        st.caption(f'Step 2 - Exact match search: DB contains "{result["normalized"]}"? → NO')
                        st.caption(
                            f'Step 3 - Fuzzy match (Levenshtein): closest is '
                            f'"{result["matched_name"]}" (distance {result["distance"]})'
                        )
                    st.caption(f'Step 4 - Decision: {result["status"].upper()}'
                               + (f' ({result["confidence"]} confidence)' if result["confidence"] else ''))

            # ---- 4. Forms & strengths on record (only when there's a match/suggestion) ----
            if result["status"] in ("match", "suggested"):
                best_name = result["matched_name"]
                rows = variants.get(best_name, [])
                if rows:
                    st.divider()
                    st.markdown(
                        f'<h4><span class="dot"></span>4. \U0001F4CB {html.escape(best_name)} - forms &amp; strengths on record</h4>',
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


# The 4 combinations this mode compares: both architectures x both preprocessing
# pipelines. Order controls both inference order and the 2x2 card grid layout below.
COMPARE_COMBOS = [
    ("CRNN", "baseline"),
    ("CRNN", "clhaa"),
    ("TrOCR", "baseline"),
    ("TrOCR", "clhaa"),
]


def _run_combo(model_name, pipeline, image, crnn_summary, trocr_summary):
    """Run one (model, pipeline) combo on `image`. Returns a result dict describing
    what happened - unavailable (no finished checkpoint), error (loading/inference
    raised), or ok (with the decoded prediction and, if known, its val CER)."""
    row = {"model": model_name, "pipeline": pipeline}
    try:
        if model_name == "CRNN":
            if crnn_pipeline_checkpoint(pipeline) is None:
                row["status"] = "unavailable"
                return row
            module, device = load_crnn(pipeline)
            image_for_model = PREP_FUNCS[pipeline](image)
            row["prediction"] = run_crnn(module, device, image_for_model)
            if not crnn_summary.empty and pipeline in crnn_summary.index:
                row["val_cer"] = crnn_summary.loc[pipeline, "best_val_cer"]
        else:
            ckpt_dir, is_full = trocr_pipeline_dir(pipeline)
            if ckpt_dir is None or not os.path.isfile(os.path.join(ckpt_dir, "model.safetensors")):
                row["status"] = "unavailable"
                return row
            model, processor, device = load_trocr(ckpt_dir)
            image_for_model = PREP_FUNCS[pipeline](image)
            row["prediction"] = run_trocr(model, processor, device, image_for_model)
            row["is_full"] = is_full
            if is_full and not trocr_summary.empty and pipeline in trocr_summary.index:
                row["val_cer"] = trocr_summary.loc[pipeline, "best_val_cer"]
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)
        return row
    row["status"] = "ok"
    return row


def render_compare_mode():
    st.markdown(
        '<div class="app-header"><h1>\U0001F4CA Compare All Models</h1>'
        '<p>Upload a single handwritten word to run it through all 4 model / preprocessing '
        'combinations side by side.</p></div>',
        unsafe_allow_html=True,
    )

    crnn_summary = load_crnn_summary()
    trocr_summary = load_trocr_summary()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        with st.container(border=True, key="compare_left_frame"):
            st.markdown('<h4><span class="dot"></span>1. \U0001F4E4 Upload image</h4>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Single word image",
                type=["png", "jpg", "jpeg", "bmp"],
                label_visibility="collapsed",
                key="compare_uploader",
            )

            image = None
            if uploaded is not None:
                try:
                    image = Image.open(uploaded)
                    image.load()  # force-decode now, so a corrupted file fails here, not later mid-pipeline
                except Exception as e:
                    st.error(f"Could not read this file as an image: {e}")
                    st.stop()
                st.image(image, use_container_width=True)

            ground_truth = st.text_input(
                "Correct answer (optional)",
                placeholder="Type the true word here to score every model",
                key="compare_ground_truth",
            )

    if uploaded is None:
        with col_right:
            with st.container(border=True, key="compare_right_frame"):
                st.markdown('<h4><span class="dot"></span>2. \U0001F4CA Comparison</h4>', unsafe_allow_html=True)
                st.caption("Upload an image on the left to compare all 4 models here.")
        return

    with st.spinner("Running all 4 models..."):
        rows = [_run_combo(m, p, image, crnn_summary, trocr_summary) for m, p in COMPARE_COMBOS]

    for row in rows:
        if row["status"] == "ok" and ground_truth:
            cer = compute_cer(row["prediction"], ground_truth)
            row["cer"] = cer
            row["accuracy"] = max(0.0, 1 - cer) * 100

    # Lowest CER wins - surfaced as a one-line summary plus a per-card badge/glow, so
    # the reader gets the headline answer before scanning all 4 cards' raw numbers.
    winner_key = None
    scored_rows = [r for r in rows if r["status"] == "ok" and "cer" in r]
    if scored_rows:
        winner = min(scored_rows, key=lambda r: r["cer"])
        winner_key = (winner["model"], winner["pipeline"])

    with col_right:
        with st.container(border=True, key="compare_right_frame"):
            st.markdown('<h4><span class="dot"></span>2. \U0001F4CA Comparison</h4>', unsafe_allow_html=True)
            if not ground_truth:
                st.caption("Type the correct answer on the left to see accuracy / CER for each model.")
            elif winner_key:
                st.markdown(
                    f'<div class="verify-box match"><span class="verify-badge match">'
                    f'🏆 Best match: {winner_key[0]} / {winner_key[1]}</span></div>',
                    unsafe_allow_html=True,
                )

            grid_cols = st.columns(2, gap="medium")
            for i, row in enumerate(rows):
                label = f'{row["model"]} / {row["pipeline"]}'
                card_key = f'compare_card_{row["model"]}_{row["pipeline"]}'
                is_winner = winner_key == (row["model"], row["pipeline"])
                with grid_cols[i % 2]:
                    if is_winner:
                        # Container `key` deterministically becomes the "st-key-<key>"
                        # class Streamlit generates, so this scoped rule reaches exactly
                        # this card and no other - see the CUSTOM_CSS comment above.
                        st.markdown(
                            f'<style>.st-key-{card_key} {{ border-color: rgba(45, 212, 191, 0.9) !important; '
                            f'box-shadow: 0 0 0 1px rgba(45, 212, 191, 0.4), 0 8px 22px rgba(45, 212, 191, 0.2) !important; }}</style>',
                            unsafe_allow_html=True,
                        )
                    with st.container(border=True, key=card_key):
                        # Always reserve the badge's slot, hidden when not the winner -
                        # otherwise the winner card is taller than its neighbours (extra
                        # badge line) and the grid rows stop lining up at the top.
                        badge_style = "" if is_winner else " visibility:hidden;"
                        st.markdown(
                            f'<span class="winner-badge" style="{badge_style}">🏆 Best match</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'<div class="model-card-title">{html.escape(label)}</div>', unsafe_allow_html=True)

                        if row["status"] == "unavailable":
                            st.caption("Checkpoint not available in this deployment.")
                            continue
                        if row["status"] == "error":
                            st.error(row["error"])
                            continue

                        st.markdown(
                            f'<span class="raw-output">{html.escape(row["prediction"]) or "(empty)"}</span>',
                            unsafe_allow_html=True,
                        )
                        caption_bits = []
                        if "val_cer" in row:
                            caption_bits.append(f'val CER {row["val_cer"]:.2%}')
                        if row["model"] == "TrOCR" and not row.get("is_full", True):
                            caption_bits.append("pilot checkpoint")
                        if caption_bits:
                            st.caption(" · ".join(caption_bits))

                        if "accuracy" in row:
                            st.markdown(
                                render_gauge_row(row["accuracy"], [("CER", f'{row["cer"]:.2%}')], size=84, stroke=8),
                                unsafe_allow_html=True,
                            )


def main():
    st.set_page_config(page_title="Doctor Handwriting OCR", page_icon="\U0001FA7A", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["Single model", "Compare all models"],
        horizontal=True,
        label_visibility="collapsed",
        key="app_mode",
    )
    st.write("")

    if mode == "Compare all models":
        render_compare_mode()
    else:
        render_single_mode()


if __name__ == "__main__":
    main()
