"""
Mock drug-name database: OCR post-correction via fuzzy matching against known medicine
names. Real-world handwriting OCR always has some error rate; this snaps a raw prediction
to the closest known name and reports how confident that match is, so the raw model
output and the suggested correction can both be shown side by side instead of silently
trusting either one alone.

Two vocabulary sources, kept both for comparison:

- v2 (default): medical_name_dataset/all_medicine_and_drug_price_data(20k)_Bangladesh.csv
  - a real, independently-sourced Bangladesh medicine database (MedEasy platform, via
  Kaggle - see DATA_PROVENANCE.md), 19,957 rows / 12,907 unique brand names, each with
  category (dosage form), generic name, strength, manufacturer, and price. This is an
  actual external reference, not derived from this project's own training data - fuzzy
  matches against it aren't "cheating" by reusing what the model was trained to output.

- v1 (legacy): doctor/data/Doctor_clean/labels.csv's deduplicated text column (1,871
  names). This is the training vocabulary itself, kept only as a comparison baseline -
  matching against it is somewhat circular (of course the model's output is close to
  something it was trained on), so it's not used by the app by default.
"""
import difflib
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEDICINE_CSV = os.path.join(
    _HERE, "..", "medical_name_dataset", "all_medicine_and_drug_price_data(20k)_Bangladesh.csv")
LEGACY_LABELS_CSV = os.path.join(_HERE, "..", "doctor", "data", "Doctor_clean", "labels.csv")


def load_vocabulary(medicine_csv=DEFAULT_MEDICINE_CSV):
    df = pd.read_csv(medicine_csv)
    names = df["medicine_name"].astype(str).str.strip()
    names = names[names.str.len() > 0]
    return sorted(names.unique())


def load_variants(medicine_csv=DEFAULT_MEDICINE_CSV):
    """medicine_name -> list of {category_name, generic_name, strength, manufacturer_name,
    price} dicts - one brand name can have several forms/strengths (e.g. Napa the tablet,
    the syrup, the IV infusion), each a separate row in the source csv."""
    df = pd.read_csv(medicine_csv)
    df["medicine_name"] = df["medicine_name"].astype(str).str.strip()
    cols = ["category_name", "generic_name", "strength", "manufacturer_name", "price"]
    variants = {}
    for name, group in df.groupby("medicine_name"):
        variants[name] = group[cols].to_dict("records")
    return variants


def load_vocabulary_v1_legacy(labels_csv=LEGACY_LABELS_CSV):
    """Deprecated: the training-data-derived vocabulary. See module docstring for why
    this isn't used by default - kept only so the two approaches can be compared."""
    df = pd.read_csv(labels_csv)
    texts = df["text"].astype(str).str.strip()
    texts = texts[texts.str.len() > 0]
    return sorted(texts.unique())


def suggest_corrections(raw_prediction, vocabulary, n=3, cutoff=0.4):
    """Up to n (name, similarity) pairs from vocabulary closest to raw_prediction, sorted
    by similarity descending. Empty list if nothing clears the cutoff (0-1 scale)."""
    if not raw_prediction:
        return []
    matches = difflib.get_close_matches(raw_prediction, vocabulary, n=n, cutoff=cutoff)
    scored = [(m, difflib.SequenceMatcher(None, raw_prediction, m).ratio()) for m in matches]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
