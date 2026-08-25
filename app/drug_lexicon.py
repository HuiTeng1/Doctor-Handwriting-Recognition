"""
Mock drug-name database: OCR post-correction via fuzzy matching against known medicine
names. Real-world handwriting OCR always has some error rate; this snaps a raw prediction
to the closest known name and reports how confident that match is, so the raw model
output and the suggested correction can both be shown side by side instead of silently
trusting either one alone.

The lookup (`verify_against_database`) is a 4-step pipeline:
  1. Normalize    - lowercase + trim the raw OCR prediction.
  2. Exact match   - does the DB contain this name verbatim (post-normalize)?
  3. Fuzzy match   - closest DB entries by Levenshtein (edit) distance.
  4. Decision      - turn the winning distance into a confidence tier (High/Medium/Low).

Two vocabulary sources, kept both for comparison:

- v2 (default): data/medical_name_dataset/all_medicine_and_drug_price_data(20k)_Bangladesh.csv
  - a real, independently-sourced Bangladesh medicine database (MedEasy platform, via
  Kaggle - see DATA_PROVENANCE.md), 19,957 rows / 12,907 unique brand names, each with
  category (dosage form), generic name, strength, manufacturer, and price. This is an
  actual external reference, not derived from this project's own training data - fuzzy
  matches against it aren't "cheating" by reusing what the model was trained to output.

- v1 (legacy): data/doctor/doctor_clean/labels.csv's deduplicated text column (1,871
  names). This is the training vocabulary itself, kept only as a comparison baseline -
  matching against it is somewhat circular (of course the model's output is close to
  something it was trained on), so it's not used by the app by default.
"""
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEDICINE_CSV = os.path.join(
    _HERE, "..", "data", "medical_name_dataset", "all_medicine_and_drug_price_data(20k)_Bangladesh.csv")
LEGACY_LABELS_CSV = os.path.join(_HERE, "..", "data", "doctor", "doctor_clean", "labels.csv")


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


def normalize_name(text):
    """Step 1: normalize - lowercase + trim."""
    return text.strip().lower()


def levenshtein_distance(a, b):
    """Edit distance: minimum single-character insertions/deletions/substitutions to
    turn a into b."""
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


def verify_against_database(raw_prediction, vocabulary, n=3, high_max=1, medium_max=3):
    """4-step mock-database verification pipeline (see module docstring).

    Returns a dict:
      status: "empty" | "match" | "suggested" | "no_match"
      normalized: the Step-1 output
      matched_name: best DB name (None only when status == "empty")
      distance: Levenshtein distance to matched_name (0 when status == "match")
      confidence: "High" | "Medium" | "Low" | None (None when status in ("empty", "match"))
      alternatives: up to n-1 further (name, distance) pairs, closest first
    """
    normalized = normalize_name(raw_prediction)
    if not normalized:
        return {"status": "empty", "normalized": normalized, "matched_name": None,
                "distance": None, "confidence": None, "alternatives": []}

    normalized_pairs = [(name, normalize_name(name)) for name in vocabulary]

    # Step 2: exact match search
    for name, key in normalized_pairs:
        if key == normalized:
            return {"status": "match", "normalized": normalized, "matched_name": name,
                    "distance": 0, "confidence": None, "alternatives": []}

    # Step 3: fuzzy match - Levenshtein distance to every DB entry
    ranked = sorted(
        ((name, levenshtein_distance(normalized, key)) for name, key in normalized_pairs),
        key=lambda pair: pair[1],
    )
    best_name, best_distance = ranked[0]

    # Step 4: verification decision
    if best_distance <= high_max:
        confidence = "High"
    elif best_distance <= medium_max:
        confidence = "Medium"
    else:
        confidence = "Low"
    status = "suggested" if confidence != "Low" else "no_match"

    return {
        "status": status,
        "normalized": normalized,
        "matched_name": best_name,
        "distance": best_distance,
        "confidence": confidence,
        "alternatives": ranked[1:n],
    }
