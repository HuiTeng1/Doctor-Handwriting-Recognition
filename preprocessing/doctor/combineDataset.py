import os
import shutil
import zipfile
import pandas as pd

# ------------------------------------------------------------------
# CONFIG — only thing you MUST set
# ------------------------------------------------------------------
# It merges 5 raw, separately-downloaded dataset folders into this same doctor_raw/
# folder's images/ + labels.csv - that merge has already happened once; the output
# (images/ + labels.csv) already exists in this project. The 5 raw source folders are
# expected to live right alongside that output, inside doctor_raw/ itself (see
# SOURCE_FOLDERS below). Running this file directly (`__main__` below) calls
# ensure_source_folders_extracted() first, which auto-unzips the 3 raw zips if you place
# them straight in doctor_raw/ (named after each dataset, e.g. "RxHand Original.zip") -
# see that function's docstring for the per-zip handling and why "Doctor Handwriting
# Recognition Dataset" extracts into a folder literally named "89". Verified against the
# live labels.csv to reproduce it byte-for-byte, so re-running this after a from-scratch
# unzip does NOT change anything downstream. images/ and labels.csv are gitignored
# already; the 5 raw source folders (and the raw zips themselves) are gitignored too (see
# .gitignore) so they don't get accidentally committed.
#
# If you're just setting this project up somewhere new and don't have those 5 raw
# folders handy: you do NOT need to run this script. You need doctor_raw/images/ (the
# already-merged output, ~58MB / 10,347 files) handed to you directly (it's gitignored -
# too large for git) and placed at data/doctor/doctor_raw/images/. labels.csv is
# already tracked in git. Once images/ is in place, start from discover_doctor_duplicates.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data", "doctor")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "doctor_raw")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_FOLDER, "images")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, "labels.csv")

SOURCE_FOLDERS = [
    os.path.join(OUTPUT_FOLDER, "89"),  # the downloaded zip is called "Doctor Handwriting
    # Recognition Dataset", but this folder must stay named "89" - that's the literal
    # `source` value already baked into the live labels.csv (from whenever this folder was
    # first extracted, years ago), and pilot_compare.py's FAMILY_MAP hardcodes "89" ->
    # "phone" for stratified sampling. Renaming this folder would silently break that
    # mapping (rows would map to NaN family) without any error.
    os.path.join(OUTPUT_FOLDER, "RxHand Original"),
    os.path.join(OUTPUT_FOLDER, "Doctor’s Handwritten Prescription BD dataset", "Testing"),
    os.path.join(OUTPUT_FOLDER, "Doctor’s Handwritten Prescription BD dataset", "Training"),
    os.path.join(OUTPUT_FOLDER, "Doctor’s Handwritten Prescription BD dataset", "Validation"),
]

# Candidate column names to look for filename/text in each csv
# (add more here if your csvs use different names)
FILENAME_COL_CANDIDATES = ["filename", "file", "file_name", "image_name", "image", "img",
                            "images", "image_path", "img_path"]
TEXT_COL_CANDIDATES = ["text", "label", "labels", "transcription", "gt", "ground_truth",
                        "medicine_name", "generic_name", "word"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ------------------------------------------------------------------
# Optional: auto-unzip the 3 raw source zips (place them directly in OUTPUT_FOLDER, named
# after the dataset itself - see the *_ZIP_SEARCH_NAME below) before combining. Each was
# packaged differently by its original source, so each needs different handling to land
# at the exact SOURCE_FOLDERS path combine_folders() expects:
#   - Doctor Handwriting Recognition Dataset: no wrapping folder inside the zip - the csv
#     and img/img/*.jpg sit right at the zip root, so this one is extracted straight into
#     a newly-created "89" folder (see the SOURCE_FOLDERS comment above for why it's "89"
#     and not the zip's own name - the zip's name is only used to *find* the zip file).
#   - RxHand Original: wraps everything in one top-level folder, but that folder is named
#     "RxHand-Handwritten Prescription Word Image Dataset" (the zip's original name), not
#     "RxHand Original" - extracted to a scratch dir, then that one folder is renamed.
#   - Doctor's Handwritten Prescription BD dataset: already wraps everything in a
#     top-level folder with the exact name expected (containing Testing/Training/
#     Validation as subfolders) - extracted directly, no renaming needed.
_DOCTOR_ZIP_SEARCH_NAME = "Doctor Handwriting Recognition Dataset"
_DOCTOR_TARGET_DIR = SOURCE_FOLDERS[0]  # "89" - see SOURCE_FOLDERS comment above
_RXHAND_ZIP_NAME = os.path.basename(os.path.normpath(SOURCE_FOLDERS[1]))
_BD_ZIP_NAME = os.path.basename(os.path.normpath(os.path.dirname(SOURCE_FOLDERS[2])))


def _normalize_apostrophes(s):
    return s.replace("’", "'").replace("‘", "'")


def _find_zip(name):
    """Accepts `name.zip`, an accidentally double-extended `name.zip.zip`, and either a
    straight or curly apostrophe in `name` vs. the file actually on disk (the BD dataset's
    original name uses a curly one; anyone retyping/renaming it by hand types a straight
    one instead)."""
    normalized_target = _normalize_apostrophes(name)
    for candidate in os.listdir(OUTPUT_FOLDER):
        if not candidate.lower().endswith(".zip"):
            continue
        stem = candidate
        for suffix in (".zip.zip", ".zip"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if _normalize_apostrophes(stem) == normalized_target:
            return os.path.join(OUTPUT_FOLDER, candidate)
    return None


def _extract_flat(zip_path, target_dir):
    """Zip has no top-level wrapping folder - extract straight into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)


def _extract_and_rename(zip_path, target_dir):
    """Zip's images land inside one top-level folder under a different name than we need
    (possibly alongside a few loose label files at the zip root, e.g. RxHand's own
    Prescription_Labels.csv/.xlsx) - extract to a scratch dir, rename that one folder to
    target_dir, then move any loose sibling files in alongside it so find_csv() still
    finds them."""
    scratch_dir = target_dir + "__unzip_tmp"
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(scratch_dir)
    entries = os.listdir(scratch_dir)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(scratch_dir, e))]
    loose_files = [e for e in entries if os.path.isfile(os.path.join(scratch_dir, e))]
    if len(subdirs) != 1:
        raise RuntimeError(f"Expected exactly one top-level folder inside {zip_path}, found folders={subdirs}")
    shutil.move(os.path.join(scratch_dir, subdirs[0]), target_dir)
    for f in loose_files:
        shutil.move(os.path.join(scratch_dir, f), os.path.join(target_dir, f))
    shutil.rmtree(scratch_dir)


def _extract_asis(zip_path, parent_dir):
    """Zip's own top-level folder already has the exact name we need - extract directly
    into parent_dir and let the zip create that folder itself."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(parent_dir)


def ensure_source_folders_extracted():
    """Unzips whichever of the 3 raw source zips are present in OUTPUT_FOLDER and not
    already extracted. Safe to call every time - skips anything already extracted, and
    leaves anything with neither a zip nor an extracted folder for combine_folders() to
    report as missing. Never touches OUTPUT_IMAGES_DIR/OUTPUT_CSV_PATH."""
    jobs = [
        (_DOCTOR_ZIP_SEARCH_NAME, _DOCTOR_TARGET_DIR, _extract_flat),
        (_RXHAND_ZIP_NAME, os.path.join(OUTPUT_FOLDER, _RXHAND_ZIP_NAME), _extract_and_rename),
        (_BD_ZIP_NAME, os.path.join(OUTPUT_FOLDER, _BD_ZIP_NAME), _extract_asis),
    ]
    for zip_search_name, target_dir, extract_fn in jobs:
        name = os.path.basename(target_dir)
        if os.path.isdir(target_dir):
            print(f"[unzip] {name}: already extracted, skipping")
            continue
        zip_path = _find_zip(zip_search_name)
        if not zip_path:
            print(f"[unzip] {name}: no zip found ({zip_search_name}.zip) - skipping, "
                  f"combine_folders() will report this source as missing")
            continue
        print(f"[unzip] {name}: extracting from {os.path.basename(zip_path)}...")
        if extract_fn is _extract_asis:
            extract_fn(zip_path, OUTPUT_FOLDER)
        else:
            extract_fn(zip_path, target_dir)
        print(f"[unzip] {name}: done -> {target_dir}")


def find_csv(subfolder):
    """Find a csv file anywhere inside subfolder (top level first, then nested)."""
    top_level = [f for f in os.listdir(subfolder) if f.lower().endswith(".csv")]
    if top_level:
        return os.path.join(subfolder, top_level[0])
    for root, _, files in os.walk(subfolder):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    return None


def find_images_dir(subfolder):
    """
    Find the folder that actually holds the images:
    prefer a subfolder with lots of image files, else fall back to subfolder itself.
    """
    best_dir, best_count = subfolder, 0
    for root, _, files in os.walk(subfolder):
        count = sum(1 for f in files if f.lower().endswith(IMAGE_EXTS))
        if count > best_count:
            best_dir, best_count = root, count
    return best_dir


def guess_column(columns, candidates):
    lower_map = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def combine_folders(source_folders=SOURCE_FOLDERS,
                     output_images_dir=OUTPUT_IMAGES_DIR,
                     output_csv_path=OUTPUT_CSV_PATH,
                     copy_files=True):
    os.makedirs(output_images_dir, exist_ok=True)
    all_rows = []
    image_counter = 1

    missing = [f for f in source_folders if not os.path.isdir(f)]
    if missing:
        raise ValueError(f"These paths don't exist / aren't folders: {missing}")

    for folder in source_folders:
        folder_name = os.path.basename(os.path.normpath(folder))

        csv_path = find_csv(folder)
        if not csv_path:
            print(f"[WARN] {folder_name}: no csv found - skipping.")
            continue

        images_dir = find_images_dir(folder)

        try:
            df = pd.read_csv(csv_path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as e:
            print(f"[WARN] {folder_name}: could not read {csv_path} ({e}). Skipping.")
            continue
        filename_col = guess_column(df.columns, FILENAME_COL_CANDIDATES)
        text_col = guess_column(df.columns, TEXT_COL_CANDIDATES)

        if not filename_col or not text_col:
            print(f"[WARN] {folder_name}: couldn't auto-detect columns "
                  f"(found columns: {list(df.columns)}). Skipping. "
                  f"Add the right names to FILENAME_COL_CANDIDATES/TEXT_COL_CANDIDATES.")
            continue

        print(f"[INFO] {folder_name}: csv='{os.path.relpath(csv_path, folder)}', "
              f"images_dir='{os.path.relpath(images_dir, folder)}', "
              f"filename_col='{filename_col}', text_col='{text_col}'")

        new_filenames = []
        missing_count = 0

        for _, row in df.iterrows():
            old_name = row[filename_col]
            old_path = os.path.join(images_dir, old_name)

            if not os.path.isfile(old_path):
                missing_count += 1
                new_filenames.append(None)
                continue

            ext = os.path.splitext(old_name)[1].lower()
            new_name = f"img_{image_counter}{ext}"
            new_path = os.path.join(output_images_dir, new_name)

            if copy_files:
                shutil.copy2(old_path, new_path)
            else:
                shutil.move(old_path, new_path)

            new_filenames.append(new_name)
            image_counter += 1

        out_df = pd.DataFrame({
            "filename": new_filenames,
            "text": df[text_col].values,
        })
        out_df["source"] = folder_name

        if missing_count > 0:
            print(f"[WARN] {folder_name}: {missing_count} images referenced in "
                  f"the csv were not found on disk and will be dropped.")

        before = len(out_df)
        out_df = out_df.dropna(subset=["filename"])
        print(f"[INFO] {folder_name}: combined {len(out_df)}/{before} rows\n")

        all_rows.append(out_df)

    if not all_rows:
        raise ValueError("No source folders were successfully processed.")

    combined = pd.concat(all_rows, ignore_index=True)

    dupes = combined["filename"].duplicated().sum()
    if dupes > 0:
        print(f"[WARN] {dupes} duplicate filenames found after combining - "
              f"check your source folder names.")

    combined.to_csv(output_csv_path, index=False)
    print(f"[INFO] Done. {len(combined)} total rows written to {output_csv_path}")
    print(f"[INFO] {len(combined)} images copied into {output_images_dir}")

    return combined


if __name__ == "__main__":
    ensure_source_folders_extracted()
    combine_folders()