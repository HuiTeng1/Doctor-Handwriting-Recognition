import os
import shutil
import pandas as pd

# ------------------------------------------------------------------
# CONFIG — only thing you MUST set
# ------------------------------------------------------------------
# It merged 5 raw, separately-downloaded dataset folders into data/Doctor/{images/,
# labels.csv} - that merge has already happened once; its output (data/Doctor/) already
# exists in this project. The 5 raw source folders below were the author's paths at the
# time and won't exist on a different machine as-is - the folders themselves still exist
# (kept outside this repo, e.g. on external storage), just not necessarily at this exact
# path. If you need to re-run this script, update SOURCE_FOLDERS to wherever you've
# placed those 5 folders on the current machine first.
#
# If you're just setting this project up somewhere new and don't have those 5 raw
# folders handy: you do NOT need to run this script. You need data/Doctor/images/ (the
# already-merged output, ~58MB / 10,347 files) handed to you directly (it's gitignored -
# too large for git) and placed at final_code/doctor/data/Doctor/images/. labels.csv is
# already tracked in git. Once images/ is in place, start from clean_doctor_dataset.py.
SOURCE_FOLDERS = [
    r"C:\Users\limhu\OneDrive\Documents\Coding Practice\AI\PreProcessing\Doctor Handwriting Recognition Dataset",
    r"C:\Users\limhu\OneDrive\Documents\Coding Practice\AI\PreProcessing\RxHand Original",
    r"C:\Users\limhu\OneDrive\Documents\Coding Practice\AI\PreProcessing\Doctor’s Handwritten Prescription BD dataset\Testing",
    r"C:\Users\limhu\OneDrive\Documents\Coding Practice\AI\PreProcessing\Doctor’s Handwritten Prescription BD dataset\Training",
    r"C:\Users\limhu\OneDrive\Documents\Coding Practice\AI\PreProcessing\Doctor’s Handwritten Prescription BD dataset\Validation",
]

# Candidate column names to look for filename/text in each csv
# (add more here if your csvs use different names)
FILENAME_COL_CANDIDATES = ["filename", "file", "file_name", "image_name", "image", "img",
                            "images", "image_path", "img_path"]
TEXT_COL_CANDIDATES = ["text", "label", "labels", "transcription", "gt", "ground_truth",
                        "medicine_name", "generic_name", "word"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DATA_DIR = "data"
OUTPUT_FOLDER = os.path.join(DATA_DIR, "Doctor")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_FOLDER, "images")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, "labels.csv")


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

        df = pd.read_csv(csv_path)
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
    combine_folders()