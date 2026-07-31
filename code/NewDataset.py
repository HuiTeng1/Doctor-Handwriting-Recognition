"""
Combine multiple doctor-handwriting folders into a single folder + single CSV.

Run this ONCE before the main preprocessing.py script, if your doctor data
lives in separate folders, each with its own images/ + labels.csv.

Input: set the 3 full paths in SOURCE_FOLDERS below.
    <full_path>/doctor_1/images/*.jpg + labels.csv
    <full_path>/doctor_2/images/*.jpg + labels.csv
    <full_path>/doctor_3/images/*.jpg + labels.csv

Output (saved next to this script, in a local "data" folder):
    data/doctor_combined/images/*.jpg
    data/doctor_combined/labels.csv
"""

import os
import shutil
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# Output goes here — relative, so it lands in a "data" folder next to this script.
DATA_DIR = "data"

# INPUT: your 3 datasets live elsewhere, so put their FULL paths here.
# Windows: use r"..." raw strings.  Mac/Linux: use "/home/you/doctor_1" style.
SOURCE_FOLDERS = [
    r"C:\full\path\to\doctor_1",
    r"C:\full\path\to\doctor_2",
    r"C:\full\path\to\doctor_3",
]

OUTPUT_FOLDER = os.path.join(DATA_DIR, "doctor_combined")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_FOLDER, "images")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, "labels.csv")


def combine_folders(source_folders=SOURCE_FOLDERS,
                     output_images_dir=OUTPUT_IMAGES_DIR,
                     output_csv_path=OUTPUT_CSV_PATH,
                     copy_files=True):
    """
    Merge multiple folders (each with images/ + labels.csv) into one.

    Renames every image as "{source_folder_name}__{original_filename}" so
    files from different folders never collide.

    copy_files=True copies images (safe, keeps originals intact).
    Set to False to move instead (frees disk space, but empties originals).
    """
    os.makedirs(output_images_dir, exist_ok=True)
    all_rows = []

    for folder in source_folders:
        csv_path = os.path.join(folder, "labels.csv")
        images_dir = os.path.join(folder, "images")
        folder_name = os.path.basename(os.path.normpath(folder))

        if not os.path.isfile(csv_path):
            print(f"[WARN] No labels.csv at {csv_path} - skipping.")
            continue

        df = pd.read_csv(csv_path)
        if "filename" not in df.columns or "text" not in df.columns:
            raise ValueError(f"{csv_path} must have 'filename' and 'text' columns")

        new_filenames = []
        missing_count = 0

        for _, row in df.iterrows():
            old_name = row["filename"]
            old_path = os.path.join(images_dir, old_name)
            new_name = f"{folder_name}__{old_name}"
            new_path = os.path.join(output_images_dir, new_name)

            if not os.path.isfile(old_path):
                missing_count += 1
                new_filenames.append(None)  # mark for dropping later
                continue 

            if copy_files:
                shutil.copy2(old_path, new_path)
            else:
                shutil.move(old_path, new_path)

            new_filenames.append(new_name)

        df["filename"] = new_filenames
        df["source"] = folder_name

        if missing_count > 0:
            print(f"[WARN] {folder_name}: {missing_count} images referenced in "
                  f"labels.csv were not found on disk and will be dropped.")

        before = len(df)
        df = df.dropna(subset=["filename"])
        print(f"[INFO] {folder_name}: combined {len(df)}/{before} rows")

        all_rows.append(df)

    if not all_rows:
        raise ValueError("No source folders were successfully processed.")

    combined = pd.concat(all_rows, ignore_index=True)

    dupes = combined["filename"].duplicated().sum()
    if dupes > 0:
        print(f"[WARN] {dupes} duplicate filenames found after combining - "
              f"check your source folder names.")

    combined.to_csv(output_csv_path, index=False)
    print(f"\n[INFO] Done. {len(combined)} total rows written to {output_csv_path}")
    print(f"[INFO] {len(combined)} images copied into {output_images_dir}")

    return combined


if __name__ == "__main__":
    combine_folders()