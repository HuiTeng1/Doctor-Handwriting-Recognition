import os
import zipfile
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'normal')
IAM_DATASET_DIR = os.path.join(NORMAL_DATA_DIR, 'I_Am_Dataset')

IAM_WORDS_ZIP = os.path.join(IAM_DATASET_DIR, 'iam_words.zip')
IAM_WORDS_TXT = os.path.join(IAM_DATASET_DIR, 'words_new.txt')
IMG_DIR = os.path.join(IAM_DATASET_DIR, 'iam_words_images')
CRNN_SPLITS_DIR = os.path.join(NORMAL_DATA_DIR, 'crnn')     # data/normal/crnn
TROCR_SPLITS_DIR = os.path.join(NORMAL_DATA_DIR, 'trocr')   # data/normal/trocr


def load_crnn_splits():
    """CRNN-specific IAM train/val/test split (70/10/20). CRNN has already been trained on this
    (val CER 0.1439) - do not change the split logic itself."""
    train_df = pd.read_csv(os.path.join(CRNN_SPLITS_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(CRNN_SPLITS_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(CRNN_SPLITS_DIR, 'test.csv'))
    return train_df, val_df, test_df


def load_trocr_splits():
    """TrOCR-specific IAM train/val split (95/5). Reproduces the split used to train
    stage1_iam_finetuned_final (same words_new.txt + iam_words.zip + seed=42, matching the
    original Colab script's logic - no separate test set, val doubles as the eval set)."""
    train_df = pd.read_csv(os.path.join(TROCR_SPLITS_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(TROCR_SPLITS_DIR, 'val.csv'))
    return train_df, val_df


def extract_images(dfs):
    """Extract the images used by dfs from iam_words.zip into IMG_DIR on demand
    (flat directory, filenames carry an i_ prefix), skipping any that already exist."""
    os.makedirs(IMG_DIR, exist_ok=True)

    needed = set()
    for df in dfs:
        needed.update(df['filename'].tolist())

    missing = [fn for fn in needed if not os.path.isfile(os.path.join(IMG_DIR, fn))]
    if not missing:
        print(f"[iam_split] Images already extracted ({len(needed)}), skipping")
        return

    print(f"[iam_split] Extracting {len(missing)} images from iam_words.zip...")
    with zipfile.ZipFile(IAM_WORDS_ZIP) as zf:
        for fn in missing:
            word_id = fn[len('i_'):-len('.png')]          # i_a05-099-01-07.png -> a05-099-01-07
            form_id = '-'.join(word_id.split('-')[:2])    # a05-099
            prefix = word_id.split('-')[0]                 # a05
            member = f'words/{prefix}/{form_id}/{word_id}.png'
            with zf.open(member) as src, open(os.path.join(IMG_DIR, fn), 'wb') as dst:
                dst.write(src.read())
    print("[iam_split] Image extraction complete")
