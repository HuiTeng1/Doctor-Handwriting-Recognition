import os, re, sys, torch
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', '..', 'preprocessing', 'normal'))

from hw_datasets import KaggleHandwritingDataModule, KaggleHandwrittenNames
from training_modules import HandwritingRecogTrainModule
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES
from ctc_decoder import best_path
from torchmetrics import CharErrorRate
from Iam_split import load_crnn_splits, extract_images, IMG_DIR

IMG_DIR_FULL = IMG_DIR
CKPT_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'checkpoint', 'normal', 'crnn')
TEST_RESULTS_PATH = os.path.join(CKPT_DIR, 'test_results.csv')


def report(results_df):
    total = len(results_df)
    exact_matches = int(results_df['is_exact_match'].sum())
    char_metric = CharErrorRate()
    cer = char_metric(results_df['prediction'].tolist(), results_df['ground_truth'].tolist()).item()
    print(f"[iam_only] Test samples: {total}")
    print(f"[iam_only] Test CER: {cer:.4f}")
    print(f"[iam_only] Test exact match: {(exact_matches/total)*100:.2f}%")


def main():
    if os.path.isfile(TEST_RESULTS_PATH):
        print(f"[iam_only] Already tested, reading existing results: {TEST_RESULTS_PATH} (delete this file to force a re-test)")
        report(pd.read_csv(TEST_RESULTS_PATH, keep_default_na=False))
        return

    _, _, test_df = load_crnn_splits()
    print(f"[iam_only] test: {len(test_df)}")

    extract_images([test_df])

    hparams = {
        'train_img_path': IMG_DIR_FULL + '/', 'val_img_path': IMG_DIR_FULL + '/', 'test_img_path': IMG_DIR_FULL + '/',
        'lr': 3e-4, 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256,
        'input_height': 36, 'input_width': 324,
        'gru_hidden_size': 128, 'gru_num_layers': 2, 'num_classes': NUM_CLASSES,
        'filename_col': 'filename', 'label_col': 'text',
    }

    use_gpu = torch.cuda.is_available()

    ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.startswith('epoch=') and f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f"No trained checkpoint found in {CKPT_DIR} - run train_iam_only.py first")

    def extract_cer(f):
        m = re.search(r'val-char-error-rate=([\d.]+)\.ckpt', f)
        return float(m.group(1)) if m else float('inf')

    best_ckpt_path = os.path.join(CKPT_DIR, sorted(ckpt_files, key=extract_cer)[0])
    train_module = HandwritingRecogTrainModule.load_from_checkpoint(
        best_ckpt_path, hparams=hparams,
        index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX,
        map_location=torch.device('cpu') if not use_gpu else None
    )
    train_module.eval()
    print(f"[iam_only] Loaded: {best_ckpt_path}")

    test_helper = KaggleHandwritingDataModule(test_df, test_df, hparams, LABEL_TO_INDEX)
    test_ds = KaggleHandwrittenNames(test_df, test_helper.transforms, LABEL_TO_INDEX,
                                    hparams['test_img_path'], filename_col='filename', label_col='text')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0,
        collate_fn=KaggleHandwritingDataModule.custom_collate)

    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['transformed_images'].to(train_module.device)
            labels = batch['labels'].cpu().numpy()
            target_lens = batch['target_lens'].cpu().numpy()
            output = train_module.model(images).permute(1, 0, 2)
            output = torch.exp(output).cpu().numpy()
            for i, pred in enumerate(output):
                predicted_string = best_path(pred, train_module.chars)
                gt_string = ''.join(INDEX_TO_LABELS[idx] for idx in labels[i][0:target_lens[i]])
                all_preds.append(predicted_string); all_gts.append(gt_string)

    results_df = pd.DataFrame({
        'filename': test_df['filename'].tolist(),
        'ground_truth': all_gts,
        'prediction': all_preds,
        'is_exact_match': [p == g for p, g in zip(all_preds, all_gts)],
    })
    results_df.to_csv(TEST_RESULTS_PATH, index=False, encoding='utf-8-sig')
    print(f"[iam_only] Results saved to: {TEST_RESULTS_PATH}")

    report(results_df)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
