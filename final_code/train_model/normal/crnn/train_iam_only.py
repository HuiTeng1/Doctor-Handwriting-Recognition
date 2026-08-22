import os, sys, torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', '..', 'preprocessing', 'normal'))

from hw_datasets import KaggleHandwritingDataModule
from training_modules import HandwritingRecogTrainModule
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES
from Iam_split import load_crnn_splits, extract_images, IMG_DIR

IMG_DIR_FULL = IMG_DIR
CKPT_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'checkpoint', 'normal', 'crnn')   # the trained checkpoint (with DONE.txt) lives here; the split source changed but the data content didn't, so this is unaffected


def main():
    pl.seed_everything(15798)

    training_done_flag = f"{CKPT_DIR}/DONE.txt"
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.isfile(training_done_flag):
        print("[iam_only] Training already complete, skipping. Run test_CrnnNormal.py to test.")
        return

    train_df, val_df, _ = load_crnn_splits()
    print(f"[iam_only] train: {len(train_df)}  val: {len(val_df)}")

    extract_images([train_df, val_df])

    hparams = {
        'train_img_path': IMG_DIR_FULL + '/', 'val_img_path': IMG_DIR_FULL + '/', 'test_img_path': IMG_DIR_FULL + '/',
        'lr': 3e-4, 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256,
        'input_height': 36, 'input_width': 324,
        'gru_hidden_size': 128, 'gru_num_layers': 2, 'num_classes': NUM_CLASSES,
        'filename_col': 'filename', 'label_col': 'text',
    }

    use_gpu = torch.cuda.is_available()

    data_module = KaggleHandwritingDataModule(train_df, val_df, hparams, LABEL_TO_INDEX)
    data_module.train_dataloader = lambda: torch.utils.data.DataLoader(
        data_module.train, batch_size=hparams['train_batch_size'], shuffle=True, pin_memory=True,
        num_workers=2, persistent_workers=True, collate_fn=KaggleHandwritingDataModule.custom_collate)

    train_module = HandwritingRecogTrainModule(hparams, index_to_labels=INDEX_TO_LABELS, label_to_index=LABEL_TO_INDEX)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPT_DIR, filename='{epoch}-{val-loss:.3f}-{val-char-error-rate:.4f}',
        save_top_k=1, monitor='val-char-error-rate', mode='min', save_last=True)
    early_stopping = EarlyStopping(monitor="val-char-error-rate", patience=15, verbose=True, mode="min")

    trainer = pl.Trainer(accelerator='gpu' if use_gpu else 'cpu', max_epochs=-1,
                        callbacks=[checkpoint_callback, early_stopping, ModelSummary(max_depth=-1)],
                        logger=False, precision=16 if use_gpu else 32,
                        gradient_clip_val=1.0)

    import glob
    last_ckpts = glob.glob(os.path.join(CKPT_DIR, 'last*.ckpt'))
    resume_ckpt = max(last_ckpts, key=os.path.getmtime) if last_ckpts else None
    if resume_ckpt:
        print(f"[iam_only] Resuming from latest checkpoint: {resume_ckpt}")
    else:
        print("[iam_only] Training from scratch")

    trainer.fit(train_module, data_module, ckpt_path=resume_ckpt)

    with open(training_done_flag, 'w') as f:
        f.write("done")
    print("[iam_only] Training complete. Run test_CrnnNormal.py to test.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()