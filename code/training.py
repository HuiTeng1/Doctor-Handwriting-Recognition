import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary

from datasets import KaggleHandwritingDataModule
from training_modules import HandwritingRecogTrainModule
from modeling import LABEL_TO_INDEX, INDEX_TO_LABELS, NUM_CLASSES


def get_data(path):
    """function to return train and validation for training by reading already processed and clean dataset"""
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(path, 'val.csv'))
    # Safety filters (data should already be clean after split_Normal.py)
    train_df = train_df[train_df.IDENTITY != 'UNREADABLE']
    val_df = val_df[val_df.IDENTITY != 'UNREADABLE']
    train_df = train_df[train_df.IDENTITY != 'EMPTY']
    val_df = val_df[val_df.IDENTITY != 'EMPTY']
    train_df = train_df.dropna(subset=['IDENTITY'])
    val_df = val_df.dropna(subset=['IDENTITY'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-loss:.3f}-{val-char-error-rate:.4f}',
                                          save_top_k=1, monitor='val-char-error-rate', mode='min', save_last=True)
    early_stopping = EarlyStopping(monitor="val-char-error-rate", patience=10, verbose=False, mode="min")
    model_summary = ModelSummary(max_depth=-1)

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=100,
                         callbacks=[checkpoint_callback, early_stopping, model_summary],
                         logger=False,  # wandb disabled
                         precision=16)
    trainer.fit(train_module, data_module)


def run_training():
    pl.seed_everything(15798)
    hparams = {
        'train_img_path': './data/Normal/images/test/',
        'lr': 1e-4,
        'val_img_path': './data/Normal/images/test/',
        'test_img_path': './data/Normal/images/test/',
        'data_path': './data/Normal',
        'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 1024, 'input_height': 36, 'input_width': 324,
        'gru_hidden_size': 128, 'gru_num_layers': 2, 'num_classes': NUM_CLASSES
    }
    label_to_index = LABEL_TO_INDEX
    index_to_labels = INDEX_TO_LABELS

    train_df, val_df = get_data(hparams['data_path'])
    data_module = KaggleHandwritingDataModule(train_df, val_df, hparams, label_to_index)
    train_module = HandwritingRecogTrainModule(hparams, index_to_labels=index_to_labels, label_to_index=label_to_index)
    train_model(train_module, data_module)


if __name__ == '__main__':
    run_training()