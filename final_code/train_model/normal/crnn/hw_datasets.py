import os
from typing import Optional

import pytorch_lightning as pl
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomRotation, RandomApply, \
    GaussianBlur, CenterCrop, RandomAutocontrast, Normalize


class KaggleHandwrittenNames(Dataset):
    def __init__(self, data, transforms, label_to_index, img_path, filename_col='FILENAME', label_col='IDENTITY'):
        self.data = data
        self.transforms = transforms
        self.img_path = img_path
        self.label_to_index = label_to_index
        # Column names differ between datasets: the Normal/Kaggle-names CSVs use
        # 'FILENAME'/'IDENTITY', the Doctor/medical CSVs use 'filename'/'text'.
        # Defaults keep this backward compatible with existing Normal-dataset scripts.
        self.filename_col = filename_col
        self.label_col = label_col

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_name = row[self.filename_col]
        image_label = row[self.label_col]
        the_image = Image.open(os.path.join(self.img_path, file_name))
        transformed_image = self.transforms(the_image)
        target_len = len(image_label)
        label_chars = list(image_label)
        image_label = torch.tensor([self.label_to_index[char] for char in label_chars])
        return {
            'transformed_image': transformed_image,
            'label': image_label,
            'target_len': target_len
        }


class KaggleHandwritingDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, hparams, label_to_index):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_batch_size = hparams['train_batch_size']
        self.val_batch_size = hparams['val_batch_size']
        # --- Preprocessing (applied to train + val + test consistently) ---
        # RandomAutocontrast(p=1.0): NOT actually random - p=1.0 means every
        # image gets its contrast auto-stretched, making faint/light strokes
        # stand out more clearly against the background.
        # Normalize(mean=[0.5], std=[0.5]): maps pixel values from [0, 1] to
        # [-1, 1], which is a standard trick to help CNN training converge
        # more stably.
        self.transforms = Compose([Resize((hparams['input_height'], hparams['input_width'])), Grayscale(),
                                   RandomAutocontrast(p=1.0),
                                   ToTensor(),
                                   Normalize(mean=[0.5], std=[0.5])])
        applier1 = RandomApply(transforms=[RandomRotation(10), GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5)
        applier2 = RandomApply(transforms=[CenterCrop((hparams['input_height'] - 1, hparams['input_width'] - 2))], p=0.5)
        self.train_transforms = Compose([applier2, Resize((hparams['input_height'], hparams['input_width'])), Grayscale(),
                                   RandomAutocontrast(p=1.0),
                                   applier1, ToTensor(),
                                   Normalize(mean=[0.5], std=[0.5])])
        self.train_img_path = hparams['train_img_path']
        self.val_img_path = hparams['val_img_path']
        self.label_to_index = label_to_index
        # Optional overrides for datasets whose CSV columns aren't FILENAME/IDENTITY
        # (e.g. the Doctor/medical CSVs use 'filename'/'text'). Not set -> old behavior.
        self.filename_col = hparams.get('filename_col', 'FILENAME')
        self.label_col = hparams.get('label_col', 'IDENTITY')

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = KaggleHandwrittenNames(self.train_data, self.train_transforms, self.label_to_index, self.train_img_path,
                                                filename_col=self.filename_col, label_col=self.label_col)
            self.val = KaggleHandwrittenNames(self.val_data, self.transforms, self.label_to_index, self.val_img_path,
                                              filename_col=self.filename_col, label_col=self.label_col)

    def custom_collate(data):
        '''
        To handle variable max seq length batch size
        '''
        transformed_images = []
        labels = []
        target_lens = []
        for d in data:
            transformed_images.append(d['transformed_image'])
            labels.append(d['label'])
            target_lens.append(d['target_len'])
        # pad_sequence pads all batch items to the longest length sequence in the batch because labels
        # can be of variable length
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        transformed_images = torch.stack(transformed_images)
        target_lens = torch.tensor(target_lens)
        return {
            'transformed_images': transformed_images,
            'labels': batch_labels,
            'target_lens': target_lens
        }

    def train_dataloader(self):
        # NOTE: On Windows, if you hit a multiprocessing/DataLoader error,
        # change num_workers=4 back to num_workers=0 here and in val_dataloader.
        return DataLoader(self.train, batch_size=self.train_batch_size, shuffle=True, pin_memory=True,
                          num_workers=2, collate_fn=KaggleHandwritingDataModule.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, shuffle=False, pin_memory=True,
                          num_workers=2, collate_fn=KaggleHandwritingDataModule.custom_collate)