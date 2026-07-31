import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import KaggleHandwritingDataModule


# ======================================================================
# 70-CLASS DICTIONARY (shared by normal base + doctor fine-tune)
#   space % ( ) - . / ? 0-9 A-Z a-z
# ======================================================================
CHARS = " %()-./?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LABEL_TO_INDEX = {c: i for i, c in enumerate(CHARS)}
INDEX_TO_LABELS = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)  # 70


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class HandwritingRecognitionCNN(nn.Module):
    """Class to enable instantiation of CNN model used for extraction of features from an image"""
    def __init__(self):
        super().__init__()
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 2), kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, stride=(1, 2), kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.image_feature_extractor(x)


class HandwritingRecognitionGRU(nn.Module):
    """Class to enable instantiation of a GRU sequential model to be applied on CNN features"""
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.gru_layer = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.output = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        recurrent_output, _ = self.gru_layer(x)
        out = self.output(recurrent_output)
        out = F.log_softmax(out, dim=2)
        return out


class HandwritingRecognition(nn.Module):
    """Class to merge both CNN and GRU architecture to perform full end to end inference for Handwriting recognition"""
    def __init__(self, gru_input_size, gru_hidden, gru_layers, num_classes):
        super().__init__()
        self.cnn_feature_extractor = HandwritingRecognitionCNN()
        self.gru = HandwritingRecognitionGRU(gru_input_size, gru_hidden, gru_layers, num_classes+1)
        self.linear1 = nn.Linear(1280, 512)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(512, 256)

    def forward(self, x):
        out = self.cnn_feature_extractor(x)
        batch_size, channels, height, width = out.size()
        out = out.view(batch_size, -1, width)
        # out is reshaped and permuted so that CNN features along width of image should be fed in sequence)
        out = out.permute(0, 2, 1)
        out = self.linear1(out)
        out = self.activation(self.linear2(out))
        out = self.gru(out)
        out = out.permute(1, 0, 2)
        return out


def test_modelling():
    pl.seed_everything(6579)
    hparams = {
        'train_img_path': './data/Normal/images/test/',
        'lr': 1e-3, 'val_img_path': './data/Normal/images/test/',
        'test_img_path': './data/Normal/images/test/',
        'data_path': './data/Normal', 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256, 'input_height': 36, 'input_width': 324, 'gru_hidden_size': 128,
        'gru_num_layers': 2, 'num_classes': NUM_CLASSES
    }
    label_to_index = LABEL_TO_INDEX

    train_df = pd.read_csv(os.path.join(hparams['data_path'], 'train.csv'))
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(os.path.join(hparams['data_path'], 'val.csv'))
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    sample_module = KaggleHandwritingDataModule(train_df, val_df, hparams, label_to_index)
    sample_module.setup()
    sample_train_module = sample_module.train_dataloader()
    sample_train_batch = next(iter(sample_train_module))
    model = HandwritingRecognition(hparams['gru_input_size'], hparams['gru_hidden_size'],
                           hparams['gru_num_layers'], hparams['num_classes'])
    output = model(sample_train_batch['transformed_images'])
    print("the output shape:", output.shape)


if __name__ == '__main__':
    test_modelling()