import os
from PIL import Image
import torch
import torch.nn as nn
from ctc_decoder import best_path, beam_search
import pytorch_lightning as pl
from torchmetrics import CharErrorRate
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from torchvision.utils import make_grid

from modeling import HandwritingRecognition, CHARS, NUM_CLASSES


class HandwritingRecogTrainModule(pl.LightningModule):
    def __init__(self, hparams, index_to_labels, label_to_index):
        super().__init__()
        # save_hyperparameters saves the parameters in the signature in the form of dict
        self.save_hyperparameters()
        self.chars = CHARS  # 70-class dictionary
        self.model = HandwritingRecognition(self.hparams['hparams']['gru_input_size'], self.hparams['hparams']['gru_hidden_size'],
                                            self.hparams['hparams']['gru_num_layers'], self.hparams['hparams']['num_classes'])
        # blank index must equal num_classes (the model outputs num_classes + 1 logits)
        self.criterion = nn.CTCLoss(blank=NUM_CLASSES, zero_infinity=True, reduction='mean')
        self.transforms = Compose([Resize((self.hparams['hparams']['input_height'], self.hparams['hparams']['input_width'])), Grayscale(),
                                                              ToTensor()])
        self.char_metric = CharErrorRate()

    def forward(self, the_image):
        out = self.model(the_image)
        out = out.permute(1, 0, 2)
        out = torch.exp(out)
        return out

    def intermediate_operation(self, batch):
        transformed_images = batch['transformed_images']
        labels = batch['labels']
        target_lens = batch['target_lens']

        output = self.model(transformed_images)

        N = output.size(1)
        input_length = output.size(0)
        input_lengths = torch.full(size=(N,), fill_value=input_length, dtype=torch.int32)

        loss = self.criterion(output, labels, input_lengths, target_lens)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, preds = self.intermediate_operation(batch)
        with torch.inference_mode():
            preds = preds.permute(1, 0, 2)
            preds = torch.exp(preds)
            ground_truth = batch['labels']
            target_lens = batch['target_lens']
            ground_truth = ground_truth.cpu().detach().numpy()
            target_lens = target_lens.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            actual_predictions = []
            for pred in preds:
                actual_predictions.append(best_path(pred, self.chars))
            exact_matches = 0
            actual_ground_truths = []
            for i, predicted_string in enumerate(actual_predictions):
                ground_truth_sample = ground_truth[i][0:target_lens[i]]
                ground_truth_string = [self.hparams.index_to_labels[index] for index in ground_truth_sample]
                ground_truth_string = ''.join(ground_truth_string)
                actual_ground_truths.append(ground_truth_string)
                if predicted_string == ground_truth_string:
                    exact_matches += 1
            exact_match_percentage = (exact_matches / len(preds)) * 100
            char_error_rate = self.char_metric(actual_predictions, actual_ground_truths)
            self.log_dict({'train-loss': loss, 'train-exact-match': exact_match_percentage,
                           'train-char_error_rate': char_error_rate}, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.intermediate_operation(batch)
        preds = preds.permute(1, 0, 2)
        preds = torch.exp(preds)
        ground_truth = batch['labels']
        target_lens = batch['target_lens']
        ground_truth = ground_truth.cpu().detach().numpy()
        target_lens = target_lens.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        actual_predictions = []
        for pred in preds:
            actual_predictions.append(best_path(pred, self.chars))
        exact_matches = 0
        actual_ground_truths = []
        for i, predicted_string in enumerate(actual_predictions):
            ground_truth_sample = ground_truth[i][0:target_lens[i]]
            ground_truth_string = [self.hparams.index_to_labels[index] for index in ground_truth_sample]
            ground_truth_string = ''.join(ground_truth_string)
            actual_ground_truths.append(ground_truth_string)
            if predicted_string == ground_truth_string:
                exact_matches += 1
        char_error_rate = self.char_metric(actual_predictions, actual_ground_truths)
        exact_match_percentage = (exact_matches / len(preds)) * 100
        self.log_dict({'val-loss': loss, 'val-exact-match': exact_match_percentage,
                       'val-char-error-rate': char_error_rate}, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['hparams']['lr'])


if __name__ == '__main__':
    print("Dictionary size (NUM_CLASSES):", NUM_CLASSES)
    print("CHARS:", repr(CHARS))