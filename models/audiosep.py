import random

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR
from transformers import logging

from models.interfaces import QueryEncoder, SSModel

logging.set_verbosity_error()


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
            self,
            ss_model: SSModel,
            waveform_mixer=None,
            query_encoder: QueryEncoder = None,
            loss_function=None,
            optimizer_type='AdamW',
            learning_rate=1e-4,
            lr_lambda_func=lambda epoch: 1.0,
            use_text_ratio=1.0
    ):
        super().__init__()
        self.ss_model = ss_model
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

    def forward(self, prompt: str, mixture, device: torch.device = None):
        text_conditions = self.query_encoder.get_query_embed(modality='text', text=[prompt], device=device)
        input_dict = {'mixture': mixture[None, None, :], 'condition': text_conditions}
        sep_segment = self.ss_model(input_dict)['waveform'].squeeze().cpu().numpy()
        return sep_segment

    def training_step(self, batch, batch_idx):
        random.seed(batch_idx)
        text, waveform = batch['audio_text']['text'], batch['audio_text']['waveform']
        mixtures, segments = self.waveform_mixer(waveform)
        conditions = self.query_encoder.get_query_embed('hybrid', text, segments.squeeze(1),
                                                        self.use_text_ratio)
        input_dict = {'mixture': mixtures[:, None, :].squeeze(1), 'condition': conditions}
        sep_segment = self.ss_model(input_dict)['waveform'].squeeze()
        sep_segment = sep_segment.squeeze()

        # (batch_size, 1, segment_samples)
        target_dict = {
            'segment': segments.squeeze(1),
        }
        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.ss_model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("Only AdamW optimizer is implemented")

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}


def get_model_class(model_type):
    if model_type == 'ResUNet30':
        from models.resunet import ResUNet30
        return ResUNet30

    else:
        raise NotImplementedError
