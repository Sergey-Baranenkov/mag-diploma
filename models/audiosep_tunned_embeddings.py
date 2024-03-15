import random

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typing
from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR
from transformers import logging
from models.interfaces import QueryEncoder, SSModel
from utils import calculate_sdr, calculate_sisdr

logging.set_verbosity_error()


class TunedQueryEncoder(pl.LightningModule, QueryEncoder):
    def __init__(self, query_encoder: QueryEncoder, device=None):
        super().__init__()
        self.query_encoder = query_encoder
        self.tuned_embedding_layer = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 512)
        ).to(device)

    def get_query_embed(
            self,
            modality: typing.Literal['text', 'audio', 'hybrid'],
            audio=None,
            text=None,
            use_text_ratio: float = 1,
            device=None
    ):
        conditions = self.query_encoder.get_query_embed(modality, audio, text, use_text_ratio, device)
        conditions = self.tuned_embedding_layer(conditions)
        return conditions


class AudioSepTunedEmbeddings(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
            self,
            ss_model: SSModel = None,
            waveform_mixer=None,
            loss_function=None,
            query_encoder: QueryEncoder = None,
            learning_rate: float = None,
            lr_lambda_func=None,
            optimizer_type='AdamW'
    ):  # Существующие параметры
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func
        self.ss_model = ss_model
        self.query_encoder = TunedQueryEncoder(query_encoder)
        self.waveform_mixer = waveform_mixer
        self.loss_function = loss_function

        self.optimizer_type = optimizer_type

    def forward(self, prompt: str, mixture, device: torch.device = None):
        text = [prompt]
        conditions = self.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        }

        sep_segment = self.ss_model(input_dict)["waveform"]

        sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
        return sep_segment

    def training_step(self, batch, batch_idx):
        sep_segment, target_dict = self.batch_forward(batch, batch_idx)
        sep_segment = sep_segment.squeeze()

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)
        self.epoch_losses.append(loss.item())
        self.log_dict({"train_loss": loss})

        return loss

    def on_train_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.epoch_losses = []
        return

    def on_train_epoch_end(self):
        loss = np.mean(self.epoch_losses)
        print('mean epoch loss:', loss)

    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.query_encoder.tuned_embedding_layer.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict

    def validation_step(self, batch, batch_idx):
        sep_segment, target_dict = self.batch_forward(batch, batch_idx)
        sep_segment = sep_segment.squeeze()
        output_dict = {
            'segment': sep_segment,
        }

        # Calculate metrics for each example in the batch
        sdr_values = []
        sisdr_values = []
        for ref, est in zip(target_dict['segment'], sep_segment.squeeze(1)):
            sdr = calculate_sdr(ref.cpu().numpy(), est.cpu().numpy())
            sisdr = calculate_sisdr(ref.cpu().numpy(), est.cpu().numpy())
            sdr_values.append(sdr)
            sisdr_values.append(sisdr)

        # Average metrics across the batch
        avg_sdr = torch.tensor(sdr_values).mean()
        avg_sisdr = torch.tensor(sisdr_values).mean()
        loss = self.loss_function(output_dict, target_dict)
        res_dict = {"val_loss": loss, "val_sdr": avg_sdr, "val_sisdr": avg_sisdr}

        self.log_dict(res_dict)

        return res_dict

    def batch_forward(self, batch, batch_idx):
        random.seed(batch_idx)

        batch_audio_text_dict = batch['audio_text']
        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']

        mixtures, segments = self.waveform_mixer(
            waveforms=batch_audio
        )

        conditions = self.query_encoder.get_query_embed(
            modality='text',
            text=batch_text,
        )

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        sep_segment = self.ss_model(input_dict)['waveform']

        return sep_segment, target_dict
