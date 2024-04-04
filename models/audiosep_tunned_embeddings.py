import random
from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typing
from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR
from transformers import logging

from models.audiosep import get_averaged_metrics
from models.interfaces import QueryEncoder, SSModel
from utils import calculate_sdr, calculate_sisdr, flatmap

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
            optimizer_type='AdamW',
            logs_per_class: bool = False,
    ):  # Существующие параметры
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func
        self.ss_model = ss_model
        self.query_encoder = TunedQueryEncoder(query_encoder)
        self.waveform_mixer = waveform_mixer
        self.loss_function = loss_function

        self.optimizer_type = optimizer_type
        self.strict_loading = False
        self.logs_per_class = logs_per_class

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

        return sep_segment, target_dict, input_dict['mixture']

    def training_step(self, batch, batch_idx):
        texts = batch['audio_text']['text']
        batch_size = len(texts)
        sep_segment, target_dict, mixtures = self.batch_forward(batch, batch_idx)
        sep_segment = sep_segment.squeeze()

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate metrics for each example in the batch
        sdr_values = defaultdict(list)
        sdr_i_values = defaultdict(list)
        sisdr_values = defaultdict(list)

        for ref, est, mixture, text in zip(target_dict['segment'], sep_segment.squeeze(1), mixtures, texts):
            ref = ref.cpu().numpy()
            est = est.detach().cpu().numpy()

            sdr_no_sep = calculate_sdr(ref=ref, est=mixture.cpu().numpy())
            sdr = calculate_sdr(ref=ref, est=est)
            sisdr = calculate_sisdr(ref=ref, est=est)

            sdri = sdr - sdr_no_sep
            sdr_values[text].append(sdr)
            sdr_i_values[text].append(sdri)
            sisdr_values[text].append(sisdr)

        if self.logs_per_class:
            for cls in sdr_values.keys():
                res_dict = get_averaged_metrics(sdr_values[cls], sdr_i_values[cls], sisdr_values[cls], 'train', cls)
                self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)
        else:
            res_dict = get_averaged_metrics(flatmap(sdr_values.values()), flatmap(sdr_i_values.values()), flatmap(sisdr_values.values()), 'train')
            self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        loss = self.loss_function(output_dict, target_dict)

        res_dict = {
            "train_loss": loss,
        }

        self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch['audio_text']['text']
        batch_size = len(texts)
        sep_segment, target_dict, mixtures = self.batch_forward(batch, batch_idx)
        sep_segment = sep_segment.squeeze()
        output_dict = {
            'segment': sep_segment,
        }

        # Calculate metrics for each example in the batch
        sdr_values = defaultdict(list)
        sdr_i_values = defaultdict(list)
        sisdr_values = defaultdict(list)
        for ref, est, mixture, text in zip(target_dict['segment'], sep_segment.squeeze(1), mixtures, texts):
            ref = ref.cpu().numpy()
            est = est.cpu().numpy()

            sdr_no_sep = calculate_sdr(ref=ref, est=mixture.cpu().numpy())
            sdr = calculate_sdr(ref=ref, est=est)
            sisdr = calculate_sisdr(ref=ref, est=est)

            sdri = sdr - sdr_no_sep
            sdr_values[text].append(sdr)
            sdr_i_values[text].append(sdri)
            sisdr_values[text].append(sisdr)

        if self.logs_per_class:
            for cls in sdr_values.keys():
                res_dict = get_averaged_metrics(sdr_values[cls], sdr_i_values[cls], sisdr_values[cls], 'val', cls)
                self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)
        else:
            res_dict = get_averaged_metrics(flatmap(sdr_values.values()), flatmap(sdr_i_values.values()), flatmap(sisdr_values.values()), 'val')
            self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        loss = self.loss_function(output_dict, target_dict)

        res_dict = {
            "train_loss": loss,
        }

        self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        return res_dict

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

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        embedding_params = {k: v for k, v in self.state_dict().items() if 'tuned_embedding_layer' in k.lower()}
        checkpoint_params = {**embedding_params}
        checkpoint['state_dict'] = checkpoint_params
        return checkpoint
