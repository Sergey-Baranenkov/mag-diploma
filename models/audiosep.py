import random
from collections import defaultdict

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR
from transformers import logging

from models.interfaces import QueryEncoder, SSModel
from utils import flatmap, calculate_sdr, calculate_sisdr

logging.set_verbosity_error()


def get_averaged_metrics(sdr_values: list,
                         sdr_i_values: list,
                         sisdr_values: list,
                         phase: str, cls_name: str = 'avg',
                         ):
    # Average metrics across the batch
    avg_sdr = torch.tensor(sdr_values).mean()
    avg_sdr_i = torch.tensor(sdr_i_values).mean()
    avg_sisdr = torch.tensor(sisdr_values).mean()

    res_dict = {
        f"{phase}_sdr_{cls_name}": avg_sdr,
        f"{phase}_sdr_i_{cls_name}": avg_sdr_i,
        f"{phase}_si_sdr_{cls_name}": avg_sisdr
    }

    return res_dict


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
            use_text_ratio=1.0,
            logs_per_class: bool = True,
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
        self.logs_per_class = logs_per_class

    def forward(self, prompt: str, mixture, device: torch.device = None):
        text_conditions = self.query_encoder.get_query_embed(modality='text', text=[prompt], device=device)
        input_dict = {'mixture': mixture[None, None, :], 'condition': text_conditions}
        sep_segment = self.ss_model(input_dict)['waveform'].squeeze().cpu().numpy()
        return sep_segment

    def batch_forward(self, batch, batch_idx):
        random.seed(batch_idx)

        batch_audio_text_dict = batch['audio_text']
        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']
        batch_mixtures = batch_audio_text_dict['mixture']

        mixtures, segments = self.waveform_mixer(
            waveforms=batch_audio,
            names=batch_text,
            mixtures = batch_mixtures
        )

        conditions = self.query_encoder.get_query_embed(
            modality='text',
            text=batch_text,
        ).to(self.dtype)

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

        res_dict = get_averaged_metrics(flatmap(sdr_values.values()), flatmap(sdr_i_values.values()), flatmap(sisdr_values.values()), 'train')
        self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        loss = self.loss_function(output_dict, target_dict)

        res_dict = {
            "train_loss": loss,
        }

        self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        if self.optimizer_type == 'fake':
            # fake loss
            fake_loss = torch.HalfTensor([1.0]).to(self.device)
            fake_loss.requires_grad = True

            return fake_loss
        else:
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

        res_dict = get_averaged_metrics(flatmap(sdr_values.values()), flatmap(sdr_i_values.values()), flatmap(sisdr_values.values()), 'val')
        self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        loss = self.loss_function(output_dict, target_dict)

        res_dict = {
            "val_loss": loss,
        }

        self.log_dict(res_dict, on_step=False, on_epoch=True, batch_size=batch_size)

        return res_dict

    def configure_optimizers(self):
        if self.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.ss_model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'fake':
            return None
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
