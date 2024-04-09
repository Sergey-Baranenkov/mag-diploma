import random
from collections import defaultdict
from typing import List

import lightning.pytorch as pl
import numpy as np
import torch
import torch.optim as optim
import typing
from huggingface_hub import PyTorchModelHubMixin
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import logging
from models.audiosep import AudioSep, get_averaged_metrics
from models.interfaces import QueryEncoder
from utils import calculate_sisdr, calculate_sdr, flatmap

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


class AudioSepLoraAndTunedEmbeddings(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
            self,
            pretrained_audiosep_model: AudioSep,
            target_modules: List[str],
            waveform_mixer,
            loss_function,
            optimizer_type,
            learning_rate,
            lr_lambda_func,
            logs_per_class: bool,
            lora_params: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'pretrained_audiosep_model',
            'waveform_mixer',
            'loss_function',
            'lr_lambda_func'
        ])

        self.epoch_losses = None
        config = LoraConfig(
            target_modules=target_modules,
            *lora_params,
        )

        model = get_peft_model(pretrained_audiosep_model, config)
        model.query_encoder = TunedQueryEncoder(model.query_encoder)
        self.model = model
        self.waveform_mixer = waveform_mixer
        self.loss_function = loss_function
        self.lr_lambda_func = lr_lambda_func
        self.strict_loading = False
        self.logs_per_class = logs_per_class

    def forward(self, prompt: str, mixture, device: torch.device = None):
        text = [prompt]
        conditions = self.model.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        }

        sep_segment = self.model.ss_model(input_dict)["waveform"]

        sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
        return sep_segment

    def batch_forward(self, batch, batch_idx):
        random.seed(batch_idx)
        text, waveform = batch['audio_text']['text'], batch['audio_text']['waveform']
        mixtures, segments = self.waveform_mixer(waveform, text)
        conditions = self.model.query_encoder.get_query_embed(
            'hybrid',
            text=text,
            audio=segments.squeeze(1),
            use_text_ratio=1.0
        )

        input_dict = {'mixture': mixtures[:, None, :].squeeze(1), 'condition': conditions}
        sep_segment = self.model.ss_model(input_dict)['waveform'].squeeze()
        target_dict = {
            'segment': segments.squeeze(1),
        }

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
        if self.hparams.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError("Only AdamW optimizer is implemented")

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        lora_params = {k: v for k, v in self.state_dict().items() if 'lora' in k.lower()}
        embedding_params = {k: v for k, v in self.state_dict().items() if 'tuned_embedding_layer' in k.lower()}
        checkpoint_params = {**lora_params, **embedding_params}
        checkpoint['state_dict'] = checkpoint_params
        return checkpoint

    def print_parameters(self):
        self.model.print_trainable_parameters()