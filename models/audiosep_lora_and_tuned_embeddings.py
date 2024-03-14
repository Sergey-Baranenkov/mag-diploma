import random
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
from models.audiosep import AudioSep
from models.interfaces import QueryEncoder

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
        )

        model = get_peft_model(pretrained_audiosep_model, config)
        model.query_encoder = TunedQueryEncoder(model.query_encoder)
        self.model = model
        self.waveform_mixer = waveform_mixer
        self.loss_function = loss_function
        self.lr_lambda_func = lr_lambda_func
        self.strict_loading = False

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

    def training_step(self, batch, batch_idx):
        random.seed(batch_idx)
        text, waveform = batch['audio_text']['text'], batch['audio_text']['waveform']
        mixtures, segments = self.waveform_mixer(waveform)
        conditions = self.model.query_encoder.get_query_embed(
            'hybrid',
            text=text,
            audio=segments.squeeze(1),
            use_text_ratio=1.0
        )

        input_dict = {'mixture': mixtures[:, None, :].squeeze(1), 'condition': conditions}
        sep_segment = self.model.ss_model(input_dict)['waveform'].squeeze()
        # (batch_size, 1, segment_samples)
        target_dict = {
            'segment': segments.squeeze(1),
        }
        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)
        self.epoch_losses.append(loss.detach().cpu().numpy())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError("Only AdamW optimizer is implemented")

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def on_train_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.epoch_losses = []
        return

    def on_train_epoch_end(self):
        loss = np.mean(self.epoch_losses)
        print('mean epoch loss:', loss)

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        # Сохраняем параметры LoRa
        lora_params = {k: v for k, v in self.state_dict().items() if 'lora' in k.lower()}

        # Добавляем параметры слоя self.embedding_layer
        embedding_params = {k: v for k, v in self.state_dict().items() if 'tuned_embedding_layer' in k.lower()}
        # Объединяем параметры LoRa и embedding_layer в один словарь
        checkpoint_params = {**lora_params, **embedding_params}

        # Обновляем словарь состояния в контрольной точке
        checkpoint['state_dict'] = checkpoint_params
        return checkpoint

    def print_parameters(self):
        self.model.print_trainable_parameters()
    # def validation_step(self, batch, batch_idx):
    #     text, waveform = batch['audio_text']['text'], batch['audio_text']['waveform']
    #     mixtures, references = self.waveform_mixer(waveform)
    #     conditions = self.query_encoder.get_query_embed('hybrid', text, references.squeeze(1), self.use_text_ratio)
    #     input_dict = {'mixture': mixtures[:, None, :], 'condition': conditions}
    #     estimated_segments = self.ss_model(input_dict)['waveform'].squeeze()
    #
    #     # Calculate metrics for each example in the batch
    #     sdr_values = []
    #     sisdr_values = []
    #     for ref, est in zip(references.squeeze(1), estimated_segments):
    #         sdr = calculate_sdr(ref.cpu().numpy(), est.cpu().numpy())
    #         sisdr = calculate_sisdr(ref.cpu().numpy(), est.cpu().numpy())
    #         sdr_values.append(sdr)
    #         sisdr_values.append(sisdr)
    #
    #     # Average metrics across the batch
    #     avg_sdr = torch.tensor(sdr_values).mean()
    #     avg_sisdr = torch.tensor(sisdr_values).mean()
    #
    #     # Log individual metrics
    #     self.log('val_sdr', avg_sdr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log('val_sisdr', avg_sisdr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     # Return the metrics for use in validation_epoch_end
    #     return {'val_sdr': avg_sdr, 'val_sisdr': avg_sisdr}
    #
    # def validation_epoch_end(self, validation_step_outputs):
    #     # Aggregate metrics from all validation steps
    #     avg_sdr = torch.stack([x['val_sdr'] for x in validation_step_outputs]).mean()
    #     avg_sisdr = torch.stack([x['val_sisdr'] for x in validation_step_outputs]).mean()
    #
    #     # Log aggregated metrics
    #     self.log('avg_val_sdr', avg_sdr, prog_bar=True)
    #     self.log('avg_val_sisdr', avg_sisdr, prog_bar=True)
    # %%
