import random

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR
from transformers import logging
from models.interfaces import QueryEncoder, SSModel
logging.set_verbosity_error()


class AudioSepTunedEmbeddings(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
            self,
            ss_model: SSModel = None,
            waveform_mixer=None,
            loss_function=None,
            query_encoder: QueryEncoder = None,
            learning_rate: float = None,
            lr_lambda_func = None,
            optimizer_type = 'AdamW'
    ):  # Существующие параметры
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func
        self.ss_model = ss_model
        self.query_encoder = query_encoder
        self.waveform_mixer = waveform_mixer
        self.loss_function = loss_function

        self.optimizer_type = optimizer_type
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, prompt: str, mixture, device: torch.device = None):
        text = [prompt]
        conditions = self.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )
        conditions = self.embedding_layer(conditions)

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        }

        sep_segment = self.ss_model(input_dict)["waveform"]

        sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
        return sep_segment

    def training_step(self, batch_data_dict, batch_idx):
        random.seed(batch_idx)

        batch_audio_text_dict = batch_data_dict['audio_text']
        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']

        mixtures, segments = self.waveform_mixer(
            waveforms=batch_audio
        )

        conditions = self.query_encoder.get_query_embed(
            modality='text',
            text=batch_text,
        )
        conditions = self.embedding_layer(conditions)

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        self.ss_model.eval()
        sep_segment = self.ss_model(input_dict)['waveform']
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

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.embedding_layer.parameters(),
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
