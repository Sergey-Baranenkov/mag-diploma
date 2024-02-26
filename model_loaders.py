from typing import Dict

import torch
from torch import nn as nn

from models.audiosep import get_model_class, AudioSep
from utils import parse_yaml


def load_ss_model(
        configs: Dict,
        checkpoint_path: str,
        query_encoder: nn.Module
) -> nn.Module:
    ss_model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]

    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # Load PyTorch Lightning model
    pl_model = AudioSep.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        waveform_mixer=None,
        query_encoder=query_encoder,
        loss_function=None,
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
        map_location=torch.device('cpu'),
    )

    return pl_model


def get_ss_model(config_yaml) -> nn.Module:
    configs = parse_yaml(config_yaml)
    ss_model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]

    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    return ss_model
