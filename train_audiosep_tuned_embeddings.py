import argparse
import logging
import os
import pathlib

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.tensorboard import SummaryWriter

from callbacks.base import CheckpointEveryNSteps
from data.datamodules import *
from data.waveform_mixers import SegmentMixer
from losses import get_loss_function
from models.audiosep_tunned_embeddings import AudioSepTunedEmbeddings
from models.clap_encoder import CLAP_Encoder
from models.resunet import *
from optimizers.lr_schedulers import get_lr_lambda
from utils import create_logging, get_dirs, get_data_module
from utils import parse_yaml
from model_loaders import load_ss_model

torch.set_float32_matmul_precision('high')

def train(args) -> NoReturn:
    r"""Train, evaluate, and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int, number of GPUs to train
        config_yaml: str
    """

    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = torch.cuda.device_count()
    # Read config file.
    configs = parse_yaml(config_yaml)

    # Configuration of data
    max_mix_num = configs['data']['max_mix_num']
    sampling_rate = configs['data']['sampling_rate']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']

    # Configuration of the trainer
    num_nodes = configs['train']['num_nodes']
    batch_size = configs['train']['batch_size_per_device'] 
    sync_batchnorm = configs['train']['sync_batchnorm'] 
    num_workers = configs['train']['num_workers']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']
    save_epoch_frequency = configs['train']['save_epoch_frequency']
    resume_checkpoint_path = args.resume_checkpoint_path
    if resume_checkpoint_path == "":
        resume_checkpoint_path = None
    else:
        logging.info(f'Finetuning AudioSep with checkpoint [{resume_checkpoint_path}]')

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    logging.info(configs)

    # data module
    data_module = get_data_module(
        config_yaml=config_yaml,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db,
        higher_db=higher_db
    )

    loss_function = get_loss_function(loss_type)

    device = torch.device('cuda')
    SS_CONFIG_PATH = './config/audiosep_base.yaml'
    CLAP_CKPT_PATH = './checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt'
    AUDIOSEP_CKPT_PATH = './checkpoint/audiosep_base_4M_steps.ckpt'

    loss_function = get_loss_function(loss_type)

    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db,
        higher_db=higher_db
    )

    query_encoder = CLAP_Encoder(
        pretrained_path=CLAP_CKPT_PATH).eval().to(device)

    configs = parse_yaml(SS_CONFIG_PATH)

    model = load_ss_model(
        configs=configs,
        checkpoint_path=AUDIOSEP_CKPT_PATH,
        query_encoder=query_encoder
    )\
        .eval()\
        .to(device)

    model.freeze()
    # pytorch-lightning model
    pl_model = AudioSepTunedEmbeddings(
        waveform_mixer=segment_mixer,
        loss_function=loss_function,
        ss_model=model.ss_model,
        query_encoder=model.query_encoder,
        learning_rate = learning_rate,
        lr_lambda_func = lr_lambda_func
    )

    checkpoint_every_n_epochs = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='{epoch}',
        every_n_epochs=save_epoch_frequency,
        save_top_k=-1
    )

    callbacks = [checkpoint_every_n_epochs]
    wandb_logger = WandbLogger(name='embeddings', project='diploma')
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        num_nodes=num_nodes,
        precision="32-true",
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=-1,
        log_every_n_steps=38,
        use_distributed_sampler=True,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model, 
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=data_module,
        ckpt_path=resume_checkpoint_path if resume_checkpoint_path else None,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        required=True,
        default='',
        help="Path of pretrained checkpoint for finetuning.",
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)