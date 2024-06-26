import argparse
import logging
import pathlib
import time

import torch
from pytorch_lightning.loggers import WandbLogger

from data.datamodules import *
from data.waveform_mixers import BalancedSegmentMixer, get_segment_mixer
from losses import get_loss_function
from model_loaders import load_ss_model
from models.audiosep import AudioSep
from models.clap_encoder import CLAP_Encoder
from models.resunet import *
from optimizers.lr_schedulers import get_lr_lambda
from utils import get_dirs, get_data_module
from utils import parse_yaml

torch.set_float32_matmul_precision('high')
torch.cuda.amp.autocast(enabled=False)

def train(args, optuna_args={}) -> NoReturn:
    timestamp = time.time()
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

    # Read config file.
    configs = parse_yaml(config_yaml)

    task_name = configs['task_name']
    # Configuration of data
    sampling_rate = configs['data']['sampling_rate']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']

    # Configuration of the trainer
    num_nodes = configs['train']['num_nodes']
    batch_size = configs['train']['batch_size_per_device']
    logs_per_class = configs['train']['logs_per_class']
    sync_batchnorm = configs['train']['sync_batchnorm'] 
    num_workers = configs['train']['num_workers']
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']
    check_val_every_n_epoch = configs['train']['check_val_every_n_epoch']
    log_every_n_steps = configs['train']['log_every_n_steps']
    precision = configs["train"]['precision']
    max_mix_num = configs['data']['max_mix_num']
    loss_type = configs['train']['loss_type']
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    segment_mixer_type = configs["train"]['segment_mixer_type']

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

    device = torch.device('cuda')
    SS_CONFIG_PATH = './config/audiosep_base.yaml'
    CLAP_CKPT_PATH = './checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt'
    AUDIOSEP_CKPT_PATH = './checkpoint/audiosep_base_4M_steps.ckpt'

    loss_function = get_loss_function(loss_type)

    segment_mixer = get_segment_mixer(segment_mixer_type)(
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
    pl_model = AudioSep(
        waveform_mixer=segment_mixer,
        loss_function=loss_function,
        ss_model=model.ss_model,
        query_encoder=model.query_encoder,
        learning_rate = learning_rate,
        lr_lambda_func = lr_lambda_func,
        logs_per_class=logs_per_class,
        optimizer_type='fake',
    )

    wandb_logger = WandbLogger(name=f'{task_name}_{timestamp}', project='diploma')
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        num_nodes=num_nodes,
        precision=precision,
        logger=wandb_logger,
        fast_dev_run=False,
        max_epochs=-1,
        log_every_n_steps=log_every_n_steps,
        use_distributed_sampler=True,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        # accumulate_grad_batches=4,
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model,
        datamodule=data_module
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

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)