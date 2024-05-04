import argparse
import logging
import os
import pathlib
import sys
import time

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.tensorboard import SummaryWriter

from callbacks.base import CheckpointEveryNSteps
from data.datamodules import *
from data.waveform_mixers import SegmentMixer, BalancedSegmentMixer, get_segment_mixer
from losses import get_loss_function
from models.audiosep_tunned_embeddings import AudioSepTunedEmbeddings
from models.clap_encoder import CLAP_Encoder
from models.resunet import *
from optimizers.lr_schedulers import get_lr_lambda
from utils import create_logging, get_dirs, get_data_module
from utils import parse_yaml
from model_loaders import load_ss_model
import optuna
from optuna.integration import PyTorchLightningPruningCallback

torch.set_float32_matmul_precision('high')

def objective(trial: optuna.trial.Trial, args):
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical("learning_rate", [1e-3, 1e-4, 1e-5])
    loss_type = trial.suggest_categorical("loss_type", ["si_sdr", "l1_wav"])
    max_mix_num = trial.suggest_categorical("max_mix_num", [2, 3, 4])

    # Call the modified training function with suggested parameters
    return train(
        args,
        {
            'learning_rate': learning_rate,
            'loss_type': loss_type,
            'max_mix_num': max_mix_num,
            'trial': trial,
            'max_epochs': 30
        })

def train(args, optuna_args={}) -> NoReturn:
    timestamp = args.timestamp
    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = torch.cuda.device_count()
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
    save_epoch_frequency = configs['train']['save_epoch_frequency']
    check_val_every_n_epoch = configs['train']['check_val_every_n_epoch']
    log_every_n_steps = configs['train']['log_every_n_steps']
    checkpoint_filename_args = configs["train"]["checkpoint_filename_args"]

    max_mix_num = optuna_args.get('max_mix_num') or configs['data']['max_mix_num']
    loss_type = optuna_args.get('loss_type') or configs['train']['loss_type']
    learning_rate = optuna_args.get('learning_rate') or float(configs['train']["optimizer"]['learning_rate'])
    precision = configs["train"]['precision']
    segment_mixer_type = configs["train"]['segment_mixer_type']
    resume_checkpoint_path = args.resume_checkpoint_path
    if resume_checkpoint_path == "":
        resume_checkpoint_path = None
    else:
        logging.info(f'Finetuning AudioSep with checkpoint [{resume_checkpoint_path}]')

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num, checkpoint_filename_args, timestamp,
    )

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
        pretrained_path=CLAP_CKPT_PATH)\
        .eval()\
        .to(device)


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
        lr_lambda_func = lr_lambda_func,
        logs_per_class=logs_per_class,
    )

    checkpoint_every_n_epochs = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='{epoch}',
        every_n_epochs=save_epoch_frequency,
        save_top_k=-1
    )

    callbacks = [checkpoint_every_n_epochs]
    if optuna_args.get('trial') is not None:
        early_stopping = PyTorchLightningPruningCallback(optuna_args['trial'], monitor="val_si_sdr_avg")
        callbacks.append(early_stopping)

    wandb_logger = WandbLogger(name=f'{task_name}_{checkpoint_filename_args}_{timestamp}', project='diploma')
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        num_nodes=num_nodes,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=optuna_args.get('max_epochs') or 50,
        log_every_n_steps=log_every_n_steps,
        use_distributed_sampler=True,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model,
        datamodule=data_module,
        ckpt_path=resume_checkpoint_path if resume_checkpoint_path else None,
    )

    val_si_sdr_avg_loss = trainer.callback_metrics["val_si_sdr_avg"]
    return val_si_sdr_avg_loss


if __name__ == "__main__":
    timestamp = time.time()
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
    args.timestamp = timestamp
    # # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study_name = f'{args.config_yaml}_{timestamp}'
    # storage_name = "sqlite:///optuna.db"
    #
    # study = optuna.create_study(direction='maximize', storage=storage_name, study_name=study_name)
    # study.optimize(lambda trial: objective(trial, args), n_trials=100)
    # print("Best trial:", study.best_trial.params)

    train(args)