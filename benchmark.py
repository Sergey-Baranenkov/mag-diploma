import argparse
import os

import torch
import numpy as np
from evaluation.evaluate_audioset import AudioSetEvaluator
from evaluation.evaluate_audiocaps import AudioCapsEvaluator
from evaluation.evaluate_vggsound import VGGSoundEvaluator
from evaluation.evaluate_music import MUSICEvaluator
from evaluation.evaluate_esc50 import ESC50Evaluator
from evaluation.evaluate_clotho import ClothoEvaluator
from evaluation.evaluate_musdb18 import MUSDB18Evaluator
from models.audiosep_lora_and_tuned_embeddings import AudioSepLoraAndTunedEmbeddings
from models.audiosep_tunned_embeddings import AudioSepTunedEmbeddings
from models.audiosep_lora import AudioSepLora
from models.clap_encoder import CLAP_Encoder

from utils import (
    parse_yaml,
    get_mean_sdr_from_dict, get_layers, )
from model_loaders import load_ss_model


def get_model(name: str, checkpoint_path: str, device=torch.device('cuda')):
    match name:
        case 'audiosep':
            SS_CONFIG_PATH = './config/audiosep_base.yaml'
            CLAP_CKPT_PATH = './checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt'
            AUDIOSEP_CKPT_PATH = './checkpoint/audiosep_base_4M_steps.ckpt'

            query_encoder = CLAP_Encoder(
                pretrained_path=CLAP_CKPT_PATH).eval().to(device)

            configs = parse_yaml(SS_CONFIG_PATH)

            model = load_ss_model(
                configs=configs,
                checkpoint_path=AUDIOSEP_CKPT_PATH,
                query_encoder=query_encoder
            ) \
                .eval() \
                .to(device)

            return model
        case 'audiosep_embeddings':
            base_model = get_model('audiosep', '')

            model = AudioSepTunedEmbeddings.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                strict=False,
                ss_model=base_model.ss_model,
                query_encoder=base_model.query_encoder,
                waveform_mixer=None,
                loss_function=None,
                optimizer_type=None,
                learning_rate=None,
                lr_lambda_func=None,
            ) \
                .eval() \
                .to(device)

            return model

        case 'audiosep_lora':
            base_model = get_model('audiosep', '')
            model = AudioSepLora.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                strict=False,
                pretrained_audiosep_model=base_model,
                loss_function=None,
                waveform_mixer=None,
                lr_lambda_func=None
            ) \
                .eval() \
                .to(device)

            merged_model = model.model.merge_and_unload()
            merged_model.query_encoder = model.model.query_encoder
            return merged_model

        case 'audiosep_lora_and_embeddings':
            base_model = get_model('audiosep', '')
            model = AudioSepLoraAndTunedEmbeddings.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                strict=False,
                pretrained_audiosep_model=base_model,
                loss_function=None,
                waveform_mixer=None,
                lr_lambda_func=None
            ) \
                .eval() \
                .to(device)

            merged_model = model.model.merge_and_unload()
            merged_model.query_encoder = model.model.query_encoder
            return merged_model

    raise Exception('unknown model')


def benchmark(model_name: str, datasets_to_test: list[str], checkpoint_path: str):
    log_dir = 'eval_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Musdb18 Evaluator
    musdb18_evaluator = MUSDB18Evaluator()
    # AudioSet Evaluators
    audioset_evaluator = AudioSetEvaluator()
    # AudioCaps Evaluator
    audiocaps_evaluator = AudioCapsEvaluator()
    # VGGSound+ Evaluator
    vggsound_evaluator = VGGSoundEvaluator()
    # Clotho Evaluator
    clotho_evaluator = ClothoEvaluator()
    # MUSIC Evaluator
    music_evaluator = MUSICEvaluator()
    # ESC-50 Evaluator
    esc50_evaluator = ESC50Evaluator()

    pl_model = get_model(model_name, checkpoint_path)
    msgs = []
    print(f'-------  Start Evaluation  -------')

    if 'clotho' in datasets_to_test:
        # evaluation on Clotho
        SISDR, SDRi = clotho_evaluator(pl_model)
        msg_clotho = "Clotho Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_clotho)
        print(msg_clotho)

    if 'vgg' in datasets_to_test:
        # evaluation on VGGSound+ (YAN)
        SISDR, SDRi = vggsound_evaluator(pl_model)
        msg_vgg = "VGGSound Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_vgg)
        print(msg_vgg)

    if 'musdb18' in datasets_to_test:
        # evaluation on MUSIC
        SISDR, SDRi = musdb18_evaluator(pl_model)
        msg_musdb = "Musdb18 Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_musdb)
        print(msg_musdb)

    if 'music' in datasets_to_test:
        # evaluation on MUSIC
        SISDR, SDRi = music_evaluator(pl_model)
        msg_music = "MUSIC Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_music)
        print(msg_music)

    if 'esc50' in datasets_to_test:
        # evaluation on ESC-50
        SISDR, SDRi = esc50_evaluator(pl_model)
        msg_esc50 = "ESC-50 Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_esc50)
        print(msg_esc50)

    if 'audioset' in datasets_to_test:
        # evaluation on AudioSet
        stats_dict = audioset_evaluator(pl_model)
        median_sdris = {}
        median_sisdrs = {}

        for class_id in range(527):
            median_sdris[class_id] = np.nanmedian(stats_dict["sdris_dict"][class_id])
            median_sisdrs[class_id] = np.nanmedian(stats_dict["sisdrs_dict"][class_id])

        SDRi = get_mean_sdr_from_dict(median_sdris)
        SISDR = get_mean_sdr_from_dict(median_sisdrs)
        msg_audioset = "AudioSet Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_audioset)
        print(msg_audioset)

    if 'audiocaps' in datasets_to_test:
        # evaluation on AudioCaps
        SISDR, SDRi = audiocaps_evaluator(pl_model)
        msg_audiocaps = "AudioCaps Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        msgs.append(msg_audiocaps)
        print(msg_audiocaps)

    # open file in write mode
    log_path = os.path.join(log_dir, 'eval_results.txt')
    with open(log_path, 'w') as fp:
        for msg in msgs:
            fp.write(msg + '\n')
    print(f'Eval log is written to {log_path} ...')
    print('-------------------------  Done  ---------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to evaluate."
    )

    parser.add_argument(
        "--datasets_to_test", nargs='+', required=True, help="Name of the model to evaluate."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        default='',
        help="Path of pretrained checkpoint",
    )

    args = parser.parse_args()

    benchmark(args.model_name, args.datasets_to_test, args.checkpoint_path)

