import numpy as np
import torch

from model_loaders import load_ss_model
from models.audiosep_lora_and_tuned_embeddings import AudioSepLoraAndTunedEmbeddings
from models.clap_encoder import CLAP_Encoder
from models.interfaces import QueryEncoder
from pipeline import separate_audio
from utils import parse_yaml
import gc


def find_nearest_embedding(class_checkpoint_combinations: (str, str), query_embedding: np.ndarray, index, threshold: float):
    D, I = index.search(query_embedding, 1)

    if D[0][0] > threshold:
        return None, None, None

    return class_checkpoint_combinations[I[0][0]][0], class_checkpoint_combinations[I[0][0]][1], D[0][0]


def get_model(name: str, query_encoder: QueryEncoder, checkpoint_path: str, device=torch.device('cuda')):
    match name:
        case 'audiosep':
            SS_CONFIG_PATH = './config/audiosep_base.yaml'
            AUDIOSEP_CKPT_PATH = './checkpoint/audiosep_base_4M_steps.ckpt'

            configs = parse_yaml(SS_CONFIG_PATH)

            model = load_ss_model(
                configs=configs,
                checkpoint_path=AUDIOSEP_CKPT_PATH,
                query_encoder=query_encoder
            ) \
                .eval() \
                .to(device)

            return model

        case 'audiosep_lora_and_embeddings':
            base_model = get_model('audiosep', query_encoder, '')
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

            return model.model.merge_and_unload()
    raise Exception('unknown model')


class Separator:
    def __init__(self, index, class_checkpoint_combinations, distance_threshold: float = 1.0,
                 device=torch.device('cuda')):
        self.index = index
        self.class_checkpoint_combinations = class_checkpoint_combinations
        self.distance_threshold = distance_threshold

        self.device = device

        clap_ckpt_path = './checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt'
        ss_config_path = './config/audiosep_base.yaml'
        audiosep_ckpt_path = './checkpoint/audiosep_base_4M_steps.ckpt'
        configs = parse_yaml(ss_config_path)

        self.query_encoder = CLAP_Encoder(pretrained_path=clap_ckpt_path) \
            .eval() \
            .to(device)

        self.model = load_ss_model(
            configs=configs,
            checkpoint_path=audiosep_ckpt_path,
            query_encoder=self.query_encoder
        ) \
            .eval() \
            .to(device)

        self.current_ckpt = None

    def separate(self, src_file: str, output_file: str, caption: str):
        cls = self.reset_model_by_caption(caption)

        # используем ближайший класс если кастомная модель иначе caption пользователя
        resulting_caption = cls if cls is not None else caption

        separate_audio(self.model, src_file, resulting_caption, output_file, self.device, use_chunk=True)

        return resulting_caption

    def reset_model_by_caption(self, caption: str) -> str:
        query_embedding = self.query_encoder.get_query_embed('text', text=[caption], device=self.device).cpu().numpy()

        cls, ckpt_path, distance = find_nearest_embedding(
            self.class_checkpoint_combinations,
            query_embedding,
            self.index,
            self.distance_threshold
        )
        print(f'class is close to {cls} with distance = {distance}')
        if ckpt_path != self.current_ckpt:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

            if ckpt_path is None:
                self.model = get_model('audiosep', self.query_encoder, '', self.device)
            else:
                self.model = get_model('audiosep_lora_and_embeddings', self.query_encoder, ckpt_path)

            self.current_ckpt = ckpt_path
            print(f'reset current model ckpt to', self.current_ckpt)

        return cls