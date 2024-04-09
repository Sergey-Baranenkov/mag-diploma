import csv
import os
import librosa
import torch
import numpy as np
from tqdm import tqdm

# Предполагается, что calculate_sdr и calculate_sisdr определены в utils
from utils import calculate_sdr, calculate_sisdr

class MUSDB18Evaluator:
    def __init__(self,
                 sampling_rate=32000,
                 audio_dir='evaluation/data/musdb18/test',
                 metadata_path='evaluation/metadata/musdb18_eval.csv'):
        self.sampling_rate = sampling_rate
        self.audio_dir = audio_dir

        # Загрузка метаданных для оценки
        with open(metadata_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            self.eval_list = [row for row in csv_reader][1:]

        self.source_types = ["bass", "drums", "vocals"]

    def __call__(self, model):
        model.eval()
        device = model.device

        sisdrs_list = {source_type: [] for source_type in self.source_types}
        sdris_list = {source_type: [] for source_type in self.source_types}

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list, desc="Evaluating"):
                idx, src, caption, mxtr = eval_data
                print(mxtr, caption)

                source_path = os.path.join(self.audio_dir, src)
                mixture_path = os.path.join(self.audio_dir, mxtr)

                source, _ = librosa.load(source_path, sr=self.sampling_rate, mono=True, duration=5, offset=1)
                mixture, _ = librosa.load(mixture_path, sr=self.sampling_rate, mono=True, duration=5, offset=1)

                # Подготовка данных для модели
                mixture_tensor = torch.tensor(mixture).to(device)
                source_tensor = torch.tensor(source).float().to(device)

                # Инференс модели
                sep_segment = model(prompt=caption, mixture=mixture_tensor, device=device)

                # Вычисление метрик
                sdr_no_sep = calculate_sdr(ref=source_tensor.cpu().numpy(), est=mixture)
                sdr = calculate_sdr(ref=source_tensor.cpu().numpy(), est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source_tensor.cpu().numpy(), est=sep_segment)

                sisdrs_list[caption].append(sisdr)
                sdris_list[caption].append(sdri)

        # Вычисление и вывод средних метрик
        mean_sisdr_list = []
        mean_sdri_list = []

        for source_class in self.source_types:
            sisdr = np.mean(sisdrs_list[source_class])
            sdri = np.mean(sdris_list[source_class])
            print(f"Class {source_class} Mean SI-SDR: {sisdr}, Mean SDRI: {sdri}")
            mean_sisdr_list.append(sisdr)
            mean_sdri_list.append(sdri)

        mean_sdri = np.mean(mean_sdri_list)
        mean_sisdr = np.mean(mean_sisdr_list)

        print(f"Mean SI-SDR: {mean_sisdr}, Mean SDRI: {mean_sdri}")
        return mean_sisdr, mean_sdri