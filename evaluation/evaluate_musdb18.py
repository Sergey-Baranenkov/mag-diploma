import csv
import os
import sys
from typing import Dict

import librosa
import lightning.pytorch as pl
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../AudioSep/')
from utils import (
    calculate_sdr,
    calculate_sisdr,
)


class MUSDB18Evaluator:
    def __init__(
            self,
            sampling_rate=32000
    ) -> None:

        self.sampling_rate = sampling_rate

        with open('evaluation/metadata/musdb18_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]

        self.eval_list = eval_list
        self.audio_dir = 'evaluation/data/musdb18/test'

        self.source_types = [
            "bass",
            "drums",
            "vocals",
           ]

    def __call__(
            self,
            pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        print(f'Evaluation on Musdb18 Test with [text label] queries.')

        pl_model.eval()
        device = pl_model.device

        sisdrs_list = {source_type: [] for source_type in self.source_types}
        sdris_list = {source_type: [] for source_type in self.source_types}

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):
                idx, src, caption, mxtr, = eval_data

                source_path = os.path.join(self.audio_dir, src)
                mixture_path = os.path.join(self.audio_dir, mxtr)

                source, fs = librosa.load(
                    source_path,
                    sr=self.sampling_rate,
                    mono=True,
                    duration=10,
                    offset=1
                )

                mixture, fs = librosa.load(
                    mixture_path,
                    sr=self.sampling_rate,
                    mono=True,
                    duration=10,
                    offset=1
                )

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                sep_segment = pl_model(caption, mixture, device)

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source, est=sep_segment)

                sisdrs_list[caption].append(sisdr)
                sdris_list[caption].append(sdri)

        mean_sisdr_list = []
        mean_sdri_list = []

        for source_class in self.source_types:
            sisdr = np.mean(sisdrs_list[source_class])
            sdri = np.mean(sdris_list[source_class])
            mean_sisdr_list.append(sisdr)
            mean_sdri_list.append(sdri)

        mean_sdri = np.mean(mean_sdri_list)
        mean_sisdr = np.mean(mean_sisdr_list)

        return mean_sisdr, mean_sdri
