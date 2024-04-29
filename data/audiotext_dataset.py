import json
import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset


def pad_with_repetition(audio, sr, target_length_sec):
    # Рассчитываем текущую длительность в секундах
    current_length_sec = audio.shape[1] / sr

    # Проверяем, нужно ли дополнение
    if current_length_sec < target_length_sec:
        # Рассчитываем, сколько раз нужно повторить аудио
        repeats = int(torch.ceil(torch.tensor(target_length_sec / current_length_sec)))

        # Повторяем аудио
        repeated_audio = audio.repeat(1, repeats)

        # Обрезаем до нужной длительности
        target_length_samples = int(target_length_sec * sr)
        padded_audio = repeated_audio[:, :target_length_samples]
    else:
        padded_audio = audio

    return padded_audio


def pad_with_silence(audio, sr, target_length_sec):
    # Рассчитываем текущую длительность в секундах
    current_length_sec = audio.shape[1] / sr

    # Проверяем, нужно ли дополнение
    if current_length_sec < target_length_sec:
        # Рассчитываем количество отсчётов, которое нужно добавить
        silence_length = int((target_length_sec - current_length_sec) * sr)

        # Создаём тензор тишины
        silence = torch.zeros((audio.shape[0], silence_length))

        # Соединяем исходный аудио с тишиной
        padded_audio = torch.cat((audio, silence), dim=1)
    else:
        padded_audio = audio

    return padded_audio


class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """

    def __init__(
            self,
            datafiles=[''],
            sampling_rate=32000,
            max_clip_len=5,
            with_mixture: bool = False
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json
        self.max_clip_len = max_clip_len
        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate
        self.with_mixture = with_mixture

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform, predefined_random_idx=None):
        # waveform: [1, samples]
        # random crop
        random_idx = 0
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1) - self.max_length) \
                if predefined_random_idx is None \
                else predefined_random_idx

            waveform = waveform[:, random_idx:random_idx + self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform, random_idx

    def _read_audio(self, index):
        try:
            audio_path = self.all_data_json[index]['wav']
            audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)
            # опциональная строчка, иначе заполнится тишиной в момент _cut_or_randomcrop
            audio_data = pad_with_repetition(audio_data, audio_rate, self.max_clip_len)
            text = self.all_data_json[index]['caption']

            # drop short utterance
            if audio_data.size(1) < self.sampling_rate * 1:
                raise Exception(f'{audio_path} is too short, drop it ..., {audio_data.size()}, {self.sampling_rate}')

            return text, audio_data, audio_rate

        except Exception as e:
            print(f'error: {e} occurs, when loading {audio_path}')
            random_index = random.randint(0, len(self.all_data_json) - 1)
            return self._read_audio(index=random_index)

    def _read_audio_mixture(self, index):
        audio_path = self.all_data_json[index]['wav']
        audio_dir = os.path.dirname(audio_path)
        mixture_path = os.path.join(audio_dir, 'mixture.wav')
        audio_data, audio_rate = torchaudio.load(mixture_path, channels_first=True)
        # опциональная строчка, иначе заполнится тишиной в момент _cut_or_randomcrop
        audio_data = pad_with_repetition(audio_data, audio_rate, self.max_clip_len)

        # drop short utterance
        if audio_data.size(1) < self.sampling_rate * 1:
            raise Exception(f'{audio_path} is too short, drop it ..., {audio_data.size()}, {self.sampling_rate}')

        return audio_data, audio_rate

    def _handle_audio(self, audio_data, audio_rate, random_idx=None):
        if audio_data.shape[0] > 1:
            # audio_data: [samples]
            audio_data = (audio_data[0] + audio_data[1]) / 2
        else:
            audio_data = audio_data.squeeze(0)

        # resample audio clip
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)

        audio_data = audio_data.unsqueeze(0)

        audio_data, random_idx = self._cut_or_randomcrop(audio_data, random_idx)

        return audio_data, audio_rate, random_idx

    def __getitem__(self, index):
        # create a source data
        text, audio_data, audio_rate = self._read_audio(index)
        audio_data, audio_rate, source_random_idx = self._handle_audio(audio_data, audio_rate)

        data_dict = {
            'text': text,
            'waveform': audio_data,
            'modality': 'audio_text',
            'mixture': None
        }

        if self.with_mixture:
            mixture_audio_data, mixture_audio_rate = self._read_audio_mixture(index)
            mixture_audio_data, mixture_audio_rate, _ = self._handle_audio(mixture_audio_data, mixture_audio_rate,
                                                                           source_random_idx)
            data_dict['mixture'] = mixture_audio_data

        return data_dict

#%%

